#!/bin/bash

# Qwen3-0.6B minimal-GPU OpenClaw Combine (Binary RL + OPD), Megatron backend.
#
# Simplifications vs the 8-GPU 4B script:
#   * actor TP=1 (0.6B full-param fits easily on one 24GB card)
#   * rollout SGLang on 1 GPU
#   * one local PRM SGLang engine serves BOTH the judge (hint gen + eval vote)
#     AND the OPD teacher log-probs (OPENCLAW_COMBINE_OPD_TEACHER_SOURCE=inference),
#     so no separate Megatron teacher GPU is needed
#   => total 3 GPUs. For GRPO-only, drop PRM entirely and use 2 GPUs.
#
# Note: topk-select (run_qwen3_4b_openclaw_topk_select.sh) still requires a
# Megatron teacher GPU; this script uses the plain combine loss instead.

pkill -9 sglang || true
sleep 3
ray stop --force || true
pkill -9 ray || true
pkill -9 python || true
sleep 3

set -ex

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

NUM_GPUS=${NUM_GPUS:-3}
ACTOR_GPUS=${ACTOR_GPUS:-1}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-1}

# Inference-side teacher: teacher log-probs come from the PRM SGLang engine.
export OPENCLAW_COMBINE_OPD_TEACHER_SOURCE="${OPENCLAW_COMBINE_OPD_TEACHER_SOURCE:-inference}"
PRM_GPUS=${PRM_GPUS:-1}
PRM_NUM_GPUS_PER_ENGINE=${PRM_NUM_GPUS_PER_ENGINE:-1}
PRM_TEACHER_GPUS=0

if (( ACTOR_GPUS + ROLLOUT_GPUS + PRM_GPUS > NUM_GPUS )); then
    echo "ACTOR_GPUS + ROLLOUT_GPUS + PRM_GPUS must be <= NUM_GPUS"
    exit 1
fi

export RAY_health_check_failure_threshold=20
export RAY_health_check_period_ms=5000
export RAY_health_check_timeout_ms=30000
export RAY_num_heartbeats_timeout=60

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
SLIME_ROOT="${REPO_ROOT}/slime"
source "${SLIME_ROOT}/scripts/models/qwen3-0.6B.sh"

HF_CKPT=${HF_CKPT:-"${REPO_ROOT}/models/qwen3-0.6B"}
REF_LOAD=${REF_LOAD:-${HF_CKPT}}
SAVE_CKPT=${SAVE_CKPT:-"${REPO_ROOT}/export/ckpt/qwen3-0.6b-openclaw-combine-3gpu"}
# Judge = same 0.6B checkpoint; no larger PRM model needed.
PRM_MODEL_PATH=${PRM_MODEL_PATH:-${HF_CKPT}}

export SGLANG_API_KEY="${SGLANG_API_KEY:-}"
export SERVED_MODEL_NAME="qwen3-0.6b"
export HOST="0.0.0.0"
export PORT="30000"
export OPENCLAW_RECORD_ENABLED="${OPENCLAW_RECORD_ENABLED:-1}"
export OPENCLAW_RECORD_FILE="${SCRIPT_DIR}/results/qwen3_0p6b_combine_3gpu_record.jsonl"
export TP="1"
export CONTEXT_LENGTH="16384"
export MEM_FRACTION_STATIC="0.6"
export REASONING_PARSER="qwen3"
export TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen25}"
export PRM_M="${PRM_M:-1}"
export OPENCLAW_OPD_TEACHER_LP_MAX_CONCURRENCY="${OPENCLAW_OPD_TEACHER_LP_MAX_CONCURRENCY:-1}"
export OPENCLAW_COMBINE_W_RL="${OPENCLAW_COMBINE_W_RL:-1.0}"
export OPENCLAW_COMBINE_W_OPD="${OPENCLAW_COMBINE_W_OPD:-1.0}"
export TRAIN_EPOCHS="${TRAIN_EPOCHS:-1}"

CKPT_ARGS=(
   --megatron-to-hf-mode bridge
   --hf-checkpoint "${HF_CKPT}"
   --ref-load "${REF_LOAD}"
   --save "${SAVE_CKPT}"
   --save-interval 100
)

ROLLOUT_ARGS=(
   --disable-rollout-global-dataset
   --rollout-function-path openclaw_combine_rollout.generate_rollout_openclaw_combine
   --num-rollout ${NUM_ROLLOUT:-100000000}
   --rollout-batch-size ${ROLLOUT_BATCH_SIZE:-16}
   --n-samples-per-prompt 1
   --rollout-max-response-len ${ROLLOUT_MAX_RESPONSE_LEN:-8192}
   --rollout-max-context-len ${ROLLOUT_MAX_CONTEXT_LEN:-16384}
   --rollout-temperature ${ROLLOUT_TEMPERATURE:-0.6}
   --reward-key score
   --num-steps-per-rollout 1
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU:-16384}
   --log-probs-chunk-size 1024
)

COMBINE_ARGS=(
   --advantage-estimator grpo
   --disable-rewards-normalization
   --loss-type custom_loss
   --custom-loss-function-path combine_loss.combine_loss_function
   --use-kl-loss
   --kl-loss-coef 0.0
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr ${LR:-1e-5}
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

EVAL_ARGS=()

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-tool-call-parser "${TOOL_CALL_PARSER}"
   --sglang-mem-fraction-static 0.6
   --sglang-context-length 16384
   --sglang-reasoning-parser qwen3
)

PRM_ARGS=(
   --prm-enable
   --prm-num-gpus "${PRM_GPUS}"
   --prm-num-gpus-per-engine "${PRM_NUM_GPUS_PER_ENGINE}"
   --prm-model-path "${PRM_MODEL_PATH}"
   --prm-m "${PRM_M}"
   --prm-temperature "${PRM_TEMPERATURE:-0.6}"
   --prm-max-new-tokens ${PRM_MAX_NEW_TOKENS:-4096}
)

CUSTOM_ARGS=(
   --custom-generate-function-path openclaw_combine_api_server.generate
   --custom-rm-path openclaw_combine_api_server.reward_func
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

WANDB_ARGS=()
if [[ "${USE_WANDB:-0}" == "1" && -n "${WANDB_API_KEY:-}" ]]; then
  WANDB_ARGS=(
    --use-wandb
    --wandb-project ${WANDB_PROJECT:-openclaw_rl}
    --wandb-group qwen3-0p6b-openclaw-combine-3gpu
    --wandb-key ${WANDB_API_KEY}
  )
fi

export OPENCLAW_EVAL_MODE="${OPENCLAW_EVAL_MODE:-1}"

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export no_proxy="127.0.0.1,${MASTER_ADDR}"
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${REPO_ROOT}/Megatron-LM:${SCRIPT_DIR}:${REPO_ROOT}/openclaw-opd:${SLIME_ROOT}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"OPENCLAW_EVAL_MODE\": \"${OPENCLAW_EVAL_MODE}\",
    \"OPENCLAW_COMBINE_W_RL\": \"${OPENCLAW_COMBINE_W_RL}\",
    \"OPENCLAW_COMBINE_W_OPD\": \"${OPENCLAW_COMBINE_W_OPD}\",
    \"OPENCLAW_COMBINE_OPD_TEACHER_SOURCE\": \"${OPENCLAW_COMBINE_OPD_TEACHER_SOURCE}\",
    \"TRAIN_EPOCHS\": \"${TRAIN_EPOCHS}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train_async.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "${ACTOR_GPUS}" \
   --rollout-num-gpus "${ROLLOUT_GPUS}" \
   --num-gpus-per-node "${NUM_GPUS}" \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${COMBINE_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${CUSTOM_ARGS[@]} \
   ${PRM_ARGS[@]}
