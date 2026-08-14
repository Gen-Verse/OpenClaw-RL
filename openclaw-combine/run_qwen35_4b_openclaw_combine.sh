#!/bin/bash
# Qwen3.5-4B OpenClaw combined training (Binary RL + OPD).

SKIP_CLUSTER_CLEANUP=${SKIP_CLUSTER_CLEANUP:-0}
if [ "${SKIP_CLUSTER_CLEANUP}" != "1" ]; then
  pkill -9 sglang
  sleep 3
  ray stop --force
  pkill -9 ray
  pkill -9 python
  sleep 3
  pkill -9 ray
  pkill -9 python
fi

set -ex

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-/tmp}"

NUM_GPUS=${NUM_GPUS:-8}
ACTOR_GPUS=${ACTOR_GPUS:-4}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-2}

# OPD teacher source: "megatron" = 1+1 PRM/teacher split; "inference" = 2+0.
OPD_SRC="${OPENCLAW_COMBINE_OPD_TEACHER_SOURCE:-megatron}"
if [ "${OPD_SRC}" = "inference" ]; then
    PRM_GPUS=${PRM_GPUS:-2}
    PRM_NUM_GPUS_PER_ENGINE=${PRM_NUM_GPUS_PER_ENGINE:-2}
    PRM_TEACHER_GPUS=${PRM_TEACHER_GPUS:-0}
else
    PRM_GPUS=${PRM_GPUS:-1}
    PRM_NUM_GPUS_PER_ENGINE=${PRM_NUM_GPUS_PER_ENGINE:-1}
    PRM_TEACHER_GPUS=${PRM_TEACHER_GPUS:-1}
fi

if (( ACTOR_GPUS + ROLLOUT_GPUS + PRM_GPUS + PRM_TEACHER_GPUS > NUM_GPUS )); then
    echo "ACTOR_GPUS + ROLLOUT_GPUS + PRM_GPUS + PRM_TEACHER_GPUS must be <= NUM_GPUS"
    echo "ACTOR_GPUS=${ACTOR_GPUS}, ROLLOUT_GPUS=${ROLLOUT_GPUS}, PRM_GPUS=${PRM_GPUS}, PRM_TEACHER_GPUS=${PRM_TEACHER_GPUS}, NUM_GPUS=${NUM_GPUS}"
    exit 1
fi

export RAY_health_check_failure_threshold=20
export RAY_health_check_period_ms=5000
export RAY_health_check_timeout_ms=30000
export RAY_num_heartbeats_timeout=60

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
SLIME_ROOT="${REPO_ROOT}/slime"
source "${SLIME_ROOT}/scripts/models/qwen3.5-4B.sh"

HF_CKPT=${HF_CKPT:-/data_storage/wyj/systems/huggingface/hub/Qwen35-4B}
REF_LOAD=${REF_LOAD:-/data_storage/wyj/systems/huggingface/hub/qwen35-4b_torch_dist}
SAVE_CKPT=${SAVE_CKPT:-${REPO_ROOT}/ckpt/qwen35-4b-openclaw-combine}
PRM_MODEL_PATH=${PRM_MODEL_PATH:-/data_storage/wyj/systems/huggingface/hub/Qwen35-4B}
PRM_TEACHER_LOAD=${PRM_TEACHER_LOAD:-${REF_LOAD}}

export SGLANG_API_KEY="${SGLANG_API_KEY}"
export SERVED_MODEL_NAME="qwen3.5-4b"
export HOST="0.0.0.0"
export PORT="30000"
export OPENCLAW_RECORD_ENABLED="${OPENCLAW_RECORD_ENABLED:-1}"
export OPENCLAW_RECORD_FILE="${SCRIPT_DIR}/results/qwen35_4b_record.jsonl"
export TP="2"
export CONTEXT_LENGTH="32768"
export MEM_FRACTION_STATIC="0.8"
export REASONING_PARSER="${REASONING_PARSER:-qwen3}"
export TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"
export SGLANG_LANGUAGE_ONLY="${SGLANG_LANGUAGE_ONLY:-1}"
# Qwen3.5 must use the raw spec path (slime_plugins/models/qwen3_5.py) to
# preserve the hybrid linear/full-attention layout. Do NOT set
# SLIME_QWEN35_TEXT_ONLY_BRIDGE or --megatron-to-hf-mode bridge here.
export PRM_M="${PRM_M:-1}"
export OPENCLAW_OPD_TEACHER_LP_MAX_CONCURRENCY="${OPENCLAW_OPD_TEACHER_LP_MAX_CONCURRENCY:-1}"
export OPENCLAW_COMBINE_W_RL="${OPENCLAW_COMBINE_W_RL:-1.0}"
export OPENCLAW_COMBINE_W_OPD="${OPENCLAW_COMBINE_W_OPD:-1.0}"
export TRAIN_EPOCHS="${TRAIN_EPOCHS:-2}"

export OPENCLAW_COMBINE_OPD_TEACHER_SOURCE="${OPD_SRC}"

CKPT_ARGS=(
   --hf-checkpoint "${HF_CKPT}"
   --ref-load "${REF_LOAD}"
   --save "${SAVE_CKPT}"
   --save-interval 100
   --rotary-base 10000000
)
if [ "${PRM_TEACHER_GPUS}" -gt 0 ]; then
    CKPT_ARGS+=(
        --prm-teacher-load "${PRM_TEACHER_LOAD}"
        --prm-teacher-num-gpus "${PRM_TEACHER_GPUS}"
    )
fi

ROLLOUT_ARGS=(
   --disable-rollout-global-dataset
   --rollout-function-path openclaw_combine_rollout.generate_rollout_openclaw_combine

   --num-rollout 100000000
   --rollout-batch-size 16
   --n-samples-per-prompt 1
   --rollout-max-response-len 8192
   --rollout-max-context-len 32768
   --rollout-temperature 0.6
   --reward-key score

   --num-steps-per-rollout 1
)

PERF_ARGS=(
   # TP=4 + SP requires the gradient fix in slime_plugins/models/hf_attention.py
   # for the hybrid linear-attention layers.
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # Stay on thd (dynamic batching). The spec path requires
   # packed_seq_params.cu_seqlens_q, which bshd does not provide.
   --use-dynamic-batch-size
   --max-tokens-per-gpu 32768
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
   --lr 1e-5
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
   --rollout-num-gpus-per-engine 2
   --sglang-tool-call-parser "${TOOL_CALL_PARSER}"
   --sglang-mem-fraction-static 0.8
   --sglang-context-length 32768
   --sglang-reasoning-parser "${REASONING_PARSER}"
)

if [ "${SGLANG_LANGUAGE_ONLY}" = "1" ]; then
  SGLANG_ARGS+=(--sglang-language-only)
fi

PRM_ARGS=(
   --prm-enable
   --prm-num-gpus "${PRM_GPUS}"
   --prm-num-gpus-per-engine "${PRM_NUM_GPUS_PER_ENGINE}"
   --prm-model-path "${PRM_MODEL_PATH}"
   --prm-m "${PRM_M}"
   --prm-temperature "${PRM_TEMPERATURE:-0.6}"
   --prm-max-new-tokens "${PRM_MAX_NEW_TOKENS:-8192}"
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

USE_WANDB=${USE_WANDB:-1}
WANDB_PROJECT=${WANDB_PROJECT:-openclaw_rl}
WANDB_KEY_VALUE=${WANDB_KEY:-${WANDB_API_KEY:-}}
if [ "${USE_WANDB}" = "1" ] && [ -n "${WANDB_KEY_VALUE}" ]; then
  WANDB_ARGS=(
    --use-wandb
    --wandb-project ${WANDB_PROJECT}
    --wandb-group qwen35-4b-openclaw-combine
    --wandb-key ${WANDB_KEY_VALUE}
  )
else
  WANDB_ARGS=()
fi

export OPENCLAW_EVAL_MODE="${OPENCLAW_EVAL_MODE:-1}"

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export no_proxy="127.0.0.1,${MASTER_ADDR}"
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${REPO_ROOT}/Megatron-LM:${SCRIPT_DIR}:${REPO_ROOT}/openclaw-opd:${SLIME_ROOT}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"FLASHINFER_WORKSPACE_BASE\": \"${FLASHINFER_WORKSPACE_BASE}\",
    \"OPENCLAW_EVAL_MODE\": \"${OPENCLAW_EVAL_MODE}\",
    \"OPENCLAW_COMBINE_W_RL\": \"${OPENCLAW_COMBINE_W_RL}\",
    \"OPENCLAW_COMBINE_W_OPD\": \"${OPENCLAW_COMBINE_W_OPD}\",
    \"OPENCLAW_COMBINE_OPD_TEACHER_SOURCE\": \"${OPENCLAW_COMBINE_OPD_TEACHER_SOURCE}\",
    \"TRAIN_EPOCHS\": \"${TRAIN_EPOCHS}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 "${SLIME_ROOT}/train_async.py" \
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
