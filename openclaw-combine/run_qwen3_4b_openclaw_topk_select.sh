#!/bin/bash

# Multi-candidate top-K OPD launcher for openclaw-combine.
#
# Drop-in counterpart to ``run_qwen3_4b_openclaw_combine.sh`` that swaps:
#   * rollout fn   :  openclaw_combine_rollout.generate_rollout_openclaw_combine
#                  -> openclaw_combine_select_rollout.
#                     generate_rollout_openclaw_combine_select
#   * loss fn      :  combine_loss.combine_loss_function
#                  -> openclaw_topk_select_loss.openclaw_topk_select_loss_function
#   * api server   :  OpenClawCombineAPIServer
#                  -> OpenClawCombineSelectAPIServer  (multi-cand hints)
#
# 9-cell support matrix (--distill-subset-mode × --hint-selection):
#                          shortest   token_optimal   sequence_optimal
#   student              :    ✓             ✓                ✓
#   overlap              :    ✓             ✓                ✓
#   teacher              :    ✓             ✓                ✓
# All cells handled by openclaw_topk_select_loss_function (see its
# docstring for the full equations and the in-kernel vs actor-side
# k* selection split).
#
# torch_dist conversion (if you don't already have these directories):
#
#   # student: Qwen3-4B-Thinking-2507
#   cd /data_storage/wyj/OpenClaw-RL/slime
#   source scripts/models/qwen3-4B.sh
#   PYTHONPATH=/data_storage/wyj/OpenClaw-RL/Megatron-LM \
#     python tools/convert_hf_to_torch_dist.py \
#       ${MODEL_ARGS[@]} \
#       --hf-checkpoint /data_storage/wyj/systems/huggingface/hub/Qwen3-4B-Thinking-2507 \
#       --rotary-base 5000000 \
#       --save /data_storage/wyj/systems/huggingface/hub/Qwen3-4B-Thinking-2507_torch_dist

pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

NUM_GPUS=${NUM_GPUS:-8}
ACTOR_GPUS=${ACTOR_GPUS:-4}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-2}

# topk-select REQUIRES megatron PRM teacher -- the inference-side
# teacher path computes single-cand teacher_log_probs only and does
# not produce per-cand top-K. Force the megatron layout (1 GPU PRM
# SGLang + 1 GPU Megatron teacher).
export OPENCLAW_COMBINE_OPD_TEACHER_SOURCE="megatron"
PRM_GPUS=${PRM_GPUS:-1}
PRM_NUM_GPUS_PER_ENGINE=${PRM_NUM_GPUS_PER_ENGINE:-1}
PRM_TEACHER_GPUS=${PRM_TEACHER_GPUS:-1}

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
source "${SLIME_ROOT}/scripts/models/qwen3-4B.sh"

# Student (HF for tokenizer + SGLang) and student-torch_dist (Megatron load).
HF_CKPT=${HF_CKPT:-/data_storage/wyj/systems/huggingface/hub/Qwen3-4B-Thinking-2507}
REF_LOAD=${REF_LOAD:-/data_storage/wyj/systems/huggingface/hub/Qwen3-4B-Thinking-2507-_torch_dist}
SAVE_CKPT=${SAVE_CKPT:-/data_storage/wyj/OpenClaw-RL/ckpt/qwen3-4b-openclaw-topk-select}

# PRM teacher: same family as the student in this setting (Qwen3-4B
# Thinking variant). Megatron teacher loads from torch_dist; SGLang PRM
# loads from HF (it cannot load torch_dist). Override either via env.
PRM_MODEL_PATH=${PRM_MODEL_PATH:-${HF_CKPT}}
PRM_TEACHER_LOAD=${PRM_TEACHER_LOAD:-${REF_LOAD}}
PRM_TEACHER_HF=${PRM_TEACHER_HF:-${HF_CKPT}}

export SGLANG_API_KEY="${SGLANG_API_KEY}"
export SERVED_MODEL_NAME="qwen3-4b"
export HOST="0.0.0.0"
export PORT="30000"
export OPENCLAW_RECORD_ENABLED="${OPENCLAW_RECORD_ENABLED:-1}"  # 0=off, 1=on
export OPENCLAW_RECORD_FILE="${SCRIPT_DIR}/results/qwen3_4b_topk_select_record.jsonl"
export TP="2"
export CONTEXT_LENGTH="32768"
export MEM_FRACTION_STATIC="0.8"
export REASONING_PARSER="qwen3"
export TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen25}"
export PRM_M="${PRM_M:-3}"  # judge votes per turn; topk-select keeps all accepted
export OPENCLAW_OPD_TEACHER_LP_MAX_CONCURRENCY="${OPENCLAW_OPD_TEACHER_LP_MAX_CONCURRENCY:-1}"

# combine_loss-style RL+OPD weighting
export OPENCLAW_TOPK_W_RL="${OPENCLAW_TOPK_W_RL:-1.0}"
export OPENCLAW_TOPK_W_OPD="${OPENCLAW_TOPK_W_OPD:-1.0}"
# clip is important in stabilizing the training
export OPENCLAW_TOPK_ADV_DIFF_CLIP="${OPENCLAW_TOPK_ADV_DIFF_CLIP:-1.0}"
export TRAIN_EPOCHS="${TRAIN_EPOCHS:-1}"

# Subset S_t selection mode for the OPD loss kernel. See
# openclaw_topk_select_loss docstring for the equations.
#   student : S_t = top-K(π_old)
#   teacher : S_t = top-K(π_T,k*)
#   overlap : S_t = top-K(π_old) ∩ top-K(π_T,k*)
OPENCLAW_TOPK_SUBSET_MODE="${OPENCLAW_TOPK_SUBSET_MODE:-student}"

# Which generated hint will be used for supervision? Three different selection modes:
#   shortest          : k* = 0 always (cand list is sorted shortest-first
#                       at the API server). M=3 candidates are still
#                       generated but only the shortest drives supervision.
#   sequence_optimal  : per-Sample argmax_k Σ_t |S^q_t ∩ S^p_{t,k}|.
#                       the best hint is the one that maximizes the overlap between the student and teacher top k in sequence level.
#   token_optimal     : k*(t) = argmax_k |S^q_t ∩ S^p_{t,k}| per token.
#                       the best hint is the one that maximizes the overlap between the student and teacher top k in token level. Empirically, these two methods have similar performance.
OPENCLAW_TOPK_HINT_SELECTION="${OPENCLAW_TOPK_HINT_SELECTION:-sequence_optimal}"

# Top-K width on both student and teacher sides.
OPENCLAW_TOPK_K="${OPENCLAW_TOPK_K:-4}"
# Max #candidate hints kept per turn at the API server 
export OPENCLAW_TOPK_MAX_CAND="${OPENCLAW_TOPK_MAX_CAND:-3}"

CKPT_ARGS=(
   --hf-checkpoint "${HF_CKPT}"
   --ref-load "${REF_LOAD}"
   --save "${SAVE_CKPT}"
   --save-interval 100
   # Qwen3 rope_theta (matches the qwen3-4B model script).
   --rotary-base 5000000
   --prm-teacher-load "${PRM_TEACHER_LOAD}"
   --prm-teacher-num-gpus "${PRM_TEACHER_GPUS}"
   --prm-teacher-hf-checkpoint "${PRM_TEACHER_HF}"
   # Teacher rope (same family as student in this setting).
   --prm-teacher-rotary-base 5000000
)

ROLLOUT_ARGS=(
   --disable-rollout-global-dataset
   --rollout-function-path openclaw_combine_select_rollout.generate_rollout_openclaw_combine_select

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
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 32768
   --log-probs-chunk-size 1024
)

TOPK_SELECT_ARGS=(
   --advantage-estimator grpo
   --disable-rewards-normalization
   --loss-type custom_loss
   --custom-loss-function-path openclaw_topk_select_loss.openclaw_topk_select_loss_function
   --distill-topk "${OPENCLAW_TOPK_K}"
   --distill-subset-mode "${OPENCLAW_TOPK_SUBSET_MODE}"
   --hint-m "${OPENCLAW_TOPK_MAX_CAND}"
   --hint-selection "${OPENCLAW_TOPK_HINT_SELECTION}"
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
   --sglang-reasoning-parser qwen3
)

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
    --wandb-group qwen3-4b-openclaw-topk-select
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
    \"PYTHONPATH\": \"${REPO_ROOT}/Megatron-LM:${SCRIPT_DIR}:${REPO_ROOT}/openclaw-opd:${REPO_ROOT}/hint_opt_exp:${SLIME_ROOT}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"OPENCLAW_EVAL_MODE\": \"${OPENCLAW_EVAL_MODE}\",
    \"OPENCLAW_COMBINE_OPD_TEACHER_SOURCE\": \"${OPENCLAW_COMBINE_OPD_TEACHER_SOURCE}\",
    \"OPENCLAW_TOPK_W_RL\": \"${OPENCLAW_TOPK_W_RL}\",
    \"OPENCLAW_TOPK_W_OPD\": \"${OPENCLAW_TOPK_W_OPD}\",
    \"OPENCLAW_TOPK_ADV_DIFF_CLIP\": \"${OPENCLAW_TOPK_ADV_DIFF_CLIP}\",
    \"OPENCLAW_TOPK_MAX_CAND\": \"${OPENCLAW_TOPK_MAX_CAND}\",
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
   ${TOPK_SELECT_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${CUSTOM_ARGS[@]} \
   ${PRM_ARGS[@]}
