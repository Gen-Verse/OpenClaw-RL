#!/bin/bash

pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

# keep stdout/stderr unbuffered in ray jobs
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

NUM_GPUS=${NUM_GPUS:-1}
ACTOR_GPUS=${ACTOR_GPUS:-1}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-1}
PRM_GPUS=${PRM_GPUS:-0}
PRM_PROVIDER=${PRM_PROVIDER:-api} # local | api
COLOCATE=${COLOCATE:-1} # 1=actor/rollout share the same GPU set
# Keep historical colocate behavior: offload both train and rollout by default.
# This avoids deterministic OOM during rollout memory resume when train stays on GPU.
OFFLOAD_TRAIN=${OFFLOAD_TRAIN:-1}     # 0=keep train actor on GPU, 1=offload actor to CPU between phases
OFFLOAD_ROLLOUT=${OFFLOAD_ROLLOUT:-1} # 1=recommend with colocate
USE_OPTIMIZER_CPU_OFFLOAD=${USE_OPTIMIZER_CPU_OFFLOAD:-0}

EFFECTIVE_PRM_GPUS="${PRM_GPUS}"
if [ "${PRM_PROVIDER}" = "api" ]; then
    EFFECTIVE_PRM_GPUS=0
fi

if [ "${COLOCATE}" = "1" ]; then
    if (( ACTOR_GPUS > NUM_GPUS )); then
        echo "When COLOCATE=1, ACTOR_GPUS must be <= NUM_GPUS"
        echo "ACTOR_GPUS=${ACTOR_GPUS}, NUM_GPUS=${NUM_GPUS}"
        exit 1
    fi
else
    if (( ACTOR_GPUS + ROLLOUT_GPUS + EFFECTIVE_PRM_GPUS > NUM_GPUS )); then
        echo "ACTOR_GPUS + ROLLOUT_GPUS + EFFECTIVE_PRM_GPUS must be <= NUM_GPUS"
        echo "ACTOR_GPUS=${ACTOR_GPUS}, ROLLOUT_GPUS=${ROLLOUT_GPUS}, EFFECTIVE_PRM_GPUS=${EFFECTIVE_PRM_GPUS}, NUM_GPUS=${NUM_GPUS}"
        exit 1
    fi
fi

export RAY_health_check_failure_threshold=20
export RAY_health_check_period_ms=5000
export RAY_health_check_timeout_ms=30000
export RAY_num_heartbeats_timeout=60

ATTENTION_BACKEND=${ATTENTION_BACKEND:-auto} # flash | fused | unfused | local | auto

# Avoid inheriting stale NVTE backend toggles from parent shell.
unset NVTE_FLASH_ATTN
unset NVTE_FUSED_ATTN
unset NVTE_UNFUSED_ATTN

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_ROOT="$(cd -- "${SCRIPT_DIR}/../slime" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
MEGATRON_ROOT="${PROJECT_ROOT}/Megatron-LM"
if [ ! -d "${MEGATRON_ROOT}/megatron" ]; then
  echo "Megatron-LM source not found at: ${MEGATRON_ROOT}"
  echo "Please ensure the repository contains Megatron-LM and path is correct."
  exit 1
fi

# Auto-load env vars for local runs (e.g. PRM_API_BASE_URL/PRM_API_MODEL/PRM_API_KEY).
# Priority: ENV_FILE (if set) > ${SCRIPT_DIR}/.env > ${SCRIPT_DIR}/../.env
ENV_FILE_PATH="${ENV_FILE:-}"
if [ -z "${ENV_FILE_PATH}" ]; then
  if [ -f "${SCRIPT_DIR}/.env" ]; then
    ENV_FILE_PATH="${SCRIPT_DIR}/.env"
  elif [ -f "${SCRIPT_DIR}/../.env" ]; then
    ENV_FILE_PATH="${SCRIPT_DIR}/../.env"
  fi
fi
if [ -n "${ENV_FILE_PATH}" ] && [ -f "${ENV_FILE_PATH}" ]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE_PATH}"
  set +a
  echo "Loaded env file: ${ENV_FILE_PATH}"
fi

source "${SLIME_ROOT}/scripts/models/qwen3-4B.sh"

HF_CKPT=${HF_CKPT:-/absolute/path/to/Qwen3-4B-Thinking-2507}
REF_LOAD=${REF_LOAD:-${HF_CKPT}}
SAVE_CKPT=${SAVE_CKPT:-/absolute/path/to/OpenClaw-RL/ckpt/qwen3-4b-openclaw-rl}
PRM_MODEL_PATH=${PRM_MODEL_PATH:-/absolute/path/to/Qwen3-4B-Thinking-2507}

export SGLANG_API_KEY="${SGLANG_API_KEY}"
export SERVED_MODEL_NAME="qwen3-4b"
export HOST="0.0.0.0"
export PORT="30000"
export OPENCLAW_RECORD_ENABLED="${OPENCLAW_RECORD_ENABLED:-1}"  # 0=off, 1=on
export OPENCLAW_RECORD_FILE="${SCRIPT_DIR}/results/qwen3_4b_record.jsonl"
export OPENCLAW_PAUSE_REQUEST_MODE="${OPENCLAW_PAUSE_REQUEST_MODE:-reject}" # reject | block
export OPENCLAW_PAUSE_BLOCK_TIMEOUT_SEC="${OPENCLAW_PAUSE_BLOCK_TIMEOUT_SEC:-600}"
export OPENCLAW_PAUSE_BLOCK_POLL_SEC="${OPENCLAW_PAUSE_BLOCK_POLL_SEC:-0.05}"
export OPENCLAW_IDLE_WAIT_TIMEOUT_SEC="${OPENCLAW_IDLE_WAIT_TIMEOUT_SEC:-180}"
export SGLANG_FLUSH_CACHE_MAX_RETRIES="${SGLANG_FLUSH_CACHE_MAX_RETRIES:-300}"
export SGLANG_FLUSH_CACHE_RETRY_INTERVAL_SEC="${SGLANG_FLUSH_CACHE_RETRY_INTERVAL_SEC:-1}"
export SGLANG_FLUSH_CACHE_REQUEST_TIMEOUT_SEC="${SGLANG_FLUSH_CACHE_REQUEST_TIMEOUT_SEC:-5}"
export SGLANG_SKIP_FLUSH_CACHE_ON_TIMEOUT="${SGLANG_SKIP_FLUSH_CACHE_ON_TIMEOUT:-1}"
USE_EXPANDABLE_SEGMENTS=${USE_EXPANDABLE_SEGMENTS:-0}
# IMPORTANT: torch_memory_saver (used by SGLang when offload_rollout is enabled)
# is incompatible with expandable_segments. Keep disabled by default.
if [ "${USE_EXPANDABLE_SEGMENTS}" = "1" ]; then
  export PYTORCH_ALLOC_CONF="expandable_segments:True"
else
  export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-}"
fi
export TP="1"
export CONTEXT_LENGTH="32768"
export MEM_FRACTION_STATIC="0.85"
export REASONING_PARSER="qwen3"
export TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen25}"
export PRM_M="${PRM_M:-3}"


CKPT_ARGS=(
   --megatron-to-hf-mode bridge
   --hf-checkpoint "${HF_CKPT}"
   --ref-load "${REF_LOAD}"
   --save "${SAVE_CKPT}"
   --save-interval 100
   --rotary-base 1000000
)

ROLLOUT_ARGS=(
   --disable-rollout-global-dataset
   --rollout-function-path openclaw_rollout.generate_rollout_openclaw

   --num-rollout 100000000
   --rollout-batch-size 2
   --n-samples-per-prompt 1
   --rollout-max-response-len 8192
   --rollout-max-context-len 32768
   --rollout-temperature 0.6
   --reward-key score

   --num-steps-per-rollout 1
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
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

GRPO_ARGS=(
   --advantage-estimator grpo
   --disable-rewards-normalization
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
   --use-precision-aware-optimizer
)

if [ "${USE_OPTIMIZER_CPU_OFFLOAD}" = "1" ]; then
  OPTIMIZER_ARGS+=(
    --optimizer-cpu-offload
    --overlap-cpu-optimizer-d2h-h2d
  )
fi



EVAL_ARGS=()

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-tool-call-parser "${TOOL_CALL_PARSER}"
   --sglang-mem-fraction-static 0.85
   --sglang-context-length 32768
   --sglang-reasoning-parser qwen3
)

PRM_ARGS=(
   --prm-enable
   --prm-provider "${PRM_PROVIDER}"
   --prm-num-gpus "${EFFECTIVE_PRM_GPUS}"
   --prm-num-gpus-per-engine 2
   --prm-m "${PRM_M}"
   --prm-temperature "${PRM_TEMPERATURE:-0.6}"
   --prm-max-new-tokens "${PRM_MAX_NEW_TOKENS:-4096}"
)

if [ "${PRM_PROVIDER}" = "api" ]; then
  # OpenAI-compatible endpoint, e.g. https://api.openai.com/v1
  PRM_ARGS+=(
    --prm-api-base-url "${PRM_API_BASE_URL}"
    --prm-api-model "${PRM_API_MODEL}"
  )
  if [ -n "${PRM_API_KEY:-}" ]; then
    PRM_ARGS+=(--prm-api-key "${PRM_API_KEY}")
  fi
  PRM_ARGS+=(--prm-api-timeout "${PRM_API_TIMEOUT:-120}")
else
  PRM_ARGS+=(--prm-model-path "${PRM_MODEL_PATH}")
fi

CUSTOM_ARGS=(
   --custom-generate-function-path openclaw_api_server.generate
   --custom-rm-path openclaw_api_server.reward_func
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend "${ATTENTION_BACKEND}"
)

if [ "${COLOCATE}" = "1" ]; then
  MISC_ARGS+=(--colocate)
fi

OFFLOAD_ARGS=()
if [ "${OFFLOAD_TRAIN}" = "1" ]; then
  OFFLOAD_ARGS+=(--offload-train)
else
  OFFLOAD_ARGS+=(--no-offload-train)
fi
if [ "${OFFLOAD_ROLLOUT}" = "1" ]; then
  OFFLOAD_ARGS+=(--offload-rollout)
else
  OFFLOAD_ARGS+=(--no-offload-rollout)
fi

if [ "${COLOCATE}" = "1" ] && [ "${OFFLOAD_TRAIN}" = "0" ] && [ "${OFFLOAD_ROLLOUT}" = "1" ]; then
  echo "Warning: COLOCATE=1 with OFFLOAD_TRAIN=0 and OFFLOAD_ROLLOUT=1 can trigger torch_memory_saver resume OOM."
fi

USE_WANDB=${USE_WANDB:-1}
WANDB_PROJECT=${WANDB_PROJECT:-openclaw_rl}
WANDB_KEY_VALUE=${WANDB_KEY:-${WANDB_API_KEY:-}}
if [ "${USE_WANDB}" = "1" ] && [ -n "${WANDB_KEY_VALUE}" ]; then
  WANDB_ARGS=(
    --use-wandb
    --wandb-project ${WANDB_PROJECT}
    --wandb-group qwen3-4b-openclaw-rl
    --wandb-key ${WANDB_KEY_VALUE}
  )
else
  WANDB_ARGS=()
fi

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export no_proxy="127.0.0.1,${MASTER_ADDR}"
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_ROOT}:${SCRIPT_DIR}:${SLIME_ROOT}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"PYTORCH_ALLOC_CONF\": \"${PYTORCH_ALLOC_CONF}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "${ACTOR_GPUS}" \
   --rollout-num-gpus "${ROLLOUT_GPUS}" \
   --num-gpus-per-node "${NUM_GPUS}" \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${OFFLOAD_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${CUSTOM_ARGS[@]} \
   ${PRM_ARGS[@]}


