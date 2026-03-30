#!/usr/bin/env bash

set -euo pipefail
set -x

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
SLIME_ROOT="${PROJECT_ROOT}/slime"
MEGATRON_ROOT="${PROJECT_ROOT}/Megatron-LM"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PYTHON_PREFIX="$("${PYTHON_BIN}" -c 'import sys; print(sys.prefix)')"
PYTHON_VERSION="$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
PYTHON_SITE_PACKAGES="${PYTHON_PREFIX}/lib/python${PYTHON_VERSION}/site-packages"
TORCH_LIB_DIR="${PYTHON_SITE_PACKAGES}/torch/lib"
CUDA_RUNTIME_LIB_DIR="${PYTHON_SITE_PACKAGES}/nvidia/cuda_runtime/lib"
CUDA_NVRTC_LIB_DIR="${PYTHON_SITE_PACKAGES}/nvidia/cuda_nvrtc/lib"
CUDNN_LIB_DIR="${PYTHON_SITE_PACKAGES}/nvidia/cudnn/lib"
CURAND_LIB_DIR="${PYTHON_SITE_PACKAGES}/nvidia/curand/lib"

if [[ ! -d "${CUDA_RUNTIME_LIB_DIR}" ]]; then
  echo "missing CUDA runtime dir: ${CUDA_RUNTIME_LIB_DIR}"
  exit 1
fi

LD_LIBRARY_PATH_PARTS=()
for lib_dir in "${PYTHON_PREFIX}/lib" "${TORCH_LIB_DIR}" "${CUDA_RUNTIME_LIB_DIR}" "${CUDA_NVRTC_LIB_DIR}" "${CUDNN_LIB_DIR}" "${CURAND_LIB_DIR}"; do
  if [[ -d "${lib_dir}" ]]; then
    LD_LIBRARY_PATH_PARTS+=("${lib_dir}")
  fi
done
LD_LIBRARY_PATH_PREFIX="$(IFS=:; echo "${LD_LIBRARY_PATH_PARTS[*]}")"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH_PREFIX}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

if [[ ! -d "${SLIME_ROOT}" ]]; then
  echo "missing SLIME_ROOT: ${SLIME_ROOT}"
  exit 1
fi

mkdir -p "${SCRIPT_DIR}/results"

CUDA_SHIM_ROOT="${SCRIPT_DIR}/results/cuda-home"
mkdir -p "${CUDA_SHIM_ROOT}/targets/x86_64-linux"
ln -sfn "${CUDA_RUNTIME_LIB_DIR}" "${CUDA_SHIM_ROOT}/lib64"
ln -sfn "${CUDA_RUNTIME_LIB_DIR}" "${CUDA_SHIM_ROOT}/targets/x86_64-linux/lib"
export CUDA_HOME="${CUDA_SHIM_ROOT}"
export CUDA_PATH="${CUDA_SHIM_ROOT}"
export CUDNN_HOME="${PYTHON_SITE_PACKAGES}/nvidia/cudnn"
export CUDNN_PATH="${CUDNN_HOME}"

ray stop --force || true
pkill -f "sglang" || true
pkill -f "code_rl_api_server" || true
sleep 2

NUM_GPUS="${NUM_GPUS:-8}"
ACTOR_GPUS="${ACTOR_GPUS:-4}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-2}"
PRM_GPUS="${PRM_GPUS:-2}"
ENABLE_PRM="${ENABLE_PRM:-1}"

EFFECTIVE_PRM_GPUS=0
if [[ "${ENABLE_PRM}" == "1" ]]; then
  EFFECTIVE_PRM_GPUS="${PRM_GPUS}"
fi

if (( ACTOR_GPUS + ROLLOUT_GPUS + EFFECTIVE_PRM_GPUS > NUM_GPUS )); then
  echo "ACTOR_GPUS + ROLLOUT_GPUS + PRM_GPUS must be <= NUM_GPUS"
  echo "ACTOR_GPUS=${ACTOR_GPUS}, ROLLOUT_GPUS=${ROLLOUT_GPUS}, PRM_GPUS=${EFFECTIVE_PRM_GPUS}, NUM_GPUS=${NUM_GPUS}"
  exit 1
fi

export RAY_health_check_failure_threshold=20
export RAY_health_check_period_ms=5000
export RAY_health_check_timeout_ms=30000
export RAY_num_heartbeats_timeout=60

source "${SLIME_ROOT}/scripts/models/qwen3-4B.sh"

HF_CKPT="${HF_CKPT:-/absolute/path/to/Qwen3-4B-Thinking-2507}"
REF_LOAD="${REF_LOAD:-${HF_CKPT}}"
SAVE_CKPT="${SAVE_CKPT:-${PROJECT_ROOT}/ckpt/qwen3-4b-code-rl}"
PRM_MODEL_PATH="${PRM_MODEL_PATH:-/absolute/path/to/Qwen3-4B-Thinking-2507}"

export SGLANG_API_KEY="${SGLANG_API_KEY:-apiKey}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-4b-2507}"
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-30000}"

export CODE_RL_RECORD_ENABLED="${CODE_RL_RECORD_ENABLED:-1}"
export CODE_RL_RECORD_FILE="${CODE_RL_RECORD_FILE:-${SCRIPT_DIR}/results/code_rl_record.jsonl}"
export CODE_RL_PRM_RECORD_FILE="${CODE_RL_PRM_RECORD_FILE:-${SCRIPT_DIR}/results/code_rl_prm_record.jsonl}"
export CODE_RL_FEEDBACK_RECORD_FILE="${CODE_RL_FEEDBACK_RECORD_FILE:-${SCRIPT_DIR}/results/code_rl_feedback_record.jsonl}"
export CODE_RL_TRACE_RECORD_FILE="${CODE_RL_TRACE_RECORD_FILE:-${SCRIPT_DIR}/results/code_rl_trace.jsonl}"
export CODE_RL_PURGE_RECORD_FILES_ON_PAUSE="${CODE_RL_PURGE_RECORD_FILES_ON_PAUSE:-0}"
export CODE_RL_SUBMIT_SIDE="${CODE_RL_SUBMIT_SIDE:-1}"
export CODE_RL_TRAIN_SIDE="${CODE_RL_TRAIN_SIDE:-0}"
export CODE_RL_REWARD_MODE="${CODE_RL_REWARD_MODE:-hybrid}"
export CODE_RL_SESSION_IDLE_FLUSH_SEC="${CODE_RL_SESSION_IDLE_FLUSH_SEC:-30}"

export TP="${TP:-2}"
export CONTEXT_LENGTH="${CONTEXT_LENGTH:-32768}"
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"
export REASONING_PARSER="${REASONING_PARSER:-qwen3}"
export TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen25}"
export PRM_M="${PRM_M:-3}"
export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION="${SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION:-true}"
export SLIME_HOST_IP="${SLIME_HOST_IP:-127.0.0.1}"

CKPT_ARGS=(
  --megatron-to-hf-mode bridge
  --hf-checkpoint "${HF_CKPT}"
  --ref-load "${REF_LOAD}"
  --save "${SAVE_CKPT}"
  --save-interval 100
  --rotary-base 5000000
)

ROLLOUT_ARGS=(
  --disable-rollout-global-dataset
  --rollout-function-path code_rl_rollout.generate_rollout_code_rl
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
  --micro-batch-size 1
  --qkv-format bshd
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
  --optimizer-cpu-offload
  --overlap-cpu-optimizer-d2h-h2d
  --use-precision-aware-optimizer
)

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine "${TP}"
  --sglang-tool-call-parser "${TOOL_CALL_PARSER}"
  --sglang-mem-fraction-static "${MEM_FRACTION_STATIC}"
  --sglang-context-length "${CONTEXT_LENGTH}"
  --sglang-reasoning-parser "${REASONING_PARSER}"
)

if [[ "${ENABLE_PRM}" == "1" ]]; then
  PRM_ARGS=(
    --prm-enable
    --prm-num-gpus "${PRM_GPUS}"
    --prm-num-gpus-per-engine "${TP}"
    --prm-model-path "${PRM_MODEL_PATH}"
    --prm-m "${PRM_M}"
    --prm-temperature "${PRM_TEMPERATURE:-0.6}"
    --prm-max-new-tokens "${PRM_MAX_NEW_TOKENS:-4096}"
  )
else
  PRM_ARGS=()
fi

CUSTOM_ARGS=(
  --custom-generate-function-path code_rl_api_server.generate
  --custom-rm-path code_rl_api_server.reward_func
)

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend unfused
  --fallback-to-eager-attn
)

USE_WANDB="${USE_WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-code_rl}"
WANDB_KEY_VALUE="${WANDB_KEY:-${WANDB_API_KEY:-}}"
if [[ "${USE_WANDB}" == "1" && -n "${WANDB_KEY_VALUE}" ]]; then
  WANDB_ARGS=(
    --use-wandb
    --wandb-project "${WANDB_PROJECT}"
    --wandb-group qwen3-4b-code-rl
    --wandb-key "${WANDB_KEY_VALUE}"
  )
else
  WANDB_ARGS=()
fi

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export no_proxy="127.0.0.1,${MASTER_ADDR}"

ray start \
  --head \
  --node-ip-address "${MASTER_ADDR}" \
  --num-gpus "${NUM_GPUS}" \
  --disable-usage-stats \
  --dashboard-host=0.0.0.0 \
  --dashboard-port=8265

RUNTIME_ENV_JSON="$(cat <<JSON
{
  "env_vars": {
    "PYTHONPATH": "${MEGATRON_ROOT}:${SCRIPT_DIR}:${SLIME_ROOT}",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "LD_LIBRARY_PATH": "${LD_LIBRARY_PATH}",
    "CUDA_HOME": "${CUDA_HOME}",
    "CUDA_PATH": "${CUDA_PATH}",
    "CUDNN_HOME": "${CUDNN_HOME}",
    "CUDNN_PATH": "${CUDNN_PATH}",
    "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "${SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION}",
    "SLIME_HOST_IP": "${SLIME_HOST_IP}"
  }
}
JSON
)"

cd "${SLIME_ROOT}"

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 train_async.py \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node "${ACTOR_GPUS}" \
  --rollout-num-gpus "${ROLLOUT_GPUS}" \
  --num-gpus-per-node "${NUM_GPUS}" \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${GRPO_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  "${MISC_ARGS[@]}" \
  "${WANDB_ARGS[@]}" \
  "${CUSTOM_ARGS[@]}" \
  "${PRM_ARGS[@]}"
