#!/usr/bin/env bash
# a3s-code-rl training on 4x H200.
# Default GPU allocation: Actor=2(TP=2), Rollout=2(TP=2), external PRM=0 local GPUs.
set -euo pipefail
set -x

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

# Ensure conda env python is used (conda activate may not work in non-interactive shells)
CONDA_ENV="${CONDA_ENV:-/mnt/shared-storage-user/lixiangtian/miniconda3/envs/openclaw-rl}"
export PATH="${CONDA_ENV}/bin:${PATH}"
export CONDA_PREFIX="${CONDA_ENV}"

PYTHON_VERSION="$("${CONDA_ENV}/bin/python3" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
PYTHON_SITE_PACKAGES="${CONDA_ENV}/lib/python${PYTHON_VERSION}/site-packages"
TORCH_LIB_DIR="${PYTHON_SITE_PACKAGES}/torch/lib"
CUDA_RUNTIME_LIB_DIR="${PYTHON_SITE_PACKAGES}/nvidia/cuda_runtime/lib"
CUDA_NVRTC_LIB_DIR="${PYTHON_SITE_PACKAGES}/nvidia/cuda_nvrtc/lib"
CUDNN_LIB_DIR="${PYTHON_SITE_PACKAGES}/nvidia/cudnn/lib"
CURAND_LIB_DIR="${PYTHON_SITE_PACKAGES}/nvidia/curand/lib"

if [[ ! -d "${CUDA_RUNTIME_LIB_DIR}" ]]; then
  echo "missing CUDA runtime dir: ${CUDA_RUNTIME_LIB_DIR}"
  exit 1
fi

LD_LIBRARY_PATH_PARTS=("${CONDA_ENV}/lib")
for lib_dir in "${TORCH_LIB_DIR}" "${CUDA_RUNTIME_LIB_DIR}" "${CUDA_NVRTC_LIB_DIR}" "${CUDNN_LIB_DIR}" "${CURAND_LIB_DIR}"; do
  if [[ -d "${lib_dir}" ]]; then
    LD_LIBRARY_PATH_PARTS+=("${lib_dir}")
  fi
done
LD_LIBRARY_PATH_PREFIX="$(IFS=:; echo "${LD_LIBRARY_PATH_PARTS[*]}")"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH_PREFIX}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
SLIME_ROOT="${PROJECT_ROOT}/slime"
MEGATRON_ROOT="${PROJECT_ROOT}/Megatron-LM"
CODE_RL_DIR="${PROJECT_ROOT}/a3s-code-rl"

if [[ ! -d "${SLIME_ROOT}" ]]; then
  echo "missing SLIME_ROOT: ${SLIME_ROOT}"; exit 1
fi

# ── Artifact root ────────────────────────────────────────────────
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}}"
RUN_ID="a3s_code_rl_qwen3_4b_$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${ARTIFACT_ROOT}/runs/${RUN_ID}"
LOG_DIR="${RUN_ROOT}/logs"
mkdir -p "${LOG_DIR}" "${RUN_ROOT}" "${CODE_RL_DIR}/results"

CUDA_SHIM_ROOT="${RUN_ROOT}/cuda-home"
mkdir -p "${CUDA_SHIM_ROOT}/targets/x86_64-linux"
ln -sfn "${CUDA_RUNTIME_LIB_DIR}" "${CUDA_SHIM_ROOT}/lib64"
ln -sfn "${CUDA_RUNTIME_LIB_DIR}" "${CUDA_SHIM_ROOT}/targets/x86_64-linux/lib"
export CUDA_HOME="${CUDA_SHIM_ROOT}"
export CUDA_PATH="${CUDA_SHIM_ROOT}"
export CUDNN_HOME="${PYTHON_SITE_PACKAGES}/nvidia/cudnn"
export CUDNN_PATH="${CUDNN_HOME}"

# ── Kill stale processes ─────────────────────────────────────────
ray stop --force 2>/dev/null || true
pkill -f "sglang" 2>/dev/null || true
pkill -f "code_rl_api_server" 2>/dev/null || true
sleep 2

# ── GPU allocation (4x H200) ────────────────────────────────────
NUM_GPUS="${NUM_GPUS:-4}"
ACTOR_GPUS="${ACTOR_GPUS:-2}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-2}"
PRM_GPUS="${PRM_GPUS:-1}"
ENABLE_PRM="${ENABLE_PRM:-1}"
PRM_BACKEND="${PRM_BACKEND:-external_openai}"  # external_openai | local_sglang | disabled
TP_TRAIN="${TP_TRAIN:-2}"   # tensor-parallel for Megatron actor
TP_SGLANG="${TP_SGLANG:-2}" # tensor-parallel for SGLang rollout / PRM

EFFECTIVE_PRM_GPUS=0
EFFECTIVE_PRM_BACKEND="disabled"
if [[ "${ENABLE_PRM}" == "1" && "${PRM_BACKEND}" == "local_sglang" ]]; then
  EFFECTIVE_PRM_GPUS="${PRM_GPUS}"
  EFFECTIVE_PRM_BACKEND="local_sglang"
elif [[ "${ENABLE_PRM}" == "1" && "${PRM_BACKEND}" == "external_openai" ]]; then
  EFFECTIVE_PRM_BACKEND="external_openai"
fi

if (( ACTOR_GPUS + ROLLOUT_GPUS + EFFECTIVE_PRM_GPUS > NUM_GPUS )); then
  echo "ACTOR_GPUS + ROLLOUT_GPUS + PRM_GPUS must be <= NUM_GPUS"
  echo "ACTOR_GPUS=${ACTOR_GPUS}, ROLLOUT_GPUS=${ROLLOUT_GPUS}, PRM_GPUS=${EFFECTIVE_PRM_GPUS}, NUM_GPUS=${NUM_GPUS}"
  exit 1
fi

if (( TP_TRAIN > ACTOR_GPUS || ACTOR_GPUS % TP_TRAIN != 0 )); then
  echo "TP_TRAIN must divide ACTOR_GPUS"
  echo "ACTOR_GPUS=${ACTOR_GPUS}, TP_TRAIN=${TP_TRAIN}"
  exit 1
fi

if (( TP_SGLANG > ROLLOUT_GPUS || ROLLOUT_GPUS % TP_SGLANG != 0 )); then
  echo "TP_SGLANG must divide ROLLOUT_GPUS"
  echo "ROLLOUT_GPUS=${ROLLOUT_GPUS}, TP_SGLANG=${TP_SGLANG}"
  exit 1
fi

# ── Model paths ──────────────────────────────────────────────────
HF_CKPT="${HF_CKPT:-/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen3-4B-Thinking-2507/snapshots/768f209d9ea81521153ed38c47d515654e938aea}"
REF_LOAD="${REF_LOAD:-${HF_CKPT}}"
SAVE_CKPT="${SAVE_CKPT:-${ARTIFACT_ROOT}/ckpt/qwen3-4b-code-rl}"
PRM_MODEL_PATH="${PRM_MODEL_PATH:-${HF_CKPT}}"
KIMI_BASE_URL="${KIMI_BASE_URL:-http://s-20260204175507-cqflp.ailab-pj.pjh-service.org.cn}"
mkdir -p "${SAVE_CKPT}"

# ── Model arch args (qwen3-4B-Instruct-2507 = qwen3-4B + rotary_base 5M) ──
source "${SLIME_ROOT}/scripts/models/qwen3-4B-Instruct-2507.sh"

# ── Serving / API config ─────────────────────────────────────────
export SGLANG_API_KEY="${SGLANG_API_KEY:-apiKey}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-4b-2507}"
export HOST="0.0.0.0"
export PORT="${PORT:-30000}"

# ── Recording config ─────────────────────────────────────────────
export CODE_RL_RECORD_ENABLED="${CODE_RL_RECORD_ENABLED:-1}"
export CODE_RL_RECORD_FILE="${CODE_RL_RECORD_FILE:-${RUN_ROOT}/code_rl_record.jsonl}"
export CODE_RL_PRM_RECORD_FILE="${CODE_RL_PRM_RECORD_FILE:-${RUN_ROOT}/code_rl_prm_record.jsonl}"
export CODE_RL_FEEDBACK_RECORD_FILE="${CODE_RL_FEEDBACK_RECORD_FILE:-${RUN_ROOT}/code_rl_feedback_record.jsonl}"
export CODE_RL_TRACE_RECORD_FILE="${CODE_RL_TRACE_RECORD_FILE:-${RUN_ROOT}/code_rl_trace.jsonl}"
export CODE_RL_PURGE_RECORD_FILES_ON_PAUSE="${CODE_RL_PURGE_RECORD_FILES_ON_PAUSE:-0}"
export CODE_RL_SUBMIT_SIDE="${CODE_RL_SUBMIT_SIDE:-0}"
export CODE_RL_TRAIN_SIDE="${CODE_RL_TRAIN_SIDE:-0}"
export CODE_RL_REWARD_MODE="${CODE_RL_REWARD_MODE:-hybrid}"
export CODE_RL_SESSION_IDLE_FLUSH_SEC="${CODE_RL_SESSION_IDLE_FLUSH_SEC:-30}"
export CODE_RL_MATCHED_CONTEXT_TOKENS="${CODE_RL_MATCHED_CONTEXT_TOKENS:-8192}"
export CODE_RL_MAX_TRAIN_TOKENS="${CODE_RL_MAX_TRAIN_TOKENS:-${CODE_RL_MATCHED_CONTEXT_TOKENS}}"
export CODE_RL_PRM_TIMEOUT_SEC="${CODE_RL_PRM_TIMEOUT_SEC:-180}"
export CODE_RL_CONTEXT_SAFETY_MARGIN="${CODE_RL_CONTEXT_SAFETY_MARGIN:-256}"

export CONTEXT_LENGTH="${CONTEXT_LENGTH:-${CODE_RL_MATCHED_CONTEXT_TOKENS}}"
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"
export REASONING_PARSER="${REASONING_PARSER:-qwen3}"
export TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen25}"
export PRM_M="${PRM_M:-3}"
export ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-2048}"
export ROLLOUT_MAX_CONTEXT_LEN="${ROLLOUT_MAX_CONTEXT_LEN:-${CONTEXT_LENGTH}}"
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-${CODE_RL_MAX_TRAIN_TOKENS}}"
export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION="${SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION:-true}"
export SLIME_HOST_IP="${SLIME_HOST_IP:-127.0.0.1}"
export CODE_RL_MAX_RESPONSE_TOKENS="${CODE_RL_MAX_RESPONSE_TOKENS:-2048}"
export CODE_RL_DROP_REPETITIVE_SAMPLES="${CODE_RL_DROP_REPETITIVE_SAMPLES:-1}"
export SLIME_PPO_RATIO_SAFE_BOUND="${SLIME_PPO_RATIO_SAFE_BOUND:-20.0}"
export CLIP_GRAD="${CLIP_GRAD:-0.5}"
export POLICY_LR="${POLICY_LR:-5e-6}"
export POLICY_WEIGHT_DECAY="${POLICY_WEIGHT_DECAY:-0.01}"
export POLICY_ADAM_EPS="${POLICY_ADAM_EPS:-1e-6}"
export POLICY_KL_LOSS_COEF="${POLICY_KL_LOSS_COEF:-0.01}"
export POLICY_EPS_CLIP_C="${POLICY_EPS_CLIP_C:-3.0}"
export POLICY_NORMALIZE_ADVANTAGES="${POLICY_NORMALIZE_ADVANTAGES:-1}"
export POLICY_USE_ROLLOUT_LOGPROBS="${POLICY_USE_ROLLOUT_LOGPROBS:-1}"
export DISABLE_BF16_REDUCED_PRECISION_MATMUL="${DISABLE_BF16_REDUCED_PRECISION_MATMUL:-1}"

if [[ "${ENABLE_PRM}" == "1" && "${PRM_BACKEND}" == "external_openai" ]]; then
  export CODE_RL_PRM_OPENAI_URL="${CODE_RL_PRM_OPENAI_URL:-${KIMI_BASE_URL}/v1/chat/completions}"
  export CODE_RL_PRM_OPENAI_MODEL_NAME="${CODE_RL_PRM_OPENAI_MODEL_NAME:-kimi-k2.5}"
  export CODE_RL_PRM_API_KEY="${CODE_RL_PRM_API_KEY:-}"
  export CODE_RL_PRM_HEALTH_URL="${CODE_RL_PRM_HEALTH_URL:-${KIMI_BASE_URL}/v1/models}"
else
  unset CODE_RL_PRM_OPENAI_URL || true
  unset CODE_RL_PRM_OPENAI_MODEL_NAME || true
  unset CODE_RL_PRM_API_KEY || true
  unset CODE_RL_PRM_HEALTH_URL || true
fi

# ── Ray ──────────────────────────────────────────────────────────
export RAY_health_check_failure_threshold=20
export RAY_health_check_period_ms=5000
export RAY_health_check_timeout_ms=30000
export RAY_num_heartbeats_timeout=60
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export no_proxy="127.0.0.1,${MASTER_ADDR}"

ray start \
  --head \
  --node-ip-address "${MASTER_ADDR}" \
  --num-gpus "${NUM_GPUS}" \
  --disable-usage-stats \
  --dashboard-host=0.0.0.0 \
  --dashboard-port=8265

# ── Checkpoint args ──────────────────────────────────────────────
CKPT_ARGS=(
  --megatron-to-hf-mode bridge
  --hf-checkpoint "${HF_CKPT}"
  --ref-load "${REF_LOAD}"
  --save "${SAVE_CKPT}"
  --save-interval 100
  --rotary-base 5000000
)

# ── Rollout ──────────────────────────────────────────────────────
ROLLOUT_ARGS=(
  --disable-rollout-global-dataset
  --rollout-function-path code_rl_rollout.generate_rollout_code_rl
  --num-rollout 100000000
  --rollout-batch-size 16
  --n-samples-per-prompt 1
  --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
  --rollout-max-context-len "${ROLLOUT_MAX_CONTEXT_LEN}"
  --rollout-temperature 0.6
  --reward-key score
  --num-steps-per-rollout 1
)

# ── Performance (adapted for 2-GPU actor with TP=2) ─────────────
PERF_ARGS=(
  --tensor-model-parallel-size "${TP_TRAIN}"
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
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
  --log-probs-chunk-size 1024
)

# ── GRPO ─────────────────────────────────────────────────────────
GRPO_ARGS=(
  --advantage-estimator grpo
  --disable-rewards-normalization
  --use-kl-loss
  --kl-loss-coef "${POLICY_KL_LOSS_COEF}"
  --kl-loss-type low_var_kl
  --entropy-coef 0.00
  --eps-clip 0.2
  --eps-clip-high 0.28
  --eps-clip-c "${POLICY_EPS_CLIP_C}"
)

if [[ "${POLICY_NORMALIZE_ADVANTAGES}" == "1" ]]; then
  GRPO_ARGS+=(--normalize-advantages)
fi

if [[ "${POLICY_USE_ROLLOUT_LOGPROBS}" == "1" ]]; then
  GRPO_ARGS+=(--use-rollout-logprobs)
fi

# ── Optimizer ────────────────────────────────────────────────────
OPTIMIZER_ARGS=(
  --optimizer adam
  --lr "${POLICY_LR}"
  --lr-decay-style constant
  --weight-decay "${POLICY_WEIGHT_DECAY}"
  --adam-beta1 0.9
  --adam-beta2 0.98
  --adam-eps "${POLICY_ADAM_EPS}"
  --clip-grad "${CLIP_GRAD}"
  --optimizer-cpu-offload
  --overlap-cpu-optimizer-d2h-h2d
  --use-precision-aware-optimizer
)

# ── SGLang ───────────────────────────────────────────────────────
SGLANG_ARGS=(
  --rollout-num-gpus-per-engine "${TP_SGLANG}"
  --sglang-tool-call-parser "${TOOL_CALL_PARSER}"
  --sglang-mem-fraction-static "${MEM_FRACTION_STATIC}"
  --sglang-context-length "${CONTEXT_LENGTH}"
  --sglang-reasoning-parser "${REASONING_PARSER}"
)

# ── PRM ──────────────────────────────────────────────────────────
if [[ "${ENABLE_PRM}" == "1" && "${PRM_BACKEND}" == "local_sglang" ]]; then
  PRM_ARGS=(
    --prm-enable
    --prm-num-gpus "${PRM_GPUS}"
    --prm-num-gpus-per-engine "${TP_SGLANG}"
    --prm-model-path "${PRM_MODEL_PATH}"
    --prm-m "${PRM_M}"
    --prm-temperature "${PRM_TEMPERATURE:-0.6}"
    --prm-max-new-tokens "${PRM_MAX_NEW_TOKENS:-4096}"
  )
else
  PRM_ARGS=()
fi

# ── Custom generate / reward ─────────────────────────────────────
CUSTOM_ARGS=(
  --custom-generate-function-path code_rl_api_server.generate
  --custom-rm-path code_rl_api_server.reward_func
)

# ── Misc ─────────────────────────────────────────────────────────
MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend unfused
  --fallback-to-eager-attn
)

if [[ "${DISABLE_BF16_REDUCED_PRECISION_MATMUL}" == "1" ]]; then
  MISC_ARGS+=(--disable-bf16-reduced-precision-matmul)
fi

# ── Wandb ────────────────────────────────────────────────────────
USE_WANDB="${USE_WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-a3s_code_rl}"
WANDB_KEY_VALUE="${WANDB_KEY:-${WANDB_API_KEY:-}}"
if [[ "${USE_WANDB}" == "1" && -n "${WANDB_KEY_VALUE}" ]]; then
  WANDB_ARGS=(
    --use-wandb
    --wandb-project "${WANDB_PROJECT}"
    --wandb-group qwen3-4b-code-rl-4gpu
    --wandb-key "${WANDB_KEY_VALUE}"
  )
else
  WANDB_ARGS=()
fi

# ── Runtime env ──────────────────────────────────────────────────
RUNTIME_ENV_JSON="$(cat <<JSON
{
  "env_vars": {
    "PYTHONPATH": "${MEGATRON_ROOT}:${CODE_RL_DIR}:${SLIME_ROOT}",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "LD_LIBRARY_PATH": "${LD_LIBRARY_PATH}",
    "CUDA_HOME": "${CUDA_HOME}",
    "CUDA_PATH": "${CUDA_PATH}",
    "CUDNN_HOME": "${CUDNN_HOME}",
    "CUDNN_PATH": "${CUDNN_PATH}",
    "no_proxy": "127.0.0.1,${MASTER_ADDR}",
    "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "${SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION}",
    "SLIME_HOST_IP": "${SLIME_HOST_IP}",
    "CODE_RL_PRM_OPENAI_URL": "${CODE_RL_PRM_OPENAI_URL:-}",
    "CODE_RL_PRM_OPENAI_MODEL_NAME": "${CODE_RL_PRM_OPENAI_MODEL_NAME:-}",
    "CODE_RL_PRM_API_KEY": "${CODE_RL_PRM_API_KEY:-}",
    "CODE_RL_PRM_HEALTH_URL": "${CODE_RL_PRM_HEALTH_URL:-}",
    "CODE_RL_PRM_TIMEOUT_SEC": "${CODE_RL_PRM_TIMEOUT_SEC}",
    "CODE_RL_CONTEXT_SAFETY_MARGIN": "${CODE_RL_CONTEXT_SAFETY_MARGIN}",
    "CODE_RL_MAX_RESPONSE_TOKENS": "${CODE_RL_MAX_RESPONSE_TOKENS}",
    "CODE_RL_DROP_REPETITIVE_SAMPLES": "${CODE_RL_DROP_REPETITIVE_SAMPLES}",
    "SLIME_PPO_RATIO_SAFE_BOUND": "${SLIME_PPO_RATIO_SAFE_BOUND}"
  }
}
JSON
)"

# ── Save launch info ─────────────────────────────────────────────
cat > "${RUN_ROOT}/launch_info.json" <<EOF
{
  "run_id": "${RUN_ID}",
  "run_root": "${RUN_ROOT}",
  "hf_ckpt": "${HF_CKPT}",
  "save_ckpt": "${SAVE_CKPT}",
  "num_gpus": ${NUM_GPUS},
  "actor_gpus": ${ACTOR_GPUS},
  "rollout_gpus": ${ROLLOUT_GPUS},
  "prm_gpus": ${EFFECTIVE_PRM_GPUS},
  "tp_train": ${TP_TRAIN},
  "tp_sglang": ${TP_SGLANG},
  "enable_prm": $([[ "${ENABLE_PRM}" == "1" ]] && echo true || echo false),
  "prm_backend": "${EFFECTIVE_PRM_BACKEND}",
  "prm_model": "$([[ "${EFFECTIVE_PRM_BACKEND}" == "external_openai" ]] && printf '%s' "${CODE_RL_PRM_OPENAI_MODEL_NAME}" || printf '%s' "${PRM_MODEL_PATH}")",
  "timestamp": "$(date -Iseconds)"
}
EOF

echo "=== RUN_ROOT: ${RUN_ROOT} ==="

# ── Launch (direct driver, avoids ray job submit env issues) ──────
export PYTHONPATH="${MEGATRON_ROOT}:${CODE_RL_DIR}:${SLIME_ROOT}:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OPENCLAW_RUNTIME_ENV_JSON="${RUNTIME_ENV_JSON}"

cd "${SLIME_ROOT}"

python3 "${PROJECT_ROOT}/openclaw-rl/train_async_direct_driver.py" \
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
