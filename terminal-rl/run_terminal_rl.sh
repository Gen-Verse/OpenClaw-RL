#!/usr/bin/env bash
# Adapted from terminal-rl/terminal-rl_qwen3-8b.sh for Qwen3-4B on a single 8xH200 node.
# Key differences from the 8B official script:
#   * uses qwen3-4B model args
#   * TP=2 (model is half the size)
#   * points ENV_SERVER_URL directly at the pool_server (no router_server proxy needed for single machine)
#   * wandb creds sourced from ../.env
#
# The script structure, arg names, and launch flow mirror the official one;
# do not reintroduce ad-hoc code patches — solve via configuration.

set -euo pipefail
set -x

log() { echo "[$(date +'%F %T')] $*"; }

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

# ── GPU layout ───────────────────────────────────────────────────
NUM_GPUS="${NUM_GPUS:-8}"
ACTOR_GPUS="${ACTOR_GPUS:-2}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-6}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
CUSTOM_CONFIG_PATH="${CUSTOM_CONFIG_PATH:-${SCRIPT_DIR}/configs/rollout_qwen3.yaml}"

export REPO_ROOT
export SLIME_DIR="${REPO_ROOT}/slime"
export MEGATRON_DIR="${MEGATRON_DIR:-${REPO_ROOT}/Megatron-LM}"

# ── Conda env activation ────────────────────────────────────────
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/home/ubuntu/miniconda3/envs/tbench-rl}"
if [[ -n "${CONDA_ENV_PATH}" ]]; then
    set +u
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV_PATH}"
    set -u
fi

# ── Load wandb creds (optional) ─────────────────────────────────
if [[ -f "${REPO_ROOT}/../.env" ]]; then
    set +u
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/../.env"
    set -u
fi

# ── CUDA_HOME for tvm_ffi / TE runtime libs ─────────────────────
export CUDA_HOME="${CUDA_HOME:-${CONDA_PREFIX}}"
export PATH="${CUDA_HOME}/bin:${PATH}"
_NV_SP="${CONDA_PREFIX}/lib/python3.12/site-packages/nvidia"
_NV_LD=""
for d in cudnn nvtx cusparse cublas cufft cusolver curand cuda_runtime cuda_nvcc cuda_nvrtc nvjitlink nccl; do
    [[ -d "${_NV_SP}/${d}/lib" ]] && _NV_LD="${_NV_LD:+${_NV_LD}:}${_NV_SP}/${d}/lib"
done
export LD_LIBRARY_PATH="${_NV_LD}:${CUDA_HOME}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
unset _NV_SP _NV_LD

# ── Megatron / model args for Qwen3-4B ──────────────────────────
# shellcheck disable=SC1091
source "${SLIME_DIR}/scripts/models/qwen3-4B.sh"

# ── Paths (require via env — match official script style) ───────
HF_CKPT="${HF_CKPT:-/nfs/models/Qwen3-4B}"
REF_LOAD="${REF_LOAD:-/nfs/models/Qwen3-4B_torch_dist}"
SAVE_CKPT="${SAVE_CKPT:-${REPO_ROOT}/terminal-rl/ckpt/qwen3-4b-terminal-rl}"
RESUME_LOAD="${RESUME_LOAD:-${REF_LOAD}}"
ROLLOUT_PROMPT_DATA="${ROLLOUT_PROMPT_DATA:-${SCRIPT_DIR}/dataset/seta_env_convert/train.jsonl}"
mkdir -p "${SAVE_CKPT}" "${REPO_ROOT}/terminal-rl/logs" "${REPO_ROOT}/terminal-rl/build_outputs"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:2048,expandable_segments:True}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

# ── Pool server (running on the same box at 18081) ──────────────
export USE_REMOTE_ENV="${USE_REMOTE_ENV:-1}"
export ENV_SERVER_PORT="${ENV_SERVER_PORT:-18081}"
export ENV_SERVER_HOST="${ENV_SERVER_HOST:-127.0.0.1}"
export ENV_SERVER_URL="${ENV_SERVER_URL:-http://${ENV_SERVER_HOST}:${ENV_SERVER_PORT}}"
export START_ENV_POOL_SERVER="${START_ENV_POOL_SERVER:-0}"
export WORKER_URLS="${WORKER_URLS:-${ENV_SERVER_URL}}"

export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_tbench}"
mkdir -p "${RAY_TMPDIR}"

# ── Training args (mirror official 8B script, with 4B tweaks) ──
CKPT_ARGS=(
    --hf-checkpoint "${HF_CKPT}"
    --ref-load "${REF_LOAD}"
    --load "${RESUME_LOAD}"
    --save "${SAVE_CKPT}"
    --save-interval 8
    --rotary-base 1000000
)

ROLLOUT_ARGS=(
    --prompt-data "${ROLLOUT_PROMPT_DATA}"
    --input-key task
    --rollout-shuffle
    --reward-key score
    --num-rollout 2000
    --rollout-batch-size 16
    --n-samples-per-prompt 8
    --rollout-max-response-len 8192
    --rollout-max-context-len 16384
    --rollout-temperature 1
    --num-steps-per-rollout 2
    --balance-data
)

EVAL_ARGS=(
    --n-samples-per-eval-prompt 16
    --eval-max-response-len 16384
    --eval-top-p 1
)

PERF_ARGS=(
    --tensor-model-parallel-size 2
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu 16384
    --log-probs-chunk-size 1024
)

GRPO_ARGS=(
    --advantage-estimator grpo
    --dynamic_history
    --use-kl-loss
    --kl-loss-coef 0.01
    --kl-loss-type k3
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
    --optimizer-cpu-offload
    --overlap-cpu-optimizer-d2h-h2d
    --use-precision-aware-optimizer
)

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 2
    --sglang-mem-fraction-static 0.6
)

MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
    # Apex isn't installed; disable fused gradient-accum kernel
    --no-gradient-accumulation-fusion
)

CUSTOM_ARGS=(
    --custom-generate-function-path generate.generate
    --custom-rollout-log-function-path rollout_log.rollout_log
    --custom-config-path "${CUSTOM_CONFIG_PATH}"
)

WANDB_ARGS=()
if [[ -n "${WANDB_API_KEY:-}" ]]; then
    WANDB_ARGS=(
        --use-wandb
        --wandb-project "${WANDB_PROJECT:-openclaw-terminal-rl}"
        --wandb-group "${WANDB_GROUP:-qwen3-4b-terminal-rl}"
        --wandb-key "${WANDB_API_KEY}"
    )
    [[ -n "${WANDB_ORG:-}" ]] && WANDB_ARGS+=( --wandb-team "${WANDB_ORG}" )
    [[ -n "${WANDB_MODE:-}" ]] && WANDB_ARGS+=( --wandb-mode "${WANDB_MODE}" )
fi

# ── Setup helpers ────────────────────────────────────────────────
check_gpus() {
    if (( ACTOR_GPUS + ROLLOUT_GPUS > NUM_GPUS )); then
        echo "ACTOR_GPUS + ROLLOUT_GPUS must be <= NUM_GPUS (got ACTOR_GPUS=${ACTOR_GPUS}, ROLLOUT_GPUS=${ROLLOUT_GPUS}, NUM_GPUS=${NUM_GPUS})"
        exit 1
    fi
}

cleanup_prev() {
    log "cleanup previous processes"
    pkill -9 sglang 2>/dev/null || true
    sleep 3
    ray stop --force 2>/dev/null || true
    pkill -9 ray 2>/dev/null || true
    pkill -9 -f "tbench-rl/bin/python" 2>/dev/null || true
    sleep 3
}

detect_nvlink() {
    local count
    count="$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || true)"
    if [[ "${count:-0}" -gt 0 ]]; then export HAS_NVLINK=1; else export HAS_NVLINK=0; fi
    log "HAS_NVLINK=${HAS_NVLINK} (detected ${count} NVLink references)"
}

start_ray_head() {
    log "start ray head"
    ray start --head \
        --node-ip-address "${MASTER_ADDR}" \
        --num-gpus "${NUM_GPUS}" \
        --disable-usage-stats \
        --dashboard-host=0.0.0.0 \
        --dashboard-port=8265 \
        --temp-dir "${RAY_TMPDIR}"
}

build_runtime_env_json() {
    CONDA_ENV_PATH="${CONDA_ENV_PATH}" \
    CONDA_PYTHON_VERSION="${CONDA_PYTHON_VERSION:-3.12}" \
    REPO_ROOT="${REPO_ROOT}" \
    SLIME_DIR="${SLIME_DIR}" \
    MEGATRON_DIR="${MEGATRON_DIR}" \
    SCRIPT_DIR="${SCRIPT_DIR}" \
    HAS_NVLINK="${HAS_NVLINK}" \
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
    USE_REMOTE_ENV="${USE_REMOTE_ENV}" \
    ENV_SERVER_URL="${ENV_SERVER_URL}" \
    CUDA_HOME="${CUDA_HOME}" \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
    python3 - <<'PY'
import json, os
conda_env = os.environ["CONDA_ENV_PATH"]
py_ver = os.environ["CONDA_PYTHON_VERSION"]
site_packages = f"{conda_env}/lib/python{py_ver}/site-packages"
parts = [
    os.environ["REPO_ROOT"],
    os.environ["SLIME_DIR"],
    os.environ["MEGATRON_DIR"],
    os.environ["SCRIPT_DIR"],
    site_packages,
]
pythonpath = ":".join([p for p in parts if p])
env_vars = {
    "PYTHONPATH": pythonpath,
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": os.environ.get("HAS_NVLINK", "0"),
    "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
    "USE_REMOTE_ENV": os.environ.get("USE_REMOTE_ENV", "0"),
    "ENV_SERVER_URL": os.environ.get("ENV_SERVER_URL", ""),
    "CUDA_HOME": os.environ.get("CUDA_HOME", ""),
    "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
    "TOKENIZERS_PARALLELISM": "false",
    "NCCL_IB_DISABLE": "1",
}
print(json.dumps({"env_vars": env_vars}))
PY
}

submit_job() {
    log "submit ray job"
    local runtime_env_json
    runtime_env_json="$(build_runtime_env_json)"

    ray job submit --address="http://127.0.0.1:8265" \
        --runtime-env-json="${runtime_env_json}" \
        -- python3 ${SLIME_DIR}/train_async.py \
        --actor-num-nodes 1 \
        --actor-num-gpus-per-node "${ACTOR_GPUS}" \
        --rollout-num-gpus "${ROLLOUT_GPUS}" \
        "${MODEL_ARGS[@]}" \
        "${CKPT_ARGS[@]}" \
        "${ROLLOUT_ARGS[@]}" \
        "${OPTIMIZER_ARGS[@]}" \
        "${GRPO_ARGS[@]}" \
        "${WANDB_ARGS[@]}" \
        "${PERF_ARGS[@]}" \
        "${EVAL_ARGS[@]}" \
        "${SGLANG_ARGS[@]}" \
        "${MISC_ARGS[@]}" \
        "${CUSTOM_ARGS[@]}"
}

cleanup_prev
check_gpus
detect_nvlink
export SCRIPT_DIR
start_ray_head
submit_job
