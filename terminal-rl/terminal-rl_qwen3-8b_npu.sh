#!/usr/bin/env bash
set -euo pipefail
set -ex

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
ulimit -n 65535

log() { echo "[$(date +'%F %T')] $*"; }

require_cmd() { command -v "$1" >/dev/null 2>&1 || { echo "[ERROR] missing cmd: $1"; exit 1; }; }

WORKER_URLS="http://localhost:18081"
ROLLOUT_PROMPT_DATA="terminal-rl/dataset/seta_env_convert/train.jsonl"

RAY_TMPDIR="ray_tmp"
# CHECK_HOST="localhost"
# ENV_SERVER_PORT="18081"

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

# set visible devices
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export RAY_DEBUG=1
export RAY_DEDUP_LOGS=0

export RAY_health_check_failure_threshold=20
export RAY_health_check_period_ms=5000
export RAY_health_check_timeout_ms=30000
export RAY_num_heartbeats_timeout=60

NUM_GPUS="${NUM_GPUS:-8}"
ACTOR_GPUS="${ACTOR_GPUS:-4}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-4}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
CUSTOM_CONFIG_PATH="${CUSTOM_CONFIG_PATH:-${SCRIPT_DIR}/configs/rollout_qwen3.yaml}"

export REPO_ROOT
export SLIME_DIR="${REPO_ROOT}/slime"
export MEGATRON_DIR="Megatron-LM"

source "${SLIME_DIR}/scripts/models/qwen3-8B.sh"

# Paths: set/export before running (no built-in defaults).
HF_HOME="${HF_HOME:-}"
HF_CKPT=Qwen3-8B
REF_LOAD=Qwen3-8B-dist-slime
SAVE_CKPT=Qwen3-8B-save
RESUME_LOAD="${RESUME_LOAD:-${SAVE_CKPT}}"
ROLLOUT_PROMPT_DATA="${ROLLOUT_PROMPT_DATA:-}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:2048,expandable_segments:True}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

export USE_REMOTE_ENV="${USE_REMOTE_ENV:-1}"
export PROVIDER_NAME="${PROVIDER_NAME:-pull}"
export ENV_SERVER_BIND_HOST="${ENV_SERVER_BIND_HOST:-0.0.0.0}"
export ENV_SERVER_PORT="${ENV_SERVER_PORT:-18080}"
export ENV_SERVER_HOST="${ENV_SERVER_HOST:-${MASTER_ADDR}}"
export ENV_SERVER_URL="${ENV_SERVER_URL:-}"
export START_ENV_POOL_SERVER="${START_ENV_POOL_SERVER:-0}"

# export RAY_TMPDIR="${RAY_TMPDIR:-}"

export WORKER_URLS="${WORKER_URLS:-}"

ROUTER_SESSION_NAME="${ROUTER_SESSION_NAME:-terminal_router}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-}"
ROUTER_PROJECT_DIR="${ROUTER_PROJECT_DIR:-${REPO_ROOT}}"
export CONDA_ENV_PATH
CONDA_PYTHON_VERSION="${CONDA_PYTHON_VERSION:-3.12}"
export CONDA_PYTHON_VERSION
ROUTER_HOST="${ROUTER_HOST:-0.0.0.0}"
ROUTER_PORT="${ROUTER_PORT:-${ENV_SERVER_PORT}}"

CHECK_HOST="${CHECK_HOST:-127.0.0.1}"
CHECK_WAIT_SECS="${CHECK_WAIT_SECS:-60}"
ROUTER_RESTART="${ROUTER_RESTART:-1}"

CKPT_ARGS=(
  --hf-checkpoint "${HF_CKPT}"
  --ref-load "${REF_LOAD}"
  # --load "${RESUME_LOAD}"
  --save "${SAVE_CKPT}"
  --save-interval 80000
  --rotary-base 1000000
)

ROLLOUT_ARGS=(
   --prompt-data "${ROLLOUT_PROMPT_DATA}"
   --input-key task
   --rollout-shuffle
   --reward-key score
   --num-rollout 250
   --rollout-batch-size 16
   --n-samples-per-prompt 4
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

if [[ -n "${WANDB_KEY:-}" ]]; then
  WANDB_ARGS=(
    --use-wandb
    --wandb-project ${WANDB_PROJECT}
    --wandb-group ${WANDB_GROUP}
    --wandb-key ${WANDB_KEY}
  )
else
  WANDB_ARGS=()
fi

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
)

CUSTOM_ARGS=(
   --custom-generate-function-path generate.generate
   --custom-rollout-log-function-path rollout_log.rollout_log
   --custom-config-path "${CUSTOM_CONFIG_PATH}"
)

check_gpus() {
  if (( ACTOR_GPUS + ROLLOUT_GPUS > NUM_GPUS )); then
    echo "ACTOR_GPUS + ROLLOUT_GPUS must be <= NUM_GPUS"
    echo "ACTOR_GPUS=${ACTOR_GPUS}, ROLLOUT_GPUS=${ROLLOUT_GPUS}, NUM_GPUS=${NUM_GPUS}"
    exit 1
  fi
}

cleanup_prev() {
  log "cleanup previous processes"
  pkill -9 sglang || true
  sleep 3
  ray stop --force || true
  pkill -9 ray || true
  pkill -9 python || true
  sleep 3
  pkill -9 ray || true
  pkill -9 python || true
}

start_router() {
  require_cmd curl
  mkdir -p "${ROUTER_PROJECT_DIR}/logs"
  local logf="${ROUTER_PROJECT_DIR}/logs/router_${ROUTER_PORT}.log"

  python -m terminal-rl.router_server \
    --host "${ROUTER_HOST}" --port "${ROUTER_PORT}" --workers "${WORKER_URLS}" \
    > "${logf}" 2>&1 &

  export ROUTER_PID=$!
  log "router started pid=${ROUTER_PID}, log=${logf}"

  sleep 1
  tail -n 50 "${logf}" || true
}

check_router() {
  require_cmd curl
  local base_url="http://${CHECK_HOST}:${ROUTER_PORT}"

  log "wait router healthz up to ${CHECK_WAIT_SECS}s: ${base_url}/healthz"
  for ((i=1; i<=CHECK_WAIT_SECS; i++)); do
    if curl -fsS "${base_url}/healthz" >/dev/null 2>&1; then
      log "router is up"
      break
    fi
    sleep 1
  done

  log "curl ${base_url}/status"
  curl -sS "${base_url}/status"
  echo
  log "curl ${base_url}/healthz"
  curl -sS "${base_url}/healthz"
  echo
}

detect_nvlink() {
  local count
  count="$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || true)"
  if [[ "${count:-0}" -gt 0 ]]; then
    export HAS_NVLINK=1
  else
    export HAS_NVLINK=0
  fi
  log "HAS_NVLINK=${HAS_NVLINK} (detected ${count} NVLink references)"
}

maybe_fill_env_server_url() {
  if [[ "${USE_REMOTE_ENV}" == "1" && -z "${ENV_SERVER_URL}" ]]; then
    export ENV_SERVER_URL="http://${ENV_SERVER_HOST}:${ENV_SERVER_PORT}"
    if [[ "${START_ENV_POOL_SERVER}" == "0" ]]; then
      export START_ENV_POOL_SERVER=1
    fi
  fi
  log "ENV_SERVER_URL=${ENV_SERVER_URL} START_ENV_POOL_SERVER=${START_ENV_POOL_SERVER}"
}

start_ray_head() {
  require_cmd ray
  log "start ray head"
  mkdir -p "${RAY_TMPDIR}"
  ray start --head \
    --node-ip-address "${MASTER_ADDR}" \
    --num-gpus "${NUM_GPUS}" \
    --disable-usage-stats \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265 \
    --temp-dir "${RAY_TMPDIR}"
}
# --temp-dir "${RAY_TMPDIR}"

build_runtime_env_json() {
  python3 - <<'PY'
import json, os

conda_env = os.environ.get("CONDA_ENV_PATH", "")
py_ver = os.environ.get("CONDA_PYTHON_VERSION", "3.12")
site_packages = f"{conda_env}/lib/python{py_ver}/site-packages" if conda_env else ""

parts = [
  os.environ.get("REPO_ROOT",""),
  os.environ.get("SLIME_PKG_DIR",""),
  os.environ.get("MEGATRON_DIR",""),
  os.environ.get("SCRIPT_DIR",""),
  site_packages,
]
pythonpath = ":".join([p for p in parts if p])

env_vars = {
  "PYTHONPATH": pythonpath,
  "CUDA_DEVICE_MAX_CONNECTIONS": "1",
  "NCCL_NVLS_ENABLE": os.environ.get("HAS_NVLINK","0"),
  "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF",""),
  "USE_REMOTE_ENV": os.environ.get("USE_REMOTE_ENV","0"),
  "ENV_SERVER_URL": os.environ.get("ENV_SERVER_URL",""),
}
print(json.dumps({"env_vars": env_vars}))
PY
}
MEGATRON_LM_PATH=Megatron-LM
MEGATRON_BRIDGE_PATH=Megatron-Bridge/src
SGLANG_PATH=sglang/python
export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-"expandable_segments:True"}


RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_LM_PATH}:${MEGATRON_BRIDGE_PATH}:${SGLANG_PATH}:${SCRIPT_DIR}:${SLIME_DIR}:$PYTHONPATH\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"PYTORCH_NPU_ALLOC_CONF\": \"${PYTORCH_NPU_ALLOC_CONF}\",
    \"ASCEND_TOOLKIT_HOME\": \"/usr/local/Ascend/cann-8.5.0/\",
    \"ASCEND_OPP_PATH\": \"/usr/local/Ascend/cann-8.5.0/opp/\",
    \"ASCEND_AICPU_PATH\": \"/usr/local/Ascend/cann-8.5.0/\",
    \"ASCEND_HOME_PATH\": \"/usr/local/Ascend/cann-8.5.0/\"
  }
}"

submit_job() {
  log "submit ray job"
  local runtime_env_json
  runtime_env_json="$(build_runtime_env_json)"

  ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 ${SLIME_DIR}/train_async.py \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node "${ACTOR_GPUS}" \
    --rollout-num-gpus "${ROLLOUT_GPUS}" \
    --skip-eval-before-train \
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

start_router
check_router

# check_gpus
# detect_nvlink
maybe_fill_env_server_url
export SCRIPT_DIR
start_ray_head
submit_job
