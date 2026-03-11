#!/bin/bash
# Eval script: run a stronger LLM (e.g. Qwen3-32B) on SWE training data.
# Starts sglang + swe_env_pool_server on the current 8-GPU machine, then
# runs eval_swe.py with the desired concurrency.
#
# Usage (from dev machine, single-node job):
#   bash run_eval_swe.sh
#
# Key env vars you can override before calling:
#   EVAL_MODEL_PATH   HF checkpoint of the eval model
#   EVAL_MODEL_NAME   LiteLLM model name passed to litellm (default: openai/eval_model)
#   EVAL_TP           Tensor-parallel degree for sglang (default: 4)
#   EVAL_GPU_IDS      Comma-separated CUDA device IDs for sglang (default: 0,1,2,3)
#   MAX_CONCURRENT    Concurrent Docker containers (default: 8)
#   MAX_INSTANCES     Number of instances to eval (default: all)
#   STEP_LIMIT        Max agent steps per instance (default: 20)
#   MAX_TOKENS        Max tokens per LLM call (default: 8192)
#   PROMPT_DATA       Path to JSONL training data
#   SWE_EXEC_SERVER_URLS  Comma-sep exec server URLs (docker exec nodes)

set -e

# ─────────────────────────────────────────────────────────────────────────────
# 0. Paths & environment
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SWE_RL_DIR="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
ROOT=/data_storage/wyj
SLIME_DIR="$(cd -- "${SWE_RL_DIR}/../slime" &>/dev/null && pwd)"

RUN_TIMESTAMP=$(date +%F_%H%M%S)
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"
RUN_LOG="${LOG_DIR}/eval_stronger_model_${RUN_TIMESTAMP}.log"
exec > >(tee -a "${RUN_LOG}") 2>&1
echo "Run log: ${RUN_LOG}"
echo "Run timestamp: ${RUN_TIMESTAMP}"

# Activate conda env
ENV_PATH=${ROOT}/systems/envs/agentic-rl-jxl
source activate "${ENV_PATH}"
export PATH="${ENV_PATH}/bin:${PATH}"
echo "python=$(which python)"

# Load .env.swe (WANDB_API_KEY, SWE config, etc.)
if [[ -f "${SWE_RL_DIR}/.env.swe" ]]; then
  set -a; source "${SWE_RL_DIR}/.env.swe"; set +a
fi

# ─────────────────────────────────────────────────────────────────────────────
# 1. Configuration — edit these or override via env vars
# ─────────────────────────────────────────────────────────────────────────────

# --- Eval LLM ---
# Qwen3-32B: needs ~64 GB in bf16. TP=4 on 4×80 GB GPUs is comfortable.
# Adjust EVAL_TP and EVAL_GPU_IDS based on your actual GPU setup.
EVAL_MODEL_PATH=${EVAL_MODEL_PATH:-${ROOT}/systems/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/latest}
EVAL_MODEL_NAME=${EVAL_MODEL_NAME:-openai/eval_model}   # name passed to litellm
EVAL_TP=${EVAL_TP:-4}                                   # tensor-parallel degree
EVAL_GPU_IDS=${EVAL_GPU_IDS:-0,1,2,3}                  # GPUs used by sglang
SGLANG_PORT=${SGLANG_PORT:-18080}
SGLANG_HOST=127.0.0.1

# --- Dataset & eval params ---
PROMPT_DATA=${PROMPT_DATA:-/data_storage/wyj/swe_gym_subset/train.jsonl}
DATA_SOURCE=${DATA_SOURCE:-swe-gym}
MAX_CONCURRENT=${MAX_CONCURRENT:-8}    # concurrent docker containers
MAX_INSTANCES=${MAX_INSTANCES:-}       # empty = run all; set e.g. 100 for a subset
STEP_LIMIT=${STEP_LIMIT:-20}
MAX_TOKENS=${MAX_TOKENS:-8192}
TEMPERATURE=${TEMPERATURE:-0}          # 0 = greedy for deterministic eval

# --- Docker exec servers (same as training) ---
export SWE_EXEC_SERVER_URLS=${SWE_EXEC_SERVER_URLS:-http://192.168.0.15:5000}

# --- Output ---
OUTPUT_DIR=${OUTPUT_DIR:-${SWE_RL_DIR}/output/eval_runs/$(echo "${EVAL_MODEL_NAME}" | tr '/:' '__')_${RUN_TIMESTAMP}}
mkdir -p "${OUTPUT_DIR}"

# --- Pool server ---
SWE_ENV_SERVER_PORT=${SWE_ENV_SERVER_PORT:-18090}
SWE_ENV_SERVER_URL=http://127.0.0.1:${SWE_ENV_SERVER_PORT}
SWE_MAX_CONTAINERS_PER_NODE=${SWE_MAX_CONTAINERS_PER_NODE:-16}

# ─────────────────────────────────────────────────────────────────────────────
# 2. Validate inputs
# ─────────────────────────────────────────────────────────────────────────────
if [[ ! -f "${PROMPT_DATA}" ]]; then
  echo "ERROR: PROMPT_DATA not found: ${PROMPT_DATA}"
  exit 1
fi

# Resolve model path (handle glob for snapshot dirs)
if [[ ! -d "${EVAL_MODEL_PATH}" ]]; then
  # Try to find actual snapshot dir
  SNAP=$(ls -d ${ROOT}/systems/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/*/ 2>/dev/null | head -1)
  if [[ -n "${SNAP}" ]]; then
    EVAL_MODEL_PATH="${SNAP%/}"
    echo "Auto-resolved EVAL_MODEL_PATH=${EVAL_MODEL_PATH}"
  else
    echo "ERROR: EVAL_MODEL_PATH not found: ${EVAL_MODEL_PATH}"
    echo "  Set EVAL_MODEL_PATH=/path/to/Qwen3-32B and retry."
    exit 1
  fi
fi

echo "============================================================"
echo "  Eval model   : ${EVAL_MODEL_NAME}"
echo "  Model path   : ${EVAL_MODEL_PATH}"
echo "  TP degree    : ${EVAL_TP}  GPUs: ${EVAL_GPU_IDS}"
echo "  Data         : ${PROMPT_DATA}"
echo "  Data source  : ${DATA_SOURCE}"
echo "  Max instances: ${MAX_INSTANCES:-all}"
echo "  Max concurrent: ${MAX_CONCURRENT}"
echo "  Step limit   : ${STEP_LIMIT}"
echo "  Max tokens   : ${MAX_TOKENS}"
echo "  Output       : ${OUTPUT_DIR}"
echo "  Exec servers : ${SWE_EXEC_SERVER_URLS}"
echo "============================================================"

# ─────────────────────────────────────────────────────────────────────────────
# 3. Cleanup trap
# ─────────────────────────────────────────────────────────────────────────────
SGLANG_PID=""
POOL_PID=""

cleanup() {
  set +e
  echo "Cleaning up background processes..."
  [[ -n "${POOL_PID}" ]]   && kill "${POOL_PID}"   2>/dev/null
  [[ -n "${SGLANG_PID}" ]] && kill "${SGLANG_PID}" 2>/dev/null
  sleep 2
  [[ -n "${SGLANG_PID}" ]] && kill -9 "${SGLANG_PID}" 2>/dev/null
  pkill -f "sglang.launch_server" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ─────────────────────────────────────────────────────────────────────────────
# 3b. Bypass proxy for local services (sglang warmup hits 127.0.0.1)
# ─────────────────────────────────────────────────────────────────────────────
NODE_IP=$(hostname -I | awk '{print $1}')
ALL_EXEC_HOSTS="$(echo "${SWE_EXEC_SERVER_URLS}" | tr ',' '\n' | sed -E 's#https?://([^:/]+).*#\1#' | tr '\n' ',' | sed 's/,$//')"
export NO_PROXY="localhost,127.0.0.1,${NODE_IP},${ALL_EXEC_HOSTS}"
export no_proxy="${NO_PROXY}"
echo "NO_PROXY=${NO_PROXY}"

# ─────────────────────────────────────────────────────────────────────────────
# 4. Start sglang server for the eval model
# ─────────────────────────────────────────────────────────────────────────────
SGLANG_LOG="${LOG_DIR}/sglang_eval_${RUN_TIMESTAMP}.log"
echo "Starting sglang for ${EVAL_MODEL_NAME} on GPUs ${EVAL_GPU_IDS} (TP=${EVAL_TP})..."

CUDA_VISIBLE_DEVICES=${EVAL_GPU_IDS} \
NO_PROXY="localhost,127.0.0.1,${NODE_IP}" \
no_proxy="localhost,127.0.0.1,${NODE_IP}" \
HTTP_PROXY="" http_proxy="" HTTPS_PROXY="" https_proxy="" \
python -m sglang.launch_server \
  --model-path "${EVAL_MODEL_PATH}" \
  --served-model-name "eval_model" \
  --tp "${EVAL_TP}" \
  --host 0.0.0.0 \
  --port "${SGLANG_PORT}" \
  --trust-remote-code \
  --mem-fraction-static 0.85 \
  --skip-server-warmup \
  > "${SGLANG_LOG}" 2>&1 &
SGLANG_PID=$!
echo "sglang PID=${SGLANG_PID}, log=${SGLANG_LOG}"

# Wait for sglang to be ready
echo "Waiting for sglang to be ready..."
for i in $(seq 1 120); do
  if curl -fsS "http://${SGLANG_HOST}:${SGLANG_PORT}/health" >/dev/null 2>&1; then
    echo "sglang ready after ${i}×5s"
    break
  fi
  if ! kill -0 "${SGLANG_PID}" 2>/dev/null; then
    echo "ERROR: sglang died. Check log: ${SGLANG_LOG}"
    exit 1
  fi
  sleep 5
done

# Quick smoke test
echo "Smoke-testing sglang endpoint..."
curl -s "http://${SGLANG_HOST}:${SGLANG_PORT}/v1/models" | python3 -c "
import sys, json
d = json.load(sys.stdin)
models = [m['id'] for m in d.get('data', [])]
print('  Available models:', models)
" || { echo "ERROR: sglang smoke test failed"; exit 1; }

# ─────────────────────────────────────────────────────────────────────────────
# 5. Start swe_env_pool_server
# ─────────────────────────────────────────────────────────────────────────────
POOL_LOG="${LOG_DIR}/swe_env_pool_eval_${RUN_TIMESTAMP}.log"
echo "Starting swe_env_pool_server on port ${SWE_ENV_SERVER_PORT}..."

PYTHONPATH="${SLIME_DIR}:${SWE_RL_DIR}:${SWE_RL_DIR}/server:${SCRIPT_DIR}:${PYTHONPATH}" \
python3 -m swe_env_pool_server \
  --host 0.0.0.0 \
  --port "${SWE_ENV_SERVER_PORT}" \
  --exec-server-urls "${SWE_EXEC_SERVER_URLS}" \
  --max-containers-per-node "${SWE_MAX_CONTAINERS_PER_NODE}" \
  > "${POOL_LOG}" 2>&1 &
POOL_PID=$!
echo "Pool server PID=${POOL_PID}, log=${POOL_LOG}"

# Wait for pool server to be ready
for i in $(seq 1 60); do
  if curl -fsS "${SWE_ENV_SERVER_URL}/healthz" >/dev/null 2>&1; then
    echo "Pool server ready after ${i}×2s"
    break
  fi
  sleep 2
done

# Check exec server health
IFS=',' read -r -a _exec_urls <<< "${SWE_EXEC_SERVER_URLS}"
for exec_url in "${_exec_urls[@]}"; do
  if ! curl -fsS --max-time 8 "${exec_url}/healthz" >/dev/null 2>&1; then
    echo "ERROR: exec server not healthy: ${exec_url}/healthz"
    exit 1
  fi
  echo "Exec server OK: ${exec_url}"
done

# ─────────────────────────────────────────────────────────────────────────────
# 6. Run evaluation
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Starting evaluation..."
echo "============================================================"

EXTRA_ARGS=()
[[ -n "${MAX_INSTANCES}" ]] && EXTRA_ARGS+=(--max-instances "${MAX_INSTANCES}")

PYTHONPATH="${SLIME_DIR}:${SWE_RL_DIR}:${SWE_RL_DIR}/server:${SCRIPT_DIR}:${PYTHONPATH}" \
OPENAI_BASE_URL="http://${SGLANG_HOST}:${SGLANG_PORT}/v1" \
OPENAI_API_KEY=dummy \
SWE_ENV_SERVER_URL="${SWE_ENV_SERVER_URL}" \
NO_PROXY="${NO_PROXY}" no_proxy="${NO_PROXY}" \
HTTP_PROXY="" http_proxy="" HTTPS_PROXY="" https_proxy="" \
python3 "${SCRIPT_DIR}/eval_swe.py" \
  --data           "${PROMPT_DATA}" \
  --data-source    "${DATA_SOURCE}" \
  --model          "${EVAL_MODEL_NAME}" \
  --api-base       "http://${SGLANG_HOST}:${SGLANG_PORT}/v1" \
  --output-dir     "${OUTPUT_DIR}" \
  --max-concurrent "${MAX_CONCURRENT}" \
  --step-limit     "${STEP_LIMIT}" \
  --max-tokens     "${MAX_TOKENS}" \
  --temperature    "${TEMPERATURE}" \
  "${EXTRA_ARGS[@]}"

echo ""
echo "============================================================"
echo "  Eval complete!"
echo "  Results: ${OUTPUT_DIR}/summary.json"
echo "============================================================"
cat "${OUTPUT_DIR}/summary.json" 2>/dev/null || true
