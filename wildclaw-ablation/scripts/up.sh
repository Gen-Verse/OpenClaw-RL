#!/usr/bin/env bash
# 一键拉起 WildClawBench 消融所需的所有组件：
#   1. 前置检查（Docker 镜像 / 任务数据 / sglang / 配置）
#   2. 生成 configs/my_api.json（从环境变量渲染模板）
#   3. 起 SGLang 模型服务（后台 + pidfile + 日志）
#   4. 健康检查（/v1/models 就绪才返回）
#
# 用法:
#   bash up.sh [base|rl] [0p6b|4b|8b]   # 起对应角色的模型服务
#   bash up.sh down                      # 停掉模型服务
#   bash up.sh base 4b                   # 例: Qwen3-4B-Instruct-2507
#
# 关键环境变量:
#   WCB_ROOT     WildClawBench 克隆目录（检查镜像/数据用）
#   BASE_CKPT    base 模型路径     (默认 <repo>/models/qwen3-0.6B)
#   RL_CKPT      RL 模型路径       (默认 <repo>/export/ckpt/wcb_grpo_qlora_ckpt)
#   PORT         服务端口          (默认 8000)
set -euo pipefail

ABLATION_ROOT="${ABLATION_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)}"
REPO_ROOT="$(cd -- "${ABLATION_ROOT}/.." &>/dev/null && pwd)"
ROLE="${1:-base}"
SIZE="${2:-0p6b}"
PORT="${PORT:-8000}"
LOG_DIR="${ABLATION_ROOT}/results/logs"
PID_FILE="${LOG_DIR}/sglang.pid"

case "${SIZE}" in
  0p6b) MODEL_DIR="qwen3-0.6B";               NAME="qwen3-0p6b" ;;
  4b)   MODEL_DIR="Qwen3-4B-Instruct-2507";   NAME="qwen3-4b-instruct" ;;
  8b)   MODEL_DIR="Qwen3-8B";                 NAME="qwen3-8b" ;;
  *) echo "unknown size: ${SIZE} (0p6b|4b|8b)" >&2; exit 2 ;;
esac

BASE_CKPT="${BASE_CKPT:-${REPO_ROOT}/models/${MODEL_DIR}}"
RL_CKPT="${RL_CKPT:-${REPO_ROOT}/export/ckpt/wcb_grpo_qlora_ckpt}"

BASE_MODEL="${NAME}-base"
RL_MODEL="${NAME}-rl"

mkdir -p "${LOG_DIR}"

log() { echo "[up] $*"; }

stop_server() {
  if [[ -f "${PID_FILE}" ]]; then
    local pid
    pid="$(cat "${PID_FILE}")"
    if kill -0 "${pid}" 2>/dev/null; then
      log "stopping sglang pid=${pid}"
      kill "${pid}" || true
      sleep 3
      kill -9 "${pid}" 2>/dev/null || true
    fi
    rm -f "${PID_FILE}"
  fi
  pkill -f "sglang.launch_server" 2>/dev/null || true
}

if [[ "${ROLE}" == "down" ]]; then
  stop_server
  log "stopped."
  exit 0
fi

# ---------- pick ckpt ----------
case "${ROLE}" in
  base) CKPT="${BASE_CKPT}"; SERVED="${BASE_MODEL}" ;;
  rl)   CKPT="${RL_CKPT}";  SERVED="${RL_MODEL}" ;;
  *) echo "unknown role: ${ROLE} (base|rl|down)" >&2; exit 2 ;;
esac

PID_FILE="${LOG_DIR}/sglang_${ROLE}_${SIZE}.pid"

# ---------- preflight ----------
log "preflight checks..."

if [[ ! -d "${CKPT}" ]]; then
  echo "[up] ERROR: model dir not found: ${CKPT}" >&2
  echo "  base 模型放到 ${BASE_CKPT}，或 export BASE_CKPT=/path/to/model" >&2
  exit 1
fi

if ! python3 -c "import sglang" 2>/dev/null; then
  echo "[up] ERROR: sglang not installed in current python env" >&2
  exit 1
fi

if [[ -n "${WCB_ROOT:-}" ]]; then
  if ! docker image inspect wildclawbench-ubuntu:v1.3 >/dev/null 2>&1; then
    echo "[up] WARN: wildclawbench-ubuntu:v1.3 image not loaded; run the image download/load steps first" >&2
  fi
  if [[ ! -d "${WCB_ROOT}/workspace" ]]; then
    echo "[up] WARN: ${WCB_ROOT}/workspace missing; hf download internlm/WildClawBench workspace first" >&2
  fi
fi

# ---------- render my_api.json ----------
MODELS_CONFIG="${ABLATION_ROOT}/configs/my_api.json"
if [[ ! -f "${MODELS_CONFIG}" ]]; then
  log "rendering ${MODELS_CONFIG} from template"
  sed -e "s#http://host.docker.internal:8000/v1#http://host.docker.internal:${PORT}/v1#" \
      -e "s#\${LOCAL_API_KEY}#${LOCAL_API_KEY:-none}#" \
      "${ABLATION_ROOT}/configs/my_api.template.json" > "${MODELS_CONFIG}"
  log "wrote ${MODELS_CONFIG} (baseUrl=http://host.docker.internal:${PORT}/v1)"
fi

# ---------- (re)start server ----------
stop_server

# reasoning parser 只给 thinking 模型；Qwen3-4B-Instruct-2507 是非 thinking，
# 加了会把 content 吞进 reasoning_content、导致 openclaw 流式解析失败。
REASONING_ARGS=()
case "${SIZE}" in
  0p6b) REASONING_ARGS=(--reasoning-parser qwen3) ;;
  4b)    REASONING_ARGS=() ;;
  # 8B thinking model: use a template that emits an empty think block so the
  # answer goes to `content` (openclaw reads content; reasoning_content breaks it)
  8b)    REASONING_ARGS=(--chat-template "${ABLATION_ROOT}/configs/qwen3_nothink_chat_template.jinja") ;;
esac

log "starting sglang: model=${CKPT} served-name=${SERVED} port=${PORT}"
nohup python3 -m sglang.launch_server \
  --model-path "${CKPT}" \
  --served-model-name "${SERVED}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --mem-fraction-static "${MEM_FRACTION:-0.85}" \
  "${REASONING_ARGS[@]}" \
  --tool-call-parser qwen25 \
  > "${LOG_DIR}/sglang_${ROLE}_${SIZE}.log" 2>&1 &
echo $! > "${PID_FILE}"

# ---------- health check ----------
log "waiting for server ready (log: ${LOG_DIR}/sglang_${ROLE}_${SIZE}.log) ..."
for i in $(seq 1 "${WAIT_SECS:-600}"); do
  if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    log "server ready at http://127.0.0.1:${PORT} (served-model-name=${SERVED})"
    echo
    echo "next:"
    echo "  export WCB_ROOT=${WCB_ROOT:-<wildclawbench clone>}"
    echo "  export BASE_MODEL=${SERVED} ABLATION_ROOT=${ABLATION_ROOT}"
    echo "  bash ${ABLATION_ROOT}/scripts/run_cycle.sh        # 完整闭环"
    echo "  # 或只跑 skill 进化:"
    echo "  bash ${ABLATION_ROOT}/scripts/run_tasks.sh train local/${SERVED} 0"
    exit 0
  fi
  if ! kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
    echo "[up] ERROR: server died; see ${LOG_DIR}/sglang_${ROLE}_${SIZE}.log" >&2
    exit 1
  fi
  sleep 2
done

echo "[up] ERROR: server not ready after ${WAIT_SECS:-600}s; see ${LOG_DIR}/sglang_${ROLE}.log" >&2
exit 1
