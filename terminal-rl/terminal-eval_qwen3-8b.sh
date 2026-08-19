#!/usr/bin/env bash
set -euo pipefail

require() {
  local name="$1"
  local value="$2"
  if [[ -z "${value}" ]]; then
    echo "[ERROR] ${name} is required" >&2
    exit 1
  fi
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
SLIME_DIR="${REPO_ROOT}/slime"

HF_CKPT="${HF_CKPT:-${MODEL_CKPT:-}}"
EVAL_PROMPT_DATA="${EVAL_PROMPT_DATA:-}"
ENV_SERVER_URL="${ENV_SERVER_URL:-}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-4}"
ROLLOUT_GPUS_PER_ENGINE="${ROLLOUT_GPUS_PER_ENGINE:-2}"
N_SAMPLES_PER_EVAL_PROMPT="${N_SAMPLES_PER_EVAL_PROMPT:-8}"
TERMINAL_EVAL_RESULTS_PATH="${TERMINAL_EVAL_RESULTS_PATH:-${REPO_ROOT}/terminal-rl/outputs/eval/results.jsonl}"

require "HF_CKPT or MODEL_CKPT" "${HF_CKPT}"
require "EVAL_PROMPT_DATA" "${EVAL_PROMPT_DATA}"
require "ENV_SERVER_URL" "${ENV_SERVER_URL}"

source "${SLIME_DIR}/scripts/models/qwen3-8B.sh"

export PYTHONUNBUFFERED=1
export REPO_ROOT
export SLIME_DIR
export TERMINAL_EVAL_RESULTS_PATH
export TERMINAL_SKILL_RETRIEVAL="${TERMINAL_SKILL_RETRIEVAL:-0}"
export TERMINAL_SKILLS_DIR="${TERMINAL_SKILLS_DIR:-}"
export TERMINAL_SKILL_TOP_K="${TERMINAL_SKILL_TOP_K:-3}"

ray stop --force || true
ray start --head --node-ip-address "${MASTER_ADDR:-127.0.0.1}" \
  --num-gpus "${ROLLOUT_GPUS}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="$(python3 - <<'PY'
import json
import os

parts = [
    os.environ.get("REPO_ROOT", ""),
    os.environ.get("SLIME_DIR", ""),
    os.path.join(os.environ.get("REPO_ROOT", ""), "Megatron-LM"),
    os.path.join(os.environ.get("REPO_ROOT", ""), "terminal-rl"),
]
env_vars = {
    "PYTHONPATH": ":".join(path for path in parts if path),
    "ENV_SERVER_URL": os.environ.get("ENV_SERVER_URL", ""),
    "TERMINAL_EVAL_RESULTS_PATH": os.environ.get("TERMINAL_EVAL_RESULTS_PATH", ""),
    "TERMINAL_SKILL_RETRIEVAL": os.environ.get("TERMINAL_SKILL_RETRIEVAL", "0"),
    "TERMINAL_SKILLS_DIR": os.environ.get("TERMINAL_SKILLS_DIR", ""),
    "TERMINAL_SKILL_TOP_K": os.environ.get("TERMINAL_SKILL_TOP_K", "3"),
}
print(json.dumps({"env_vars": env_vars}))
PY
)"

ray job submit --address="http://127.0.0.1:8265" --runtime-env-json="${RUNTIME_ENV_JSON}" -- \
  python3 "${SLIME_DIR}/eval_only.py" \
  --actor-num-nodes 0 \
  --actor-num-gpus-per-node 0 \
  --rollout-num-gpus "${ROLLOUT_GPUS}" \
  --rollout-num-gpus-per-engine "${ROLLOUT_GPUS_PER_ENGINE}" \
  --hf-checkpoint "${HF_CKPT}" \
  --ref-load "${HF_CKPT}" \
  --eval-prompt-data terminal_heldout "${EVAL_PROMPT_DATA}" \
  --eval-input-key task \
  --n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL_PROMPT}" \
  --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN:-8192}" \
  --eval-max-context-len "${EVAL_MAX_CONTEXT_LEN:-16384}" \
  --rollout-temperature "${EVAL_TEMPERATURE:-0}" \
  --custom-generate-function-path generate.generate \
  --custom-eval-rollout-log-function-path terminal_eval_log.log_eval_rollout_data \
  "${MODEL_ARGS[@]}"

python3 "${SCRIPT_DIR}/terminal_report.py" \
  --results "${TERMINAL_EVAL_RESULTS_PATH}" \
  --output "${TERMINAL_EVAL_RESULTS_PATH%.jsonl}.summary.json" \
  --pass-at-k "${N_SAMPLES_PER_EVAL_PROMPT}"
