#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
VARIANT="${2:-}"
if [[ "${MODE}" != "train" && "${MODE}" != "eval" ]]; then
  echo "usage: $0 <train|eval> <rl_only|rl_skill|base|skill_only>" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
TERMINAL_RL="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${TERMINAL_RL}/.." &>/dev/null && pwd)"
RESULT_ROOT="${TERMINAL_RESULT_ROOT:-${TERMINAL_RL}/outputs}"
SKILLS_DIR="${TERMINAL_SKILLS_DIR:-${REPO_ROOT}/slime-coding-agent/skills/generated}"

case "${VARIANT}" in
  base)
    export MODEL_CKPT="${BASE_CKPT:?BASE_CKPT is required}"
    export TERMINAL_SKILL_RETRIEVAL=0
    ;;
  rl_only)
    export MODEL_CKPT="${RL_CKPT:?RL_CKPT is required}"
    export TERMINAL_SKILL_RETRIEVAL=0
    ;;
  skill_only)
    export MODEL_CKPT="${BASE_CKPT:?BASE_CKPT is required}"
    export TERMINAL_SKILL_RETRIEVAL=1
    ;;
  rl_skill)
    export MODEL_CKPT="${RL_CKPT:?RL_CKPT is required}"
    export TERMINAL_SKILL_RETRIEVAL=1
    ;;
  *)
    echo "unknown variant: ${VARIANT}" >&2
    exit 2
    ;;
esac

export TERMINAL_SKILLS_DIR="${SKILLS_DIR}"
export TERMINAL_EVAL_RESULTS_PATH="${RESULT_ROOT}/eval/${VARIANT}.jsonl"

if [[ "${MODE}" == "train" ]]; then
  if [[ "${VARIANT}" != "rl_only" && "${VARIANT}" != "rl_skill" ]]; then
    echo "only rl_only and rl_skill are trainable variants" >&2
    exit 2
  fi
  export HF_CKPT="${MODEL_CKPT}"
  export ROLLOUT_PROMPT_DATA="${ROLLOUT_PROMPT_DATA:?ROLLOUT_PROMPT_DATA is required}"
  export EVAL_PROMPT_DATA="${EVAL_PROMPT_DATA:?EVAL_PROMPT_DATA is required}"
  export TERMINAL_METRICS_PATH="${RESULT_ROOT}/metrics/${VARIANT}.jsonl"
  export TERMINAL_TRAJECTORY_LOG="${RESULT_ROOT}/trajectories/${VARIANT}.jsonl"
  bash "${TERMINAL_RL}/terminal-rl_qwen3-8b.sh"
else
  export HF_CKPT="${MODEL_CKPT}"
  export EVAL_PROMPT_DATA="${EVAL_PROMPT_DATA:?EVAL_PROMPT_DATA is required}"
  bash "${TERMINAL_RL}/terminal-eval_qwen3-8b.sh"
fi
