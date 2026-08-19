#!/usr/bin/env bash
# 跑一个消融变体：只在 held-out eval 切分上评测（不碰 train 切分）。
# 用法: SIZE=<0p6b|4b|8b> bash run_variant.sh <base|skill_only|rl_only|rl_skill> [task.md 冒烟]
set -euo pipefail

VARIANT="${1:-}"
if [[ -z "${VARIANT}" ]]; then
  echo "usage: $0 <base|skill_only|rl_only|rl_skill> [task.md]" >&2
  exit 2
fi

ABLATION_ROOT="${ABLATION_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)}"

# 尺寸 -> served model name（与 up.sh 对齐）
SIZE="${SIZE:-0p6b}"
case "${SIZE}" in
  0p6b) SIZE_NAME="qwen3-0p6b" ;;
  4b)   SIZE_NAME="qwen3-4b-instruct" ;;
  8b)   SIZE_NAME="qwen3-8b" ;;
  *) echo "unknown SIZE: ${SIZE} (0p6b|4b|8b)" >&2; exit 2 ;;
esac

INJECT_SKILLS=0
case "${VARIANT}" in
  base)       MODEL="${BASE_MODEL:-${SIZE_NAME}-base}" ;;
  skill_only) MODEL="${BASE_MODEL:-${SIZE_NAME}-base}"; INJECT_SKILLS=1 ;;
  rl_only)    MODEL="${RL_MODEL:-${SIZE_NAME}-rl}" ;;
  rl_skill)   MODEL="${RL_MODEL:-${SIZE_NAME}-rl}"; INJECT_SKILLS=1 ;;
  *) echo "unknown variant: ${VARIANT}" >&2; exit 2 ;;
esac
MODEL_ID="${MODEL_ID:-local/${MODEL}}"

echo "[variant] ${VARIANT} model=${MODEL_ID} skills=${INJECT_SKILLS}"

if [[ -n "${2:-}" ]]; then
  # 冒烟：单任务直跑
  RUN_NAME="${VARIANT}" \
  bash "${ABLATION_ROOT}/scripts/run_tasks.sh" "${2}" "${MODEL_ID}" "${INJECT_SKILLS}"
else
  # 正式：held-out eval 切分
  RUN_NAME="${VARIANT}" \
  bash "${ABLATION_ROOT}/scripts/run_tasks.sh" eval "${MODEL_ID}" "${INJECT_SKILLS}"
fi
