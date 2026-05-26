#!/usr/bin/env bash
# TB-overfit capacity probe: train Qwen3-8B directly on tbench_test 86 task.
# Wraps run_qwen3_8b_experiment.sh, only overrides:
#   - ROLLOUT_PROMPT_DATA → tbench_test_convert/train.jsonl
#   - SAVE_CKPT path → qwen3-8b-tboverfit
#   - WANDB_GROUP → qwen3-8b-tboverfit-probe
#   - Fresh ckpt dir (no resume from run-3)
#
# Use this ckpt ONLY for capacity probe — DO NOT submit to TB leaderboard
# (eval set used as training set → inherently contaminated).
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export ROLLOUT_PROMPT_DATA="${SCRIPT_DIR}/dataset/tbench_test_convert/train.jsonl"
export SAVE_CKPT="${SCRIPT_DIR}/ckpt/qwen3-8b-tboverfit"
export RESUME_LOAD="${SAVE_CKPT}"
export WANDB_GROUP="qwen3-8b-tboverfit-probe"

# Sanity: verify ROLLOUT_PROMPT_DATA exists + has 86 lines (tbench_test full set)
if [[ ! -f "${ROLLOUT_PROMPT_DATA}" ]]; then
    echo "[ERROR] ROLLOUT_PROMPT_DATA not found: ${ROLLOUT_PROMPT_DATA}"; exit 1
fi
n_tasks=$(wc -l < "${ROLLOUT_PROMPT_DATA}")
echo "[tboverfit] training data: ${ROLLOUT_PROMPT_DATA} (${n_tasks} tasks)"
echo "[tboverfit] save_ckpt:     ${SAVE_CKPT}"
echo "[tboverfit] wandb_group:   ${WANDB_GROUP}"

# Make sure ckpt dir is fresh
mkdir -p "${SAVE_CKPT}"

exec "${SCRIPT_DIR}/run_qwen3_8b_experiment.sh"
