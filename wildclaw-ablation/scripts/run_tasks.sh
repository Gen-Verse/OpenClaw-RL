#!/usr/bin/env bash
# 按任务清单逐个跑 WildClawBench（run_batch.py 单次只接一个 --task）。
# 用法: bash run_tasks.sh <train|eval|任务列表文件|单个task.md> <model_id> [inject_skills 0|1]
# ROLLOUTS_PER_TASK: 每个任务重复次数（GRPO 组内需要 >1 条 rollout）
set -euo pipefail

SPLIT_KEY="${1:?train|eval|path-to-list}"
MODEL_ID="${2:?model id, e.g. local/qwen3-0p6b-base}"
INJECT_SKILLS="${3:-0}"

ABLATION_ROOT="${ABLATION_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)}"
WCB_ROOT="${WCB_ROOT:?set WCB_ROOT}"
REPO_ROOT="$(cd -- "${ABLATION_ROOT}/.." &>/dev/null && pwd)"
SPLIT_JSON="${SPLIT_JSON:-${ABLATION_ROOT}/configs/split.json}"
MODELS_CONFIG="${MODELS_CONFIG:-${ABLATION_ROOT}/configs/my_api.json}"
SKILLS_SRC="${SKILLS_SRC:-${ABLATION_ROOT}/skills/generated}"
RUN_NAME="${RUN_NAME:-run}"
RESULTS_DIR="${ABLATION_ROOT}/results/${RUN_NAME}"
ROLLOUTS_PER_TASK="${ROLLOUTS_PER_TASK:-1}"

# ---- resolve task list ----
if [[ "${SPLIT_KEY}" == "train" || "${SPLIT_KEY}" == "eval" ]]; then
  LIST_FILE="${RESULTS_DIR}/.tasks_${SPLIT_KEY}.txt"
  mkdir -p "${RESULTS_DIR}"
  python3 - "${SPLIT_JSON}" "${SPLIT_KEY}" > "${LIST_FILE}" <<'PY'
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
print("\n".join(data[sys.argv[2]]))
PY
elif [[ "${SPLIT_KEY}" == *.md ]]; then
  LIST_FILE="${RESULTS_DIR}/.tasks_single.txt"
  mkdir -p "${RESULTS_DIR}"
  printf '%s\n' "${SPLIT_KEY}" > "${LIST_FILE}"
else
  LIST_FILE="${SPLIT_KEY}"
fi

# ---- lobster workspace (skill variants) ----
LOBSTER_ARGS=()
if [[ "${INJECT_SKILLS}" == "1" ]]; then
  WORKSPACE_DIR="${RESULTS_DIR}/lobster_workspace"
  rm -rf "${WORKSPACE_DIR}"
  mkdir -p "${WORKSPACE_DIR}/skills"
  count=0
  if [[ -d "${SKILLS_SRC}" ]]; then
    # EvolvingSkillStore 布局: <skills_root>/<name>/SKILL.md
    for d in "${SKILLS_SRC}"/*/; do
      [[ -f "${d}SKILL.md" ]] || continue
      name="$(basename "${d}")"
      mkdir -p "${WORKSPACE_DIR}/skills/${name}"
      cp "${d}SKILL.md" "${WORKSPACE_DIR}/skills/${name}/SKILL.md"
      count=$((count + 1))
    done
  fi
  echo "[run_tasks] skills injected: ${count}"
  LOBSTER_ARGS=(--lobster-workspace "${WORKSPACE_DIR}")
fi

mkdir -p "${RESULTS_DIR}/raw"
cd "${WCB_ROOT}"

total=0
for ((rep=1; rep<=ROLLOUTS_PER_TASK; rep++)); do
  while IFS= read -r task; do
    [[ -n "${task}" ]] || continue
    total=$((total + 1))
    echo "[run_tasks] (${total}, rep ${rep}/${ROLLOUTS_PER_TASK}) ${task}"
    python3 eval/run_batch.py \
      --task "${task}" \
      --models-config "${MODELS_CONFIG}" \
      --model "${MODEL_ID}" \
      "${LOBSTER_ARGS[@]}"
    if [[ -d output ]]; then
      mkdir -p "${RESULTS_DIR}/raw"
      cp -r output/. "${RESULTS_DIR}/raw/"
    fi
  done < "${LIST_FILE}"
done

if [[ -f output/summary_all.json ]]; then
  cp output/summary_all.json "${RESULTS_DIR}/summary_all.json"
fi
echo "[run_tasks] done: ${total} task(s) -> ${RESULTS_DIR}"
