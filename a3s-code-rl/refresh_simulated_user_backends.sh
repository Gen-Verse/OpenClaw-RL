#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

DEFAULT_A3S_CODE_PYTHON_BIN="python3"
if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python3" ]]; then
  DEFAULT_A3S_CODE_PYTHON_BIN="${CONDA_PREFIX}/bin/python3"
elif [[ -n "${CONDA_ENV:-}" && -x "${CONDA_ENV}/bin/python3" ]]; then
  DEFAULT_A3S_CODE_PYTHON_BIN="${CONDA_ENV}/bin/python3"
fi

export A3S_CODE_PYTHON_BIN="${A3S_CODE_PYTHON_BIN:-${DEFAULT_A3S_CODE_PYTHON_BIN}}"
export A3S_CODE_CONFIG_ROOT="${A3S_CODE_CONFIG_ROOT:-${SCRIPT_DIR}/generated_configs}"
export A3S_CODE_SIMULATED_USER_BACKENDS_FILE="${A3S_CODE_SIMULATED_USER_BACKENDS_FILE:-${SCRIPT_DIR}/simulated_user_backends.json}"
export A3S_CODE_SIMULATED_USER_PROBE_TIMEOUT_SEC="${A3S_CODE_SIMULATED_USER_PROBE_TIMEOUT_SEC:-20}"
export A3S_CODE_SIMULATED_USER_SLOW_LATENCY_MS="${A3S_CODE_SIMULATED_USER_SLOW_LATENCY_MS:-3000}"

export SIMULATED_USER_PROBE_MODEL_URLS="${SIMULATED_USER_PROBE_MODEL_URLS:-http://s-20260204175507-cqflp.ailab-pj.pjh-service.org.cn/v1/chat/completions,http://s-20260304112131-p7qvl.ailab-pj.pjh-service.org.cn/v1/chat/completions,http://s-20260304115506-xsb7j.ailab-pj.pjh-service.org.cn/v1/chat/completions,http://s-20260304151647-c9kjf.ailab-pj.pjh-service.org.cn/v1/chat/completions,http://s-20260304094348-lphmr.ailab-pj.pjh-service.org.cn/v1/chat/completions,http://s-20250920125849-njm55.ailab-pj.pjh-service.org.cn/v1/chat/completions,http://s-20260129181145-27qvr.ailab-pj.pjh-service.org.cn/v1/chat/completions}"
export SIMULATED_USER_PROBE_MODEL_NAMES="${SIMULATED_USER_PROBE_MODEL_NAMES:-kimi-k2.5,qwen3.5-397b,minimax2.5,glm-5,deepseek-v3.2,intern-s1,intern-s1-pro}"
export SIMULATED_USER_PROBE_API_KEYS="${SIMULATED_USER_PROBE_API_KEYS:-,,,,,,}"

export SIMULATED_USER_MODEL_URLS="${SIMULATED_USER_PROBE_MODEL_URLS}"
export SIMULATED_USER_MODEL_NAMES="${SIMULATED_USER_PROBE_MODEL_NAMES}"
export SIMULATED_USER_API_KEYS="${SIMULATED_USER_PROBE_API_KEYS}"

mkdir -p "${A3S_CODE_CONFIG_ROOT}"

"${A3S_CODE_PYTHON_BIN}" -u "${SCRIPT_DIR}/check_simulated_user_backends.py" "$@"
