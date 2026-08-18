#!/bin/bash

# SWE-Bench RL — default algorithm (trajectory mode).
#
# Qwen3-4B-Instruct-2507, 4 GPU nodes (32 GPU), colocate actor+rollout.
# Scaffold (bash backtick, submit sentinel, remote Docker pool) is IDENTICAL to
# the baseline per-turn swe-rl script; only the RL core changes.
#
# Algorithm profile (see docs/four-way-algorithm-comparison.md):
#   * Sample organization: trajectory mode (1 trajectory = 1 Sample)
#   * Advantage estimator: GRPO (with std normalization)
#   * KL loss:             coef=0 (nominally on, effectively off)
#   * Clip ratio:          0.2 / 0.28 (DAPO asymmetric)
#   * Entropy:             0
#   * No compact filter, no Dr.GRPO length norm (those are rllm variant)
#
# Differences from run_swe_rl_4b_remote_4nodes_colocate_b32.sh (per-turn baseline):
#   * Drops --dynamic_history (trajectory mode doesn't need per-turn → broadcast)
#   * KL coef 0.0001 → 0.0
#   * eps-clip-high 0.2 → 0.28 (DAPO)
#   * custom-generate/rm path → generate_with_swe.{generate,reward_func}

pkill -9 sglang || true
sleep 3
ray stop --force || true
pkill -9 ray || true
pkill -9 python || true
sleep 3
pkill -9 ray || true
pkill -9 python || true

set -ex

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SWE_RL_DIR="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
SLIME_DIR="$(cd -- "${SWE_RL_DIR}/../slime" &>/dev/null && pwd)"
EXPORT_ROOT=${EXPORT_ROOT:-"${SWE_RL_DIR}/../export"}
mkdir -p "${EXPORT_ROOT}/ckpt" "${EXPORT_ROOT}/swe_rollouts"
RUN_TIMESTAMP=${RUN_TIMESTAMP:-$(date +%F_%H%M%S)}
LOG_DIR=${LOG_DIR:-"${SCRIPT_DIR}/logs"}
mkdir -p "${LOG_DIR}"
RUN_LOG=${RUN_LOG:-"${LOG_DIR}/run_swe_rl_4b_${RUN_TIMESTAMP}.log"}
exec > >(tee -a "${RUN_LOG}") 2>&1
echo "Run log: ${RUN_LOG}"
echo "Run timestamp: ${RUN_TIMESTAMP}"

source "${SLIME_DIR}/scripts/models/qwen3-4B-Instruct-2507.sh"
MEGATRON_LM_PATH=${MEGATRON_LM_PATH:-"${SWE_RL_DIR}/../Megatron-LM"}

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export RAY_health_check_failure_threshold=${RAY_health_check_failure_threshold:-20}
export RAY_health_check_period_ms=${RAY_health_check_period_ms:-5000}
export RAY_health_check_timeout_ms=${RAY_health_check_timeout_ms:-30000}
export RAY_num_heartbeats_timeout=${RAY_num_heartbeats_timeout:-60}

# ---- Cluster topology (colocate: actor and rollout share all GPUs) ----
NUM_NODES=${NUM_NODES:-4}
NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-8}
# All 32 GPUs shared: TP=4, DP=32/4=8; rollout TP=2 → 16 engines
ACTOR_GPUS_PER_NODE=${ACTOR_GPUS_PER_NODE:-8}
ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-2}

MLP_ROLE_INDEX=${MLP_ROLE_INDEX:-0}
MASTER_ADDR="${MLP_WORKER_0_HOST:-${MASTER_ADDR:-$(hostname -I | awk '{print $1}')}}"
_WORKER_IP_VAR="MLP_WORKER_${MLP_ROLE_INDEX}_HOST"
NODE_IP="${!_WORKER_IP_VAR:-${WORKER_IP:-$(hostname -I | awk '{print $1}')}}"
export MASTER_ADDR
echo "MLP_ROLE_INDEX=${MLP_ROLE_INDEX}, MASTER_ADDR=${MASTER_ADDR}, NODE_IP=${NODE_IP}"

# Load environment overrides early (before remote Docker config and pre-flight cleanup).
if [[ -f "${SWE_RL_DIR}/.env.swe" ]]; then
  source "${SWE_RL_DIR}/.env.swe"
fi
# Load .env for WANDB_API_KEY and other secrets
if [[ -f "${SWE_RL_DIR}/../.env" ]]; then
  set -a; source "${SWE_RL_DIR}/../.env"; set +a
fi

# ---- Remote Docker (SWE env pool) ----
export SWE_EXEC_SERVER_URLS=${SWE_EXEC_SERVER_URLS:-}
if [[ -z "${SWE_EXEC_SERVER_URLS}" ]]; then
  echo "SWE_EXEC_SERVER_URLS is required, e.g. http://<exec-1>:5000,http://<exec-2>:5000"
  exit 1
fi
export SWE_MAX_CONTAINERS_PER_NODE=${SWE_MAX_CONTAINERS_PER_NODE:-48}
_n_exec_nodes=$(echo "${SWE_EXEC_SERVER_URLS}" | tr ',' '\n' | wc -l)
export SWE_MAX_CONCURRENT=${SWE_MAX_CONCURRENT:-$(( SWE_MAX_CONTAINERS_PER_NODE * _n_exec_nodes ))}
export SWE_CONTAINER_PIDS_LIMIT=${SWE_CONTAINER_PIDS_LIMIT:-1024}
export SWE_CONTAINER_MEMORY=${SWE_CONTAINER_MEMORY:-8g}
export SWE_STRICT_NO_TEST_PATCH=${SWE_STRICT_NO_TEST_PATCH:-1}
export SWE_STRICT_NO_CONFIG_PATCH=${SWE_STRICT_NO_CONFIG_PATCH:-1}
export SWE_TEST_PATCH_POLICY_SCOPE=${SWE_TEST_PATCH_POLICY_SCOPE:-eval_tests_only}
# Self-evolution skill bridge (all optional, off by default)
export SWE_SKILL_RETRIEVAL="${SWE_SKILL_RETRIEVAL:-0}"
export SWE_SKILLS_DIR="${SWE_SKILLS_DIR:-}"
export SWE_SKILL_TOP_K="${SWE_SKILL_TOP_K:-3}"
export SWE_TRAJECTORY_LOG="${SWE_TRAJECTORY_LOG:-}"
export EVOLUTION_SERVER_URL="${EVOLUTION_SERVER_URL:-}"
SWE_ENV_SERVER_BIND_HOST=${SWE_ENV_SERVER_BIND_HOST:-0.0.0.0}
SWE_ENV_SERVER_PORT=${SWE_ENV_SERVER_PORT:-18090}
SWE_ENV_SERVER_URL=${SWE_ENV_SERVER_URL:-"http://${MASTER_ADDR}:${SWE_ENV_SERVER_PORT}"}
export SWE_ENV_SERVER_URL
CES_SSH_USER=${CES_SSH_USER:-root}
CES_SSH_KEY=${CES_SSH_KEY:-}

ALL_EXEC_HOSTS="$(echo "${SWE_EXEC_SERVER_URLS}" | tr ',' '\n' | sed -E 's#https?://([^:/]+).*#\1#' | tr '\n' ',' | sed 's/,$//')"
export NO_PROXY="localhost,127.0.0.1,${MASTER_ADDR},${NODE_IP},${ALL_EXEC_HOSTS}"
export no_proxy="${NO_PROXY}"

# Proxy for external access (grading calls make_test_spec -> git clone github.com)
export HTTP_PROXY=${HTTP_PROXY:-}
export HTTPS_PROXY=${HTTPS_PROXY:-}
export http_proxy="${HTTP_PROXY}"
export https_proxy="${HTTPS_PROXY}"

# Start env pool server on head node
SWE_POOL_PID=""
_cleanup_ecs_containers() {
  IFS=',' read -r -a _urls <<< "${SWE_EXEC_SERVER_URLS}"
  for url in "${_urls[@]}"; do
    host="$(echo "${url}" | sed -E 's#https?://([^:/]+).*#\1#')"
    echo "Cleaning up Docker containers on CES node ${host}..."
    ssh_key_args=()
    if [[ -n "${CES_SSH_KEY}" ]]; then
      ssh_key_args=(-i "${CES_SSH_KEY}")
    fi
    ssh "${ssh_key_args[@]}" -o StrictHostKeyChecking=no -o ConnectTimeout=5 "${CES_SSH_USER}@${host}" \
      'docker ps -q --filter "name=swe-" | xargs -r docker rm -f' 2>/dev/null || true
  done
}
cleanup() {
  set +e
  if [[ -n "${SWE_POOL_PID}" ]] && kill -0 "${SWE_POOL_PID}" 2>/dev/null; then
    kill "${SWE_POOL_PID}" || true
  fi
  if [[ ${MLP_ROLE_INDEX} -eq 0 ]]; then
    _cleanup_ecs_containers
  fi
}
trap cleanup EXIT INT TERM

if [[ ${MLP_ROLE_INDEX} -eq 0 ]]; then
  echo "Pre-flight: killing residual pool server on port ${SWE_ENV_SERVER_PORT}..."
  pkill -f "swe_env_pool_server" 2>/dev/null || true
  sleep 1
  if ss -ltnp 2>/dev/null | grep -q ":${SWE_ENV_SERVER_PORT} "; then
    fuser -k "${SWE_ENV_SERVER_PORT}/tcp" 2>/dev/null || true
    sleep 1
  fi

  echo "Pre-flight: cleaning residual Docker containers on CES nodes..."
  _cleanup_ecs_containers

  SWE_POOL_LOG=${SWE_POOL_LOG:-"${LOG_DIR}/swe_env_pool_server_${RUN_TIMESTAMP}.log"}
  HTTP_PROXY= HTTPS_PROXY= http_proxy= https_proxy= \
  PYTHONPATH="${SLIME_DIR}:${SWE_RL_DIR}:${SWE_RL_DIR}/server:${PYTHONPATH}" \
  python3 -m swe_env_pool_server \
    --host "${SWE_ENV_SERVER_BIND_HOST}" \
    --port "${SWE_ENV_SERVER_PORT}" \
    --exec-server-urls "${SWE_EXEC_SERVER_URLS}" \
    --max-containers-per-node "${SWE_MAX_CONTAINERS_PER_NODE}" \
    > "${SWE_POOL_LOG}" 2>&1 &
  SWE_POOL_PID=$!
  echo "SWE env pool server PID=${SWE_POOL_PID}, log=${SWE_POOL_LOG}"

  for i in {1..60}; do
    if curl -fsS "${SWE_ENV_SERVER_URL}/healthz" >/dev/null 2>&1; then
      echo "SWE env pool server is ready: ${SWE_ENV_SERVER_URL}"
      break
    fi
    sleep 2
  done

  IFS=',' read -r -a _exec_urls <<< "${SWE_EXEC_SERVER_URLS}"
  for exec_url in "${_exec_urls[@]}"; do
    if ! curl -fsS --max-time 8 "${exec_url}/healthz" >/dev/null; then
      echo "WARNING: SWE exec server is not healthy: ${exec_url}/healthz"
    fi
  done
fi

# ---- Checkpoint ----
HF_CKPT=${HF_CKPT:-}
if [[ -z "${HF_CKPT}" ]]; then
  echo "HF_CKPT is required (path to model checkpoint/snapshot)."
  exit 1
fi
REF_LOAD=${REF_LOAD:-${HF_CKPT}}
CKPT_ARGS=(
  --hf-checkpoint "${HF_CKPT}"
  --ref-load "${REF_LOAD}"
  --save "${SAVE_CKPT:-${EXPORT_ROOT}/ckpt/swe-rl-4b_${RUN_TIMESTAMP}}"
  --save-interval 1
  --megatron-to-hf-mode bridge
)

# ---- Dataset ----
DEBUG_MODE=${DEBUG_MODE:-0}
if [[ "${DEBUG_MODE}" == "1" ]]; then
  PROMPT_DATA=${PROMPT_DATA:-${SWE_RL_DIR}/../data/swegym_293/train_with_eval_script.parquet}
  NUM_ROLLOUT=80
else
  PROMPT_DATA=${PROMPT_DATA:-${SWE_RL_DIR}/../data/swegym_293/train_with_eval_script.parquet}
  NUM_ROLLOUT=500
fi
if [[ ! -f "${PROMPT_DATA}" ]]; then
  echo "Missing prompt dataset: ${PROMPT_DATA}"
  exit 1
fi

# ---- Rollout ----
# rollout-batch-size=32, n=8 → 256 trajectories; 256 / DP=8 = 32 per rank
ROLLOUT_ARGS=(
  --prompt-data "${PROMPT_DATA}"
  --input-key prompt
  --metadata-key instance
  --rollout-shuffle
  --reward-key score
  --num-rollout "${NUM_ROLLOUT}"
  --rollout-batch-size 32
  --n-samples-per-prompt 8
  --rollout-max-response-len 4096
  --rollout-max-context-len 32768
  --rollout-temperature 1.4
  --rollout-top-p 0.95
  --num-steps-per-rollout 1
  --dynamic-sampling-filter-path "slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std"
)

# ---- Performance ----
# 32 GPU shared: TP=4, DP=32/4=8
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
  --max-tokens-per-gpu 9216
  --log-probs-chunk-size 1024
  --balance-data
)

# ---- GRPO (default: DAPO asymmetric clip, KL effectively off) ----
GRPO_ARGS=(
  --advantage-estimator grpo
  # No --dynamic_history: trajectory mode is 1 sample/traj already
  --use-kl-loss
  --kl-loss-coef 0.0
  --kl-loss-type low_var_kl
  --entropy-coef 0.00
  --eps-clip 0.2
  --eps-clip-high 0.28
)

# ---- Optimizer (aligned with ProRL: lr=1e-6, wd=0.01, beta2=0.999) ----
OPTIMIZER_ARGS=(
  --optimizer adam
  --lr 1e-6
  --lr-decay-style constant
  --weight-decay 0.01
  --adam-beta1 0.9
  --adam-beta2 0.999
  --optimizer-cpu-offload
  --overlap-cpu-optimizer-d2h-h2d
  --use-precision-aware-optimizer
)

# ---- SGLang ----
# Colocate: SGLang sleeps during training (TorchMemorySaver releases all VRAM)
# so mem-fraction-static can be high — all memory is returned when training starts
SGLANG_ARGS=(
  --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
  --sglang-mem-fraction-static 0.85
  --sglang-router-port 30000
)

# ---- Custom generate (trajectory mode, default algo) ----
CUSTOM_ARGS=(
  --custom-generate-function-path generate_with_swe.generate
  --custom-rm-path generate_with_swe.reward_func
)

# ---- W&B ----
WANDB_KEY_VALUE=${WANDB_KEY:-${WANDB_API_KEY:-}}
if [ -n "${WANDB_KEY_VALUE}" ]; then
  WANDB_ARGS=(
    --use-wandb
    --wandb-project "${WANDB_PROJECT:-slime_swe}"
    --wandb-group qwen3-4B-instruct-swe-rl
    --wandb-key "${WANDB_KEY_VALUE}"
  )
else
  WANDB_ARGS=()
fi

# ---- Misc ----
MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)

# ---- SWE rollout env vars ----
SWE_SAVE_TRAJ_DIR=${SWE_SAVE_TRAJ_DIR:-${EXPORT_ROOT}/swe_rollouts/swe-rl-4b_${RUN_TIMESTAMP}}
mkdir -p "${SWE_SAVE_TRAJ_DIR}"
SWE_ROLLOUT_TIMEOUT=${SWE_ROLLOUT_TIMEOUT:-1800}
SWE_EVAL_TIMEOUT=${SWE_EVAL_TIMEOUT:-300}
SWE_DOCKER_REGISTRY=${SWE_DOCKER_REGISTRY:-slime-agent-cn-beijing.cr.volces.com}

# ---- NVLink detection ----
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "${NVLINK_COUNT}" -gt 0 ]; then
  HAS_NVLINK=1
else
  HAS_NVLINK=0
fi
# No expandable_segments — TorchMemorySaver (colocate sleep/wakeup) is incompatible with it
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"max_split_size_mb:2048"}

# ---- Ray cluster ----
if [[ ${MLP_ROLE_INDEX} -eq 0 ]]; then
  ray start --head --node-ip-address "${NODE_IP}" --num-gpus "${NUM_GPUS_PER_NODE}" \
    --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
else
  sleep 30
  ray start --address="${MASTER_ADDR}:6379" --num-gpus "${NUM_GPUS_PER_NODE}" \
    --node-ip-address "${NODE_IP}"
fi

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_LM_PATH}:${SWE_RL_DIR}:${SWE_RL_DIR}/server:${SLIME_DIR}\",
    \"PYTHONUNBUFFERED\": \"${PYTHONUNBUFFERED}\",
    \"PYTHONFAULTHANDLER\": \"${PYTHONFAULTHANDLER}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"${PYTORCH_CUDA_ALLOC_CONF}\",
    \"SWE_ENV_SERVER_URL\": \"${SWE_ENV_SERVER_URL}\",
    \"SWE_MAX_CONCURRENT\": \"${SWE_MAX_CONCURRENT}\",
    \"SWE_STRICT_NO_TEST_PATCH\": \"${SWE_STRICT_NO_TEST_PATCH}\",
    \"SWE_STRICT_NO_CONFIG_PATCH\": \"${SWE_STRICT_NO_CONFIG_PATCH}\",
    \"SWE_TEST_PATCH_POLICY_SCOPE\": \"${SWE_TEST_PATCH_POLICY_SCOPE}\",
    \"SWE_SAVE_TRAJ_DIR\": \"${SWE_SAVE_TRAJ_DIR}\",
    \"SWE_SKILL_RETRIEVAL\": \"${SWE_SKILL_RETRIEVAL}\",
    \"SWE_SKILLS_DIR\": \"${SWE_SKILLS_DIR}\",
    \"SWE_SKILL_TOP_K\": \"${SWE_SKILL_TOP_K}\",
    \"SWE_TRAJECTORY_LOG\": \"${SWE_TRAJECTORY_LOG}\",
    \"EVOLUTION_SERVER_URL\": \"${EVOLUTION_SERVER_URL}\",
    \"SWE_ROLLOUT_TIMEOUT\": \"${SWE_ROLLOUT_TIMEOUT}\",
    \"SWE_EVAL_TIMEOUT\": \"${SWE_EVAL_TIMEOUT}\",
    \"SWE_DOCKER_REGISTRY\": \"${SWE_DOCKER_REGISTRY}\",
    \"SWE_CONTAINER_PIDS_LIMIT\": \"${SWE_CONTAINER_PIDS_LIMIT}\",
    \"SWE_CONTAINER_MEMORY\": \"${SWE_CONTAINER_MEMORY}\",
    \"HTTP_PROXY\": \"${HTTP_PROXY}\",
    \"HTTPS_PROXY\": \"${HTTPS_PROXY}\",
    \"http_proxy\": \"${http_proxy}\",
    \"https_proxy\": \"${https_proxy}\",
    \"NO_PROXY\": \"${NO_PROXY}\",
    \"no_proxy\": \"${no_proxy}\",
    \"HF_HOME\": \"${HF_HOME:-${HOME}/.cache/huggingface}\",
    \"WANDB_API_KEY\": \"${WANDB_API_KEY:-}\"
  }
}"

RAY_JOB_SUBMISSION_ID=${RAY_JOB_SUBMISSION_ID:-"swe_rl_4b_$(date +%Y%m%d_%H%M%S)"}

if [[ ${MLP_ROLE_INDEX} -eq 0 ]]; then
  ray job submit --address="http://${MASTER_ADDR}:8265" \
    --submission-id "${RAY_JOB_SUBMISSION_ID}" \
    --no-wait \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -u "${SLIME_DIR}/train.py" \
    --actor-num-nodes "${NUM_NODES}" \
    --actor-num-gpus-per-node "${ACTOR_GPUS_PER_NODE}" \
    --colocate \
    ${MODEL_ARGS[@]} \
    ${CKPT_ARGS[@]} \
    ${ROLLOUT_ARGS[@]} \
    ${OPTIMIZER_ARGS[@]} \
    ${GRPO_ARGS[@]} \
    ${WANDB_ARGS[@]} \
    ${PERF_ARGS[@]} \
    ${SGLANG_ARGS[@]} \
    ${MISC_ARGS[@]} \
    ${CUSTOM_ARGS[@]}

  set +e
  ray job logs --address="http://${MASTER_ADDR}:8265" "${RAY_JOB_SUBMISSION_ID}" -f --log-style=record
  RAY_LOG_EXIT=$?
  RAY_STATUS_OUTPUT=$(ray job status --address="http://${MASTER_ADDR}:8265" "${RAY_JOB_SUBMISSION_ID}" --log-style=record 2>&1)
  echo "${RAY_STATUS_OUTPUT}"
  set -e
  if [[ "${RAY_STATUS_OUTPUT}" == *"SUCCEEDED"* ]]; then
    exit 0
  fi
  echo "Ray job failed (submission id: ${RAY_JOB_SUBMISSION_ID}, logs exit: ${RAY_LOG_EXIT})"
  exit 1
else
  echo "Worker node ${MLP_ROLE_INDEX} joined the cluster. Waiting for job to finish..."
  while ray status >/dev/null 2>&1; do
    sleep 60
  done
  echo "Ray cluster stopped. Worker node ${MLP_ROLE_INDEX} exiting."
fi
