#!/bin/bash

# SWE-Bench RL training (Qwen3-8B) on 5 nodes with PRM + REMOTE Docker exec.
#
# Layout (strict non-colocate):
#   4 train/policy nodes: actor (TP=4) + SGLang rollout engines
#   1 dedicated PRM node:  PRM SGLang engine (8 GPUs)
#
# Scheduler should inject:
#   MLP_ROLE_INDEX=0/1/2/3/4
#   MLP_WORKER_{0..4}_HOST=<ip>

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
RUN_LOG=${RUN_LOG:-"${LOG_DIR}/run_swe_rl_8b_prm_5nodes_remote_${RUN_TIMESTAMP}.log"}
exec > >(tee -a "${RUN_LOG}") 2>&1
echo "Run log: ${RUN_LOG}"
echo "Run timestamp: ${RUN_TIMESTAMP}"

source "${SLIME_DIR}/scripts/models/qwen3-8B.sh"
MEGATRON_LM_PATH=${MEGATRON_LM_PATH:-"${SWE_RL_DIR}/../Megatron-LM"}

MINISWE_DIR="${SWE_RL_DIR}/mini-swe-agent"
MINISWE_VERSION="v1.12.0"
if ! python3 -c "import minisweagent" 2>/dev/null; then
  if [ ! -d "${MINISWE_DIR}" ]; then
    git clone --branch "${MINISWE_VERSION}" --depth 1 \
      https://github.com/SWE-agent/mini-swe-agent.git "${MINISWE_DIR}"
  fi
  pip install -e "${MINISWE_DIR}"
fi

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export MSWEA_DOCKER_EXEC_MODE=api
export RAY_health_check_failure_threshold=${RAY_health_check_failure_threshold:-20}
export RAY_health_check_period_ms=${RAY_health_check_period_ms:-5000}
export RAY_health_check_timeout_ms=${RAY_health_check_timeout_ms:-30000}
export RAY_num_heartbeats_timeout=${RAY_num_heartbeats_timeout:-60}

# ---------------------------------------------------------------------------
# Cluster layout: 5 nodes = 4 train/policy + 1 PRM
# ---------------------------------------------------------------------------
NUM_NODES=${NUM_NODES:-5}
TRAIN_NUM_NODES=${TRAIN_NUM_NODES:-4}
NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-8}
ACTOR_GPUS_PER_NODE=${ACTOR_GPUS_PER_NODE:-4}
ROLLOUT_GPUS_PER_NODE=${ROLLOUT_GPUS_PER_NODE:-4}
ROLLOUT_GPUS_TOTAL=${ROLLOUT_GPUS_TOTAL:-$((TRAIN_NUM_NODES * ROLLOUT_GPUS_PER_NODE))}
ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-2}
PRM_GPUS_TOTAL=${PRM_GPUS_TOTAL:-8}
PRM_GPUS_PER_ENGINE=${PRM_GPUS_PER_ENGINE:-8}
PRM_ENABLE=${PRM_ENABLE:-1}

if (( ACTOR_GPUS_PER_NODE + ROLLOUT_GPUS_PER_NODE > NUM_GPUS_PER_NODE )); then
  echo "ACTOR_GPUS_PER_NODE + ROLLOUT_GPUS_PER_NODE must be <= NUM_GPUS_PER_NODE"
  echo "ACTOR_GPUS_PER_NODE=${ACTOR_GPUS_PER_NODE}, ROLLOUT_GPUS_PER_NODE=${ROLLOUT_GPUS_PER_NODE}, NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE}"
  exit 1
fi

if (( TRAIN_NUM_NODES > NUM_NODES )); then
  echo "TRAIN_NUM_NODES must be <= NUM_NODES"
  echo "TRAIN_NUM_NODES=${TRAIN_NUM_NODES}, NUM_NODES=${NUM_NODES}"
  exit 1
fi

# Strict non-colocate layout validation (same as GUI PRM script)
if [[ "${PRM_ENABLE:-0}" == "1" ]]; then
  PRM_NUM_NODES=$((NUM_NODES - TRAIN_NUM_NODES))
  TRAIN_POLICY_GPUS_CAP=$((TRAIN_NUM_NODES * NUM_GPUS_PER_NODE))
  TRAIN_POLICY_GPUS_REQ=$((TRAIN_NUM_NODES * ACTOR_GPUS_PER_NODE + ROLLOUT_GPUS_TOTAL))
  PRM_GPUS_CAP=$((PRM_NUM_NODES * NUM_GPUS_PER_NODE))

  if (( PRM_NUM_NODES <= 0 )); then
    echo "PRM_ENABLE=1 requires dedicated PRM nodes: NUM_NODES must be > TRAIN_NUM_NODES."
    echo "NUM_NODES=${NUM_NODES}, TRAIN_NUM_NODES=${TRAIN_NUM_NODES}"
    exit 1
  fi
  if (( TRAIN_POLICY_GPUS_REQ != TRAIN_POLICY_GPUS_CAP )); then
    echo "Strict non-colocate check failed for train/policy layout."
    echo "requested=${TRAIN_POLICY_GPUS_REQ}, capacity=${TRAIN_POLICY_GPUS_CAP}"
    echo "Expected: TRAIN_NUM_NODES*ACTOR_GPUS_PER_NODE + ROLLOUT_GPUS_TOTAL == TRAIN_NUM_NODES*NUM_GPUS_PER_NODE"
    exit 1
  fi
  if (( PRM_GPUS_TOTAL != PRM_GPUS_CAP )); then
    echo "Strict non-colocate check failed for PRM layout."
    echo "PRM_GPUS_TOTAL=${PRM_GPUS_TOTAL}, dedicated_capacity=${PRM_GPUS_CAP}"
    echo "Expected: PRM_GPUS_TOTAL == (NUM_NODES-TRAIN_NUM_NODES)*NUM_GPUS_PER_NODE"
    exit 1
  fi

  TOTAL_AVAILABLE_GPUS=$((NUM_NODES * NUM_GPUS_PER_NODE))
  TOTAL_REQUESTED_GPUS=$((TRAIN_NUM_NODES * ACTOR_GPUS_PER_NODE + ROLLOUT_GPUS_TOTAL + PRM_GPUS_TOTAL))
  if (( TOTAL_REQUESTED_GPUS > TOTAL_AVAILABLE_GPUS )); then
    echo "Requested GPUs exceed cluster capacity when PRM is enabled."
    echo "requested=${TOTAL_REQUESTED_GPUS}, available=${TOTAL_AVAILABLE_GPUS}"
    echo "actor=$((TRAIN_NUM_NODES * ACTOR_GPUS_PER_NODE)), rollout=${ROLLOUT_GPUS_TOTAL}, prm=${PRM_GPUS_TOTAL}"
    exit 1
  fi
fi

# ---------------------------------------------------------------------------
# Multi-node role info
# ---------------------------------------------------------------------------
MLP_ROLE_INDEX=${MLP_ROLE_INDEX:-0}
MASTER_ADDR="${MLP_WORKER_0_HOST:-${MASTER_ADDR:-$(hostname -I | awk '{print $1}')}}"
_WORKER_IP_VAR="MLP_WORKER_${MLP_ROLE_INDEX}_HOST"
NODE_IP="${!_WORKER_IP_VAR:-${WORKER_IP:-$(hostname -I | awk '{print $1}')}}"
export MASTER_ADDR
echo "MLP_ROLE_INDEX=${MLP_ROLE_INDEX}, MASTER_ADDR=${MASTER_ADDR}, NODE_IP=${NODE_IP}"

# ---------------------------------------------------------------------------
# Remote SWE env pool / exec configs
# ---------------------------------------------------------------------------
export SWE_ENV_SERVER_BIND_HOST=${SWE_ENV_SERVER_BIND_HOST:-0.0.0.0}
export SWE_ENV_SERVER_PORT=${SWE_ENV_SERVER_PORT:-18090}
export SWE_ENV_SERVER_HOST=${SWE_ENV_SERVER_HOST:-${MASTER_ADDR}}
export SWE_ENV_SERVER_URL=${SWE_ENV_SERVER_URL:-http://${SWE_ENV_SERVER_HOST}:${SWE_ENV_SERVER_PORT}}
export SWE_EXEC_SERVER_URLS=${SWE_EXEC_SERVER_URLS:-http://NODE1_IP:5000}
export SWE_MAX_CONTAINERS_PER_NODE=${SWE_MAX_CONTAINERS_PER_NODE:-8}
export SWE_MAX_CONCURRENT=${SWE_MAX_CONCURRENT:-4}

ALL_EXEC_HOSTS="$(echo "${SWE_EXEC_SERVER_URLS}" | tr ',' '\n' | sed -E 's#https?://([^:/]+).*#\1#' | tr '\n' ',' | sed 's/,$//')"
export NO_PROXY="localhost,127.0.0.1,${MASTER_ADDR},${NODE_IP},${ALL_EXEC_HOSTS}"
export no_proxy="${NO_PROXY}"

SWE_POOL_PID=""
cleanup() {
  set +e
  if [[ -n "${SWE_POOL_PID}" ]] && kill -0 "${SWE_POOL_PID}" 2>/dev/null; then
    kill "${SWE_POOL_PID}" || true
  fi
}
trap cleanup EXIT INT TERM

# Head node starts the SWE env pool server
if [[ ${MLP_ROLE_INDEX} -eq 0 ]]; then
  SWE_POOL_LOG=${SWE_POOL_LOG:-"${LOG_DIR}/swe_env_pool_server_5nodes_prm.log"}
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
      echo "ERROR: SWE exec server is not healthy: ${exec_url}/healthz"
      exit 1
    fi
  done
fi

# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
HF_CKPT=${HF_CKPT:-/data_storage/wyj/systems/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}
REF_LOAD=${REF_LOAD:-${HF_CKPT}}
PRM_MODEL_PATH=${PRM_MODEL_PATH:-/data_storage/wyj/systems/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/531c80e289d6cff3a7cd8c0db8110231d23a6f7a}

if [[ "${PRM_ENABLE:-0}" == "1" && -z "${PRM_MODEL_PATH}" ]]; then
  echo "PRM_ENABLE=1 requires PRM_MODEL_PATH to be set explicitly."
  echo "Example:"
  echo "  PRM_MODEL_PATH=/path/to/prm-model bash run_swe_rl_8b_prm_5nodes_remote.sh"
  exit 1
fi

CKPT_ARGS=(
  --hf-checkpoint "${HF_CKPT}"
  --ref-load "${REF_LOAD}"
  --save "${SAVE_CKPT:-${EXPORT_ROOT}/ckpt/swe-rl-8b-prm-5nodes-remote_${RUN_TIMESTAMP}}"
  --save-interval 5
  --megatron-to-hf-mode bridge
)

ENABLE_RESUME_LOAD=${ENABLE_RESUME_LOAD:-0}
RESUME_LOAD=${RESUME_LOAD:-${EXPORT_ROOT}/ckpt/swe-rl-8b-prm-5nodes-remote}
if [[ "${ENABLE_RESUME_LOAD}" == "1" ]]; then
  CKPT_ARGS+=(--load "${RESUME_LOAD}")
  echo "Resume load enabled: ${RESUME_LOAD}"
else
  echo "Resume load disabled (ENABLE_RESUME_LOAD=${ENABLE_RESUME_LOAD})"
fi

# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------
DEBUG_MODE=${DEBUG_MODE:-0}
if [[ "${DEBUG_MODE}" == "1" ]]; then
  PROMPT_DATA=${PROMPT_DATA:-/data_storage/wyj/swe_gym_subset/train_10.jsonl}
  NUM_ROLLOUT=80
  N_SAMPLES=2
else
  PROMPT_DATA=${PROMPT_DATA:-/data_storage/wyj/swe_gym_subset/train.jsonl}
  NUM_ROLLOUT=500
  N_SAMPLES=4
fi
if [[ ! -f "${PROMPT_DATA}" ]]; then
  echo "Missing prompt dataset: ${PROMPT_DATA}"
  exit 1
fi

ROLLOUT_ARGS=(
  --prompt-data "${PROMPT_DATA}"
  --input-key text
  --metadata-key metadata
  --rollout-shuffle
  --reward-key score
  --num-rollout "${NUM_ROLLOUT}"
  --rollout-batch-size 8
  --n-samples-per-prompt "${N_SAMPLES}"
  --rollout-max-response-len 4096
  --rollout-max-context-len 32768
  --rollout-temperature 1
  --num-steps-per-rollout 1
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
  --balance-data
)

GRPO_ARGS=(
  --advantage-estimator step_wise
  --dynamic_history
  --use-kl-loss
  --kl-loss-coef 0.00
  --kl-loss-type low_var_kl
  --entropy-coef 0.00
  --eps-clip 0.2
  --eps-clip-high 0.28
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

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
  --sglang-mem-fraction-static 0.6
  --sglang-router-port 30000
)

CUSTOM_ARGS=(
  --custom-generate-function-path generate_with_swe_remote.generate
  --custom-rm-path generate_with_swe_remote.reward_func
)

PRM_ARGS=(
  --prm-m "${PRM_M:-3}"
  --prm-num-gpus "${PRM_GPUS_TOTAL}"
  --prm-num-gpus-per-engine "${PRM_GPUS_PER_ENGINE}"
  --prm-step-coef "${PRM_STEP_COEF:-1.0}"
  --prm-temperature "${PRM_TEMPERATURE:-1.0}"
  --prm-max-new-tokens "${PRM_MAX_NEW_TOKENS:-4096}"
)

if [[ "${PRM_ENABLE:-0}" == "1" ]]; then
  PRM_ARGS+=(--prm-enable)
  PRM_ARGS+=(--prm-model-path "${PRM_MODEL_PATH}")
fi

WANDB_KEY_VALUE=${WANDB_KEY:-${WANDB_API_KEY:-}}
if [ -n "${WANDB_KEY_VALUE}" ]; then
  WANDB_ARGS=(
    --use-wandb
    --wandb-project slime_swe
    --wandb-group qwen3-8B-rl_swe_prm_5nodes_remote
    --wandb-key "${WANDB_KEY_VALUE}"
  )
else
  WANDB_ARGS=()
fi

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)

# ---------------------------------------------------------------------------
# SWE litellm / env configs
# ---------------------------------------------------------------------------
if [[ -f "${SWE_RL_DIR}/.env.swe" ]]; then
  source "${SWE_RL_DIR}/.env.swe"
fi
OPENAI_BASE_URL=${OPENAI_BASE_URL:-auto}
OPENAI_API_KEY=${OPENAI_API_KEY:-dummy}
LITELLM_MODEL_REGISTRY_PATH=${LITELLM_MODEL_REGISTRY_PATH:-"${SWE_RL_DIR}/litellm.json"}
SWE_LITELLM_MODEL_NAME=${SWE_LITELLM_MODEL_NAME:-openai/Qwen/Qwen3-8B}
SWE_SAVE_TRAJ_DIR=${SWE_SAVE_TRAJ_DIR:-${EXPORT_ROOT}/swe_rollouts/swe-rl-8b-prm-5nodes-remote_${RUN_TIMESTAMP}}
mkdir -p "${SWE_SAVE_TRAJ_DIR}"
echo "SWE rollout artifacts dir: ${SWE_SAVE_TRAJ_DIR}"
echo "SWE_ENV_SERVER_URL=${SWE_ENV_SERVER_URL}, SWE_EXEC_SERVER_URLS=${SWE_EXEC_SERVER_URLS}"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "${NVLINK_COUNT}" -gt 0 ]; then
  HAS_NVLINK=1
else
  HAS_NVLINK=0
fi
echo "HAS_NVLINK: ${HAS_NVLINK} (detected ${NVLINK_COUNT} NVLink references)"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"max_split_size_mb:2048,expandable_segments:True"}

# ---------------------------------------------------------------------------
# Ray cluster
# ---------------------------------------------------------------------------
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
    \"OPENAI_BASE_URL\": \"${OPENAI_BASE_URL}\",
    \"OPENAI_API_KEY\": \"${OPENAI_API_KEY}\",
    \"LITELLM_MODEL_REGISTRY_PATH\": \"${LITELLM_MODEL_REGISTRY_PATH}\",
    \"SWE_LITELLM_MODEL_NAME\": \"${SWE_LITELLM_MODEL_NAME}\",
    \"SWE_SAVE_TRAJ_DIR\": \"${SWE_SAVE_TRAJ_DIR}\",
    \"SWE_CONFIG_PATH\": \"${SWE_RL_DIR}/swebench.yaml\",
    \"SWE_ENV_SERVER_URL\": \"${SWE_ENV_SERVER_URL}\",
    \"SWE_MAX_CONCURRENT\": \"${SWE_MAX_CONCURRENT}\",
    \"MSWEA_DOCKER_EXEC_MODE\": \"${MSWEA_DOCKER_EXEC_MODE:-api}\",
    \"NO_PROXY\": \"${NO_PROXY}\",
    \"no_proxy\": \"${no_proxy}\"
  }
}"

RAY_JOB_SUBMISSION_ID=${RAY_JOB_SUBMISSION_ID:-"swe_rl_8b_prm_5nodes_remote_$(date +%Y%m%d_%H%M%S)"}

if [[ ${MLP_ROLE_INDEX} -eq 0 ]]; then
  ray job submit --address="http://${MASTER_ADDR}:8265" \
    --submission-id "${RAY_JOB_SUBMISSION_ID}" \
    --no-wait \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -u train_async.py \
    --actor-num-nodes "${TRAIN_NUM_NODES}" \
    --actor-num-gpus-per-node "${ACTOR_GPUS_PER_NODE}" \
    --rollout-num-gpus "${ROLLOUT_GPUS_TOTAL}" \
    ${MODEL_ARGS[@]} \
    ${CKPT_ARGS[@]} \
    ${ROLLOUT_ARGS[@]} \
    ${OPTIMIZER_ARGS[@]} \
    ${GRPO_ARGS[@]} \
    ${WANDB_ARGS[@]} \
    ${PERF_ARGS[@]} \
    ${SGLANG_ARGS[@]} \
    ${MISC_ARGS[@]} \
    ${CUSTOM_ARGS[@]} \
    ${PRM_ARGS[@]}

  echo "Following live Ray logs for ${RAY_JOB_SUBMISSION_ID}"
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
