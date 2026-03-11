#!/bin/bash

# SWE-Bench RL training (Qwen3-32B) on 8 nodes with REMOTE Docker exec.
# Scheduler should inject:
#   MLP_ROLE_INDEX=0/1/2/3/4/5/6/7
#   MLP_WORKER_0_HOST=<head_ip>
#   MLP_WORKER_1_HOST .. MLP_WORKER_7_HOST=<worker_ips>

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
EXPORT_ROOT=${EXPORT_ROOT:-"${SWE_RL_DIR}/output"}
mkdir -p "${EXPORT_ROOT}/ckpt" "${EXPORT_ROOT}/swe_rollouts"
RUN_TIMESTAMP=${RUN_TIMESTAMP:-$(date +%F_%H%M%S)}
LOG_DIR=${LOG_DIR:-"${SCRIPT_DIR}/logs"}
mkdir -p "${LOG_DIR}"
RUN_LOG=${RUN_LOG:-"${LOG_DIR}/run_swe_rl_32b_remote_8nodes_${RUN_TIMESTAMP}.log"}
exec > >(tee -a "${RUN_LOG}") 2>&1
echo "Run log: ${RUN_LOG}"
echo "Run timestamp: ${RUN_TIMESTAMP}"

source "${SLIME_DIR}/scripts/models/qwen3-32B.sh"
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

NUM_NODES=${NUM_NODES:-8}
NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-8}
ACTOR_GPUS_PER_NODE=${ACTOR_GPUS_PER_NODE:-4}
ROLLOUT_GPUS_PER_NODE=${ROLLOUT_GPUS_PER_NODE:-4}
ROLLOUT_GPUS_TOTAL=${ROLLOUT_GPUS_TOTAL:-$((NUM_NODES * ROLLOUT_GPUS_PER_NODE))}
ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-4}

if (( ACTOR_GPUS_PER_NODE + ROLLOUT_GPUS_PER_NODE > NUM_GPUS_PER_NODE )); then
  echo "ACTOR_GPUS_PER_NODE + ROLLOUT_GPUS_PER_NODE must be <= NUM_GPUS_PER_NODE"
  exit 1
fi

MLP_ROLE_INDEX=${MLP_ROLE_INDEX:-0}
MASTER_ADDR="${MLP_WORKER_0_HOST:-${MASTER_ADDR:-$(hostname -I | awk '{print $1}')}}"
_WORKER_IP_VAR="MLP_WORKER_${MLP_ROLE_INDEX}_HOST"
NODE_IP="${!_WORKER_IP_VAR:-${WORKER_IP:-$(hostname -I | awk '{print $1}')}}"
export MASTER_ADDR
echo "MLP_ROLE_INDEX=${MLP_ROLE_INDEX}, MASTER_ADDR=${MASTER_ADDR}, NODE_IP=${NODE_IP}"

# Remote SWE env pool / exec configs
export SWE_ENV_SERVER_BIND_HOST=${SWE_ENV_SERVER_BIND_HOST:-0.0.0.0}
export SWE_ENV_SERVER_PORT=${SWE_ENV_SERVER_PORT:-18090}
export SWE_ENV_SERVER_HOST=${SWE_ENV_SERVER_HOST:-${MASTER_ADDR}}
export SWE_ENV_SERVER_URL=${SWE_ENV_SERVER_URL:-http://${SWE_ENV_SERVER_HOST}:${SWE_ENV_SERVER_PORT}}
export SWE_EXEC_SERVER_URLS=${SWE_EXEC_SERVER_URLS:-http://NODE1_IP:5000,http://NODE2_IP:5000,http://NODE3_IP:5000,http://NODE4_IP:5000,http://NODE5_IP:5000,http://NODE6_IP:5000,http://NODE7_IP:5000,http://NODE8_IP:5000,http://NODE9_IP:5000,http://NODE10_IP:5000,http://NODE11_IP:5000,http://NODE12_IP:5000,http://NODE13_IP:5000,http://NODE14_IP:5000,http://NODE15_IP:5000,http://NODE16_IP:5000,http://NODE17_IP:5000,http://NODE18_IP:5000}
export SWE_MAX_CONTAINERS_PER_NODE=${SWE_MAX_CONTAINERS_PER_NODE:-15}
export SWE_MAX_CONCURRENT=${SWE_MAX_CONCURRENT:-128}

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

if [[ ${MLP_ROLE_INDEX} -eq 0 ]]; then
  SWE_POOL_LOG=${SWE_POOL_LOG:-"${LOG_DIR}/swe_env_pool_server_8nodes.log"}
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

HF_CKPT=${HF_CKPT:-/data_storage/wyj/systems/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/d47b0d4ae4b48fde975756bf360a63a9cca8d470}
REF_LOAD=${REF_LOAD:-${HF_CKPT}}
CKPT_ARGS=(
  --hf-checkpoint "${HF_CKPT}"
  --ref-load "${REF_LOAD}"
  --save "${SAVE_CKPT:-${EXPORT_ROOT}/ckpt/swe-rl-32b-remote-8nodes_${RUN_TIMESTAMP}}"
  --save-interval 2
  --megatron-to-hf-mode bridge
)

DEBUG_MODE=${DEBUG_MODE:-0}
if [[ "${DEBUG_MODE}" == "1" ]]; then
  PROMPT_DATA=${PROMPT_DATA:-/data_storage/wyj/swe_verified_full/train_subset_360.jsonl}
  NUM_ROLLOUT=40
  N_SAMPLES=2
else
  PROMPT_DATA=${PROMPT_DATA:-/data_storage/wyj/swe_verified_full/train_subset_360.jsonl}
  NUM_ROLLOUT=2000
  N_SAMPLES=8
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
  # GRPO group size: number of sampled trajectories per prompt.
  --n-samples-per-prompt "${N_SAMPLES}"
  # Max newly generated tokens for each rollout step.
  --rollout-max-response-len 4096
  # Total context budget seen by the model during rollout.
  --rollout-max-context-len 16384
  --rollout-temperature 1
  --num-steps-per-rollout 1
)

PERF_ARGS=(
  --tensor-model-parallel-size 8
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu 8192
  --log-probs-chunk-size 512
  --balance-data
)

GRPO_ARGS=(
  --advantage-estimator grpo
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
  --sglang-mem-fraction-static 0.55
  --sglang-router-port 30000
)

CUSTOM_ARGS=(
  --custom-generate-function-path generate_with_swe_remote.generate
  --custom-rm-path generate_with_swe_remote.reward_func
)

WANDB_KEY_VALUE=${WANDB_KEY:-${WANDB_API_KEY:-}}
if [ -n "${WANDB_KEY_VALUE}" ]; then
  WANDB_ARGS=(
    --use-wandb
    --wandb-project "${WANDB_PROJECT:-slime_swe}"
    --wandb-group qwen3-32B-rl_swe_remote_8nodes
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

if [[ -f "${SWE_RL_DIR}/.env.swe" ]]; then
  source "${SWE_RL_DIR}/.env.swe"
fi
OPENAI_BASE_URL=${OPENAI_BASE_URL:-auto}
OPENAI_API_KEY=${OPENAI_API_KEY:-dummy}
LITELLM_MODEL_REGISTRY_PATH=${LITELLM_MODEL_REGISTRY_PATH:-"${SWE_RL_DIR}/litellm.json"}
SWE_LITELLM_MODEL_NAME=${SWE_LITELLM_MODEL_NAME:-openai/Qwen/Qwen3-32B}
SWE_SAVE_TRAJ_DIR=${SWE_SAVE_TRAJ_DIR:-${EXPORT_ROOT}/swe_rollouts/swe-rl-32b-remote-8nodes_${RUN_TIMESTAMP}}
mkdir -p "${SWE_SAVE_TRAJ_DIR}"
echo "SWE rollout artifacts dir: ${SWE_SAVE_TRAJ_DIR}"
echo "SWE_ENV_SERVER_URL=${SWE_ENV_SERVER_URL}, SWE_EXEC_SERVER_URLS=${SWE_EXEC_SERVER_URLS}"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "${NVLINK_COUNT}" -gt 0 ]; then
  HAS_NVLINK=1
else
  HAS_NVLINK=0
fi
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"max_split_size_mb:2048,expandable_segments:True"}

if [[ ${MLP_ROLE_INDEX} -eq 0 ]]; then
  ray start --head --node-ip-address "${NODE_IP}" --num-gpus "${NUM_GPUS_PER_NODE}" \
    --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
else
  sleep 60
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

RAY_JOB_SUBMISSION_ID=${RAY_JOB_SUBMISSION_ID:-"swe_rl_32b_remote_8nodes_$(date +%Y%m%d_%H%M%S)"}

if [[ ${MLP_ROLE_INDEX} -eq 0 ]]; then
  ray job submit --address="http://${MASTER_ADDR}:8265" \
    --submission-id "${RAY_JOB_SUBMISSION_ID}" \
    --no-wait \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -u "${SLIME_DIR}/train_async.py" \
    --actor-num-nodes "${NUM_NODES}" \
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
