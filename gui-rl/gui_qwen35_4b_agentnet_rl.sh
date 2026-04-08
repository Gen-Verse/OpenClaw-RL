#!/bin/bash
# Qwen3.5-4B GUI-RL training with AgentNet offline dataset.
# No cloud VM or env pool server required.
#
# Required environment variables (must be set before running):
#   AGENTNET_JSONL_PATH   Path to agentnet_ubuntu_5k.jsonl
#   AGENTNET_IMAGE_DIR    Path to extracted AgentNet PNG screenshots
#   HF_CKPT              Path to Qwen3.5-4B model
#
# Image extraction (run once on the training machine):
#   cd $AGENTNET_IMAGE_DIR/../
#   zip -s 0 images.zip --out images-full.zip && unzip images-full.zip -d extracted/
#   export AGENTNET_IMAGE_DIR=$(pwd)/extracted
#
# Run:
#   cd gui-rl
#   bash gui_qwen35_4b_agentnet_rl.sh

###############################################################################
# User config: provide these through environment variables when needed.
###############################################################################

# Official Megatron-Bridge checkout that contains qwen35_vl_bridge/provider.
OFFICIAL_MBRIDGE_ROOT="${OFFICIAL_MBRIDGE_ROOT:-}"
OFFICIAL_MBRIDGE_SRC="${OFFICIAL_MBRIDGE_ROOT}/src"
MEGATRON_LM_PATH="${OFFICIAL_MBRIDGE_ROOT}/3rdparty/Megatron-LM"

# HuggingFace Qwen3.5-4B checkpoint directory.
HF_CKPT="${HF_CKPT:-}"

# AgentNet dataset files.
AGENTNET_JSONL_PATH="${AGENTNET_JSONL_PATH:-}"
AGENTNET_IMAGE_DIR="${AGENTNET_IMAGE_DIR:-}"

# Optional naming.
GUI_PROJECT_NAME="${GUI_PROJECT_NAME:-slime_gui-qwen35-4b-agentnet}"
SAVE_CKPT="${SAVE_CKPT:-}"

# Optional resource settings.
NUM_GPUS=4
ACTOR_GPUS=2
ROLLOUT_GPUS=2
ROLLOUT_BATCH_SIZE=4
ROLLOUT_NUM_GPUS_PER_ENGINE=1
GUI_EVAL_INTERVAL=""
ENABLE_RESUME_LOAD=0
RESUME_LOAD=""

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
SLIME_DIR="$(cd -- "${SCRIPT_DIR}/../slime" &>/dev/null && pwd)"
MODEL_ARGS_ROTARY_BASE=10000000 source "${SLIME_DIR}/scripts/models/qwen3.5-4B-VL.sh"

if [[ -z "${SAVE_CKPT}" ]]; then
  SAVE_CKPT="${SCRIPT_DIR}/../ckpt/gui-qwen35-4b-agentnet"
fi
if [[ -z "${RESUME_LOAD}" ]]; then
  RESUME_LOAD="${SCRIPT_DIR}/../ckpt/gui-qwen35-4b-agentnet"
fi

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

NVIDIA_BIN_DIR=${NVIDIA_BIN_DIR:-/usr/local/nvidia/bin}
NVIDIA_LIB_DIR=${NVIDIA_LIB_DIR:-/usr/local/nvidia/lib64}
if [[ -d "${NVIDIA_BIN_DIR}" ]]; then
  export PATH="${NVIDIA_BIN_DIR}:${PATH}"
fi
if [[ -d "${NVIDIA_LIB_DIR}" ]]; then
  export LD_LIBRARY_PATH="${NVIDIA_LIB_DIR}:${LD_LIBRARY_PATH:-}"
fi

export RAY_health_check_failure_threshold=${RAY_health_check_failure_threshold:-20}
export RAY_health_check_period_ms=${RAY_health_check_period_ms:-5000}
export RAY_health_check_timeout_ms=${RAY_health_check_timeout_ms:-30000}
export RAY_num_heartbeats_timeout=${RAY_num_heartbeats_timeout:-60}

if (( ACTOR_GPUS + ROLLOUT_GPUS > NUM_GPUS )); then
  echo "ACTOR_GPUS + ROLLOUT_GPUS must be <= NUM_GPUS"
  exit 1
fi

if [[ -z "${OFFICIAL_MBRIDGE_ROOT}" ]]; then
  echo "ERROR: OFFICIAL_MBRIDGE_ROOT is not set"
  exit 1
fi
if [[ -z "${HF_CKPT}" ]]; then
  echo "ERROR: HF_CKPT is not set"
  exit 1
fi
if [[ -z "${AGENTNET_JSONL_PATH}" ]]; then
  echo "ERROR: AGENTNET_JSONL_PATH is not set"
  exit 1
fi
if [[ ! -f "${AGENTNET_JSONL_PATH}" ]]; then
  echo "ERROR: AGENTNET_JSONL_PATH does not exist: ${AGENTNET_JSONL_PATH}"
  exit 1
fi
if [[ -z "${AGENTNET_IMAGE_DIR}" ]]; then
  echo "ERROR: AGENTNET_IMAGE_DIR is not set"
  exit 1
fi
if [[ ! -d "${AGENTNET_IMAGE_DIR}" ]]; then
  echo "ERROR: AGENTNET_IMAGE_DIR does not exist: ${AGENTNET_IMAGE_DIR}"
  exit 1
fi
export AGENTNET_JSONL_PATH
export AGENTNET_IMAGE_DIR

REF_LOAD=${REF_LOAD:-${HF_CKPT}}
if [[ ! -e "${HF_CKPT}" ]]; then
  echo "ERROR: HF_CKPT is not set to a valid path: ${HF_CKPT}"
  exit 1
fi
if [[ ! -d "${OFFICIAL_MBRIDGE_SRC}" ]]; then
  echo "ERROR: OFFICIAL_MBRIDGE_SRC does not exist: ${OFFICIAL_MBRIDGE_SRC}"
  exit 1
fi
if [[ ! -d "${MEGATRON_LM_PATH}" ]]; then
  echo "ERROR: MEGATRON_LM_PATH does not exist: ${MEGATRON_LM_PATH}"
  exit 1
fi
export HF_CKPT
export OFFICIAL_MBRIDGE_ROOT
export OFFICIAL_MBRIDGE_SRC
export MEGATRON_LM_PATH

# --- Project naming ---
export GUI_RESULT_DIR=${GUI_RESULT_DIR:-"${SCRIPT_DIR}/results"}
export GUI_RESULT_DIR="${GUI_RESULT_DIR}/${GUI_PROJECT_NAME}"

# --- Agent config (reuse Qwen3.5 agent) ---
export GUI_AGENT_CLASS_PATH=${GUI_AGENT_CLASS_PATH:-"agents.qwen35_agent.Qwen35AgentLocal"}
export GUI_COORDINATE_TYPE=${GUI_COORDINATE_TYPE:-"relative"}
export MULTIMODAL_KEYS=${MULTIMODAL_KEYS:-'{"image":"images"}'}
# Clean result dir
if [[ -n "${GUI_RESULT_DIR}" && "${GUI_RESULT_DIR}" != "/" ]]; then
  rm -rf "${GUI_RESULT_DIR}"
fi
mkdir -p "${GUI_RESULT_DIR}"

CKPT_ARGS=(
  --hf-checkpoint ${HF_CKPT}
  --ref-load ${REF_LOAD}
  --save "${SAVE_CKPT}"
  --save-interval 20
)

if [[ "${ENABLE_RESUME_LOAD}" == "1" ]]; then
  CKPT_ARGS+=(--load "${RESUME_LOAD}")
fi

# --- Rollout args ---
# AgentNet offline + GRPO still needs multiple trajectories per prompt group.
# With n-samples-per-prompt=1, group-normalized rewards collapse to 0 and
# policy gradient becomes degenerate. Keep this aligned with other GUI-RL GRPO
# scripts unless switching to a non-group-based estimator.
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-8}
if (( N_SAMPLES_PER_PROMPT < 2 )); then
  echo "ERROR: gui_qwen35_4b_agentnet_rl.sh requires N_SAMPLES_PER_PROMPT >= 2 for GRPO."
  echo "If you really want single-sample training, switch estimator or disable reward normalization explicitly."
  exit 1
fi
# dynamic_history can expand one rollout into hundreds of step-wise samples.
# Keep the train-side dynamic global batch size capped at the script's native
# global batch size (= rollout_batch_size * n_samples_per_prompt // num_steps_per_rollout = 16).
# The previous 32-sample cap fixed Ray OOM in data_preprocess, but still made
# the first actor_train step numerically unstable (grad norm exploded to ~7.8e14
# and then Megatron's bucket grad check observed NaN). Staying at 16 keeps the
# per-step gradient scale aligned with the base script while avoiding the old
# unbounded dynamic_gbs expansion.
SLIME_MAX_DYNAMIC_GLOBAL_BATCH_SIZE=${SLIME_MAX_DYNAMIC_GLOBAL_BATCH_SIZE:-16}

ROLLOUT_ARGS=(
  --data-source-path gui_data_source.AgentNetDataSource
  --reward-key score
  --num-rollout 1000
  --rollout-batch-size ${ROLLOUT_BATCH_SIZE}
  --n-samples-per-prompt ${N_SAMPLES_PER_PROMPT}
  --rollout-max-response-len 512
  --rollout-temperature 1.0
  --gui-max-steps 10
  --gui-max-image-history-length 3
  --num-steps-per-rollout 2
)

IN_FLIGHT_SAMPLES_ESTIMATE=$(( ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT ))
echo "Configured rollout-batch-size x n-samples-per-prompt = ${IN_FLIGHT_SAMPLES_ESTIMATE}"

EVAL_ARGS=(
  --eval-temperature 0.0
  --gui-eval-max-steps 10
  --n-samples-per-eval-prompt 1
)
if [ -n "${GUI_EVAL_INTERVAL}" ]; then
  EVAL_ARGS+=(--eval-interval "${GUI_EVAL_INTERVAL}")
fi

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr 1e-6
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.95
  --optimizer-cpu-offload
  --overlap-cpu-optimizer-d2h-h2d
  --use-precision-aware-optimizer
)

PERF_ARGS=(
  --tensor-model-parallel-size 2
  --sequence-parallel
  --pipeline-model-parallel-size 1
  # CP must be 1: VLM multimodal tensors are not sliced across CP ranks
  --context-parallel-size 1
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 32
  --megatron-to-hf-mode bridge
  --use-dynamic-batch-size
  --max-tokens-per-gpu 1024
)

GRPO_ARGS=(
  --advantage-estimator grpo
  # AgentNet offline reward is task-level static metadata, not an online
  # rollout-dependent preference signal. Samples duplicated from the same task
  # therefore share the same scalar reward, so group reward normalization would
  # collapse the whole group to zero advantage.
  --disable-rewards-normalization
  --dynamic_history
  --use-kl-loss
  --kl-loss-type low_var_kl
  --kl-loss-coef 0.01
)

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine ${ROLLOUT_NUM_GPUS_PER_ENGINE}
  --sglang-mem-fraction-static 0.85
)

# Offline generate: no env pool server needed
CUSTOM_ARGS=(
  --custom-generate-function-path generate_with_agentnet.generate
  --custom-rm-path generate_with_agentnet.reward_func
)

WANDB_ARGS=()

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=$( [ "$NVLINK_COUNT" -gt 0 ] && echo 1 || echo 0 )
echo "HAS_NVLINK: $HAS_NVLINK"

export PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:2048

ray start --head --node-ip-address 127.0.0.1 --num-gpus ${NUM_GPUS} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${OFFICIAL_MBRIDGE_SRC}:${MEGATRON_LM_PATH}:${SCRIPT_DIR}:${SLIME_DIR}\",
    \"PYTHONUNBUFFERED\": \"${PYTHONUNBUFFERED}\",
    \"PYTHONFAULTHANDLER\": \"${PYTHONFAULTHANDLER}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"PYTORCH_ALLOC_CONF\": \"${PYTORCH_ALLOC_CONF}\",
    \"PATH\": \"${PATH}\",
    \"LD_LIBRARY_PATH\": \"${LD_LIBRARY_PATH:-}\",
    \"OFFICIAL_MBRIDGE_ROOT\": \"${OFFICIAL_MBRIDGE_ROOT}\",
    \"OFFICIAL_MBRIDGE_SRC\": \"${OFFICIAL_MBRIDGE_SRC}\",
    \"MEGATRON_LM_PATH\": \"${MEGATRON_LM_PATH}\",
    \"HF_CKPT\": \"${HF_CKPT}\",
    \"SLIME_DEBUG_QWEN35_EXPORT\": \"${SLIME_DEBUG_QWEN35_EXPORT:-0}\",
    \"SLIME_QWEN35_DISABLE_LINEAR_FC1_RECHUNK\": \"${SLIME_QWEN35_DISABLE_LINEAR_FC1_RECHUNK:-1}\",
    \"SLIME_QWEN35_DEBUG_EMPTY_TRAIN_TENSORS\": \"${SLIME_QWEN35_DEBUG_EMPTY_TRAIN_TENSORS:-1}\",
    \"SLIME_QWEN35_DEBUG_TRAIN_ANOMALY\": \"${SLIME_QWEN35_DEBUG_TRAIN_ANOMALY:-0}\",
    \"SLIME_QWEN35_DEBUG_LOSS_METRICS\": \"${SLIME_QWEN35_DEBUG_LOSS_METRICS:-0}\",
    \"SLIME_MAX_DYNAMIC_GLOBAL_BATCH_SIZE\": \"${SLIME_MAX_DYNAMIC_GLOBAL_BATCH_SIZE}\",
    \"AGENTNET_JSONL_PATH\": \"${AGENTNET_JSONL_PATH}\",
    \"AGENTNET_IMAGE_DIR\": \"${AGENTNET_IMAGE_DIR}\",
    \"GUI_AGENT_CLASS_PATH\": \"${GUI_AGENT_CLASS_PATH}\",
    \"GUI_COORDINATE_TYPE\": \"${GUI_COORDINATE_TYPE}\",
    \"GUI_RESULT_DIR\": \"${GUI_RESULT_DIR}\"
  }
}"

echo "===== RUNTIME_ENV_JSON ====="
echo "${RUNTIME_ENV_JSON}" | python3 -m json.tool

RAY_JOB_SUBMISSION_ID=${RAY_JOB_SUBMISSION_ID:-"gui_qwen35_agentnet_$(date +%Y%m%d_%H%M%S)"}

ray job submit --address="http://127.0.0.1:8265" \
  --submission-id "${RAY_JOB_SUBMISSION_ID}" \
  --no-wait \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 -u "${SLIME_DIR}/train.py" \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node ${ACTOR_GPUS} \
  --rollout-num-gpus ${ROLLOUT_GPUS} \
  --multimodal-keys "${MULTIMODAL_KEYS}" \
  ${MODEL_ARGS[@]} \
  ${CKPT_ARGS[@]} \
  ${ROLLOUT_ARGS[@]} \
  ${EVAL_ARGS[@]} \
  ${PERF_ARGS[@]} \
  ${OPTIMIZER_ARGS[@]} \
  ${GRPO_ARGS[@]} \
  ${SGLANG_ARGS[@]} \
  ${WANDB_ARGS[@]} \
  ${CUSTOM_ARGS[@]}

echo "Following live Ray logs for ${RAY_JOB_SUBMISSION_ID}"
set +e
ray job logs --address="http://127.0.0.1:8265" "${RAY_JOB_SUBMISSION_ID}" -f --log-style=record
RAY_LOG_EXIT=$?
RAY_STATUS_OUTPUT=$(ray job status --address="http://127.0.0.1:8265" "${RAY_JOB_SUBMISSION_ID}" --log-style=record 2>&1)
echo "${RAY_STATUS_OUTPUT}"
set -e

if [[ "${RAY_STATUS_OUTPUT}" == *"SUCCEEDED"* ]]; then
  exit 0
fi

echo "Ray job failed (submission id: ${RAY_JOB_SUBMISSION_ID}, logs exit: ${RAY_LOG_EXIT})"
exit 1
