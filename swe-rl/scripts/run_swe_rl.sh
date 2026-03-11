#!/bin/bash

# SWE-Bench RL training with Mini-SWE-Agent + slime GRPO
#
# Prerequisites:
#   pip install minisweagent
#   python preprocess_swegym.py --output_dir ~/data/swe_gym_subset
#
# Usage:
#   bash run_swe_rl.sh

# for rerun the task (same style as gui/retool scripts)
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
source "${SLIME_DIR}/scripts/models/qwen3-4B.sh"
MEGATRON_LM_PATH=${MEGATRON_LM_PATH:-"${SWE_RL_DIR}/../Megatron-LM"}

# ── Auto-install mini-swe-agent (editable) ────────────────────────────
MINISWE_DIR="${SWE_RL_DIR}/mini-swe-agent"
MINISWE_VERSION="v1.12.0"
if ! python3 -c "import minisweagent" 2>/dev/null; then
    if [ ! -d "${MINISWE_DIR}" ]; then
        echo "Cloning mini-swe-agent ${MINISWE_VERSION}..."
        git clone --branch "${MINISWE_VERSION}" --depth 1 \
            https://github.com/SWE-agent/mini-swe-agent.git "${MINISWE_DIR}"
    fi
    echo "Installing mini-swe-agent from local source (editable mode)..."
    pip install -e "${MINISWE_DIR}"
fi

# keep stdout/stderr unbuffered in ray jobs
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

# use docker api mode to avoid timeout
export MSWEA_DOCKER_EXEC_MODE=api

# reduce false ray node death under heavy initialization
export RAY_health_check_failure_threshold=${RAY_health_check_failure_threshold:-20}
export RAY_health_check_period_ms=${RAY_health_check_period_ms:-5000}
export RAY_health_check_timeout_ms=${RAY_health_check_timeout_ms:-30000}
export RAY_num_heartbeats_timeout=${RAY_num_heartbeats_timeout:-60}

# default to 8 GPUs if not set by scheduler
NUM_GPUS=${NUM_GPUS:-8}
ACTOR_GPUS=${ACTOR_GPUS:-4}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-4}

if (( ACTOR_GPUS + ROLLOUT_GPUS > NUM_GPUS )); then
    echo "ACTOR_GPUS + ROLLOUT_GPUS must be <= NUM_GPUS"
    echo "ACTOR_GPUS=${ACTOR_GPUS}, ROLLOUT_GPUS=${ROLLOUT_GPUS}, NUM_GPUS=${NUM_GPUS}"
    exit 1
fi

# ── Checkpoints ───────────────────────────────────────────────────────
HF_CKPT=${HF_CKPT:-/data_storage/wyj/systems/huggingface/hub/Qwen3-4B}
REF_LOAD=${REF_LOAD:-${HF_CKPT}}

CKPT_ARGS=(
   --hf-checkpoint ${HF_CKPT}
   --ref-load ${REF_LOAD}
   --save ${SAVE_CKPT:-${EXPORT_ROOT}/ckpt/swe-rl}
   --save-interval 20
)

ENABLE_RESUME_LOAD=${ENABLE_RESUME_LOAD:-0}
RESUME_LOAD=${RESUME_LOAD:-${EXPORT_ROOT}/ckpt/swe-rl}
if [[ "${ENABLE_RESUME_LOAD}" == "1" ]]; then
    CKPT_ARGS+=(--load "${RESUME_LOAD}")
    echo "Resume load enabled: ${RESUME_LOAD}"
else
    echo "Resume load disabled (ENABLE_RESUME_LOAD=${ENABLE_RESUME_LOAD})"
fi

# ── Rollout ───────────────────────────────────────────────────────────
# DEBUG_MODE=1: use 10-sample subset for quick end-to-end validation
DEBUG_MODE=${DEBUG_MODE:-0}
if [[ "${DEBUG_MODE}" == "1" ]]; then
    PROMPT_DATA=${HOME}/data/swe_gym_subset/train_10.jsonl
    NUM_ROLLOUT=10
    N_SAMPLES=2
    echo "DEBUG_MODE=1: using train_10.jsonl, num-rollout=10, n-samples=2"
else
    PROMPT_DATA=${HOME}/data/swe_gym_subset/train.jsonl
    NUM_ROLLOUT=500
    N_SAMPLES=4
fi

ROLLOUT_ARGS=(
   --prompt-data ${PROMPT_DATA}
   --input-key text
   --metadata-key metadata
   --rollout-shuffle
   --reward-key score
   --num-rollout ${NUM_ROLLOUT}
   --rollout-batch-size 8
   --n-samples-per-prompt ${N_SAMPLES}
   --rollout-max-response-len 4096
   --rollout-max-context-len 32768
   --rollout-temperature 1
   --num-steps-per-rollout 1
)

# ── Eval (disabled — no eval dataset configured yet) ──────────────────
EVAL_ARGS=(
)

# ── Performance / parallelism ─────────────────────────────────────────
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

   --megatron-to-hf-mode bridge

   --use-dynamic-batch-size
   --max-tokens-per-gpu 16384
   --log-probs-chunk-size 1024
)

# ── GRPO ──────────────────────────────────────────────────────────────
GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

# ── Optimizer ─────────────────────────────────────────────────────────
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

# ── SGLang ────────────────────────────────────────────────────────────
# --sglang-router-port must match OPENAI_BASE_URL in .env.swe (port 30000)
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.6
   --sglang-router-port 30000
)

# ── Custom generate / reward ──────────────────────────────────────────
CUSTOM_ARGS=(
   --custom-generate-function-path generate_with_swe.generate
   --custom-rm-path generate_with_swe.reward_func
)

# ── Weights & Biases ─────────────────────────────────────────────────
WANDB_KEY_VALUE=${WANDB_KEY:-${WANDB_API_KEY:-}}
if [ -n "${WANDB_KEY_VALUE}" ]; then
    WANDB_ARGS=(
       --use-wandb
       --wandb-project slime_swe
       --wandb-group qwen3-4B-rl_swe
       --wandb-key ${WANDB_KEY_VALUE}
    )
else
    WANDB_ARGS=()
fi

# ── Miscellaneous ─────────────────────────────────────────────────────
MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

# ── Launch ────────────────────────────────────────────────────────────

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"max_split_size_mb:2048,expandable_segments:True"}

# Prevent HTTP_PROXY from intercepting local Ray dashboard requests
export NO_PROXY=localhost,127.0.0.1,HEAD_NODE_IP,0.0.0.0
export no_proxy=localhost,127.0.0.1,HEAD_NODE_IP,0.0.0.0

# Load Mini-SWE-Agent environment variables
source "${SWE_RL_DIR}/.env.swe"
SWE_LITELLM_MODEL_NAME=${SWE_LITELLM_MODEL_NAME:-openai/Qwen/Qwen3-4B} # default to Qwen3-4B, change to your model name if needed
SWE_SAVE_TRAJ_DIR=${SWE_SAVE_TRAJ_DIR:-${EXPORT_ROOT}/swe_rollouts}
echo "SWE rollout artifacts dir: ${SWE_SAVE_TRAJ_DIR}"

ray start --head --node-ip-address 127.0.0.1 --num-gpus ${NUM_GPUS} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_LM_PATH}:${SWE_RL_DIR}:${SWE_RL_DIR}/server:${SLIME_DIR}\",
    \"PYTHONUNBUFFERED\": \"${PYTHONUNBUFFERED}\",
    \"PYTHONFAULTHANDLER\": \"${PYTHONFAULTHANDLER}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"${PYTORCH_CUDA_ALLOC_CONF}\",
    \"OPENAI_BASE_URL\": \"${OPENAI_BASE_URL}\",
    \"OPENAI_API_KEY\": \"${OPENAI_API_KEY}\",
    \"LITELLM_MODEL_REGISTRY_PATH\": \"${LITELLM_MODEL_REGISTRY_PATH}\",
    \"SWE_LITELLM_MODEL_NAME\": \"${SWE_LITELLM_MODEL_NAME}\",
    \"SWE_SAVE_TRAJ_DIR\": \"${SWE_SAVE_TRAJ_DIR}\",
    \"SWE_CONFIG_PATH\": \"${SWE_RL_DIR}/swebench.yaml\",
    \"SWE_MAX_CONCURRENT\": \"${SWE_MAX_CONCURRENT:-3}\",
    \"MSWEA_DOCKER_EXEC_MODE\": \"${MSWEA_DOCKER_EXEC_MODE:-api}\",
    \"HTTP_PROXY\": \"http://PROXY_IP:3128\",
    \"HTTPS_PROXY\": \"http://PROXY_IP:3128\",
    \"NO_PROXY\": \"localhost,127.0.0.1,HEAD_NODE_IP,0.0.0.0\",
    \"no_proxy\": \"localhost,127.0.0.1,HEAD_NODE_IP,0.0.0.0\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train_async.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node ${ACTOR_GPUS} \
   --rollout-num-gpus ${ROLLOUT_GPUS} \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${CUSTOM_ARGS[@]}
