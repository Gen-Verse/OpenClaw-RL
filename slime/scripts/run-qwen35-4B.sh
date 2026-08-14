#!/bin/bash

pkill -9 sglang || true
sleep 3
ray stop --force || true
pkill -9 ray || true
pkill -9 python || true
sleep 3
pkill -9 ray || true
pkill -9 python || true

set -ex

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_DIR="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"

OFFICIAL_MBRIDGE_ROOT=${OFFICIAL_MBRIDGE_ROOT:-"/data_storage/wyj/OpenClaw-RL/Megatron-Bridge-qwen35"}
OFFICIAL_MBRIDGE_SRC="${OFFICIAL_MBRIDGE_ROOT}/src"
MEGATRON_LM_PATH=${MEGATRON_LM_PATH:-"${OFFICIAL_MBRIDGE_ROOT}/3rdparty/Megatron-LM"}

if [[ ! -d "${OFFICIAL_MBRIDGE_SRC}" ]]; then
  echo "OFFICIAL_MBRIDGE_SRC does not exist: ${OFFICIAL_MBRIDGE_SRC}"
  exit 1
fi
if [[ ! -d "${MEGATRON_LM_PATH}" ]]; then
  echo "MEGATRON_LM_PATH does not exist: ${MEGATRON_LM_PATH}"
  exit 1
fi

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "${NVLINK_COUNT}" -gt 0 ]; then
  HAS_NVLINK=1
else
  HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

# With the official Megatron-Bridge-qwen35 stack, Qwen3.5-4B uses the newer
# gated-delta-net flag style from the qwen35 VL provider path.
# The training data here is still text-only; "VL" in the model script name is
# about bridge/runtime compatibility, not that the rollout data must be multimodal.
source "${SCRIPT_DIR}/models/qwen3.5-4B-VL.sh"

HF_CKPT=${HF_CKPT:-"/data_storage/wyj/systems/huggingface/hub/Qwen35-4B"}
REF_LOAD=${REF_LOAD:-"${HF_CKPT}"}
SAVE_CKPT=${SAVE_CKPT:-"/data_storage/wyj/OpenClaw-RL/ckpt/qwen35-4b-math-rl"}
ENABLE_RESUME_LOAD=${ENABLE_RESUME_LOAD:-0}
RESUME_LOAD=${RESUME_LOAD:-"${SAVE_CKPT}"}

PROMPT_DATA=${PROMPT_DATA:-"/data_storage/wyj/OpenClaw-RL1/data/dapo-math-17k/dapo-math-17k.jsonl"}
EVAL_AIME_DATA=${EVAL_AIME_DATA:-"/data_storage/wyj/OpenClaw-RL1/data/aime-2024/aime-2024.jsonl"}

NUM_GPUS=${NUM_GPUS:-8}
ACTOR_GPUS=${ACTOR_GPUS:-4}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-4}
ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}
SGLANG_LANGUAGE_ONLY=${SGLANG_LANGUAGE_ONLY:-1}
SGLANG_MEMORY_SAVER_CUDA_GRAPH=${SGLANG_MEMORY_SAVER_CUDA_GRAPH:-false}

if (( ACTOR_GPUS + ROLLOUT_GPUS > NUM_GPUS )); then
  echo "ACTOR_GPUS + ROLLOUT_GPUS must be <= NUM_GPUS"
  echo "ACTOR_GPUS=${ACTOR_GPUS}, ROLLOUT_GPUS=${ROLLOUT_GPUS}, NUM_GPUS=${NUM_GPUS}"
  exit 1
fi

CKPT_ARGS=(
  --megatron-to-hf-mode bridge
  --hf-checkpoint "${HF_CKPT}"
  --ref-load "${REF_LOAD}"
  --save "${SAVE_CKPT}"
  --save-interval 20
)

if [[ "${ENABLE_RESUME_LOAD}" == "1" ]]; then
  CKPT_ARGS+=(--load "${RESUME_LOAD}")
fi

ROLLOUT_ARGS=(
  --prompt-data "${PROMPT_DATA}"
  --input-key prompt
  --label-key label
  --apply-chat-template
  --rollout-shuffle
  --rm-type deepscaler
  --num-rollout 3000
  --rollout-batch-size 32
  --n-samples-per-prompt 8
  --rollout-max-response-len 8192
  --rollout-temperature 1
  --num-steps-per-rollout 2
  --balance-data
)

EVAL_ARGS=(
  --eval-interval 20
  --eval-prompt-data aime "${EVAL_AIME_DATA}"
  --n-samples-per-eval-prompt 16
  --eval-max-response-len 16384
  --eval-top-p 1
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
  --max-tokens-per-gpu 9216
)

GRPO_ARGS=(
  --advantage-estimator grpo
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

if [[ -n "${WANDB_KEY:-}" ]]; then
  WANDB_ARGS=(
    --use-wandb
    --wandb-project "${WANDB_PROJECT:-test_slime}"
    --wandb-group "${WANDB_GROUP:-qwen35-4B-math-rl}"
    --wandb-key "${WANDB_KEY}"
  )
else
  WANDB_ARGS=()
fi

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
  --sglang-mem-fraction-static 0.7
)
if [[ "${SGLANG_LANGUAGE_ONLY}" == "1" ]]; then
  SGLANG_ARGS+=(--sglang-language-only)
fi

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"max_split_size_mb:2048,expandable_segments:True"}
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${OFFICIAL_MBRIDGE_SRC}:${MEGATRON_LM_PATH}:${SLIME_DIR}:${SCRIPT_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"SGLANG_MEMORY_SAVER_CUDA_GRAPH\": \"${SGLANG_MEMORY_SAVER_CUDA_GRAPH}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"${PYTORCH_CUDA_ALLOC_CONF}\",
    \"OFFICIAL_MBRIDGE_ROOT\": \"${OFFICIAL_MBRIDGE_ROOT}\",
    \"OFFICIAL_MBRIDGE_SRC\": \"${OFFICIAL_MBRIDGE_SRC}\",
    \"MEGATRON_LM_PATH\": \"${MEGATRON_LM_PATH}\",
    \"HF_CKPT\": \"${HF_CKPT}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 "${SLIME_DIR}/train_async.py" \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node "${ACTOR_GPUS}" \
  --rollout-num-gpus "${ROLLOUT_GPUS}" \
  ${MODEL_ARGS[@]} \
  ${CKPT_ARGS[@]} \
  ${ROLLOUT_ARGS[@]} \
  ${OPTIMIZER_ARGS[@]} \
  ${GRPO_ARGS[@]} \
  ${WANDB_ARGS[@]} \
  ${PERF_ARGS[@]} \
  ${EVAL_ARGS[@]} \
  ${SGLANG_ARGS[@]} \
  ${MISC_ARGS[@]}
