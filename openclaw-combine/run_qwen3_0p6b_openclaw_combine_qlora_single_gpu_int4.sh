#!/bin/bash

# Qwen3-0.6B single-GPU OpenClaw Combine (Binary RL + OPD) with INT4 QLoRA.
# Same architecture as the 4B script: FSDP trainer + SGLang rollout colocated
# on one GPU, with rollout offload between phases and an external PRM API.

set -euo pipefail
set -x

# NOTE: do NOT enable expandable_segments here. TorchMemorySaver (used by
# --offload-rollout / colocate sleep-wakeup) is incompatible with it and the
# SGLang engine will abort during load_model.
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-"max_split_size_mb:2048"}
export SLIME_TRAINING_SAMPLES_FILE="../results/samples.jsonl"
export PYTHONUNBUFFERED=1
export SLIME_LOGIT_CHUNK_SIZE=${SLIME_LOGIT_CHUNK_SIZE:-512}
export PYTHONFAULTHANDLER=1
export OPENCLAW_GATEWAY_TOKEN="${OPENCLAW_GATEWAY_TOKEN:-}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
export OPENCLAW_GATEWAY_URL="${OPENCLAW_GATEWAY_URL:-}"
export OPENCLAW_WORKSPACE="$HOME/.openclaw/workspace"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-}"   # point to your external LLM
export EXTERNAL_MODEL="${EXTERNAL_MODEL:-}"    # model name for the external LLM
export SLIME_TRAIN_MAX_SEQ_LEN=${SLIME_TRAIN_MAX_SEQ_LEN:-4096}
export SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK=True
export SGLANG_LOGITS_PROCESSER_CHUNK_SIZE=${SGLANG_LOGITS_PROCESSER_CHUNK_SIZE:-128}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
SLIME_ROOT="${REPO_ROOT}/slime"

NUM_GPUS=${NUM_GPUS:-1}
ACTOR_GPUS=${ACTOR_GPUS:-1}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-1}

# Model weights live under <repo>/models/ by default.
HF_CKPT=${HF_CKPT:-"${REPO_ROOT}/models/qwen3-0.6B"}
REF_LOAD=${REF_LOAD:-${HF_CKPT}}
SAVE_CKPT=${SAVE_CKPT:-"${REPO_ROOT}/export/ckpt/qwen3_0p6b_openclaw_combine_single_gpu_int4_qlora_ckpt"}

# External PRM API (OpenAI-compatible)
PRM_EXTERNAL_API_BASE=${PRM_EXTERNAL_API_BASE:-${OPENAI_BASE_URL:-}}
PRM_EXTERNAL_MODEL=${PRM_EXTERNAL_MODEL:-${EXTERNAL_MODEL:-}}
PRM_EXTERNAL_API_KEY=${PRM_EXTERNAL_API_KEY:-${OPENAI_API_KEY:-}}

if [[ -z "${PRM_EXTERNAL_API_BASE}" || -z "${PRM_EXTERNAL_MODEL}" ]]; then
  echo "PRM_EXTERNAL_API_BASE and PRM_EXTERNAL_MODEL are required (or set OPENAI_BASE_URL / EXTERNAL_MODEL)."
  exit 1
fi

export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-0.6b-int4}"
export HOST="0.0.0.0"
export PORT="${PORT:-30000}"

export OPENCLAW_RECORD_ENABLED="${OPENCLAW_RECORD_ENABLED:-1}"
export OPENCLAW_RECORD_FILE="${SCRIPT_DIR}/results/qwen3_0p6b_single_gpu_int4_qlora_record.jsonl"
export OPENCLAW_EVAL_MODE="${OPENCLAW_EVAL_MODE:-1}"

export OPENCLAW_COMBINE_W_RL="${OPENCLAW_COMBINE_W_RL:-1.0}"
export OPENCLAW_COMBINE_W_OPD="${OPENCLAW_COMBINE_W_OPD:-1.0}"
export TRAIN_EPOCHS="${TRAIN_EPOCHS:-1}"
export PRM_M="${PRM_M:-1}"
export OPENCLAW_OPD_TEACHER_LP_MAX_CONCURRENCY="${OPENCLAW_OPD_TEACHER_LP_MAX_CONCURRENCY:-1}"

ray stop --force || true
pkill -9 sglang || true
pkill -9 ray || true

export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export no_proxy="127.0.0.1,${MASTER_ADDR}"
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

CKPT_ARGS=(
  --hf-checkpoint "${HF_CKPT}"
  --ref-load "${REF_LOAD}"
  --save "${SAVE_CKPT}"
  --save-interval 20
)

ROLLOUT_ARGS=(
  --disable-rollout-global-dataset
  --rollout-function-path openclaw_combine_rollout.generate_rollout_openclaw_combine
  --num-rollout ${NUM_ROLLOUT:-20000}
  --rollout-batch-size ${ROLLOUT_BATCH_SIZE:-16}
  --n-samples-per-prompt 1
  --rollout-max-response-len ${ROLLOUT_MAX_RESPONSE_LEN:-4096}
  --rollout-max-context-len ${ROLLOUT_MAX_CONTEXT_LEN:-16384}
  --rollout-temperature ${ROLLOUT_TEMPERATURE:-0.6}
  --reward-key score
  --num-steps-per-rollout 1
)

COMBINE_ARGS=(
  --advantage-estimator grpo
  --disable-rewards-normalization
  --loss-type custom_loss
  --custom-loss-function-path combine_loss.combine_loss_function
  --use-kl-loss
  --kl-loss-coef 0.0
  --kl-loss-type low_var_kl
  --entropy-coef 0.00
  --eps-clip 0.2
  --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr ${LR:-1e-5}
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
)

PERF_ARGS=(
  --micro-batch-size ${MICRO_BATCH_SIZE:-1}
  --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU:-8192}
  --gradient-checkpointing
)

# FSDP QLoRA (INT4 base + LoRA adapters)
QLORA_ARGS=(
  --use-lora
  --lora-rank ${LORA_RANK:-4}
  --lora-alpha ${LORA_ALPHA:-4}
  --lora-target-modules "${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"
  --fsdp-load-in-4bit
  --fsdp-bnb-4bit-quant-type ${FSDP_BNB_4BIT_QUANT_TYPE:-nf4}
  --fsdp-bnb-4bit-compute-dtype ${FSDP_BNB_4BIT_COMPUTE_DTYPE:-bfloat16}
  --fsdp-bnb-4bit-use-double-quant
  --fsdp-prepare-model-for-kbit-training
)

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine 1
  --sglang-context-length ${SGLANG_CONTEXT_LENGTH:-16384}
  --sglang-mem-fraction-static ${SGLANG_MEM_FRACTION_STATIC:-0.6}
  --sglang-reasoning-parser ${SGLANG_REASONING_PARSER:-qwen3}
  --sglang-tool-call-parser ${SGLANG_TOOL_CALL_PARSER:-qwen25}
)

# External PRM: no local PRM engine / GPU allocation.
PRM_ARGS=(
  --prm-enable
  --prm-use-external-api
  --prm-num-gpus 1
  --prm-m ${PRM_M}
  --prm-temperature ${PRM_TEMPERATURE:-0.6}
  --prm-max-new-tokens ${PRM_MAX_NEW_TOKENS:-4096}
  --prm-external-api-base "${PRM_EXTERNAL_API_BASE}"
  --prm-external-model "${PRM_EXTERNAL_MODEL}"
)
if [[ -n "${PRM_EXTERNAL_API_KEY}" ]]; then
  PRM_ARGS+=(--prm-external-api-key "${PRM_EXTERNAL_API_KEY}")
fi

CUSTOM_ARGS=(
  --custom-generate-function-path openclaw_combine_api_server.generate
  --custom-rm-path openclaw_combine_api_server.reward_func
)

WANDB_ARGS=()
if [[ "${USE_WANDB:-0}" == "1" && -n "${WANDB_API_KEY:-}" ]]; then
  WANDB_ARGS=(
    --use-wandb
    --wandb-project ${WANDB_PROJECT:-openclaw_rl_int4}
    --wandb-group qwen3-0p6b-openclaw-combine-int4-qlora
    --wandb-key ${WANDB_API_KEY}
  )
fi

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${SCRIPT_DIR}:${SCRIPT_DIR}/../openclaw-opd:${SLIME_ROOT}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"OPENCLAW_EVAL_MODE\": \"${OPENCLAW_EVAL_MODE}\",
    \"OPENCLAW_COMBINE_W_RL\": \"${OPENCLAW_COMBINE_W_RL}\",
    \"OPENCLAW_COMBINE_W_OPD\": \"${OPENCLAW_COMBINE_W_OPD}\",
    \"TRAIN_EPOCHS\": \"${TRAIN_EPOCHS}\",
    \"PYTORCH_ALLOC_CONF\": \"${PYTORCH_ALLOC_CONF}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 train.py \
  --train-backend fsdp \
  --offload-rollout \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node "${ACTOR_GPUS}" \
  --rollout-num-gpus "${ROLLOUT_GPUS}" \
  --num-gpus-per-node "${NUM_GPUS}" \
  --colocate \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${COMBINE_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  "${PRM_ARGS[@]}" \
  "${QLORA_ARGS[@]}" \
  "${CUSTOM_ARGS[@]}" \
  "${WANDB_ARGS[@]}"
