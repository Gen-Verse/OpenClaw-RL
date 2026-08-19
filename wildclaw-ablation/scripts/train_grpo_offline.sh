#!/usr/bin/env bash
# Offline Binary-GRPO on WCB train-split trajectories (FSDP + INT4 QLoRA, single GPU).
# Data: .pt produced by wcb_to_rl_dataset.py (tokens + rollout_log_probs + rewards).
set -euo pipefail
set -x

ABLATION_ROOT="${ABLATION_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)}"
REPO_ROOT="$(cd -- "${ABLATION_ROOT}/.." &>/dev/null && pwd)"
SLIME_ROOT="${REPO_ROOT}/slime"

HF_CKPT="${HF_CKPT:?path to base model, e.g. <repo>/models/qwen3-0.6B}"
RL_DATA="${RL_DATA:?path to .pt from wcb_to_rl_dataset.py, may contain {rollout_id}}"
SAVE_CKPT="${SAVE_CKPT:-${REPO_ROOT}/export/ckpt/wcb_grpo_qlora_ckpt}"

# training data shape must match: samples = ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"

export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-"max_split_size_mb:2048"}
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

ray stop --force || true
pkill -9 sglang || true
pkill -9 ray || true

export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus 1 --disable-usage-stats \
  --dashboard-host=0.0.0.0 --dashboard-port=8265

cd "${SLIME_ROOT}"
python3 train.py \
  --train-backend fsdp \
  --debug-train-only \
  --load-debug-rollout-data "${RL_DATA}" \
  --hf-checkpoint "${HF_CKPT}" \
  --ref-load "${HF_CKPT}" \
  --save "${SAVE_CKPT}" \
  --save-interval ${SAVE_INTERVAL:-10} \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node 1 \
  --num-gpus-per-node 1 \
  --colocate \
  --num-rollout ${NUM_ROLLOUT:-1} \
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}" \
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}" \
  --rollout-max-context-len ${MAX_CONTEXT_LEN:-16384} \
  --advantage-estimator grpo \
  --use-rollout-logprobs \
  --eps-clip 0.2 \
  --eps-clip-high 0.28 \
  --use-kl-loss \
  --kl-loss-coef ${KL_COEF:-0.0} \
  --kl-loss-type low_var_kl \
  --entropy-coef 0.0 \
  --optimizer adam \
  --lr ${LR:-1e-5} \
  --lr-decay-style constant \
  --weight-decay 0.1 \
  --adam-beta1 0.9 \
  --adam-beta2 0.98 \
  --micro-batch-size ${MICRO_BATCH_SIZE:-1} \
  --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU:-8192} \
  --gradient-checkpointing \
  --use-lora \
  --lora-rank ${LORA_RANK:-4} \
  --lora-alpha ${LORA_ALPHA:-4} \
  --lora-target-modules "${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}" \
  --fsdp-load-in-4bit \
  --fsdp-bnb-4bit-quant-type nf4 \
  --fsdp-bnb-4bit-compute-dtype bfloat16 \
  --fsdp-bnb-4bit-use-double-quant \
  --fsdp-prepare-model-for-kbit-training
