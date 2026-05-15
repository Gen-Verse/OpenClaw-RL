#!/bin/bash

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
ulimit -n 65535

# keep stdout/stderr unbuffered in ray jobs
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export RAY_DEBUG=1
export RAY_DEDUP_LOGS=0

# default to 8 GPUs if not set by scheduler
NUM_GPUS=${NUM_GPUS:-8}
ACTOR_GPUS=${ACTOR_GPUS:-2}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-4}
PRM_GPUS=${PRM_GPUS:-2}

if (( ACTOR_GPUS + ROLLOUT_GPUS + PRM_GPUS > NUM_GPUS )); then
    echo "ACTOR_GPUS + ROLLOUT_GPUS + PRM_GPUS must be <= NUM_GPUS"
    echo "ACTOR_GPUS=${ACTOR_GPUS}, ROLLOUT_GPUS=${ROLLOUT_GPUS}, PRM_GPUS=${PRM_GPUS}, NUM_GPUS=${NUM_GPUS}"
    exit 1
fi

# set visible devices
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050

# Increase Ray heartbeat/health-check timeouts to reduce false node failures under heavy init.
export RAY_health_check_failure_threshold=20
export RAY_health_check_period_ms=5000
export RAY_health_check_timeout_ms=30000
export RAY_num_heartbeats_timeout=60

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_DIR="$(cd -- "${SCRIPT_DIR}/../slime" &>/dev/null && pwd)"
MEGATRON_LM_PATH=Megatron-LM
MEGATRON_BRIDGE_PATH=Megatron-Bridge/src
SGLANG_PATH=sglang/python
source "${SLIME_DIR}/scripts/models/qwen3-4B.sh"

HF_CKPT=qwen3-4b-sft-SGLang-RL
REF_LOAD=qwen3-4b-sft-SGLang-RL-dist-slime
SAVE_CKPT=qwen3-4b-sft-SGLang-RL-save
# RESUME_LOAD=/workspace/OpenClaw-RL/ckpt/qwen3-4b-retool-rl
# Use the existing run id to continue plotting on the same W&B curve.
#WANDB_RESUME=${WANDB_RESUME:-must}

PRM_MODEL_PATH=Qwen3-4B-Instruct-2507
# DYNAMIC_HISTORY=1

CKPT_ARGS=(
   --hf-checkpoint ${HF_CKPT}
   --ref-load ${REF_LOAD}
   # --load ${RESUME_LOAD}
   --save ${SAVE_CKPT}
   --save-interval 50
   --rotary-base 5000000
)

ROLLOUT_ARGS=(
   --prompt-data dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --reward-key score
   --num-rollout 220
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-max-context-len 16384
   --rollout-temperature 1
   --num-steps-per-rollout 2
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 50
   --eval-prompt-data aime aime-2024.jsonl
   --n-samples-per-eval-prompt 16
   --eval-max-response-len 16384
   --eval-max-context-len 32768
   --eval-top-p 1
   --eval-reward-key acc
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 16384
   --log-probs-chunk-size 1024
)

GRPO_ARGS=(
   --advantage-estimator step_wise
   --use-kl-loss
   --kl-loss-coef 0.01
   --kl-loss-type k3
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

WANDB_ARGS=(
   # --use-wandb
   # --wandb-project slime_retool
   # --wandb-group qwen3-4B-rl_retool
   # --wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.6
   # ======================= NPU 添加参数 =======================
   --sglang-device npu
)

PRM_ARGS=(
   --prm-enable
   --prm-num-gpus "${PRM_GPUS}"
   --prm-num-gpus-per-engine 2
   --prm-model-path "${PRM_MODEL_PATH}"
   --prm-m "${PRM_M:-1}"
   --prm-step-coef "${PRM_STEP_COEF:-1.0}"
   --prm-temperature "${PRM_TEMPERATURE:-0.6}"
   --prm-max-new-tokens "${PRM_MAX_NEW_TOKENS:-4096}"
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
)

CUSTOM_ARGS=(
   --custom-generate-function-path generate_with_retool.generate
   --custom-rm-path generate_with_retool.reward_func
)

DYNAMIC_HISTORY_ARGS=()
if [[ "${DYNAMIC_HISTORY:-0}" == "1" ]]; then
  DYNAMIC_HISTORY_ARGS+=(--dynamic_history)
fi

export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-"expandable_segments:True"}

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# Build the runtime environment JSON with proper variable substitution
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_LM_PATH}:${MEGATRON_BRIDGE_PATH}:${SGLANG_PATH}:${SCRIPT_DIR}:${SLIME_DIR}:$PYTHONPATH\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"PYTORCH_NPU_ALLOC_CONF\": \"${PYTORCH_NPU_ALLOC_CONF}\",
    \"ASCEND_TOOLKIT_HOME\": \"/usr/local/Ascend/cann-8.5.0/\",
    \"ASCEND_OPP_PATH\": \"/usr/local/Ascend/cann-8.5.0/opp/\",
    \"ASCEND_AICPU_PATH\": \"/usr/local/Ascend/cann-8.5.0/\",
    \"ASCEND_HOME_PATH\": \"/usr/local/Ascend/cann-8.5.0/\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train_async.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node ${ACTOR_GPUS} \
   --rollout-num-gpus ${ROLLOUT_GPUS} \
   --num-gpus-per-node ${NUM_GPUS} \
   --skip-eval-before-train \
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
   ${CUSTOM_ARGS[@]} \
   ${PRM_ARGS[@]} \
   ${DYNAMIC_HISTORY_ARGS[@]}
