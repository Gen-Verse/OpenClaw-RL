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

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

# default to 8 GPUs per node if not set by the cluster scheduler
NUM_GPUS=${NUM_GPUS:-8}

# Increase Ray heartbeat/health-check timeouts to prevent worker nodes
# from being mistakenly marked as dead during heavy initialization
export RAY_health_check_failure_threshold=20
export RAY_health_check_period_ms=5000
export RAY_health_check_timeout_ms=30000
export RAY_num_heartbeats_timeout=60

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/glm4-9B.sh"

CKPT_ARGS=(
   --hf-checkpoint /data_storage/wyj/systems/huggingface/hub/GLM-Z1-9B-0414
   --ref-load /data_storage/wyj/systems/huggingface/hub/GLM-Z1-9B-0414_torch_dist
   --load /data_storage/wyj/slime_export/ckpt/GLM-Z1-9B-0414_slime_4nodes/
   --save /data_storage/wyj/slime_export/ckpt/GLM-Z1-9B-0414_slime_4nodes/
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data /data_storage/wyj/slime_export/data/dapo-math-17k/dapo-math-17k.jsonl
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

   --global-batch-size 256
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data aime /data_storage/wyj/slime_export/data/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 16
   --eval-max-response-len 16384
   --eval-top-p 1
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 2
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 4608
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
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project test_slime
   --wandb-group test_slime_glm4-9B-4nodes
   --wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
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


# Resolve node IPs from the cluster scheduler's MLP_WORKER_*_HOST env vars
# to ensure all nodes use IPs on the same reachable network
MASTER_ADDR="${MLP_WORKER_0_HOST:-${MASTER_ADDR:-$(hostname -I | awk '{print $1}')}}"
_WORKER_IP_VAR="MLP_WORKER_${MLP_ROLE_INDEX}_HOST"
NODE_IP="${!_WORKER_IP_VAR:-${WORKER_IP:-$(hostname -I | awk '{print $1}')}}"

echo "MLP_ROLE_INDEX=${MLP_ROLE_INDEX}, MASTER_ADDR=${MASTER_ADDR}, NODE_IP=${NODE_IP}"

# launch the master node of ray in container
if [[ ${MLP_ROLE_INDEX} -eq 0 ]]; then
  ray start --head --node-ip-address ${NODE_IP} --num-gpus ${NUM_GPUS} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
else
  sleep 30
  ray start --address=${MASTER_ADDR}:6379 --num-gpus ${NUM_GPUS} --node-ip-address ${NODE_IP}
fi

# Build the runtime environment JSON with proper variable substitution
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/data_storage/wyj/slime_export/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

if [[ ${MLP_ROLE_INDEX} -eq 0 ]]; then
  ray job submit --address="http://${MASTER_ADDR}:8265" \
     --runtime-env-json="${RUNTIME_ENV_JSON}" \
     #-- python3 train.py \
     -- python3 train_async.py \
     --actor-num-nodes 4 \
     --actor-num-gpus-per-node 4 \
     --rollout-num-gpus 16 \
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
else
  # Worker nodes: stay alive until the ray node is stopped
  echo "Worker node ${MLP_ROLE_INDEX} joined the cluster. Waiting for job to finish..."
  while ray status > /dev/null 2>&1; do
    sleep 60
  done
  echo "Ray cluster stopped. Worker node ${MLP_ROLE_INDEX} exiting."
fi
