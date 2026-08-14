#!/bin/bash
# Qwen3-4B OpenClaw multi-candidate top-K OPD launcher (2-node, 16-GPU).
#
# Same pipeline as run_qwen3_4b_openclaw_topk_select.sh, scaled across two
# 8-GPU nodes with the requested doubled resource layout.
#
# 16-GPU layout (2 nodes x 8 GPUs)
# --------------------------------
#   Node A (8): ACTOR_GPUS=8                         (Megatron training, TP=8)
#   Node B (8): ROLLOUT_GPUS=4                       (SGLang student, 2 engines x TP=2)
#               + PRM_GPUS=2                         (SGLang PRM hint gen, 1 engine x TP=2)
#               + PRM_TEACHER_GPUS=2                 (Megatron teacher logp, TP=2)
#
# Slime allocates bundles in order actor -> rollout -> prm -> prm_teacher
# (see slime/ray/placement_group.py), so 8 / 4+2+2 packs cleanly across two
# 8-GPU nodes. All TP groups stay intra-node: actor TP=8 on node A; rollout
# and PRM/teacher TP=2 groups on node B.
#
# Why actor TP=8
# --------------
# The PRM teacher path forces teacher_TP = prm_teacher_num_gpus. With
# PRM_TEACHER_GPUS=2 the teacher has DP=1; using all 8 actor GPUs as TP=8
# also keeps actor DP=1, avoiding a DP mismatch between actor-side top-k
# log-prob calculation and the teacher log-prob pool.
#
# Multi-node bring-up uses the same MLP_ROLE_INDEX-driven launcher as
# run_qwen3_8b_openclaw_topk_select_2nodes.sh.

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
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"max_split_size_mb:2048,expandable_segments:True"}
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-/tmp}"

# Unset proxy to avoid distributed startup issues across nodes.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

# Per-node + total GPU counts. 16-GPU job spread across 2x8 nodes.
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NUM_NODES=${NUM_NODES:-2}
NUM_GPUS=${NUM_GPUS:-$((GPUS_PER_NODE * NUM_NODES))}

# topk-select REQUIRES megatron PRM teacher -- the inference-side teacher
# path computes single-cand teacher_log_probs only and does not produce
# per-cand top-K. Force the megatron layout.
export OPENCLAW_COMBINE_OPD_TEACHER_SOURCE="megatron"

ACTOR_GPUS=${ACTOR_GPUS:-8}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-4}
PRM_GPUS=${PRM_GPUS:-2}
PRM_NUM_GPUS_PER_ENGINE=${PRM_NUM_GPUS_PER_ENGINE:-2}
PRM_TEACHER_GPUS=${PRM_TEACHER_GPUS:-2}

if (( ACTOR_GPUS + ROLLOUT_GPUS + PRM_GPUS + PRM_TEACHER_GPUS > NUM_GPUS )); then
    echo "ACTOR_GPUS + ROLLOUT_GPUS + PRM_GPUS + PRM_TEACHER_GPUS must be <= NUM_GPUS"
    echo "ACTOR_GPUS=${ACTOR_GPUS}, ROLLOUT_GPUS=${ROLLOUT_GPUS}, PRM_GPUS=${PRM_GPUS}, PRM_TEACHER_GPUS=${PRM_TEACHER_GPUS}, NUM_GPUS=${NUM_GPUS}"
    exit 1
fi

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
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
SLIME_ROOT="${REPO_ROOT}/slime"
MEGATRON_LM_PATH=${MEGATRON_LM_PATH:-"${REPO_ROOT}/Megatron-LM"}
source "${SLIME_ROOT}/scripts/models/qwen3-4B.sh"

# Student (HF for tokenizer + SGLang) and student-torch_dist (Megatron load).
HF_CKPT=${HF_CKPT:-/data_storage/wyj/systems/huggingface/hub/Qwen3-4B-Thinking-2507}
REF_LOAD=${REF_LOAD:-/data_storage/wyj/systems/huggingface/hub/Qwen3-4B-Thinking-2507-_torch_dist}
SAVE_CKPT=${SAVE_CKPT:-${REPO_ROOT}/ckpt/qwen3-4b-openclaw-topk-select-2nodes}

# PRM teacher: same family as the student in this setting. Megatron teacher
# loads torch_dist; SGLang PRM loads HF.
PRM_MODEL_PATH=${PRM_MODEL_PATH:-${HF_CKPT}}
PRM_TEACHER_LOAD=${PRM_TEACHER_LOAD:-${REF_LOAD}}
PRM_TEACHER_HF=${PRM_TEACHER_HF:-${HF_CKPT}}

export SGLANG_API_KEY="${SGLANG_API_KEY}"
export SERVED_MODEL_NAME="qwen3-4b"
export HOST="0.0.0.0"
export PORT="30000"
export OPENCLAW_RECORD_ENABLED="${OPENCLAW_RECORD_ENABLED:-1}"  # 0=off, 1=on
export OPENCLAW_RECORD_FILE="${SCRIPT_DIR}/results/qwen3_4b_topk_select_2nodes_record.jsonl"
export TP="2"
export CONTEXT_LENGTH="32768"
export MEM_FRACTION_STATIC="0.8"
export REASONING_PARSER="qwen3"
export TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen25}"
export PRM_M="${PRM_M:-3}"  # judge votes per turn; topk-select keeps all accepted
export OPENCLAW_OPD_TEACHER_LP_MAX_CONCURRENCY="${OPENCLAW_OPD_TEACHER_LP_MAX_CONCURRENCY:-1}"

# combine_loss-style RL+OPD weighting
export OPENCLAW_TOPK_W_RL="${OPENCLAW_TOPK_W_RL:-1.0}"
export OPENCLAW_TOPK_W_OPD="${OPENCLAW_TOPK_W_OPD:-1.0}"
export OPENCLAW_TOPK_ADV_DIFF_CLIP="${OPENCLAW_TOPK_ADV_DIFF_CLIP:-1.0}"
export TRAIN_EPOCHS="${TRAIN_EPOCHS:-1}"

# Subset S_t selection mode for the OPD loss kernel.
#   student : S_t = top-K(pi_old)
#   teacher : S_t = top-K(pi_T,k*)
#   overlap : S_t = top-K(pi_old) intersect top-K(pi_T,k*)
OPENCLAW_TOPK_SUBSET_MODE="${OPENCLAW_TOPK_SUBSET_MODE:-student}"

# Hint-selection mode (k* picker per turn at the actor side):
#   shortest          : k* = 0 always; only the shortest cand supervises.
#   sequence_optimal  : argmax_k Sum_t |S^q_t intersect S^p_{t,k}| per Sample.
#   token_optimal     : k*(t) = argmax_k |S^q_t intersect S^p_{t,k}| per token.
OPENCLAW_TOPK_HINT_SELECTION="${OPENCLAW_TOPK_HINT_SELECTION:-sequence_optimal}"

# Top-K width on both student and teacher sides.
OPENCLAW_TOPK_K="${OPENCLAW_TOPK_K:-4}"
export OPENCLAW_TOPK_MAX_CAND="${OPENCLAW_TOPK_MAX_CAND:-3}"

CKPT_ARGS=(
   --hf-checkpoint "${HF_CKPT}"
   --ref-load "${REF_LOAD}"
   --save "${SAVE_CKPT}"
   --save-interval 100
   # Qwen3-4B-Thinking rope_theta used by the existing single-node launcher.
   --rotary-base 5000000
   --prm-teacher-load "${PRM_TEACHER_LOAD}"
   --prm-teacher-num-gpus "${PRM_TEACHER_GPUS}"
   --prm-teacher-hf-checkpoint "${PRM_TEACHER_HF}"
   --prm-teacher-rotary-base 5000000
)

ROLLOUT_ARGS=(
   --disable-rollout-global-dataset
   --rollout-function-path openclaw_combine_select_rollout.generate_rollout_openclaw_combine_select

   --num-rollout 100000000
   --rollout-batch-size 16
   --n-samples-per-prompt 1
   --rollout-max-response-len 8192
   --rollout-max-context-len 32768
   --rollout-temperature 0.6
   --reward-key score

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
   --max-tokens-per-gpu 32768
   --log-probs-chunk-size 1024
)

TOPK_SELECT_ARGS=(
   --advantage-estimator grpo
   --disable-rewards-normalization
   --loss-type custom_loss
   --custom-loss-function-path openclaw_topk_select_loss.openclaw_topk_select_loss_function
   --distill-topk "${OPENCLAW_TOPK_K}"
   --distill-subset-mode "${OPENCLAW_TOPK_SUBSET_MODE}"
   --hint-m "${OPENCLAW_TOPK_MAX_CAND}"
   --hint-selection "${OPENCLAW_TOPK_HINT_SELECTION}"
   --use-kl-loss
   --kl-loss-coef 0.0
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-5
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

EVAL_ARGS=()

SGLANG_ARGS=(
   # 4 rollout GPUs -> 2 SGLang engines, TP=2 each.
   --rollout-num-gpus-per-engine 2
   --sglang-tool-call-parser "${TOOL_CALL_PARSER}"
   --sglang-mem-fraction-static 0.8
   --sglang-context-length 32768
   --sglang-reasoning-parser qwen3
   --sglang-max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS:-64}"
)

PRM_ARGS=(
   --prm-enable
   --prm-num-gpus "${PRM_GPUS}"
   --prm-num-gpus-per-engine "${PRM_NUM_GPUS_PER_ENGINE}"
   --prm-model-path "${PRM_MODEL_PATH}"
   --prm-m "${PRM_M}"
   --prm-temperature "${PRM_TEMPERATURE:-0.6}"
   --prm-max-new-tokens "${PRM_MAX_NEW_TOKENS:-8192}"
)

CUSTOM_ARGS=(
   --custom-generate-function-path openclaw_combine_api_server.generate
   --custom-rm-path openclaw_combine_api_server.reward_func
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

USE_WANDB=${USE_WANDB:-1}
WANDB_PROJECT=${WANDB_PROJECT:-openclaw_rl}
WANDB_KEY_VALUE=${WANDB_KEY:-${WANDB_API_KEY:-}}
if [ "${USE_WANDB}" = "1" ] && [ -n "${WANDB_KEY_VALUE}" ]; then
  WANDB_ARGS=(
    --use-wandb
    --wandb-project "${WANDB_PROJECT}"
    --wandb-group qwen3-4b-openclaw-topk-select-2nodes
    --wandb-key "${WANDB_KEY_VALUE}"
  )
else
  WANDB_ARGS=()
fi

export OPENCLAW_EVAL_MODE="${OPENCLAW_EVAL_MODE:-1}"

# ---------------------------------------------------------------------------
# Multi-node identity. MLP_ROLE_INDEX=0 is the head node; >0 workers join the
# head Ray cluster and sit idle until the Ray job finishes or the cluster drops.
# ---------------------------------------------------------------------------
MLP_ROLE_INDEX=${MLP_ROLE_INDEX:-0}
MASTER_ADDR="${MLP_WORKER_0_HOST:-${MASTER_ADDR:-$(hostname -I | awk '{print $1}')}}"
_WORKER_IP_VAR="MLP_WORKER_${MLP_ROLE_INDEX}_HOST"
NODE_IP="${!_WORKER_IP_VAR:-${WORKER_IP:-$(hostname -I | awk '{print $1}')}}"

export MASTER_ADDR
export no_proxy="127.0.0.1,${MASTER_ADDR}"
echo "MLP_ROLE_INDEX=${MLP_ROLE_INDEX}, MASTER_ADDR=${MASTER_ADDR}, NODE_IP=${NODE_IP}"

if [[ ${MLP_ROLE_INDEX} -eq 0 ]]; then
   ray start --head \
      --node-ip-address "${NODE_IP}" \
      --num-gpus "${GPUS_PER_NODE}" \
      --disable-usage-stats \
      --dashboard-host=0.0.0.0 \
      --dashboard-port=8265
else
   sleep 30
   ray start \
      --address="${MASTER_ADDR}:6379" \
      --num-gpus "${GPUS_PER_NODE}" \
      --node-ip-address "${NODE_IP}"
fi

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_LM_PATH}:${SCRIPT_DIR}:${REPO_ROOT}/openclaw-opd:${REPO_ROOT}/hint_opt_exp:${SLIME_ROOT}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"FLASHINFER_WORKSPACE_BASE\": \"${FLASHINFER_WORKSPACE_BASE}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"${PYTORCH_CUDA_ALLOC_CONF}\",
    \"OPENCLAW_EVAL_MODE\": \"${OPENCLAW_EVAL_MODE}\",
    \"OPENCLAW_COMBINE_OPD_TEACHER_SOURCE\": \"${OPENCLAW_COMBINE_OPD_TEACHER_SOURCE}\",
    \"OPENCLAW_TOPK_W_RL\": \"${OPENCLAW_TOPK_W_RL}\",
    \"OPENCLAW_TOPK_W_OPD\": \"${OPENCLAW_TOPK_W_OPD}\",
    \"OPENCLAW_TOPK_ADV_DIFF_CLIP\": \"${OPENCLAW_TOPK_ADV_DIFF_CLIP}\",
    \"OPENCLAW_TOPK_MAX_CAND\": \"${OPENCLAW_TOPK_MAX_CAND}\",
    \"TRAIN_EPOCHS\": \"${TRAIN_EPOCHS}\"
  }
}"

cd "${SLIME_ROOT}"

TRAIN_ENTRY=${TRAIN_ENTRY:-train_async.py}
RAY_JOB_SUBMISSION_ID=${RAY_JOB_SUBMISSION_ID:-"qwen3_4b_openclaw_topk_select_2nodes_$(date +%Y%m%d_%H%M%S)"}

if [[ ${MLP_ROLE_INDEX} -eq 0 ]]; then
   ray job submit --address="http://${MASTER_ADDR}:8265" \
      --submission-id "${RAY_JOB_SUBMISSION_ID}" \
      --no-wait \
      --runtime-env-json="${RUNTIME_ENV_JSON}" \
      -- python3 -u "${TRAIN_ENTRY}" \
      --actor-num-nodes 1 \
      --actor-num-gpus-per-node "${ACTOR_GPUS}" \
      --rollout-num-gpus "${ROLLOUT_GPUS}" \
      --num-gpus-per-node "${GPUS_PER_NODE}" \
      ${MODEL_ARGS[@]} \
      ${CKPT_ARGS[@]} \
      ${ROLLOUT_ARGS[@]} \
      ${OPTIMIZER_ARGS[@]} \
      ${TOPK_SELECT_ARGS[@]} \
      ${PERF_ARGS[@]} \
      ${EVAL_ARGS[@]} \
      ${SGLANG_ARGS[@]} \
      ${MISC_ARGS[@]} \
      ${WANDB_ARGS[@]} \
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
   echo "Worker node ${MLP_ROLE_INDEX} joined the cluster. Waiting for Ray to stay up..."
   while ray status > /dev/null 2>&1; do
      sleep 60
   done
   echo "Ray cluster stopped. Worker node ${MLP_ROLE_INDEX} exiting."
fi
