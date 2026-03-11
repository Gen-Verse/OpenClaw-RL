# SWE-RL

Reinforcement learning on [SWE-Bench](https://swe-bench.github.io/) / [SWE-Gym](https://github.com/SWE-Gym/SWE-Gym) using the [slime](../slime) training framework.

The agent is trained to fix real GitHub issues by writing bash commands inside isolated Docker containers. Evaluation is performed by applying the agent's git patch and running the official test suite.

---

## Architecture

```
┌─ ECS Docker Node(s) ──────────────┐
│  server/swe_exec_server.py (:5000) │  ← pre-installed on each Docker node
│    /container/create               │
│    /container/exec                 │
│    /container/diff                 │
│    /container/evaluate             │
│    /container/destroy              │
└────────────────▲───────────────────┘
                 │ HTTP
┌─ GPU Head Node─┼───────────────────┐
│  server/swe_env_pool_server.py     │  ← started by training script
│    (:18090)  load-balances leases  │
└────────────────▲───────────────────┘
                 │ HTTP
┌─ RolloutManager (Ray Actor) ───────┐
│  swe_env_client.py                 │
│  generate_with_swe_remote.py       │  ← slime generate + reward_func
│  swe_prm.py  (optional PRM)        │
└────────────────────────────────────┘
```

**Data flow for one SWE-Bench instance:**
1. `generate()` allocates a remote Docker container via `swe_env_client.allocate()`
2. Agent loop: LLM generates THOUGHT + bash → remote exec → observation
3. Agent submits patch → a fresh eval container runs `git apply` + test suite → `resolved?`
4. Tokens are encoded with `message_utils.py` → `Sample(tokens, loss_mask, reward)` returned

---

## Directory Structure

```
swe-rl/
├── generate_with_swe_remote.py   # slime entry point (generate + reward_func)
├── swe_env_client.py             # async HTTP client for pool server
├── swe_context_manager.py        # head+tail context window management
├── swe_prm.py                    # PRM step-wise evaluator (optional)
├── swe_utils.py                  # Docker image naming helper
├── message_utils.py              # multi-turn token encoding + loss mask
├── swebench.yaml                 # agent prompt templates (Jinja2)
├── litellm.json                  # LiteLLM model cost registry
│
├── server/                       # remote Docker infrastructure
│   ├── swe_exec_server.py        # runs on each ECS Docker node
│   ├── swe_env_pool_server.py    # runs on GPU head node
│   └── setup_ecs_seed.sh         # one-time ECS node initialisation
│
├── scripts/                      # training launch scripts
│   ├── run_swe_rl_32b_remote_8nodes.sh         # 32B, 8 nodes, fresh start
│   ├── run_swe_rl_32b_remote_8nodes_resume.sh  # 32B, 8 nodes, auto-resume (requires Megatron ckpt)
│   ├── run_swe_rl_32b_remote_4nodes.sh         # 32B, 4 nodes
│   ├── run_swe_rl_8b_remote_2nodes.sh          # 8B, 2 nodes
│   └── run_swe_rl_8b_prm_5nodes_remote.sh      # 8B + PRM, 5 nodes
│
├── eval/                         # standalone evaluation
│   ├── eval_swe.py               # run any LLM on SWE data (no training loop)
│   └── run_eval_swe.sh           # wrapper: starts sglang + pool server + eval
│
├── data/                         # data preparation
│   ├── preprocess_swe_dataset.py # HuggingFace → slime JSONL
│   └── pull_swe_images.sh        # pull Docker images to ECS nodes
│
├── mini-swe-agent/               # third-party dependency (pinned v1.12.0)
└── output/                       # runtime artifacts (created on first run)
    ├── ckpt/                     # training checkpoints
    ├── eval_runs/                # standalone eval results
    └── swe_rollouts/             # rollout trajectory artifacts
```

---

## Step 0 — Prepare Data

Download a SWE dataset and convert it to the slime JSONL format:

```bash
cd swe-rl/

# SWE-Gym training set
python data/preprocess_swe_dataset.py \
  --train-data-source SumanthRH/SWE-Gym-Subset \
  --train-split train \
  --output-dir ~/data/swe_gym_subset

# SWE-bench Verified
python data/preprocess_swe_dataset.py \
  --train-data-source SumanthRH/SWE-bench_Verified \
  --train-split test \
  --output-dir ~/data/swe_verified
```

Output schema (one JSON per line):
```json
{
  "text": "<problem_statement>",
  "metadata": {"instance": {...}, "data_source": "SumanthRH/SWE-Gym-Subset"}
}
```

---

## Step 1 — Set Up ECS Docker Nodes

Each Docker node requires `server/swe_exec_server.py` running as a systemd service on port **5000**.

### 1.1 Upload files

```bash
SWE_RL_DIR=swe-rl
SEED_IP=<ECS public IP>

scp ${SWE_RL_DIR}/server/swe_exec_server.py   root@${SEED_IP}:~/
scp ${SWE_RL_DIR}/server/setup_ecs_seed.sh    root@${SEED_IP}:~/
scp ${SWE_RL_DIR}/data/pull_swe_images.sh     root@${SEED_IP}:~/
scp ~/data/swe_gym_subset/train.jsonl          root@${SEED_IP}:~/train.jsonl
```

### 1.2 Run setup on ECS (installs Docker, registers systemd service, pulls images)

```bash
ssh root@${SEED_IP}
bash ~/setup_ecs_seed.sh        # takes 2–5 hours (image pull)
```

### 1.3 Verify

```bash
curl http://localhost:5000/healthz
# {"ok": true, "running_containers": "0"}

curl http://localhost:5000/images | python3 -m json.tool | grep count
# "count": xxxx
```

### 1.4 Pull images on existing nodes (optional, skip if using the custom image)

```bash
TRAIN=~/train.jsonl bash ~/pull_swe_images.sh        # all images
N=10 TRAIN=~/train.jsonl bash ~/pull_swe_images.sh   # first 10 only
```

After setup, snapshot the ECS into a **custom image** so all subsequent nodes start pre-warmed. See `docs/en/SWE_REMOTE_DOCKER.md` for detailed instructions.

---

## Step 2 — Configure Exec Server URLs

Set `SWE_EXEC_SERVER_URLS` to a comma-separated list of all Docker node addresses:

```bash
export SWE_EXEC_SERVER_URLS="http://172.16.0.10:5000,http://172.16.0.11:5000,http://172.16.0.12:5000"
```

This can be set per-run or written into `swe-rl/.env.swe` and sourced before launching:

```bash
# swe-rl/.env.swe
SWE_EXEC_SERVER_URLS="http://172.16.0.10:5000,..."
HF_CKPT=/path/to/Qwen3-32B
PROMPT_DATA=/path/to/train.jsonl
```

---

## Step 3 — Training

All training scripts are in `scripts/`. Run from the repository root:

```bash
bash swe-rl/scripts/run_swe_rl_32b_remote_8nodes.sh
```

### Available configurations

| Script | Model | Nodes | GPUs | Notes |
|---|---|---|---|---|
| `run_swe_rl_32b_remote_8nodes.sh` | Qwen3-32B | 8 | 64 | fresh run; timestamped checkpoint dir, loads from HF checkpoint via bridge mode |
| `run_swe_rl_32b_remote_8nodes_resume.sh` | Qwen3-32B | 8 | 64 | auto-resume; fixed checkpoint path, re-running continues from last iteration. **Requires a pre-converted Megatron `torch_dist` checkpoint** as `REF_LOAD` for the first run |
| `run_swe_rl_32b_remote_4nodes.sh` | Qwen3-32B | 4 | 32 | smaller scale |
| `run_swe_rl_8b_remote_2nodes.sh` | Qwen3-8B | 2 | 16 | lightweight experiment |
| `run_swe_rl_8b_prm_5nodes_remote.sh` | Qwen3-8B | 5 | 40 | with PRM step rewards |

### Prerequisite for `_resume` script — convert HF checkpoint to Megatron format

`run_swe_rl_32b_remote_8nodes_resume.sh` requires a pre-converted Megatron `torch_dist` checkpoint as `REF_LOAD` (for the first run). Run **once** on a GPU node before starting training:

```bash
cd ../slime
source scripts/models/qwen3-32B.sh          # loads MODEL_ARGS for Qwen3-32B

PYTHONPATH=${MEGATRON_LM_PATH} torchrun --nproc-per-node 8 \
  tools/convert_hf_to_torch_dist.py \
  ${MODEL_ARGS[@]} \
  --hf-checkpoint /path/to/Qwen3-32B \
  --save ../export/ckpt/qwen3-32b_torch_dist
```

The output path must match `REF_LOAD` in the script (default: `${EXPORT_ROOT}/ckpt/qwen3-32b_torch_dist`). This conversion is **not needed** for `run_swe_rl_32b_remote_8nodes.sh`, which loads directly from HF via bridge mode.

---

### Quick start (8-node 32B)

### Key training hyperparameters (from `run_swe_rl_32b_remote_8nodes.sh`)

| Parameter | Default | Description |
|---|---|---|
| `--rollout-max-context-len` | 16384 | total context window (triggers context management) |
| `--rollout-max-response-len` | 4096 | max tokens per LLM generation step |
| `--n-samples-per-prompt` | 8 | rollout group size (GRPO) |
| `--dynamic_history` | enabled | one training sample per rollout step |
| `--lr` | 1e-6 | learning rate |
| `SWE_MAX_CONCURRENT` | 128 | max concurrent Docker containers across all nodes |
| `SWE_MAX_CONTAINERS_PER_NODE` | 15 | per-node container cap |

### Outputs

All scripts write to `EXPORT_ROOT` (default: `swe-rl/output/`). Override with `SAVE_CKPT` and `SWE_SAVE_TRAJ_DIR` env vars.

| Script | Checkpoint path | Rollout artifacts path |
|---|---|---|
| `run_swe_rl_32b_remote_8nodes.sh` | `output/ckpt/swe-rl-32b-remote-8nodes_<timestamp>/` | `output/swe_rollouts/swe-rl-32b-remote-8nodes_<timestamp>/` |
| `run_swe_rl_32b_remote_8nodes_resume.sh` | `output/ckpt/swe-rl-32b-remote-8nodes_<fixed>/` | `output/swe_rollouts/swe-rl-32b-remote-8nodes_<timestamp>/` |
| `run_swe_rl_32b_remote_4nodes.sh` | `output/ckpt/swe-rl-32b-remote-4nodes_<timestamp>/` | `output/swe_rollouts/swe-rl-32b-remote-4nodes_<timestamp>/` |
| `run_swe_rl_8b_remote_2nodes.sh` | `output/ckpt/swe-rl-8b-remote-2nodes_<timestamp>/` | `output/swe_rollouts/swe-rl-8b-remote-2nodes_<timestamp>/` |
| `run_swe_rl_8b_prm_5nodes_remote.sh` | `output/ckpt/swe-rl-8b-prm-5nodes-remote_<timestamp>/` | `output/swe_rollouts/swe-rl-8b-prm-5nodes-remote_<timestamp>/` |

Each rollout artifact directory contains:
```
<instance_id>__<ts>/
├── traj.json    # full conversation + step_debug
├── patch.diff   # git patch submitted by the agent
└── meta.json    # model, sampling params
```

---

## Step 4 — Standalone Evaluation

Run any LLM (e.g. Qwen3-32B as oracle) on SWE data without a training loop. Useful for establishing baselines or debugging the eval pipeline.

```bash
cd swe-rl/

# Required
export SWE_EXEC_SERVER_URLS="http://172.16.0.10:5000,..."
export EVAL_MODEL_PATH=/path/to/Qwen3-32B
export PROMPT_DATA=/path/to/train.jsonl
export OUTPUT_DIR=/path/to/eval_runs/my_run

bash eval/run_eval_swe.sh
```

Or call the Python script directly:

```bash
python eval/eval_swe.py \
  --data           /path/to/train.jsonl \
  --model          openai/Qwen/Qwen3-32B \
  --api-base       http://<ROUTER_IP>:8000/v1 \
  --output-dir     /path/to/eval_runs/my_run \
  --max-concurrent 8 \
  --max-instances  50 \
  --step-limit     20 \
  --max-tokens     8192
```

Eval output per instance:
```
<output_dir>/<instance_id>__<ts>/
├── traj.json      # full conversation + step_debug
├── patch.diff     # git patch submitted by the agent
└── meta.json      # model, args

<output_dir>/summary.json   # aggregate: resolve_rate, avg_steps, ...
<output_dir>/results.jsonl  # per-instance rows
```

---


## Context Management

Long rollouts can exceed the model's context window. `swe_context_manager.py` automatically truncates when `--rollout-max-context-len` is set, using a **head + tail** strategy:

```
[system, problem]
[turn_0, turn_1]            ← head (30% of budget): early exploration
[... 10 turn(s) omitted ...]
[turn_12, ..., turn_19]     ← tail (70% of budget): recent history
```

With `--dynamic_history`, one training sample is created per rollout step, trained on **exactly the truncated context the model saw**, ensuring training and rollout are perfectly aligned.

Example:
```
rollout_max_context_len = 16384
max_new_tokens          = 4096
max_input_tokens        = 12288   (auto-computed)
head_budget             ≈ 3450    (30%)
tail_budget             ≈ 8050    (70%)
```

See `docs/en/CONTEXT_MANAGEMENT.md` for full details.

---

## PRM (Process Reward Model)

By default training uses **outcome reward** only: resolved → +1, failed → −1.

With `--prm-enable`, `swe_prm.py` evaluates each step asynchronously during rollout:
- Constructs a judge prompt: issue description + history + current step + execution result
- Sends to a separate LLM (PRM model) via SGLang router
- m votes (default m=3), scores +1/−1 extracted from `\boxed{}`
- Final reward: `outcome_reward + prm_step_coef × mean_step_score`

Enable in the training script via:

```bash
--prm-enable
--prm-model-path /path/to/prm_model
--prm-m 3
--prm-num-gpus 8
--prm-step-coef 1.0
```

See `docs/en/SWE_PRM.md` for the full design.
