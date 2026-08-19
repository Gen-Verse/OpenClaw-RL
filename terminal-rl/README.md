# Terminal RL

RL training for terminal agents. The agent interacts with Docker-hosted environments and is trained with GRPO (optional PRM).

## Architecture & requirements

This setup has two independent components:

- **Training machine**: runs task router + Ray + training scripts. Requires a GPU environment and the training dependencies. Connects to workers via `WORKER_URLS`.
- **Remote workers:**: run the pool server and execute tasks inside Docker. Requires Docker and network reachability from the training machine (default port **18081**).

---

## Directory structure

```
terminal-rl/
├── README.md
├── router_server.py          # Task router service
├── generate.py               # Generation entry
├── rollout_agent.py          # Agent abstraction
├── agent_runner.py           # Orchestrate agent execution
├── env_client.py             # Environment client
├── inference_client.py       # Inference client
├── sample_builders.py        # Build training samples from rollout outcomes
├── request_utils.py
├── custom_types.py
├── rollout_log.py
├── terminal-rl_qwen3-8b.sh              # Qwen3-8B training script
├── terminal-rl_qwen3-8b_prm_2nodes.sh   # Qwen3-8B PRM 2-node training script
├── agent/
│   ├── camel_agent.py        # Rollout agent based on camel-ai ChatAgent
│   └── prm_agent.py          # PRM scoring agent
├── configs/
│   └── rollout_qwen3.yaml    # Rollout config for Qwen3 models
├── data_utils/
│   ├── download.py           # Dataset download
│   ├── convert_task_to_dataset.py  # Convert tasks to training JSONL
│   └── load_tasks.py
├── remote/                   # Remote workers and Docker environment
│   ├── README.md             # Worker deployment guide
│   ├── pool_server.py        # Task pool server (port 18081)
│   ├── terminal_env.py       # Terminal environment wrapper
│   ├── run_pool_server.sh
│   ├── setup.sh
│   ├── compose_override.yaml
│   └── docker_compose_utils.py
└── dataset/                  # Dataset directory
```

## Instructions

### 0. Start remote workers (pool server)

Follow [remote/README.md](remote/README.md) on each worker to start `pool_server` (it should be reachable at e.g. `http://<worker-ip>:18081`).

### 1. Clone the repo

From a directory of your choice:

```bash
git clone https://github.com/Gen-Verse/OpenClaw-RL.git
cd OpenClaw-RL
```

### 2. Prepare dataset (download + convert)

Download a supported dataset under `terminal-rl/dataset/`:

```bash
export DATASET_DIR="terminal-rl/dataset"
python terminal-rl/data_utils/download.py seta_env
```

Convert tasks into training JSONL:

```bash
python terminal-rl/data_utils/convert_task_to_dataset.py \
  --tasks_dir terminal-rl/dataset/seta_env
```

The `seta_env` dataset corresponds to the task dataset published in: [camel-ai/seta-env](https://github.com/camel-ai/seta-env/tree/main/Dataset).

### 3. Run training

On the training machine, set the required environment variables:

```bash
# Hugging Face cache / model paths
export HF_HOME="/path/to/huggingface"
export MODEL_CKPT="/path/to/model"
export REF_LOAD="/path/to/reference_model_dir"
export SAVE_CKPT="/path/to/save/checkpoints"

# Dataset + workers
export ROLLOUT_PROMPT_DATA="/path/to/train.jsonl"
export WORKER_URLS="http://worker1:18081,http://worker2:18081"

# Logging
export WANDB_KEY="your-wandb-key"
```

Then run (from repo root):

```bash
bash terminal-rl/terminal-rl_qwen3-8b.sh
```

---

### PRM training (optional)

To enable PRM scoring with the 2-node script, add:

```bash
export PRM_ENABLE=1
export PRM_MODEL_PATH="/path/to/prm-model"
export PRM_M=3
export PRM_STEP_COEF=1.0
export PRM_TEMPERATURE=0.0
export PRM_MAX_NEW_TOKENS=4096
# Optional: use an external PRM endpoint instead of framework-hosted engines
export PRM_SGLANG_URL="http://<prm-router-ip>:<prm-router-port>"
```

Then run:

```bash
bash terminal-rl/terminal-rl_qwen3-8b_prm_2nodes.sh
```

---

## Notes

- `WORKER_URLS` must point to already-running pool servers.
- The rollout agent implementation in `terminal-rl/agent/camel_agent.py` is based on [camel-ai/camel](https://github.com/camel-ai/camel)'s `ChatAgent`.

---

## Terminal self-evolution experiment

The terminal-only experiment closes the loop from RL training to held-out evaluation and Skill reuse:

```text
training rollouts -> success/failure trajectories -> failure skill groups
-> optional Skill retrieval -> held-out terminal evaluation
```

Prepare a deterministic train / held-out split before training:

```bash
python terminal-rl/data_utils/prepare_terminal_splits.py \
  --tasks-dir terminal-rl/dataset/seta_env \
  --dataset-dir terminal-rl/dataset \
  --output-dir terminal-rl/dataset/seta_evolution_split \
  --holdout-fraction 0.2 \
  --seed terminal-evolution-v1
```

Set the following for an RL run:

```bash
export ROLLOUT_PROMPT_DATA="$PWD/terminal-rl/dataset/seta_evolution_split/train.jsonl"
export EVAL_PROMPT_DATA="$PWD/terminal-rl/dataset/seta_evolution_split/heldout.jsonl"
export TERMINAL_METRICS_PATH="$PWD/terminal-rl/outputs/metrics/rl_only.jsonl"
export TERMINAL_EVAL_RESULTS_PATH="$PWD/terminal-rl/outputs/eval/rl_only.jsonl"
export TERMINAL_TRAJECTORY_LOG="$PWD/terminal-rl/outputs/trajectories/rl_only.jsonl"
export EVOLUTION_SERVER_URL="http://<evolution-server>:8010"
```

`rollout_log.py` writes `terminal/accuracy` against `rollout/step`; use
`terminal_training_report.py` to export the curve. Held-out runs write task-level JSONL, and
`terminal_report.py` reports `Pass@1`, `Pass@k`, resolve rate, average steps, and Skill retrieval rate.

Run the four comparison variants with `scripts/run_terminal_variant.sh`:

```text
base       = base checkpoint, no Skill retrieval
rl_only    = RL checkpoint, no Skill retrieval
skill_only = base checkpoint, retrieved Skills
rl_skill   = RL checkpoint, retrieved Skills
```

Skills are disabled by default. Enable them only for the `skill_only` and `rl_skill` variants:

```bash
export TERMINAL_SKILLS_DIR="$PWD/slime-coding-agent/skills/generated"
export TERMINAL_SKILL_RETRIEVAL=1
```

The held-out evaluation never publishes trajectories to the evolution server, preventing test
instances from becoming training data or Skills.

