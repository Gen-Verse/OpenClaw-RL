# Tool-Call RL: Teaching LLMs to Solve Math with Code

This directory is based on [slime](https://github.com/THUDM/slime)'s implementation of [Retool](https://github.com/ReTool-RL/ReTool).

This directory trains language models to solve mathematical problems by **calling a Python code interpreter** during reasoning. The model learns when and how to write code, execute it in a sandbox, interpret the results, and produce a final answer — all through reinforcement learning.


## Training Modes

| Mode | Script | Model | Nodes | Description |
|---|---|---|---|---|
| **SFT** | `retool_qwen3_4b_sft.sh` | Qwen3-4B | 1 | Supervised fine-tuning on ReTool-SFT dataset |
| **RL (GRPO)** | `retool_qwen3_4b_rl.sh` | Qwen3-4B | 1 | RL with sequence-level answer-correctness rewards |
| **PRM + RL** | `retool_qwen3_4b_prm_rl.sh` | Qwen3-4B | 1 | RL with per-step PRM scoring (step-wise advantage) |
| **RL (32B)** | `retool_qwen25_32b_4nodes_rl.sh` | Qwen2.5-32B | 4 | Multi-node GRPO for larger model |
| **PRM + RL (32B)** | `retool_qwen25_32b_prm_5nodes_rl.sh` | Qwen2.5-32B | 5 | Multi-node RL with dedicated PRM node |

## Prerequisites

- SLIME framework and Megatron-LM in sibling directories
- Python environment with `requirements.txt` installed
- For SFT → RL pipeline: download the datasets and model checkpoints (see below)

```bash
pip install -r requirements.txt
```

## Step-by-Step Guide

### Option A: SFT then RL (full pipeline)

**1. Download datasets and base model:**

```bash
# SFT data and base model
huggingface-cli download --repo-type dataset JoeYing/ReTool-SFT --local-dir /path/to/ReTool-SFT
huggingface-cli download Qwen/Qwen3-4B-Instruct-2507 --local-dir /path/to/Qwen3-4B-Instruct-2507

# RL data
huggingface-cli download --repo-type dataset BytedTsinghua-SIA/DAPO-Math-17k --local-dir /path/to/dapo-math-17k
huggingface-cli download --repo-type dataset zhuzilin/aime-2024 --local-dir /path/to/aime-2024
```

**2. Convert checkpoint to torch_dist format:**

```bash
cd slime
source scripts/models/qwen3-4B.sh
PYTHONPATH=/path/to/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /path/to/Qwen3-4B-Instruct-2507 \
    --rotary-base 5000000 \
    --save /path/to/Qwen3-4B-Instruct-2507_torch_dist
```

**3. Process SFT data and run SFT:**

```bash
python toolcall-rl/sft_data_processing.py
bash toolcall-rl/retool_qwen3_4b_sft.sh
```

**4. Process RL data and run RL:**

```bash
python toolcall-rl/rl_data_preprocess.py
bash toolcall-rl/retool_qwen3_4b_rl.sh
```

### Option B: Skip SFT, start from pre-trained checkpoint

```bash
# Download our SFT checkpoint directly
huggingface-cli download font-info/qwen3-4b-sft-SGLang-RL --local-dir /path/to/qwen3-4b-sft

# Convert to torch_dist
cd slime
source scripts/models/qwen3-4B.sh
PYTHONPATH=/path/to/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /path/to/qwen3-4b-sft \
    --rotary-base 5000000 \
    --save /path/to/qwen3-4b-sft_torch_dist

# or for qwen3.5:
source scripts/models/qwen3-4B.sh
PYTHONPATH=/path/to/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /path/to/qwen3.5-4b \
    --save /path/to/qwen3.5-4b-torch_dist

# Download RL data and run
huggingface-cli download --repo-type dataset BytedTsinghua-SIA/DAPO-Math-17k --local-dir /path/to/dapo-math-17k
huggingface-cli download --repo-type dataset zhuzilin/aime-2024 --local-dir /path/to/aime-2024
python toolcall-rl/rl_data_preprocess.py

# Set checkpoint paths and run RL
export HF_CKPT=/path/to/qwen3-4b-sft
export REF_LOAD=/path/to/qwen3-4b-sft_torch_dist
cd slime
bash ../toolcall-rl/retool_qwen3_4b_rl.sh
```

### PRM-Augmented RL

Adds a Process Reward Model that scores each tool-calling step, providing per-step rewards instead of only a final-answer signal:

```bash
export HF_CKPT=/path/to/qwen3-4b-sft
export REF_LOAD=/path/to/qwen3-4b-sft_torch_dist
export PRM_MODEL_PATH=/path/to/Qwen3-4B

cd slime
bash ../toolcall-rl/retool_qwen3_4b_prm_rl.sh
```

GPU partition for PRM mode (single node, 8 GPUs): 2 actor + 4 rollout + 2 PRM (configurable via `ACTOR_GPUS`, `ROLLOUT_GPUS`, `PRM_GPUS`).

### Multi-Node (32B)

For training larger models across multiple nodes:

```bash
# 4-node GRPO (32 GPUs total)
bash toolcall-rl/retool_qwen25_32b_4nodes_rl.sh

# 5-node PRM + RL (4 train nodes + 1 dedicated PRM node)
bash toolcall-rl/retool_qwen25_32b_prm_5nodes_rl.sh
```

Multi-node scripts enforce a strict non-colocate layout: train/policy nodes and PRM nodes are fully separated to avoid GPU contention.

## Tool Format

The model uses XML-tagged tool calls:

```
<tool_call>
{"name": "code_interpreter", "arguments": {"code": "print(2 + 2)"}}
</tool_call>
```

Execution results are returned in `<interpreter>` tags. The model continues reasoning with the result until it produces a final answer:

```
Answer: \boxed{42}
```

## Sandbox Safety

The `PythonSandbox` executes code in isolated subprocess with:

- **Time limit**: 120 seconds per execution
- **Memory limit**: 4 GB per process
- **Module allowlist**: only `math`, `random`, `datetime`, `collections`, `itertools`, `functools`, `operator`, `statistics`, `decimal`, `fractions`
- **Dangerous pattern detection**: blocks `os`, `sys`, `subprocess`, `eval`, `exec`, `open`, `__import__`, etc.
- **Concurrency control**: up to 32 concurrent sandbox processes with automatic memory cleanup

## Key Environment Variables

| Variable | Default | Description |
|---|---|---|
| `HF_CKPT` | *(script-specific)* | HuggingFace checkpoint path |
| `REF_LOAD` | *(script-specific)* | Reference model (torch_dist format) for KL regularization |
| `SAVE_CKPT` | *(script-specific)* | Output directory for saved checkpoints |
| `PROMPT_DATA` | *(script-specific)* | Path to training JSONL (DAPO-Math-17k) |
| `WANDB_KEY` | — | Weights & Biases API key |
| `NUM_GPUS` | `8` | Total GPUs per node |
| `ACTOR_GPUS` | `4` | GPUs for the training actor |
| `ROLLOUT_GPUS` | `4` | GPUs for the SGLang rollout engine |
| `PRM_GPUS` | `2` | GPUs for PRM judge (PRM mode only) |
| `PRM_MODEL_PATH` | — | Checkpoint for the PRM judge model |
| `PRM_M` | `1` | Number of independent PRM votes per step |
| `PRM_STEP_COEF` | `1.0` | Weight of PRM step score in final reward |

## File Structure

```text
toolcall-rl/
├── README.md
├── requirements.txt
│
├── retool_qwen3_4b_sft.sh              # SFT on ReTool-SFT dataset
├── retool_qwen3_4b_rl.sh               # RL with GRPO (single node, 4B)
├── retool_qwen3_4b_prm_rl.sh           # PRM + RL with step-wise advantage (single node, 4B)
├── retool_qwen25_32b_4nodes_rl.sh      # Multi-node GRPO (4 nodes, 32B)
├── retool_qwen25_32b_prm_5nodes_rl.sh  # Multi-node PRM + RL (5 nodes, 32B)
│
├── generate_with_retool.py              # Multi-turn generation loop with tool execution
├── tool_sandbox.py                      # Isolated Python sandbox with safety checks
├── sft_data_processing.py              # Convert ReTool-SFT → parquet for SLIME SFT
└── rl_data_preprocess.py               # Convert DAPO-Math-17k → JSONL for SLIME RL
```
