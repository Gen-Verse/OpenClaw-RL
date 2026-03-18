# OpenClaw Tinker

Unified training framework for OpenClaw on [Tinker](https://tinker.build) cloud infrastructure. Supports three training methods through a single entry point:

| Method | Flag | Description |
|--------|------|-------------|
| **RL** | `--method rl` | GRPO with PRM (Process Reward Model) scoring and at-least-one guarantee |
| **OPD** | `--method opd` | On-Policy Distillation via hindsight hints + teacher log-probs |
| **Combined** | `--method combine` | Weighted combination of OPD and RL advantages |

## Quick Start

```bash
export TINKER_API_KEY="your-tinker-api-key"

# Combined method
python run.py --method combine --model-name Qwen/Qwen3-8B --prm-m 1 --batch-size 16 --w-opd 1.0 --w-rl 1.0

# RL method 
python run.py --method rl --model-name Qwen/Qwen3-8B --prm-m 3 --batch-size 16

# OPD method
python run.py --method opd --model-name Qwen/Qwen3-8B --prm-m 1 --batch-size 16
```

## Architecture

```
run.py                 CLI entry point (--method {rl, opd, combine})
├── config.py          Unified TinkerConfig dataclass
├── trainer.py         Training loop: rollout → score → build datums → forward_backward → optim_step
│   ├── rollout.py     RolloutWorker: launches API proxy, feeds prompts, collects completions
│   │   └── api_server.py   OpenAI-compatible proxy with method-specific subclasses
│   ├── scorers.py     PRMScorer / OPDScorer / CombinedScorer
│   └── data_formatter.py   TrainingSample → Tinker Datum conversion
```

### Key Components

- **Trainer** (`trainer.py`): Orchestrates the full loop. Creates two Tinker clients — a LoRA training client (policy model) and a base sampling client (teacher/judge). Handles checkpoint saving at configurable intervals and graceful shutdown.

- **RolloutWorker** (`rollout.py`): Spins up a local OpenAI-compatible API server that forwards requests to the Tinker policy model. External environments (OpenClaw tasks) connect to this server. Completed sessions are queued for scoring and training.

- **API Server** (`api_server.py`): Base class `_BaseServer` provides shared infrastructure (Tinker forwarding, auth, streaming, tokenization, record management). Three subclasses handle method-specific logic:
  - `OpenClawRLServer` — PRM scoring with at-least-one guarantee
  - `OpenClawOPDServer` — Hint judge + teacher log-probs, drops turns without next_state
  - `OpenClawCombineServer` — Three-way dispatch (opd+rl / opd-only / rl-only)

- **Scorers** (`scorers.py`): Each scorer evaluates completed sessions and produces `TrainingSample` objects with rewards and optional teacher log-probs.

- **Data Formatter** (`data_formatter.py`): Converts `TrainingSample` batches into Tinker `Datum` objects for training. RL/OPD use scalar GRPO advantages; Combined computes per-token `w_opd * teacher_adv + w_rl * reward`.

## Configuration

All parameters can be set via CLI flags or environment variables:

### Model
| Flag | Env Var | Default | Description |
|------|---------|---------|-------------|
| `--model-name` | `MODEL_NAME` | `Qwen/Qwen3-4B-Instruct-2507` | Policy model (must be Tinker-supported) |
| `--lora-rank` | `LORA_RANK` | `32` | LoRA rank for training |
| `--teacher-model-name` | `TEACHER_MODEL_NAME` | same as policy | Teacher/judge model (base, no LoRA) |

### Training
| Flag | Env Var | Default | Description |
|------|---------|---------|-------------|
| `--learning-rate` | `LEARNING_RATE` | `1e-4` | Optimizer learning rate |
| `--batch-size` | `BATCH_SIZE` | `4` | Samples per training step |
| `--max-steps` | `MAX_STEPS` | `1000` | Total training steps |
| `--loss-fn` | `LOSS_FN` | `ppo` | Tinker loss: `ppo`, `importance_sampling`, `cispo` |
| `--kl-loss-coef` | `KL_LOSS_COEF` | `0.0` | KL penalty coefficient |
| `--save-interval` | `SAVE_INTERVAL` | `20` | Save checkpoint every N steps |
| `--resume-from-ckpt` | `RESUME_FROM_CKPT` | | Resume from checkpoint path |

### Method-Specific
| Flag | Env Var | Default | Method | Description |
|------|---------|---------|--------|-------------|
| `--w-opd` | `OPENCLAW_COMBINE_W_OPD` | `1.0` | combine | OPD advantage weight |
| `--w-rl` | `OPENCLAW_COMBINE_W_RL` | `1.0` | combine | RL advantage weight |
| `--eval-mode` | `EVAL_MODE` | `false` | opd | Enable PRM eval scoring alongside OPD |

### PRM / Hint Judge
| Flag | Env Var | Default | Description |
|------|---------|---------|-------------|
| `--prm-m` | `PRM_M` | `3` | Number of judge samples (majority voting) |
| `--prm-temperature` | `PRM_TEMPERATURE` | `0.6` | Sampling temperature for judge |
| `--prm-max-tokens` | `PRM_MAX_TOKENS` | `4096` | Max tokens for judge response |

### Proxy Server
| Flag | Env Var | Default | Description |
|------|---------|---------|-------------|
| `--proxy-host` | `PROXY_HOST` | `0.0.0.0` | API server bind host |
| `--proxy-port` | `PROXY_PORT` | `30000` | API server bind port |
| `--served-model-name` | `SERVED_MODEL_NAME` | `qwen3-4b` | Model name in OpenAI API responses |
| `--api-key` | `SGLANG_API_KEY` | | API key for proxy authentication |
| `--default-turn-type` | `DEFAULT_TURN_TYPE` | `main` | Default turn type when client omits `X-Turn-Type` header (`main` or `side`) |

## Training Methods

### RL (`--method rl`)

Standard GRPO reinforcement learning with Process Reward Model scoring:

1. Policy model generates responses via the API proxy
2. PRM evaluates each turn by scoring the next state (majority vote over M samples)
3. Rewards: `+1` (correct), `-1` (incorrect), `0` (uncertain)
4. **At-least-one guarantee**: If all turns in a session score ≤ 0, the best turn gets reward = +1
5. GRPO advantages (scalar reward broadcast) → Tinker Datum → training step

### OPD (`--method opd`)

On-Policy Distillation using hindsight hints and teacher knowledge:

1. Policy model generates responses; environment provides next_state observations
2. Hint judge extracts key information from next_state into a concise hint
3. Teacher model scores the response (with hint context) to get token-level log-probs
4. Advantage = reverse KL from teacher: `-kl_coef * (student_lp - teacher_lp)`
5. All samples get reward = 1.0 (no explicit reward signal)
6. Optional `--eval-mode`: also compute PRM eval scores for monitoring

### Combined (`--method combine`)

Weighted combination with three-way sample dispatch:

- **OPD+RL samples** (have both next_state and reward): get both advantage components
- **OPD-only samples** (next_state but no reward): only teacher KL advantage
- **RL-only samples** (reward but no next_state): only scalar reward advantage

Combined advantage per token:
```
combined_adv_i = w_opd * (-kl_coef * (student_lp_i - teacher_lp_i)) + w_rl * reward
```

## Tinker Integration

This project uses the [Tinker](https://tinker.build) cloud platform for:

- **LoRA Training**: `create_lora_training_client_async(base_model=..., rank=...)` for the policy model
- **Sampling**: `create_sampling_client_async(base_model=...)` for the teacher/judge model
- **Training ops**: `forward_backward_async()` + `optim_step_async()` per step
- **Checkpointing**: `save_weights_and_get_sampling_client_async()` to update the policy sampling client
- **Loss functions**: Supports `ppo`, `importance_sampling`, `cispo`
