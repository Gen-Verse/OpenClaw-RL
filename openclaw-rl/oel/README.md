# Online Experiential Learning with On-Policy Context Distillation

Experience-augmented on-policy distillation for personalized agents: extract structured experience within each session or accumulate it across sessions, build stronger teacher signals via experience-augmented prompts, and train the student policy on-policy.

## Core Pipeline

For each session:

1. Student serves responses with current policy (bare prompt, no experience).
2. Teacher sees the same prompt **augmented with experience** and provides token-level supervision.
3. After session ends, extract structured experience items from the full conversation.
4. Experience can either accumulate across sessions (capped at 2048 tokens) or be extracted fresh from the current session only. **Default: single-session post-extraction (`SESSION_EXPERIENCE=2`).**
5. Training uses top-K reverse KL divergence between student and experience-augmented teacher.

Over time, the student internalizes experience without needing it in the prompt at inference time.

## Method: OEL/OPCD-style (Experience-Augmented Distillation)

Post-hoc experience extraction and teacher replay (default mode):

1. **Phase 1**: Student generates all turns without teacher overhead.
2. **Phase 2**: After session ends, extract ONE structured experience from the entire conversation.
3. **Phase 3**: Replay teacher evaluation on ALL turns with the complete experience.
4. Submit all samples to training queue.

This ensures every turn (including Turn 1) benefits from full-session experience.

```bash
# Qwen3-1.7B, full-parameter, Megatron backend (paper default)
bash run_qwen3_1.7b_openclaw_oel_online.sh
```

Key args:

```bash
--loss-type custom_loss \
--custom-loss-function-path oel_distillation_loss.oel_distillation_loss_function \
--distill-topk 50 \
--disable-compute-advantages-and-returns \
--entropy-coef 0.00
```

## Prerequisites

| Requirement | Detail |
|-------------|--------|
| **Docker image** | `slimerl/slime:latest` ([DockerHub](https://hub.docker.com/r/slimerl/slime)) — includes SGLang, Megatron-LM, Flash-Attention, Ray |
| **GPUs** | 4x NVIDIA H100 80GB (or equivalent) |
| **Model weights** | [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) — download to `models/Qwen3-1.7B` |
| **OpenAI API key** | For GPT-4.1 simulator and GPT-4o evaluator — set `OPENAI_API_KEY` env var |

```bash
# 1. Pull image & create container
docker run -d --name openclaw-slime --gpus all --network host --shm-size 64g \
  -v $(pwd)/models:/workspace/models \
  -v $(pwd):/workspace/OpenClaw-RL \
  slimerl/slime:latest sleep infinity

# 2. Download model weights (if not already present)
huggingface-cli download Qwen/Qwen3-1.7B --local-dir models/Qwen3-1.7B

# 3. Start training (inside container)
docker exec -it openclaw-slime bash
cd /workspace/OpenClaw-RL/openclaw-rl/oel
bash run_qwen3_1.7b_openclaw_oel_online.sh

# 4. Run evaluation (separate terminal, host or container)
cd openclaw-rl/oel/eval
export OPENAI_API_KEY="sk-..."
python3 gsm8k_personal_agent.py \
    --method oel \
    --training-rounds 40 \
    --eval-every 2 \
    --problem-file ../data/hard_problems_train.json \
    --eval-problem-file ../data/hard_problems_eval.json
```

## Experiment Setting

| Setting | Value |
|---------|-------|
| **Model** | Qwen3-1.7B (full-parameter, Megatron backend) |
| **Dataset** | Hard GSM8K: 36 train + 36 eval problems (baseline accuracy <= 0.25) |
| **Training rounds** | 40 |
| **Rollout batch size** | 16 |
| **Learning rate** | 1e-5 (constant) |
| **Top-K** | 50 (teacher distribution) |
| **Experience max length** | 2048 tokens |
| **Extraction prompt** | Concrete do/don't rules with dedup (see `EXPERIENCE_UPDATE_PROMPT` in code) |
| **Simulator** | GPT-4.1 |
| **Evaluator** | GPT-4o (5-vote majority) |
| **Hardware** | 4x H100 80GB (actor=2, rollout=1, PRM=1) |

## Key Results

OEL/OPCD-style on **Hard GSM8K**, Qwen3-1.7B, 40 rounds:

| Metric | Value |
|--------|-------|
| Baseline | 26.6 |
| Peak Student | **46.0** (+19.4) |
| OOD (Out-of-Distribution) | 56–70 (stable, no forgetting) |

Reproduce: see [Prerequisites](#prerequisites) above.

## File Layout

```text
openclaw-rl/oel/
├── README.md
├── train_async.py                               # Symlink → ../../slime/train_async.py (SLIME training entry, required by Ray job working-dir)
├── openclaw_oel_api_server.py                   # API server: experience extraction + teacher query + sample submission
├── openclaw_oel_rollout.py                      # Rollout bridge to SLIME trainer
├── oel_distillation_loss.py                     # Top-K reverse KL loss (Megatron + FSDP)
├── run_qwen3_1.7b_openclaw_oel_online.sh        # Training script (paper default)
├── data/
│   ├── hard_problems_train.json                 # 36 hard GSM8K problems for training
│   └── hard_problems_eval.json                  # 36 hard GSM8K problems for evaluation
└── eval/
    ├── gsm8k_personal_agent.py                  # Experiment runner (training + evaluation loop)
    ├── personalization_evaluator.py             # GPT-4o-based score evaluator
    ├── openai_api.py                            # OpenAI API client (configure via OPENAI_API_KEY)
    └── select_hard_problems.py                  # Hard problem selection (requires full GSM8K)
```
