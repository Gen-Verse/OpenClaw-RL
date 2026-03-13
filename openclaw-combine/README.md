# Combined Binary RL + On-Policy Distillation

*Let us build on each other's strengths and offset each other's weaknesses.*

This method runs Binary RL (GRPO) and On-Policy Distillation (OPD) simultaneously, combining evaluative and directional signals into a single training objective. In our experiments, this achieves significant performance gains over either method alone.

## Why Combine?

| Dimension | Binary RL | OPD | Combined |
|---|---|---|---|
| Signal type | Evaluative (good/bad) | Directional | Evaluative + directional |
| Advantage | Sequence-level scalar | Token-level directional | Mixed sequence and token-level |
| Density | All scored turns | Hint-accepted turns only | All scored turns |
| Feedback type | User / environment | Explicit corrections | Both implicit and explicit feedback |
| Signal richness | 1 scalar per sample | 1 value per token | 1 value per token |

Binary RL accepts every scored turn, requires no hint extraction, and works with any next-state signal — including terse, implicit reactions (a user simply re-asking a question) or structured environment outputs (exit codes, test verdicts). OPD should be enabled additionally when the interaction stream is likely to carry rich directive content: users who give explicit corrections ("don't use that library", "check the file first"), or environments that produce detailed error traces (SWE diffs, compiler diagnostics).

In practice, Binary RL provides broad gradient coverage across all turns, while OPD provides high-resolution, per-token corrections on the subset of turns where directive signals are available.

## Combined Advantage

Both branches share the same PPO clipped surrogate loss — only the advantage computation differs. The combined advantage is:

$$A_t = w_{\text{binary}} \, r_{\text{final}} + w_{\text{opd}} \left( \log \pi_{\text{teacher}}(a_t \mid s_{\text{enhanced}}) - \log \pi_\theta(a_t \mid s_t) \right)$$

where $w_{\text{binary}} = w_{\text{opd}} = 1$ by default. OPD samples carry `reward=0` so their GRPO advantage is zero; RL samples carry `teacher_logp ≈ rollout_logp` so their teacher advantage is approximately zero. Each branch naturally dominates for its own sample type, and the combined advantage is simply their sum.

## Per-Turn Pipeline

For each main-line turn, after the next state arrives:

1. Run `m` hint-judge votes and `m` eval votes concurrently.
2. If the hint is accepted (longest non-trivial positive hint), emit one **OPD** sample with teacher log-probs.
3. If the eval score is `+1` or `−1`, emit one **RL** sample with the scalar reward.
4. A single turn can contribute both sample types.
5. Training batch fires when collected sample count reaches `rollout_batch_size`.

## How to Run

```bash
cd slime
bash ../openclaw-combine/run_qwen3_4b_openclaw_combine.sh
```

### Qwen3.5-4B

Use the Qwen3.5 launch script and pass the local checkpoint path through environment variables:

```bash
HF_CKPT=/path/to/Qwen3.5-4B \
PRM_MODEL_PATH=/path/to/Qwen3.5-4B \
bash ../openclaw-combine/run_qwen3.5_4b_openclaw_combine.sh
```

### Key Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OPENCLAW_COMBINE_W_RL` | `1.0` | Weight $w_{\text{binary}}$ for the GRPO advantage |
| `OPENCLAW_COMBINE_W_OPD` | `1.0` | Weight $w_{\text{opd}}$ for the teacher advantage |
| `PRM_M` | `1` | Number of independent judge/eval votes per turn |

All other variables (`NUM_GPUS`, `ACTOR_GPUS`, `HF_CKPT`, etc.) are shared with the Binary RL and OPD scripts — see the [main README](../README.md) for the full list.

## File Structure

```text
openclaw-combine/
├── README.md
├── run_qwen3_4b_openclaw_combine.sh   # Launch script
├── run_qwen3.5_4b_openclaw_combine.sh # Qwen3.5-4B launch script
├── openclaw_combine_api_server.py     # Async proxy: hint judge + PRM eval + sample submission
├── openclaw_combine_rollout.py        # Rollout bridge to SLIME trainer
├── combine_loss.py                    # Weighted advantage: w_rl * GRPO + w_opd * teacher
└── results/                           # Runtime records (auto-created)
```
