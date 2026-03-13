# On-Policy Distillation (OPD) with Hindsight Hints

Online distillation for agentic tool-use: use next-turn feedback to extract hindsight hints, build a stronger teacher signal, and train the student policy on-policy.

## Core Pipeline

For each main-line turn:

1. Serve response with current policy and keep rollout log-probs.
2. When next state arrives (user reply / env feedback), judge `(response, next_state)` for hindsight usefulness.
3. Run `m` judge votes; each vote returns `+1/-1` and optional hint.
4. Keep the longest non-trivial positive hint; if none exists, drop the sample.
5. Append hint to prompt and query teacher log-probs on the original response tokens.
6. Submit training sample to SLIME.

This turns delayed feedback into token-level supervision without hand-labeled trajectories.

## Option A (Default): Token-Level OPD

Teacher signal per token:

$$A_t=\log\pi_{\text{teacher}}(a_t\mid s+\text{hint})-\log\pi_\theta(a_t\mid s)$$

Training uses PPO-style clipped policy loss with the above token-level advantage, plus KL loss:

$$\mathcal{L}=\mathcal{L}_{pg}+\beta_{KL}\mathcal{L}_{KL}$$

Default script:

```bash
cd slime
bash ../openclaw-opd/run_qwen3_4b_openclaw_opd.sh
```

## Option B: Top-K Logits Distillation (SDFT/SDPO-style)

Following [SDFT](https://arxiv.org/abs/2601.19897) and [SDPO](https://arxiv.org/abs/2601.20802), instead of single-token teacher targets, distill teacher top-K distribution per position. But note that we use teacher top k instead of student top k (setting in their original paper), see issue #7. e will compare teacher top-K and student top-K later.

- Teacher query: `input_top_logprobs` (`K` tokens per position).
- Stored fields: `teacher_topk_log_probs [T,K]`, `teacher_topk_indices [T,K]`.
- Loss: reverse KL over `K+1` bins (top-K + tail mass):

$$D_{KL}\left(\pi_\theta^{K+1}\|\pi_{teacher}^{K+1}\right)=\sum_{k=1}^{K+1}\pi_\theta^{(k)}\left(\log\pi_\theta^{(k)}-\log\pi_{teacher}^{(k)}\right)$$

Tail bin uses:

$$\log p_{tail}=\log\left(1-\exp(\mathrm{logsumexp}(\log p_1,\dots,\log p_K))\right)$$

### Strict Compatibility Design

Top-K is implemented as an additive extension:

- Legacy token-level OPD path is unchanged.
- `teacher_log_probs [T]` keeps original meaning for legacy path.
- Top-K uses separate fields only (`teacher_topk_log_probs`, `teacher_topk_indices`).
- Top-K loss is external custom loss (not a built-in core loss switch).
- Top-K teacher query is off by default (`--distill-topk 0`).

### How to Run Top-K

```bash
cd slime
bash ../openclaw-opd/run_qwen3_4b_openclaw_opd_topk.sh
```

Equivalent key args:

```bash
--loss-type custom_loss \
--custom-loss-function-path topk_distillation_loss.topk_distillation_loss_function \
--distill-topk 50 \
--disable-compute-advantages-and-returns \
--entropy-coef 0.00
```

## File Layout

```text
openclaw-opd/
├── README.md
├── run_qwen3_4b_openclaw_opd.sh            # Token-level OPD (default)
├── run_qwen3_4b_openclaw_opd_topk.sh       # Top-K custom-loss path
├── topk_distillation_loss.py               # Reverse-KL top-K loss (external custom loss)
├── openclaw_opd_api_server.py              # Async judge + teacher query + sample submission
├── openclaw_opd_rollout.py                 # Rollout bridge to SLIME trainer
└── results/                                # Runtime records (auto-created)
```
