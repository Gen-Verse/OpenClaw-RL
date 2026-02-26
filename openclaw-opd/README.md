#  On-Policy Distillation by Extract Hindsight Hints

Online policy distillation for agentic tool-use, using **textual hindsight hints** extracted from environment feedback to construct a stronger teacher signal.

## Method Overview

Unlike binary RL which assigns a scalar reward, OPD constructs a **token-level teacher distribution** by augmenting the original prompt with a hindsight hint, then distills this improved distribution back into the student policy.

For each **main-line turn**, the system:

1. Forwards the request to the policy model (SGLang) and collects the response with per-token log-probabilities.
2. When the **next turn** arrives, the next state (user reply / environment feedback) reveals whether the previous response was helpful.
3. A **judge model** (served on the PRM GPUs) evaluates the (response, next_state) pair `m` times. Each evaluation produces:
   - A binary decision: `\boxed{1}` (the next state reveals useful hindsight) or `\boxed{-1}` (no useful signal).
   - If positive: a **textual hint** wrapped in `[HINT_START]...[HINT_END]` — a concise, actionable description of what the response should have done differently.
4. **Hint selection**: Among all votes scored `+1` with a non-trivial hint (>10 chars), the longest hint is selected. If no valid hint exists, **this sample is dropped entirely** from training.
5. The selected hint is appended to the original prompt as `[user's hint / instruction]\n{hint}`, creating an **enhanced prompt**.
6. **Teacher log-probs** are computed by running the enhanced prompt + original response through the teacher model. This gives $\log\pi_{\text{teacher}}(a_t \mid s_{\text{enhanced}})$ — what the model would have predicted if it had known the hint.
7. The sample is submitted for training with these teacher log-probs as the distillation target.

### Advantage Estimation (On-Policy Distillation)

The advantage at each token is the **log-probability gap** between the hint-enhanced teacher and the current student:

$$A_t = \log\pi_{\text{teacher}}(a_t \mid s + \text{hint}) - \log\pi_{\theta}(a_t \mid s)$$

Intuitively:
- When $A_t > 0$: the teacher (with the hint) assigns higher probability to token $a_t$ than the student — the student should increase this probability.
- When $A_t < 0$: the teacher considers this token less likely given the hint — the student should decrease this probability.

This provides a **token-level, directional** training signal that is richer than a single scalar reward.

### Policy Gradient Loss

The same PPO-style clipped surrogate is used:

$$\rho_t = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{old}}(a_t \mid s_t)}$$

$$\mathcal{L}_{\text{pg}} = -\mathbb{E}_t\Big[\min\big(\rho_t A_t,\ \text{clip}(\rho_t,\, 1-\varepsilon,\, 1+\varepsilon_{\text{high}}) \cdot A_t\big)\Big]$$

But now $A_t$ is the token-level teacher-student gap rather than a broadcast scalar reward.


### Total Loss

$$\mathcal{L} = \mathcal{L}_{\text{pg}} + \beta_{\text{KL}} \cdot \mathcal{L}_{\text{KL}}$$

where $\beta_{\text{KL}} = 0.02$. Entropy bonus is disabled ($\beta_{\text{ent}} = 0$).


## How to Run

```bash
cd slime
bash ../openclaw-opd/run_qwen3_4b_openclaw_opd.sh
```

## File Structure

```
openclaw-opd/
├── README.md
├── run_qwen3_4b_openclaw_opd.sh        # Launch script
├── openclaw_opd_api_server.py           # FastAPI proxy + judge + teacher log-probs + sample submission
├── openclaw_opd_rollout.py              # Async rollout worker (bridges API server ↔ SLIME trainer)
└── results/                             # Runtime records (auto-created)
```
