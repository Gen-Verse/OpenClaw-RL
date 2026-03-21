# Binary Reward Summarized from Next State

Online RL for agentic tool-use, using binary process reward signals from environment feedback.

## Method Overview

The policy model is deployed as an OpenAI-compatible chat proxy. External environments (e.g. OpenClaw) send multi-turn conversations through this proxy. For each **main-line turn**, the system:

1. Forwards the request to the policy model (served by SGLang) and collects the response along with per-token log-probabilities.
2. When the **next turn** arrives, its user/environment message serves as the "next state" for the previous turn.
3. A **Process Reward Model (PRM)** judges the previous response quality given the next state (could be user or env feedback). It produces `m` independent evaluations via majority vote, scoring each turn as `+1` (good), `-1` (bad), or `0` (neutral).
4. The majority-voted score becomes the scalar reward for that turn.
5. Turns that never receive a next state (i.e. the last turn in a session) are excluded from training (`loss_mask = 0`), unless they are the only turn in the session (at-least-one guarantee).
### Learned Reward Model from User Feedback (Thumbs-Down)

In addition to the LLM-based PRM, OpenClaw-RL includes a **Learned Reward Model** that learns directly from user 👎 feedback.

1. **Feedback Collection**: Each mainline turn is initially recorded as a positive sample. When the user explicitly sends a thumbs-down (for example via `/v1/feedback`), that turn is rewritten as a negative sample in the feedback store.
2. **Background Training**: When enabled, the Reward Model is trained continuously in a background thread on collected feedback data. This is intended for offline or idle-time adaptation and uses balanced sampling to handle the rarity of negative feedback.
3. **Opt-In Adaptive Scoring**: By default, the Reward Model only collects feedback and trains offline. It influences turn rewards only if `REWARD_MODEL_SCORE_ENABLE=1` and the feedback store has passed a warm-up threshold.
4. **Protected Feedback Endpoint**: `/v1/feedback` uses the same bearer-token auth path as `/v1/chat/completions`, so external clients cannot inject feedback without the configured API key.

Configurations for the Reward Model (via environment variables):
- `REWARD_MODEL_ENABLE`: `1` to enable feedback collection and background RM training (default in launch script: `0`).
- `REWARD_MODEL_SCORE_ENABLE`: `1` to let the RM influence rewards after warm-up (default: `0`).
- `REWARD_MODEL_MIN_RECORDS`: Minimum total feedback records before RM scoring activates (default: `50`).
- `REWARD_MODEL_MIN_NEGATIVES`: Minimum thumbs-down records before RM scoring activates (default: `5`).
- `REWARD_MODEL_TRAIN_INTERVAL`: Background training frequency in seconds (default: `300`).
- `REWARD_MODEL_LR`: Learning rate for reward model training (default: `1e-5`).
- `OPENCLAW_FEEDBACK_STORE_FILE`: Path to the JSONL file for storing feedback (default: `results/feedback_store.jsonl`).

### Advantage Estimation (GRPO)

Advantages are computed using **Group Relative Policy Optimization (GRPO)**. For each sample with scalar reward `r`, the advantage is broadcast uniformly to all response tokens:

$$A_t = r, \quad \forall t \in \text{response tokens}$$

No reward normalization is applied (`--disable-rewards-normalization`).

### Policy Gradient Loss

Standard PPO-style clipped surrogate objective with asymmetric clipping:

$$\rho_t = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{old}}(a_t \mid s_t)}$$

$$\mathcal{L}_{\text{pg}} = -\mathbb{E}_t\Big[\min\!\big(\rho_t A_t,\ \text{clip}(\rho_t,\, 1-\varepsilon,\, 1+\varepsilon_{\text{high}}) \cdot A_t\big)\Big]$$

where $\varepsilon = 0.2$, $\varepsilon_{\text{high}} = 0.28$.

### Total Loss

$$\mathcal{L} = \mathcal{L}_{\text{pg}} + \beta_{\text{KL}} \cdot \mathcal{L}_{\text{KL}}$$

where $\beta_{\text{KL}} = 0.02$. Entropy bonus is disabled ($\beta_{\text{ent}} = 0$).



## How to Run

```bash
cd slime
bash ../openclaw-rl/run_qwen3_4b_openclaw_rl.sh
```



## File Structure

```
openclaw-rl/
├── README.md
├── feedback_store.py                # Persistent thumbs-down / positive feedback store
├── reward_model.py                  # Learned reward model + background trainer
├── run_qwen3_4b_openclaw_rl.sh     # Launch script
├── openclaw_api_server.py           # FastAPI proxy + PRM scoring + optional RM integration
├── openclaw_rollout.py              # Async rollout worker (bridges API server ↔ SLIME trainer)
└── results/                         # Runtime records (auto-created)
```
