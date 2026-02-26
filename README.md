<div align="center">
  <h1 align="center">
    <img src="assets/spacer.png" alt="" width="23" height="40" align="absmiddle" />
    OpenClaw-RL<!--
--><sup>
    <img src="assets/clawistool.png" alt="Claw-RL logo" width="23" height="40" align="absmiddle" />
    <sup>
  </h1>

  <p><b>Empowering OpenClaw with RL — Train a personalized agent simply by talking to it.</b></p>
</div>


<p align="center">
  <img src="https://img.shields.io/badge/⚡_Fully_Async-yellow?style=for-the-badge" alt="Fully Async" />
  <img src="https://img.shields.io/badge/💰_Zero_API_Keys-blue?style=for-the-badge" alt="Zero API Keys" />
  <img src="https://img.shields.io/badge/🤖_Personalized-success?style=for-the-badge" alt="Personalized" />
  <img src="https://img.shields.io/badge/🛠️_Auto_Optimization-orange?style=for-the-badge" alt="Auto" />
  <img src="https://img.shields.io/badge/💬_Language_Feedback-purple?style=for-the-badge" alt="Language Feedback" />
  <br><br>
  <a href="https://openclaw.ai"><img src="https://img.shields.io/badge/OpenClaw-Plugin-orange?style=flat-square" alt="OpenClaw Plugin" /></a>
  <a href="https://github.com/THUDM/slime"><img src="https://img.shields.io/badge/Slime-Based-purple?style=flat-square" alt="Slime Based" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License MIT" /></a>
</p>

<p align="center">
  <video src="https://github.com/user-attachments/assets/0db6333d-4ebb-457b-9b03-bc0be9dff17e" controls width="200"></video>
</p>





## 📰 News

- **[2026/xx]** 🔥 We release **OpenClaw-RL v1** — a fully asynchronous RL framework for training personalized AI agents from natural conversation feedback. 

---

## 💡 TL;DR

> **OpenClaw-RL** is a fully asynchronous reinforcement learning framework that turns everyday conversations into training signals for personalized AI agents.

Most RL-for-LLM systems assume centralized, batch-mode training with pre-collected datasets. **OpenClaw-RL** takes a fundamentally different approach: it wraps your self-hosted model in [OpenClaw](https://openclaw.ai) as an OpenAI-compatible API, intercepts live multi-turn conversations, and continuously optimizes the policy in the background — all without interrupting your usage.


<p align="center">
  <img src="assets/rlserver.png"  alt="Overview"  width="600">
</p>

## 🌈 Features

### Fully Asynchronous 4-Component Architecture
OpenClaw-RL decouples **agent serving**, **rollout collection**, **PRM judging**, and **policy training** into independent async loops. None of them block one another — the model serves requests while training runs in the background, and PRM evaluation happens concurrently with new conversations.

### Self-Hosted & Private by Design
The entire stack (model, PRM, training) runs on **your own infrastructure**. Conversation data never leaves your system. No external API keys required.

### From Conversation to Gradient — Automatically
You don't need to manually label data. The system automatically:
- Classifies API messages into **main-line** (trainable) vs. **side** (non-trainable) turns
- Uses the next user/environment message as a natural "next state" signal
- Runs PRM evaluation asynchronously with majority voting for robust scoring
- Submits ready samples to the trainer as they become available

### Two Learning Paradigms in One Framework

**Binary RL (GRPO):** A Process Reward Model scores each turn as good/bad/neutral based on the next-state feedback. The scalar reward is used with GRPO advantage estimation and PPO-style clipped surrogate loss.

**On-Policy Distillation (OPD):** When the next state reveals useful hindsight, a judge model extracts a textual hint. This hint augments the original prompt to create an "enhanced teacher," whose token-level log-probability gap with the student becomes a directional advantage signal — richer than any scalar reward.

### Production-Ready Engineering
- **Session-aware training:** Multi-turn conversations are tracked per-session with proper turn ordering
- **Graceful weight updates:** Submission pauses during model updates, then resumes — no data corruption
- **At-least-one guarantee (Binary RL):** Every session contributes at least one effective training sample
- **Hint quality filtering (OPD):** Only the longest, most informative hint among `m` votes is selected; trivial hints are discarded
- **Teacher log-prob optimization (OPD):** Only response-suffix log-probs are computed to reduce peak memory
- **Record & debug:** All conversations and PRM evaluations are logged to JSONL for analysis

---



## 🎯 Roadmap

Our long-term goal is to **advance personalized, practically useful agents with reinforcement learning**. The roadmap has two tracks:

#### Track 1 — Personal Agent Optimization (Small-Scale but Personal)
✅ **Release v1:** Fully async OpenClaw-RL framework with Binary RL + OPD  
⬜ Broader model family support & more efficient serving  
⬜ Best recipe discovery via large-scale experiments  
⬜ Beyond the policy: extend learning to skills and memory  

#### Track 2 — General Agents Optimization (Scalable Infra)
⬜ **Next (2–3 weeks):** Scalable agentic RL infra for general agents (computer-use first)

---

## 🔧 Quick Start

### 1. RL Server Environment

### Prerequisites

- **Hardware:** 8× GPUs (default; configurable via `NUM_GPUS`, `ACTOR_GPUS`, `ROLLOUT_GPUS`, `PRM_GPUS`)
- **Software:** CUDA 12.9, Python 3.12
- **Framework:** [Slime](https://github.com/THUDM/slime) (our base RL framework)

For detailed environment setup, see [Slime](https://github.com/THUDM/slime) or [`./instructions/README.md`](./instructions/README.md).








### 2. Start the RL Server

We provide two methods (RL servers):

| Method | Signal Type | How It Works | When to Use |
|---|---|---|---|
| **[Binary RL](./openclaw-rl/)** | Scalar (+1/−1/0) | PRM judges response quality from next-state feedback via majority vote → GRPO | Abundant implicit feedback (likes, env success/failure) |
| **[On-Policy Distillation (OPD)](./openclaw-opd/)** | Token-level directional | Extract hindsight hints from next-state → construct enhanced teacher → token-level distillation | Rich textual feedback; need directional improvement |


Choose your optimization method:

<details>
<summary><b>Option A: Binary RL</b> — Best for implicit feedback (likes/dislikes, env success/failure)</summary>

```bash
cd slime
bash ../openclaw-rl/run_qwen3_4b_openclaw_rl.sh
```

The PRM will automatically judge response quality from next-state feedback. We recommend providing frequent feedback (e.g., 👍/👎) to help the model optimize effectively.

See [`./openclaw-rl/README.md`](./openclaw-rl/README.md) for algorithm details.
</details>

<details>
<summary><b>Option B: On-Policy Distillation (OPD)</b> — Best for rich textual feedback</summary>

```bash
cd slime
bash ../openclaw-opd/run_qwen3_4b_openclaw_opd.sh
```

The system extracts hindsight hints from your feedback and distills them into the policy at the token level. We recommend providing concrete feedback (e.g., "you should have checked the file first" or "don't use that library").

See [`./openclaw-opd/README.md`](./openclaw-opd/README.md) for algorithm details.
</details>

Once running, the model is served as an OpenAI-compatible API at:
```
http://<HOST_IP>:30000/v1
```

where `<HOST_IP>` is the **IP address** of the machine running the RL server (e.g. `115.190.98.251`). The port `30000` is the default and can be changed via the `PORT` environment variable.

**Take note of this endpoint** — you will need it when configuring OpenClaw in the next step.



### 3. OpenClaw Setup

Install OpenClaw from the version bundled in this repository (we will update it regularly):

Then configure OpenClaw to route requests to your RL server. Open your `openclaw.json` (or the equivalent settings file) and add a provider entry under `"models"` → `"providers"`:

```json
{
  "models": {
    "providers": {
      "qwen": {
        "baseUrl": "http://<HOST_IP>:30000/v1",
        "apiKey": "apiKey",
        "api": "openai-completions",
        "models": [
          {
            "id": "qwen3-4b",
            "name": "Qwen3 4B",
            "reasoning": true,
            "input": ["text"],
            "cost": {
              "input": 0,
              "output": 0,
              "cacheRead": 0,
              "cacheWrite": 0
            },
            "contextWindow": 32768,
            "maxTokens": 8192
          }
        ]
      }
    }
  }
}
```

Replace `<HOST_IP>` with the IP address of your RL server machine. The `apiKey` should match the `SGLANG_API_KEY` you set when starting the server.

That's it — start chatting with your OpenClaw agent. The RL server will automatically collect conversation trajectories, compute rewards, and train the model. Your agent gets better the more you use it.


#### Configurations

Before launching, set these important environment variables as needed:

| Variable | Default | Description |
|---|---|---|
| `NUM_GPUS` | `8` | Total GPUs available on the machine |
| `ACTOR_GPUS` | `4` | GPUs allocated to the training actor |
| `ROLLOUT_GPUS` | `2` | GPUs allocated to rollout generation |
| `PRM_GPUS` | `2` | GPUs allocated to the Process Reward Model |
| `HF_CKPT` | (see script) | Path to the base HuggingFace checkpoint |
| `PRM_MODEL_PATH` | (see script) | Path to the reward model HuggingFace checkpoint |
| `SAVE_CKPT` | (see script) | Path to the saved HuggingFace checkpoint |
| `SGLANG_API_KEY` | — | API key for the SGLang serving endpoint |

You can check more details about configurations in `./instructions`.



```
@misc{openclawrl,
  author       = {Wang, Yinjie and Wang, Mengdi and Yang, Ling},
  title        = {OpenClaw-RL},
  year         = {2026},
  organization = {GitHub},
  url          = {https://github.com/Gen-Verse/OpenClaw-RL},
}
```

## 🙏 Acknowledgements

This work aims to explore more effective paradigms for Agentic RL. Our implementation builds upon the excellent codebases of [slime](https://github.com/THUDM/slime), [OpenClaw](https://github.com/openclaw/openclaw) and [Open-AgentRL](https://github.com/Gen-Verse/Open-AgentRL). We sincerely thank these projects for their valuable insights and high-quality implementations, which have greatly facilitated our research.



---




