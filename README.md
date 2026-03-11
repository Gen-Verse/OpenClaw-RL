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
  <a href="./pdf/OpenClaw_RL.pdf"><img src="https://img.shields.io/badge/📄_Tech_Report-red?style=flat-square" alt="Tech Report" /></a>
  <a href="https://yinjjiew.github.io/projects/openclawrl"><img src="https://img.shields.io/badge/Blog-Page-blue?style=flat-square" alt="OpenClaw-RL Blog" /></a>
  <a href="https://openclaw.ai"><img src="https://img.shields.io/badge/OpenClaw-Plugin-orange?style=flat-square" alt="OpenClaw Plugin" /></a>
  <a href="https://github.com/THUDM/slime"><img src="https://img.shields.io/badge/Slime-Based-purple?style=flat-square" alt="Slime Based" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License MIT" /></a>
</p>

<p align="center">
  <video src="https://github.com/user-attachments/assets/a58aacad-3c1d-47aa-bbd1-cf8c5f36de6f" controls width="200"></video>
</p>









## 📰 News

- **[2026/3/11]** 🔥 We have released our [technique report](./pdf/OpenClaw_RL.pdf)!
- **[2026/3/11]** 🔥 Huge updates today! We release a [new combination method](./openclaw-combine), along with an [interesting evaluation](./openclaw-test) of these OpenClaw-RL methods. Track 2 is released too, featuring scalable RL implementations for general agent settings across [terminal](./terminal-rl), [GUI](./gui-rl), [SWE](./swe-rl), and [tool-call](./toolcall-rl) scenarios. We only focus on real-world settings!
- **[2026/3/3]** 🙌 Working with the authors of [SDFT](https://arxiv.org/abs/2601.19897) and [SDPO](https://arxiv.org/abs/2601.20802), we have integrated their methods into [openclaw-opd](./openclaw-opd). We welcome the integration of novel and effective methods!
- **[2026/3/3]** 📺 Check out these community tutorial videos on OpenClaw-RL: [**Video 1**](https://www.youtube.com/watch?v=5xnm1vB7G64) | [**Video 2**](https://www.youtube.com/watch?v=ZtN6Gg_bdJE)
- **[2026/2/26]** 🔥 We release **OpenClaw-RL v1** — a fully asynchronous RL framework for training personalized AI agents from natural conversation feedback. 

---

## 💡 TL;DR

> **OpenClaw-RL** is a fully asynchronous reinforcement learning framework that turns everyday conversations into training signals for personalized AI agents, and supports training general agents with large-scale environment parallelization.

Most RL-for-LLM systems assume centralized, batch-mode training with pre-collected datasets. **OpenClaw-RL** takes a fundamentally different approach: it wraps your self-hosted model in [OpenClaw](https://openclaw.ai) as an OpenAI-compatible API, intercepts live multi-turn conversations, and continuously optimizes the policy in the background — all without interrupting your usage.


<p align="center">
  <img src="assets/framework.png"  alt="Overview"  width="600">
</p>

<details>
<summary><b>🌈 Features</b></summary>

<br>

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

</details>

---



## 🎯 Roadmap

Our long-term goal is to **advance personalized, practically useful agents with reinforcement learning**. The roadmap has two tracks:

#### Track 1 — Personal Agent Optimization (Small-Scale but Personal)
✅ **Release Track 1:** Fully async OpenClaw-RL framework with Binary RL + OPD  
✅ Best recipe discovery via demonstration experiments  
⬜ Support Lora Training & low-precision training/inference  
⬜ Deploy training on [Tinker](https://thinkingmachines.ai/tinker/)  
⬜ Beyond the policy: extend learning to skills and memory  

#### Track 2 — General Agents Optimization (Scalable Infra)
✅ **Release Track 2:** Scalable agentic RL infra for general agents  
⬜ Support more cloud services

---

## 🔧 Quick Start

### 1. RL Server for Personal OpenClaw Agent

### Prerequisites

- **Hardware:** 8× GPUs (default; configurable via `NUM_GPUS`, `ACTOR_GPUS`, `ROLLOUT_GPUS`, `PRM_GPUS`)
- **Software:** CUDA 12.9, Python 3.12
- **Framework:** [Slime](https://github.com/THUDM/slime) (our base RL framework)

For detailed environment setup, see [Slime](https://github.com/THUDM/slime) or [`./instructions/README.md`](./instructions/README.md).








### 2. Start the RL Server

We provide three methods (RL servers):

| Dimension | [Binary RL](./openclaw-rl/) | [OPD](./openclaw-opd) | [Combined](./openclaw-combine) |
|---|---|---|---|
| Signal type | Evaluative (good / bad) | Directional | Evaluative + directional |
| Advantage | Sequence-level scalar | Token-level directional | Mixed sequence and token-level |
| Density | All scored turns | Hint-accepted turns only | All scored turns |
| Feedback type | User / environment | Explicit corrections | Both implicit and explicit feedback |
| Signal richness | 1 scalar per sample | 1 value per token | 1 value per token |



Choose your optimization method:


<details>
<summary><b>Option A: Combination Method</b> — Recommended !</summary>

```bash
cd slime
bash ../openclaw-combine/run_qwen3_4b_openclaw_combine.sh
```

This method combines binary RL and OPD to achieve the best optimization.

See [`./openclaw-combine/README.md`](./openclaw-combine/README.md) for algorithm details.
</details>


<details>
<summary><b>Option B: Binary RL</b> — Best for implicit feedback (likes/dislikes, env success/failure)</summary>

```bash
cd slime
bash ../openclaw-rl/run_qwen3_4b_openclaw_rl.sh
```

The PRM will automatically judge response quality from next-state feedback. We recommend providing frequent feedback (e.g., 👍/👎) to help the model optimize effectively.

See [`./openclaw-rl/README.md`](./openclaw-rl/README.md) for algorithm details.
</details>

<details>
<summary><b>Option C: On-Policy Distillation (OPD)</b> — Best for rich textual feedback</summary>

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

We also provide an interesting case for evaluation. A student who uses OpenClaw to do homework, does not want to be found using AI. A teacher who also uses OpenClaw to grade student's homework, wants the comments to be specific and friendly.

<details>
<summary><b>Eval Setting</b> — Both student and teacher use AI!</summary>

See [`./openclaw-test/README.md`](./openclaw-test/README.md) for setup and algorithm details.
</details>


### 3. OpenClaw Setup

Install OpenClaw from the version bundled in this repository (we will update it regularly):

<details>
<summary><b>Then configure OpenClaw to route requests to your RL server. </b></summary>

<br>

Open your `openclaw.json` (or the equivalent settings file) and add a provider entry under `"models"` → `"providers"`:

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

</details>



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

You can check more details about configurations in [`./instructions`](./instructions) .

## 📚 Documentation

### Understanding the Algorithms

| Topic | Description | Link |
|-------|-------------|------|
| **Teacher Top-K** | Why we use top-K distillation for memory efficiency | [Quick Guide](./docs/teacher_topk_quick.md) · [Deep Dive](./docs/teacher_topk_explained.md) |

### Why Teacher Top-K?

A common question: **"Why does OpenClaw-RL use teacher top-K instead of full vocabulary distillation?"**

**Short answer**: Top-K reduces memory by **2500×** while preserving **95%+** of distillation quality.

For a 128K vocabulary model:
- Full distillation: 1 MB per token → 32 GB for 4K sequence (infeasible!)
- Top-50 distillation: 400 bytes per token → 12.8 MB for 4K sequence (practical!)

**Learn more:**
- [Quick explanation](./docs/teacher_topk_quick.md) (5 min read)
- [Detailed technical guide](./docs/teacher_topk_explained.md) (15 min read)


## 📖 Citation

```
@misc{wang2026openclawrl,
  author       = {Wang, Yinjie and Chen, Xuyang and Jin, Xiaolong and Wang, Mengdi and Yang, Ling},
  title        = {OpenClaw-RL},
  year         = {2026},
  organization = {GitHub},
  url          = {https://github.com/Gen-Verse/OpenClaw-RL},
}

@article{wang2026rlanything,
  title={RLAnything: Forge Environment, Policy, and Reward Model in Completely Dynamic RL System},
  author={Wang, Yinjie and Xie, Tianbao and Shen, Ke and Wang, Mengdi and Yang, Ling},
  journal={arXiv preprint arXiv:2602.02488},
  year={2026}
}

@article{yu2025demystify,
  title={Demystifying Reinforcement Learning in Agentic Reasoning},
  author={Yu, Zhaochen and Yang, Ling and Zou, Jiaru and Yan, Shuicheng and Wang, Mengdi},
  journal={arXiv preprint arXiv:2510.11701},
  year={2025}
}
```

## 🙏 Acknowledgements

This work aims to explore more effective paradigms for Agentic RL. Our implementation builds upon the excellent codebases of [slime](https://github.com/THUDM/slime), [OpenClaw](https://github.com/openclaw/openclaw) and [Open-AgentRL](https://github.com/Gen-Verse/Open-AgentRL). We sincerely thank these projects for their valuable insights and high-quality implementations, which have greatly facilitated our research.



---




