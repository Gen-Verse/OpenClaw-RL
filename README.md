<div align="center">
  <h1 align="center">
    <img src="assets/spacer.png" alt="" width="23" height="40" align="absmiddle" />
    OpenClaw-RL<!--
--><sup>
    <img src="assets/clawistool.png" alt="Claw-RL logo" width="23" height="40" align="absmiddle" />
    <sup>
  </h1>

  <p><b>Empowering OpenClaw with RL — Train a personalized agent simply by talking to it.</b></p>
  <p><b>Scalable RL in real-world settings — Agentic RL for terminal, GUI, SWE, and tool-call settings.</b></p>
</div>


<p align="center">
  <img src="https://img.shields.io/badge/⚡_Fully_Async-yellow?style=for-the-badge" alt="Fully Async" />
  <img src="https://img.shields.io/badge/💰_Zero_API_or_Zero_GPU-blue?style=for-the-badge" alt="Zero API or Zero GPU" />
  <img src="https://img.shields.io/badge/🤖_Personalized-success?style=for-the-badge" alt="Personalized" />
  <img src="https://img.shields.io/badge/🛠️_Auto_Optimization-orange?style=for-the-badge" alt="Auto" />
  <img src="https://img.shields.io/badge/💬_Language_Feedback-purple?style=for-the-badge" alt="Language Feedback" />
  <img src="https://img.shields.io/badge/🧠_Hybrid_RL-red?style=for-the-badge" alt="Hybrid RL" />
  <img src="https://img.shields.io/badge/🌍_Real_World_Agentic_RL-green?style=for-the-badge" alt="General Agentic RL" />
  <br><br>
  <a href="https://arxiv.org/abs/2603.10165"><img src="https://img.shields.io/badge/📄_Tech_Report-red?style=flat-square" alt="Tech Report" /></a>
  <a href="https://yinjjiew.github.io/projects/openclawrl1"><img src="https://img.shields.io/badge/Blog-Page-blue?style=flat-square" alt="OpenClaw-RL Blog" /></a>
  <a href="https://openclaw.ai"><img src="https://img.shields.io/badge/OpenClaw-Plugin-orange?style=flat-square" alt="OpenClaw Plugin" /></a>
  <a href="https://github.com/THUDM/slime"><img src="https://img.shields.io/badge/Slime-Supported-purple?style=flat-square" alt="Slime Based" /></a>
  <a href="https://thinkingmachines.ai/tinker/"><img src="https://img.shields.io/badge/Tinker-Supported-yellow?style=flat-square" alt="Tinker Supported" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License MIT" /></a>
</p>

<p align="center">
  <video src="https://github.com/user-attachments/assets/a58aacad-3c1d-47aa-bbd1-cf8c5f36de6f" controls width="200"></video>
</p>









## 📰 News

- **[2026/3/13]** 🚀 OpenClaw-RL now supports both local GPU and cloud ([Tinker](https://thinkingmachines.ai/tinker/)) deployment. Launch with [**one line of code**](#combinemethod) — Hybrid RL, OPD, and Binary RL all supported!
- **[2026/3/12]** 🔥 We support LoRA training now!
- **[2026/3/10]** 🔥 We have released our [**Technical Report**](https://arxiv.org/abs/2603.10165)! 🏆 Ranked **#1** on [HuggingFace Daily Papers](https://huggingface.co/papers/2603.10165)!
- **[2026/3/10]** 🔥 Huge updates today! We released a [new combination method](./openclaw-combine), along with an [interesting evaluation](./openclaw-test) of these OpenClaw-RL methods. Track 2 is released too, featuring scalable RL implementations for general agent settings across [terminal](./terminal-rl), [GUI](./gui-rl), [SWE](./swe-rl), and [tool-call](./toolcall-rl) scenarios. We only focus on real-world settings!
- **[2026/3/3]** 🙌 Working with the authors of [SDFT](https://arxiv.org/abs/2601.19897) and [SDPO](https://arxiv.org/abs/2601.20802), we have integrated their methods into [openclaw-opd](./openclaw-opd). We welcome the integration of novel and effective methods!
- **[2026/3/3]** 📺 Check out these community tutorial videos on OpenClaw-RL: [Video 1](https://www.youtube.com/watch?v=5xnm1vB7G64) | [Video 2](https://www.youtube.com/watch?v=ZtN6Gg_bdJE)
- **[2026/2/26]** 🔥 We release **OpenClaw-RL v1** — a fully asynchronous RL framework for training personalized AI agents from natural conversation feedback. 

---

## 💡 TL;DR

> **OpenClaw-RL** is a fully asynchronous reinforcement learning framework that turns everyday conversations into training signals for personalized AI agents, and supports training general agents with large-scale environment parallelization.

Most RL-for-LLM systems assume centralized, batch-mode training with pre-collected datasets. **OpenClaw-RL** takes a fundamentally different approach: it wraps your self-hosted model in [OpenClaw](https://openclaw.ai) as an OpenAI-compatible API, intercepts live multi-turn conversations, and continuously optimizes the policy in the background — all without interrupting your usage.


<p align="center">
  <img src="assets/framework.pdf"  alt="Overview"  width="600">
</p>



> **Highlights:** Fully async 4-component loop · Self-hosted & private · Zero manual labeling · Three learning paradigms (Binary RL / OPD / Combine) · Personal + General agent support

<details>
<summary><b>🌈 Features</b></summary>

### Fully Asynchronous 4-Component Architecture
OpenClaw-RL decouples **agent serving**, **rollout collection**, **PRM/judge evaluation**, and **policy training** into independent async loops. None of them block one another: the model continues serving requests while training runs in the background, and judging happens concurrently with new interactions.

### Self-Hosted & Private by Design
The entire stack, including the **policy model**, **judge/PRM**, and **trainer**, runs on **your own infrastructure**. Conversation data stays within your system, and no third-party model API is required.

### From Feedback to Gradient — Automatically
You do not need to manually label data. The system automatically:
- Organizes multi-turn interactions into session-aware training trajectories
- Classifies API messages into **main-line** (trainable) vs. **side** (non-trainable) turns
- Uses the next user, environment, or tool feedback as a natural "next-state" signal
- Runs PRM/judge evaluation asynchronously, with majority voting when needed for more robust scoring
- Submits ready samples to the trainer as they become available

### Three Optimization Methods in One Framework

**Binary RL (GRPO):** A Process Reward Model scores each turn based on next-state feedback. The scalar reward is then used with GRPO advantage estimation and a PPO-style clipped surrogate loss.

**On-Policy Distillation (OPD):** When the next state reveals useful hindsight, a judge model extracts a textual hint. This hint augments the original prompt to create an enhanced teacher, whose token-level log-probability gap with the student becomes a directional advantage signal richer than any scalar reward.

**Combination Method:** OpenClaw-RL further combines Binary RL and OPD in a unified training recipe, leveraging the dense scalar supervision of Binary RL together with the richer token-level directional signal from OPD. This combination achieves stronger and more robust optimization than either method alone.

### From Personal Agents to Real-World Agentic RL
The same framework supports both personalized OpenClaw optimization and scalable RL for **terminal**, **GUI**, **SWE**, and **tool-call** agents in real-world settings.



</details>

---



## 🎯 Roadmap

Our long-term goal is to **advance personalized, practically useful agents with reinforcement learning**. The roadmap has two tracks:

#### Track 1 — [Personal Agent Optimization](#personalagent) (Small-Scale but Personal)
✅ **Release Track 1:** Fully async OpenClaw-RL framework with Binary RL + OPD  
✅ Best recipe discovery via demonstration experiments  
✅ Support LoRA Training  
✅ Deploy training on [Tinker](https://thinkingmachines.ai/tinker/)  
⬜ Support low-precision training/inference  
⬜ Beyond the policy: extend learning to skills and memory  

#### Track 2 — [General Agents Optimization](#generalagent) (Scalable Infra)
✅ **Release Track 2:** Scalable agentic RL infra for general agents  
⬜ Support more cloud services  



## 🤝 Contributing

We welcome contributions that integrate new learning methods into the OpenClaw-RL framework! The integration of [SDFT](https://arxiv.org/abs/2601.19897) / [SDPO](https://arxiv.org/abs/2601.20802) into [openclaw-opd](./openclaw-opd), and [supporting LoRA](https://github.com/Gen-Verse/OpenClaw-RL/pull/23) are great examples of successful community contributions.

**Highly wanted contributions:**
- 🤖 **Qwen3.5 model support with slime** — launch scripts and model configs for the Qwen3.5 family
- 🔧 **Low-precision training examples** — FP8/INT4 training scripts for existing methods

<details>
<summary><b>📋 Full contribution guidelines & feature wishlist</b></summary>


# Call for Contributions

We welcome community contributions to OpenClaw-RL! This document outlines our contribution principles and the features we'd love help with.

## Contribution Guidelines

OpenClaw-RL is organized as a collection of **self-contained method folders** (e.g., `openclaw-rl/`, `openclaw-opd/`, `openclaw-combine/`), each sitting alongside the shared `slime/` training framework and `openclaw/` runtime.

Contributions generally fall into two categories:

### Adding a new method or deployment target

Create a new top-level folder (parallel to existing ones like `openclaw-opd/`). All method-specific code — launch scripts, custom loss functions, rollout logic, API server adapters, data processing, and the README — should live inside this folder.

### Extending an existing method

For changes within an existing method folder — such as supporting a new model family, adding a LoRA variant, or a low-precision example — **add new files** (e.g., a new `.sh` script, a new data processing script) rather than modifying existing ones. This way the original working examples stay intact and your addition can be reviewed independently.

### General principles

1. **Do not modify the core framework.** Avoid changes to `slime/`, `Megatron-LM/`, or `openclaw/` unless absolutely necessary. The framework exposes extension points (`--custom-loss-function-path`, `--rollout-function-path`, `--custom-generate-function-path`, `--custom-rm-path`, etc.) specifically so that new methods can plug in without touching shared code. If a framework change is truly needed, please open a separate PR for it with a clear justification.

2. **Include documentation.** For a new method folder, add a `README.md` explaining what the method does, how to run it, key environment variables, and file structure. For additions to existing folders, update the existing `README.md` with a new section. See [`openclaw-combine/README.md`](./openclaw-combine/README.md) or [`toolcall-rl/README.md`](./toolcall-rl/README.md) for good examples.

3. **Follow existing conventions.** Use the same shell script structure (GPU partitioning, `CKPT_ARGS`, `ROLLOUT_ARGS`, `OPTIMIZER_ARGS`, etc.), environment variable naming, and `ray job submit` launch pattern used by the existing methods.





## Highly Preferred Features


### 1. 🤖 Qwen3.5 Model Support of slime

**Type:** Extend existing method folders

**Goal:** Add launch scripts and model configurations for the Qwen3.5 family across existing methods.

**Requirements:**

- Add new `.sh` scripts for Qwen3.5 in relevant method folders (e.g., `openclaw-combine/run_qwen35_4b_openclaw_combine.sh`).
- Add the corresponding model config in `slime/scripts/models/` if Qwen3.5 requires different architecture parameters (hidden size, num layers, etc.) from Qwen3.
- Verify and document any changes needed for tokenizer, chat template, reasoning parser, or tool-call parser compatibility.
- Update READMEs to list Qwen3.5 as a supported model.


### 2. 🔧 Low-Precision Training/Inference Examples

**Type:** Extend existing method folders

**Goal:** Add low-precision (e.g., INT8/INT4 inference, BF16/FP8 training) example scripts to existing method folders, enabling users to run OpenClaw-RL on consumer-grade hardware with fewer GPUs.

**Requirements:**

- Add **new** `.sh` scripts within existing method folders — do not modify existing scripts.
- Low-precision inference: demonstrate launching the SGLang rollout engine with quantized weights (e.g., AWQ/GPTQ INT4) to reduce VRAM for the serving side.
- Low-precision training: if supported by the Megatron backend, demonstrate FP8 or mixed-precision configurations that reduce training memory.
- Update the corresponding `README.md` in each method folder with a new section documenting these scripts.

---

If you're interested in any of these, feel free to open an issue to discuss your approach before submitting a PR. We're happy to provide guidance and review!


</details>




## 📝 Contents

- [Personal OpenClaw Optimization](#personalagent)
  - [Combination Method (Binary RL + OPD)](#combinemethod)
  - [Binary RL](#binaryrlmethod)
  - [On-policy Distillation](#opdmethod)
  - [Method Evaluation](#evalmethod)
- [Agentic RL in Real World Settings](#agentrl)
  - [Terminal Agent](#terminal)
  - [GUI Agent](#gui)
  - [SWE Agent](#swe)
  - [Tool-call Agent](#toolcall)

---



<a id="personalagent"></a>
## 🔧 Personal Agent Optimization Quick Start

### 1. Deployment Options

#### Don't have any money?

- **Hardware:** 8× GPUs (default; configurable via `NUM_GPUS`, `ACTOR_GPUS`, `ROLLOUT_GPUS`, `PRM_GPUS`)
- **Software:** CUDA 12.9, Python 3.12
- **Framework:** [Slime](https://github.com/THUDM/slime) (our base RL framework)

For detailed environment setup, see [Slime](https://github.com/THUDM/slime) or [`./instructions/README.md`](./instructions/README.md).



#### Don't have a GPU?

Create a [Tinker API](https://github.com/THUDM/slime). That's all you need. But note that Tinker only supports LoRA, which may not be as effective as full fine-tuning. So we are still testing it.





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

<a id="combinemethod"></a>
<details>
<summary><b>Option A: Combination Method</b> — Recommended !</summary>

```bash
cd slime
bash ../openclaw-combine/run_qwen3_4b_openclaw_combine.sh
```

This method combines binary RL and OPD to achieve the best optimization.

See [`./openclaw-combine/README.md`](./openclaw-combine/README.md) for algorithm details.

**With LoRA** (parameter-efficient, fewer GPUs):
```bash
bash ../openclaw-combine/run_qwen3_4b_openclaw_combine_lora.sh
```

<a id="combinemethodtinker"></a>
**With Tinker** (No GPUs at all)
```bash
cd openclaw-tinker
python run.py --method combine --model-name Qwen/Qwen3-8B --batch-size 16 --prm-m 1 --w-opd 1.0 --w-rl 1.0
```

see [`./openclaw-tinker/README.md`](./openclaw-tinker/README.md) for setup details.

</details>


<a id="binaryrlmethod"></a>
<details>
<summary><b>Option B: Binary RL</b> — Best for implicit feedback (likes/dislikes, env success/failure)</summary>

```bash
cd slime
bash ../openclaw-rl/run_qwen3_4b_openclaw_rl.sh
```

The PRM will automatically judge response quality from next-state feedback. We recommend providing frequent feedback (e.g., 👍/👎) to help the model optimize effectively.

See [`./openclaw-rl/README.md`](./openclaw-rl/README.md) for algorithm details.

**With LoRA** (parameter-efficient, fewer GPUs):
```bash
bash ../openclaw-rl/run_qwen3_4b_openclaw_rl_lora.sh
```

**With Tinker** (No GPUs at all)
```bash
cd openclaw-tinker
python run.py --method rl --model-name Qwen/Qwen3-8B --batch-size 16 --prm-m 3
```

see [`./openclaw-tinker/README.md`](./openclaw-tinker/README.md) for setup details.


</details>


<a id="opdmethod"></a>
<details>
<summary><b>Option C: On-Policy Distillation (OPD)</b> — Best for rich textual feedback</summary>

```bash
cd slime
bash ../openclaw-opd/run_qwen3_4b_openclaw_opd.sh
```

The system extracts hindsight hints from your feedback and distills them into the policy at the token level. We recommend providing concrete feedback (e.g., "you should have checked the file first" or "don't use that library").

See [`./openclaw-opd/README.md`](./openclaw-opd/README.md) for algorithm details.

**With LoRA** (parameter-efficient, fewer GPUs):
```bash
bash ../openclaw-opd/run_qwen3_4b_openclaw_opd_topk_lora.sh
```

**With Tinker** (No GPUs at all)
```bash
cd openclaw-tinker
python run.py --method opd --model-name Qwen/Qwen3-8B --batch-size 16 --prm-m 1
```

see [`./openclaw-tinker/README.md`](./openclaw-tinker/README.md) for setup details.



</details>

Once running, the model is served as an OpenAI-compatible API at:
```
http://<HOST_IP>:30000/v1
```

where `<HOST_IP>` is the **IP address** of the machine running the RL server (e.g. `115.190.98.251`). The port `30000` is the default and can be changed via the `PORT` environment variable.

**Take note of this endpoint** — you will need it when configuring OpenClaw in the next step.

We also provide an interesting case for evaluation. A student who uses OpenClaw to do homework, does not want to be found using AI. A teacher who also uses OpenClaw to grade student's homework, wants the comments to be specific and friendly.

<a id="evalmethod"></a>
<details>
<summary><b>Evaluation Setting</b> — Both student and teacher use AI!</summary>

We find that, under the combined optimization method, OpenClaw needs only 36 problem-solving interactions in the student setting and 24 grading interactions in the teacher setting to achieve a significant and clearly visible improvement.

<p align="center">
  <img src="assets/openclawrl1performance.png"  alt="Overview"  width="750">
</p>

See [`./openclaw-test/README.md`](./openclaw-test/README.md) for setup and algorithm details.
</details>


### 3. OpenClaw Setup

Install OpenClaw from the version bundled in this repository (we will update it regularly):

<details>
<summary><b>Then configure OpenClaw to route requests to your RL server. </b></summary>

Open your `openclaw.json` (or the equivalent settings file) and add a provider entry under `"models"` → `"providers"`:

Example of Slime-based RL server:

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

Example of Tinker-based RL server:


```json
{
  "models": {
    "providers": {
      "openclaw-rl": {
        "baseUrl": "http://localhost:30000/v1",
        "apiKey": "no-auth-needed",
        "api": "openai-completions",
        "models": [
          {
            "id": "qwen3-4b-lora",
            "name": "Qwen3 4B (OpenClaw-RL LoRA)",
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



That's it — start chatting with your OpenClaw agent. The RL server will automatically collect conversation trajectories, compute rewards, and train the model. Your agent gets better the more you use it.

</details>




---

<a id="agentrl"></a>
## 🔧 Agentic RL in Real-world Settings

The same asynchronous RL backbone that powers our personal-agent setting can also support large-scale optimization for these broader real-world environments.

| Setting | Environment | Next-state signal | Horizon |
|---|---|---|---|
| Terminal | Shell execution sandbox | stdout/stderr, exit code | Long |
| GUI | Screen state + accessibility tree | Visual state diff, task progress | Long |
| SWE | Code repository + test suite | Test verdicts, diff, lint output | Long |
| Tool-call | API/function execution | Return values, error traces | Medium |

<a id="terminal"></a>
### 🖥️ Terminal Agent — the most widely used computer-use agent

```bash
cd slime
bash ../terminal-rl/terminal_qwen3_8b_rl.sh
```


See [`./terminal-rl/README.md`](./terminal-rl/README.md) for setup details.


<a id="gui"></a>
### 📟 GUI Agent — the most general computer-use agent

```bash
cd slime
bash ../gui-rl/gui_qwen3vl_8b_rl.sh
```


See [`./gui-rl/README.md`](./gui-rl/README.md) for setup details.

<a id="swe"></a>
### 👨‍💻 SWE Agent — software engineering agent

```bash
cd slime
bash ../swe-rl/run_swe_rl_32b_remote_8nodes.sh
```


See [`./swe-rl/README.md`](./swe-rl/README.md) for setup details.

<a id="toolcall"></a>
### 🛠️ Tool-call Agent — the most practical agent

```bash
cd slime
bash ../toolcall-rl/retool_qwen3_4b_rl.sh
```

See [`./toolcall-rl/README.md`](./toolcall-rl/README.md) for setup details.





## 📖 Citation

```
@article{wang2026openclawrl,
  title={OpenClaw-RL: Train Any Agent Simply by Talking},
  author={Wang, Yinjie and Chen, Xuyang and Jin, Xiaolong and Wang, Mengdi and Yang, Ling},
  journal={arXiv preprint arXiv:2603.10165},
  year={2026}
}

@article{wang2026rlanything,
  title={RLAnything: Forge Environment, Policy, and Reward Model in Completely Dynamic RL System},
  author={Wang, Yinjie and Xie, Tianbao and Shen, Ke and Wang, Mengdi and Yang, Ling},
  journal={arXiv preprint arXiv:2602.02488},
  year={2026}
}
```

## 🙏 Acknowledgements

This work aims to explore more effective paradigms for Agentic RL. Our implementation builds upon the excellent codebases of [slime](https://github.com/THUDM/slime), [OpenClaw](https://github.com/openclaw/openclaw), [Tinker](https://thinkingmachines.ai/tinker/) and [Open-AgentRL](https://github.com/Gen-Verse/Open-AgentRL). 

We also build terminal RL using [SETA](https://github.com/camel-ai/seta)'s dataset and agent framework, GUI RL using [OSWorld](https://github.com/xlang-ai/OSWorld)'s evaluation scripts, SWE RL using [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent)'s evaluation scripts, and tool-call RL based on the work of [Retool](https://github.com/ReTool-RL/ReTool).

We sincerely thank these projects for their valuable insights and high-quality implementations, which have greatly facilitated our research.

## ⚠️ Reminder

When using OpenClaw-RL, please do not provide sensitive personal information during conversations with the model. Also, make sure to keep your API keys secure and never expose them in prompts, logs, or shared files.


---




