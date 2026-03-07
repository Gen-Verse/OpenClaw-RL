# openclaw-training-free

> **Training-Free LLM Improvement** — accumulate conversational wisdom, inject it at inference time. Zero GPU. Zero parameter updates. Zero training cost.

---

## What Is This?

`openclaw-training-free` is a lightweight add-on to [OpenClaw-RL](https://github.com/Gen-Verse/OpenClaw-RL) that improves LLM responses **without touching model weights**.

Existing approaches in this repo update parameters to improve the model:

| Approach | Parameter Update | 
|---|---|
| `openclaw-rl` (GRPO) | ✅ Yes | 
| `openclaw-opd` (OPD) | ✅ Yes |
| **`openclaw-training-free` (experience)** | ❌ **No** | 

The core idea: **every user follow-up is an implicit training signal**. When a user corrects, clarifies, or asks again, they are telling you exactly what the previous response got wrong. We make that signal explicit (via a Judge LLM), store it as an *experience*, and inject it back into future prompts — forming a self-improving **experience loop**.

```
         ┌─────────────────── Experience Loop ────────────────────┐
         │                                                         │
  User   │   ┌──────────────┐    enriched     ┌───────────────┐   │
 request ──▶ │PromptInjector│ ──── prompt ───▶│  Upstream LLM │   │
         │   └──────┬───────┘                 └───────┬───────┘   │
         │          │ retrieve                        │ response  │
         │          │                                 │           │
         │   ┌──────▼───────┐                         │           │
         │   │ ExperienceStore│ ◀──── store hint ──────┤           │
         │   └──────────────┘                         │           │
         │                          ┌─────────────────▼──────┐    │
         │                          │  ExperienceExtractor   │    │
         │   next user message ────▶│  (Judge LLM infers     │    │
         │                          │   what went wrong)     │    │
         │                          └────────────────────────┘    │
         └─────────────────────────────────────────────────────────┘
```

Over time, the experience store becomes a **playbook of lessons learned** — and every future request benefits from it automatically.

---

## How It Works (Step by Step)

**Step 1 — INJECT**
When a request arrives, retrieve the most relevant past experiences from the store and prepend them to the system prompt. The model sees "lessons learned" before generating its response.

**Step 2 — RESPOND**
Forward the enriched prompt to the upstream LLM and return its response to the user — transparently, with no latency overhead on the critical path.

**Step 3 — EXTRACT (async)**
On the *next* user message, a background worker pairs `(previous response, current user message)` and calls a Judge LLM to infer: *"What was lacking in the previous response that caused this follow-up?"* The result is a concise, actionable improvement hint.

**Step 4 — STORE**
Good hints are saved to the experience store (a local JSON file). On the next similar request, Step 1 retrieves and injects them — completing the loop.

---

## Installation

```bash
cd openclaw-training-free
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, no GPU, no torch.

---

## Quick Start

**1. Start the proxy server**

```bash
export UPSTREAM_BASE_URL=https://api.openai.com   # your LLM provider
export UPSTREAM_API_KEY=sk-xxxxxxxx
python training_free_server.py
```

**2. Point your client at the proxy**

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="anything",          # auth handled by the proxy
)
```

**3. Chat normally — improvement happens automatically**

```python
resp = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Explain transformers"}],
)
print(resp.choices[0].message.content)
```

No code changes needed in your application. The proxy accumulates experience transparently. Responses improve over time without any fine-tuning.

---

## Configuration

All settings via environment variables:

| Variable | Required | Default | Description |
|---|---|---|---|
| `UPSTREAM_BASE_URL` | **Yes** | — | Base URL of the real LLM API |
| `UPSTREAM_API_KEY` | **Yes** | — | API key for the upstream LLM |
| `JUDGE_BASE_URL` | No | same as upstream | Base URL for the Judge LLM |
| `JUDGE_API_KEY` | No | same as upstream | API key for the Judge LLM |
| `JUDGE_MODEL` | No | `gpt-4o-mini` | Model used for hint extraction |
| `EXPERIENCE_STORE_PATH` | No | `experiences.json` | Path to the experience JSON file |
| `TOP_K_EXPERIENCES` | No | `3` | Experiences injected per request |
| `PORT` | No | `8080` | Server listen port |

---

## Architecture

```
openclaw-training-free/
├── training_free_server.py    # FastAPI proxy server (entrypoint)
├── experience_extractor.py    # Judge LLM client — extracts hints
├── experience_store.py        # Keyword-indexed JSON experience library
├── prompt_injector.py         # Injects retrieved hints into system prompt
├── requirements.txt
├── __init__.py
└── README.md
```

| Component | Responsibility |
|---|---|
| **Server** (`training_free_server.py`) | OpenAI-compatible proxy. Manages session history, coordinates injection and async extraction. Background `asyncio` worker for non-blocking hint extraction. |
| **Extractor** (`experience_extractor.py`) | Calls Judge LLM to analyze `(response, next_user_message)` pairs. Filters out trivial/empty hints. |
| **Store** (`experience_store.py`) | Thread-safe, keyword-indexed JSON file. Retrieves experiences by keyword overlap with the current query. |
| **Injector** (`prompt_injector.py`) | Retrieves top-K experiences and prepends them to the system message with a character budget. |

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/v1/chat/completions` | POST | OpenAI-compatible (streaming & non-streaming) |
| `/v1/experiences/stats` | GET | Store statistics and queue depth |
| `/health` | GET | Health check |

---

## Comparison with Related Work

| | openclaw-rl | openclaw-opd | **openclaw-training-free** |
|---|---|---|---|
| Parameter updates | ✅ Yes | ✅ Yes | ❌ No |
| GPU required | ✅ Yes | ✅ Yes | ❌ No |
| Improves over time | ✅ | ✅ | ✅ (via experience injection) |
| Needs reward signal | ✅ Explicit | ❌ Auto (OPD) | ❌ Auto (next user msg) |
| Ceiling | Model capacity + training | Model capacity + training | Model capacity only |

**When to use training-free:**
- You can't afford GPU training
- You want to start improving immediately (no data collection phase)
- Your task is open-ended conversation (hard to define a reward function)
- You want a complement to RL — run training-free first, collect high-signal experiences, then use them to bootstrap RL training data

