"""OpenClaw OEL (Online Experiential Learning) API Server.

Unified framework supporting both online (OPCD-style) and offline (OEL-style)
experiential learning for language model training.

Modes (controlled by OPENCLAW_OEL_MODE env var):
  - "online"       : OPCD mode — continuous training with live experience accumulation
  - "extract"      : OEL Phase 1 — collect trajectories and extract experience (no training)
  - "deploy"       : OEL Phase 2 — collect trajectories, save to disk (no training)
  - "consolidate"  : OEL Phase 3 — load trajectories + experience, produce training samples

Reference: OEL (arXiv 2603.16856), OPCD (arXiv 2602.12275)
"""

import asyncio
import collections
import copy
import json
import logging
import os
import queue
import random
import re
import threading
import time
from itertools import count
from typing import Any

import httpx
import torch
import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from slime.utils.processing_utils import load_tokenizer
from slime.utils.types import Sample

_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_CYAN = "\033[36m"
_MAGENTA = "\033[35m"
_RESET = "\033[0m"
logger = logging.getLogger(__name__)

_BOXED_RE = re.compile(r"\\boxed\{([-+]?\d)\}")
_NON_STANDARD_BODY_KEYS = {"session_id", "session_done", "turn_type"}

# ---------------------------------------------------------------------------
# Experience prompts
# ---------------------------------------------------------------------------

EXPERIENCE_UPDATE_PROMPT_V1 = """\
You are an AI language model that continuously refines its internal experience.

Here is the latest interaction (the user's question and your answer):
{LATEST_EXPERIENCE}

Here is the previous experience:
# Experience
{PREVIOUS_EXPERIENCE}

Your task:
Based on the latest interaction and the previous experience, generate an additional experience for future learning.

Rules:
- The experience you generate MUST be formatted strictly as a markdown list where each item starts with "- EXPERIENCE ITEM:", one per line:
- EXPERIENCE ITEM: ...
- EXPERIENCE ITEM: ...
- EXPERIENCE ITEM: ...
- The experience you generate will be directly appended to the previous experience.
- The change should introduce a general, high-level, widely applicable insight, not a detail from the specific interaction. The updated experience must remain concise, structured, and meaningful.
- Focus on insights about: natural writing style, step-by-step explanation quality, tool usage patterns, and user interaction strategies.
- If the new insight conflicts with any previous experience item, you can describe the conflict and provide a resolution in the new item.

After careful reasoning step by step, output the final result in exactly this format:

Additional Experience:
# Experience
- EXPERIENCE ITEM: ...
- EXPERIENCE ITEM: ...
- EXPERIENCE ITEM: ...
"""

EXPERIENCE_UPDATE_PROMPT_V2 = """\
You are an AI assistant that extracts actionable lessons from user interactions.

Here is the latest interaction between a user and an assistant:
{LATEST_EXPERIENCE}

Here is the previous accumulated experience:
# Experience
{PREVIOUS_EXPERIENCE}

Your task:
Extract 1-3 NEW, specific, actionable lessons from this interaction that are NOT already covered by the previous experience.

Rules:
- Each lesson MUST be a concrete do/don't rule with specific examples. BAD: "Use natural language". GOOD: "Write calculations as '16 minus 7 is 9' instead of '16 - 7 = 9'".
- Pay close attention to what the USER praised, criticized, or requested — the user's feedback is the most important signal.
- Do NOT repeat or rephrase any insight already present in the previous experience. If the interaction only reinforces existing lessons, output exactly: "No new experience."
- Format strictly as:
- EXPERIENCE ITEM: ...
- EXPERIENCE ITEM: ...

Output the final result in exactly this format:

Additional Experience:
# Experience
- EXPERIENCE ITEM: ...
"""

EXPERIENCE_SOLVE_PROMPT_TEMPLATE = """\
Given previous learned experience:
# Experience
{experience}

Apply the relevant experience to respond to the following:
{prompt}"""


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _flatten_message_content(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return " ".join(parts) if parts else ""
    return str(content) if content is not None else ""


def _normalize_messages_for_template(messages: list[dict]) -> list[dict]:
    out = []
    for msg in messages:
        m = dict(msg)
        if m.get("role") == "developer":
            m["role"] = "system"
        raw = m.get("content")
        if not isinstance(raw, str) and raw is not None:
            m["content"] = _flatten_message_content(raw)
        if m.get("tool_calls"):
            m["tool_calls"] = [_normalize_tool_call(tc) for tc in m["tool_calls"]]
        out.append(m)
    return out


def _normalize_tool_call(tc: dict) -> dict:
    tc = dict(tc)
    fn = tc.get("function")
    if isinstance(fn, dict):
        fn = dict(fn)
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                fn["arguments"] = json.loads(args)
            except (json.JSONDecodeError, TypeError):
                fn["arguments"] = {}
        tc["function"] = fn
    return tc


def _extract_logprobs_from_chat_response(choice: dict[str, Any]) -> list[float]:
    logprobs_obj = choice.get("logprobs")
    if not isinstance(logprobs_obj, dict):
        return []
    content = logprobs_obj.get("content")
    if not isinstance(content, list):
        return []
    return [float(item.get("logprob", 0.0)) for item in content if isinstance(item, dict)]


def _build_prm_eval_prompt(response_text: str, next_state_text: str, next_state_role: str = "user") -> list[dict]:
    """Build PRM evaluation prompt for monitoring."""
    system = (
        "You are a process reward model (PRM) evaluating an AI assistant.\n"
        "You will see the assistant's output and the subsequent next state.\n"
        "Your task: decide whether the assistant's output **successfully fulfilled** the user's intent "
        "at that step, using the next state as evidence.\n\n"
        "## Understanding the next state's role\n"
        "- role='user': A reply from the user.\n"
        "- role='tool': The return value of a tool the assistant invoked. "
        "This content was NOT available before the assistant's action — "
        "it exists BECAUSE the assistant called the tool. "
        "A successful, non-error tool output means the assistant's action worked correctly "
        "and should be scored positively.\n\n"
        "## Scoring rules\n"
        "- \\boxed{1} (good): The next state shows the task progressed as expected.\n"
        "- \\boxed{-1} (bad): The next state signals the assistant's output was wrong, "
        "incomplete, or unwanted.\n"
        "- \\boxed{0} (neutral): The next state is ambiguous.\n\n"
        "Think step-by-step, then give your final score inside \\boxed{}."
    )
    user = (
        f"## Assistant output\n{response_text}\n\n"
        f"## Next state [role: {next_state_role}]\n{next_state_text}\n\n"
        "First, classify the next state: is it (a) positive progression, "
        "(b) a correction / redo / change request, or (c) ambiguous? "
        "Then assign \\boxed{1}, \\boxed{-1}, or \\boxed{0}."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _parse_prm_eval_score(text: str) -> int | None:
    matches = _BOXED_RE.findall(text)
    if not matches:
        return None
    val = int(matches[-1])
    if val in (1, -1, 0):
        return val
    return None


def _prm_eval_majority_vote(scores: list[int | None]) -> float:
    valid = [s for s in scores if s is not None]
    if not valid:
        return 0.0
    counter = collections.Counter(valid)
    top = counter.most_common(1)[0]
    if list(counter.values()).count(top[1]) > 1:
        return 0.0
    return float(top[0])


# ---------------------------------------------------------------------------
# Top-level Slime integration functions
# ---------------------------------------------------------------------------

async def reward_func(args, sample_or_samples, **kwargs):
    if isinstance(sample_or_samples, list):
        return [{"score": s.reward.get("score", 0.0) if isinstance(s.reward, dict) else 0.0} for s in sample_or_samples]
    s = sample_or_samples
    return {"score": s.reward.get("score", 0.0) if isinstance(s.reward, dict) else 0.0}


async def generate(args, sample: Sample, sampling_params, evaluation: bool = False) -> Sample:
    tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
    messages = sample.prompt if isinstance(sample.prompt, list) else [{"role": "user", "content": str(sample.prompt)}]
    input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    payload = {
        "input_ids": input_ids,
        "sampling_params": sampling_params,
        "return_logprob": True,
    }
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    async with httpx.AsyncClient(timeout=None) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        output = response.json()
    text = output.get("text", "")
    meta = output.get("meta_info", {})
    pairs = meta.get("output_token_logprobs", [])
    if isinstance(pairs, list) and pairs:
        token_ids = [int(p[1]) for p in pairs if isinstance(p, (list, tuple)) and len(p) >= 2]
        logprobs = [float(p[0]) for p in pairs if isinstance(p, (list, tuple)) and len(p) >= 2]
    else:
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        logprobs = [0.0] * len(token_ids)
    sample.tokens = input_ids + token_ids
    sample.response = text
    sample.response_length = len(token_ids)
    sample.rollout_log_probs = logprobs
    sample.loss_mask = [1] * len(token_ids)
    sample.status = Sample.Status.COMPLETED
    return sample


# ---------------------------------------------------------------------------
# OpenClaw OEL API Server
# ---------------------------------------------------------------------------

class OpenClawOELAPIServer:
    """Unified API server for Online Experiential Learning.

    Supports both online (OPCD-style continuous) and offline (OEL-style
    iterative) experiential learning modes.
    """

    # OEL modes
    MODE_ONLINE = "online"         # OPCD mode: continuous online training
    MODE_EXTRACT = "extract"       # OEL Phase 1: trajectory + experience extraction
    MODE_DEPLOY = "deploy"         # OEL Phase 2: trajectory collection only
    MODE_CONSOLIDATE = "consolidate"  # OEL Phase 3: load data + distillation training

    def __init__(self, args, output_queue: queue.Queue, submission_enabled: threading.Event):
        self.args = args
        self.output_queue = output_queue
        self.submission_enabled = submission_enabled
        self.tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        self.sglang_chat_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/v1/chat/completions"
        self.sglang_health_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/health"
        self.expected_api_key = os.getenv("SGLANG_API_KEY", "")
        self.host = os.getenv("HOST", "0.0.0.0")
        self.port = int(os.getenv("PORT", "30000"))
        self.served_model_name = os.getenv("SERVED_MODEL_NAME", "qwen3-4b")

        self._index_counter = count(0)
        self._group_counter = count(0)
        self._turn_counts: dict[str, int] = {}
        self._pending_turn_data: dict[str, dict[int, dict]] = {}
        self._pending_records: dict[str, dict[str, Any]] = {}

        # PRM engine config (used for experience extraction + eval + teacher logprobs)
        self._prm_enabled = getattr(args, "prm_enable", False)
        self._prm_m = int(os.getenv("PRM_M", getattr(args, "prm_m", 3)))
        self._prm_temperature = float(getattr(args, "prm_temperature", 0.6))
        self._prm_max_tokens = int(getattr(args, "prm_max_new_tokens", 4096))
        self._teacher_lp_max_concurrency = int(os.getenv("OPENCLAW_OPD_TEACHER_LP_MAX_CONCURRENCY", "3"))
        self._teacher_lp_semaphore = asyncio.Semaphore(max(1, self._teacher_lp_max_concurrency))
        self.distill_topk = int(getattr(args, "distill_topk", 0))
        self._use_topk_distillation = self.distill_topk > 0
        # When using student-based top-K selection, request more teacher logprobs
        # so we can accurately look up teacher probs at student's top-K positions.
        self._topk_source = os.getenv("OPENCLAW_OEL_TOPK_SOURCE", "teacher").strip().lower()
        if self._topk_source == "student":
            # Request 20x more logprobs from teacher to cover student's top-K
            self._teacher_topk_request_size = min(self.distill_topk * 20, 2000) if self.distill_topk > 0 else 0
            logger.info("[OpenClaw-OEL] topk_source=student, teacher_request_size=%d", self._teacher_topk_request_size)
        else:
            self._teacher_topk_request_size = self.distill_topk
        prm_ip = getattr(args, "prm_router_ip", None)
        prm_port = getattr(args, "prm_router_port", None)
        self._prm_url = f"http://{prm_ip}:{prm_port}/generate" if prm_ip and prm_port else ""
        self._prm_tokenizer = None
        if self._prm_enabled:
            prm_path = getattr(args, "prm_model_path", None) or args.hf_checkpoint
            self._prm_tokenizer = load_tokenizer(prm_path, trust_remote_code=True)
            logger.info("[OpenClaw-OEL] PRM enabled: url=%s m=%d", self._prm_url, self._prm_m)

        # ----- OEL mode -----
        self._mode = os.getenv("OPENCLAW_OEL_MODE", self.MODE_ONLINE).strip().lower()
        logger.info("[OpenClaw-OEL] mode=%s", self._mode)

        # ----- Experience extraction prompt -----
        # OPENCLAW_OEL_EXTRACTION_PROMPT: "v1" (general), "v2" (specific + dedup, default),
        #   or a file path to a custom prompt template (must contain {LATEST_EXPERIENCE} and {PREVIOUS_EXPERIENCE})
        _ext_prompt_cfg = os.getenv("OPENCLAW_OEL_EXTRACTION_PROMPT", "v2").strip()
        if _ext_prompt_cfg.lower() == "v1":
            self._extraction_prompt_template = EXPERIENCE_UPDATE_PROMPT_V1
            logger.info("[OpenClaw-OEL] extraction prompt: v1 (general)")
        elif _ext_prompt_cfg.lower() == "v2":
            self._extraction_prompt_template = EXPERIENCE_UPDATE_PROMPT_V2
            logger.info("[OpenClaw-OEL] extraction prompt: v2 (specific + dedup)")
        elif os.path.isfile(_ext_prompt_cfg):
            with open(_ext_prompt_cfg, "r", encoding="utf-8") as f:
                self._extraction_prompt_template = f.read()
            logger.info("[OpenClaw-OEL] extraction prompt: loaded from %s", _ext_prompt_cfg)
        else:
            self._extraction_prompt_template = EXPERIENCE_UPDATE_PROMPT_V2
            logger.warning("[OpenClaw-OEL] unknown extraction prompt '%s', falling back to v2", _ext_prompt_cfg)

        # ----- Experience accumulation -----
        self._experience_text: str = "No previous experience."
        self._experience_lock = threading.Lock()
        self._experience_max_tokens = int(os.getenv("OPENCLAW_OEL_EXPERIENCE_MAX_LENGTH",
                                                     os.getenv("OPENCLAW_OPCD_EXPERIENCE_MAX_LENGTH", "2048")))
        self._experience_session_count = 0
        self._experience_item_count = 0

        # No-accumulate mode: each session's extracted experience REPLACES the global
        # experience instead of appending to it. The next session sees only the
        # experience from the most-recently-completed session.
        self._no_accumulate = os.getenv("OPENCLAW_OEL_NO_ACCUMULATE", "0") == "1"

        # Per-session experience mode: experience is extracted per-turn within a session
        # and NOT accumulated across sessions. Each session starts fresh.
        _ses_exp = os.getenv("OPENCLAW_OEL_SESSION_EXPERIENCE", "0").strip().lower()
        self._session_experience_mode = _ses_exp == "1"
        self._session_experience: dict[str, str] = {}  # session_id -> experience text
        self._session_experience_lock = threading.Lock()
        if self._session_experience_mode:
            logger.info("[OpenClaw-OEL] session-experience mode ENABLED: per-session, per-turn extraction, no cross-session accumulation")

        # Replay mode: after a session completes, extract ONE experience from the
        # entire conversation and then replay teacher evaluation on every turn with
        # that experience.  This ensures all turns (including turn 1) benefit from
        # the full-session experience.
        self._replay_mode = _ses_exp == "2"
        if self._replay_mode:
            logger.info("[OpenClaw-OEL] REPLAY mode ENABLED: post-hoc experience extraction + teacher replay")

        # Multi-experience pool (OEL consolidate mode)
        self._multi_experience = os.getenv("OPENCLAW_OEL_MULTI_EXPERIENCE", "0") == "1"
        self._experience_pool: list[str] = []
        self._experience_pool_lock = threading.Lock()

        # Load experience from file or experience_list.txt
        self._load_initial_experience()

        # Conversation tracking for experience extraction
        self._session_conversations: dict[str, list[dict]] = {}
        self._session_conv_lock = threading.Lock()

        # Experience save directory
        record_file = os.getenv("OPENCLAW_RECORD_FILE", "results/record.jsonl")
        self._experience_save_dir = os.path.join(
            os.path.dirname(record_file) if os.path.dirname(record_file) else "results",
            "experiences",
        )
        os.makedirs(self._experience_save_dir, exist_ok=True)

        # Deploy save directory (OEL deploy/consolidate modes)
        self._deploy_save_dir = os.getenv("OPENCLAW_OEL_DEPLOY_SAVE_DIR", "")
        if self._deploy_save_dir:
            os.makedirs(self._deploy_save_dir, exist_ok=True)

        # Background tasks for teacher evaluation
        self._oel_tasks: dict[str, dict[int, asyncio.Task]] = {}

        # Record and eval
        self._record_file = os.getenv("OPENCLAW_RECORD_FILE", "") if os.getenv("OPENCLAW_RECORD_ENABLED", "0") == "1" else ""
        if self._record_file:
            rec_dir = os.path.dirname(self._record_file)
            if rec_dir:
                os.makedirs(rec_dir, exist_ok=True)
            open(self._record_file, "w").close()
            logger.info("[OpenClaw-OEL] record file initialized: %s", self._record_file)

        self._prm_record_file = os.getenv("OPENCLAW_PRM_RECORD_FILE", "")
        if not self._prm_record_file and self._record_file and self._prm_enabled:
            base, ext = os.path.splitext(self._record_file)
            self._prm_record_file = f"{base}_prm{ext}"
        if self._prm_record_file:
            prm_rec_dir = os.path.dirname(self._prm_record_file)
            if prm_rec_dir:
                os.makedirs(prm_rec_dir, exist_ok=True)
            open(self._prm_record_file, "w").close()

        self._eval_mode = os.getenv("OPENCLAW_EVAL_MODE", "0") == "1"
        self._eval_scores: list[float] = []
        self._eval_scores_lock = threading.Lock()
        if self._eval_mode:
            logger.info("[OpenClaw-OEL] eval mode enabled")

        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self.app = self._build_app()

    # -----------------------------------------------------------------------
    # OEL: Experience initialization
    # -----------------------------------------------------------------------

    def _load_initial_experience(self):
        """Load experience from file(s) on startup.

        Supports:
          - OPENCLAW_OEL_EXPERIENCE_PATH: single experience file or experience_list.txt
          - Multi-experience pool for consolidate mode
        """
        exp_path = os.getenv("OPENCLAW_OEL_EXPERIENCE_PATH", "")
        if not exp_path or not os.path.exists(exp_path):
            return

        # Check if it's an experience_list.txt (containing paths to multiple experience files)
        if exp_path.endswith("experience_list.txt") or exp_path.endswith(".list"):
            self._load_experience_list(exp_path)
        else:
            # Single experience file
            try:
                with open(exp_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                if text:
                    with self._experience_lock:
                        self._experience_text = text
                    logger.info(
                        "[OpenClaw-OEL] loaded experience from %s (%d chars)",
                        exp_path, len(text),
                    )
            except OSError as e:
                logger.warning("[OpenClaw-OEL] failed to load experience from %s: %s", exp_path, e)

    def _load_experience_list(self, list_path: str):
        """Load multiple experience files from an experience_list.txt.

        Each line is a path to an experience file. All are loaded into the
        experience pool for random sampling during consolidate mode.
        """
        try:
            with open(list_path, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
        except OSError as e:
            logger.warning("[OpenClaw-OEL] failed to read experience list %s: %s", list_path, e)
            return

        pool = []
        for path in lines:
            if not os.path.exists(path):
                logger.warning("[OpenClaw-OEL] experience file not found: %s", path)
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                if text:
                    pool.append(text)
            except OSError as e:
                logger.warning("[OpenClaw-OEL] failed to read experience file %s: %s", path, e)

        if pool:
            with self._experience_pool_lock:
                self._experience_pool = pool
            # Use the last experience as default (most recent)
            with self._experience_lock:
                self._experience_text = pool[-1]
            logger.info(
                "[OpenClaw-OEL] loaded %d experience files from %s",
                len(pool), list_path,
            )
        else:
            logger.warning("[OpenClaw-OEL] no valid experience files found in %s", list_path)

    def get_experience_for_turn(self, session_id: str | None = None) -> str:
        """Get experience text for the current turn.

        In session-experience mode, returns per-session experience (no cross-session).
        In multi-experience mode (consolidate), randomly samples from the pool.
        In single-experience mode (online), returns the current accumulated experience.
        """
        if (self._session_experience_mode or self._replay_mode) and session_id:
            with self._session_experience_lock:
                exp = self._session_experience.get(session_id)
                if exp is not None:
                    return exp
            # Fall through to global experience if no per-session entry
            if self._session_experience_mode or self._replay_mode:
                return "No previous experience."
        if self._multi_experience:
            with self._experience_pool_lock:
                if self._experience_pool:
                    return random.choice(self._experience_pool)
        with self._experience_lock:
            return self._experience_text

    # -----------------------------------------------------------------------
    # FastAPI app
    # -----------------------------------------------------------------------

    def _build_app(self) -> FastAPI:
        app = FastAPI(title="OpenClaw OEL Proxy")
        app.state.owner = self

        @app.get("/healthz")
        async def healthz():
            return {"ok": True}

        @app.post("/v1/chat/completions")
        async def chat_completions(
            request: Request,
            authorization: str | None = Header(default=None),
            x_session_id: str | None = Header(default=None),
            x_turn_type: str | None = Header(default=None),
            x_session_done: str | None = Header(default=None),
        ):
            owner: OpenClawOELAPIServer = request.app.state.owner
            await owner._check_auth(authorization)
            if not owner.submission_enabled.is_set():
                raise HTTPException(status_code=503, detail="submission paused for weight update")

            body = await request.json()
            session_id = x_session_id or body.get("session_id") or "unknown"
            turn_type = (x_turn_type or body.get("turn_type") or "side").strip().lower()
            session_done = (
                (x_session_done and x_session_done.strip().lower() in {"1", "true", "yes", "on"})
                or str(body.get("session_done", "")).strip().lower() in {"1", "true", "yes", "on"}
            )

            stream = bool(body.get("stream", False))
            result = await owner._handle_request(
                body, session_id=session_id, turn_type=turn_type, session_done=session_done
            )
            if stream:
                return StreamingResponse(owner._stream_response(result), media_type="text/event-stream")
            return JSONResponse(content=result["response"])

        @app.delete("/v1/sessions/{session_id}")
        async def delete_session(session_id: str, request: Request):
            owner: OpenClawOELAPIServer = request.app.state.owner
            owner._cleanup_session(session_id)
            return {"ok": True}

        return app

    async def _check_auth(self, authorization: str | None):
        if not self.expected_api_key:
            return
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="missing bearer token")
        token = authorization.split(" ", 1)[1].strip()
        if token != self.expected_api_key:
            raise HTTPException(status_code=401, detail="invalid api key")

    # -----------------------------------------------------------------------
    # Request handling
    # -----------------------------------------------------------------------

    async def _handle_request(
        self,
        body: dict[str, Any],
        session_id: str,
        turn_type: str,
        session_done: bool,
    ) -> dict[str, Any]:
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise HTTPException(status_code=400, detail="messages must be a non-empty list")

        tools = body.get("tools")
        forward_body = {k: v for k, v in body.items() if k not in _NON_STANDARD_BODY_KEYS}
        forward_body["stream"] = False
        forward_body.pop("stream_options", None)
        forward_body["logprobs"] = True
        forward_body["top_logprobs"] = 1
        if "model" not in forward_body:
            forward_body["model"] = self.served_model_name

        # Forward to SGLang
        async with httpx.AsyncClient(timeout=None) as client:
            sglang_resp = await client.post(self.sglang_chat_url, json=forward_body)
            if sglang_resp.status_code != 200:
                logger.error("[OpenClaw-OEL] SGLang returned %d: %s", sglang_resp.status_code, sglang_resp.text[:1000])
                sglang_resp.raise_for_status()
            output = sglang_resp.json()

        choice = output.get("choices", [{}])[0]
        assistant_msg = choice.get("message", {})
        tool_calls = assistant_msg.get("tool_calls") or []
        content = assistant_msg.get("content") or ""
        reasoning = assistant_msg.get("reasoning_content") or ""
        logger.info(
            "%s[OpenClaw-OEL] [%s] session=%s prompt_msgs=%d%s",
            _YELLOW, turn_type, session_id, len(messages), _RESET,
        )
        logger.info(
            "%s[OpenClaw-OEL] [%s] session=%s thinking=%d chars, response:\n%s%s",
            _RED, turn_type, session_id, len(reasoning), content[:200], _RESET,
        )

        if turn_type == "main":
            prev_turn_num = self._turn_counts.get(session_id, 0)

            # Flush record from previous turn
            if prev_turn_num > 0 and messages:
                self._flush_pending_record(session_id, messages[-1])

            # Previous turn now has next_state → fire teacher task or store for replay
            if prev_turn_num > 0 and messages:
                prev_turn_data = self._pending_turn_data.get(session_id, {}).get(prev_turn_num)
                if prev_turn_data is not None:
                    if self._replay_mode:
                        # Defer teacher evaluation to session end; just store next_state
                        prev_turn_data["_replay_next_state"] = messages[-1]
                    else:
                        self._fire_teacher_task(session_id, prev_turn_num, prev_turn_data, messages[-1])

            # Process current turn
            response_msg = dict(assistant_msg)
            if response_msg.get("content") is None:
                response_msg["content"] = ""
            norm_msgs = _normalize_messages_for_template(messages)
            norm_resp = _normalize_messages_for_template([response_msg])[0]
            full_norm = norm_msgs + [norm_resp]

            prompt_text = self.tokenizer.apply_chat_template(
                norm_msgs, tools=tools, tokenize=False, add_generation_prompt=True
            )
            full_text = self.tokenizer.apply_chat_template(
                full_norm, tools=tools, tokenize=False, add_generation_prompt=False
            )
            response_text = full_text[len(prompt_text):] if full_text.startswith(prompt_text) else full_text
            prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            response_ids = self.tokenizer(response_text, add_special_tokens=False)["input_ids"]

            if not response_ids and not response_text.strip():
                logger.info("[OpenClaw-OEL] MAIN session=%s -> empty response, skipping", session_id)
                output["session_id"] = session_id
                return {"response": output}

            response_logprobs = _extract_logprobs_from_chat_response(choice)
            if len(response_logprobs) > len(response_ids):
                response_logprobs = response_logprobs[: len(response_ids)]
            elif len(response_logprobs) < len(response_ids):
                response_logprobs = response_logprobs + [0.0] * (len(response_ids) - len(response_logprobs))

            self._turn_counts[session_id] = prev_turn_num + 1
            turn_num = self._turn_counts[session_id]
            turn_data = {
                "prompt_ids": prompt_ids,
                "response_ids": response_ids,
                "response_logprobs": response_logprobs,
                "prompt_text": prompt_text,
                "response_text": response_text,
                "messages": messages,
                "tools": tools,
                "has_next_state": False,
            }
            self._pending_turn_data.setdefault(session_id, {})[turn_num] = turn_data
            self._buffer_record(session_id, turn_num, messages, prompt_text, response_text, tool_calls)

            # Track conversation for experience extraction
            self._record_conversation_turn(session_id, messages, content, tool_calls)

            # Per-session experience: extract after each turn so the next turn's teacher can use it
            if self._session_experience_mode and self._prm_enabled:
                self._safe_create_task(self._extract_session_experience(session_id))

            logger.info(
                "[OpenClaw-OEL] MAIN session=%s turn=%d prompt_tokens=%d response_tokens=%d",
                session_id, turn_num, len(prompt_ids), len(response_ids),
            )

            # In extract/deploy modes, don't submit training samples
            if self._mode not in (self.MODE_EXTRACT, self.MODE_DEPLOY):
                self._maybe_submit_ready_samples(session_id)
        else:
            logger.info("[OpenClaw-OEL] SIDE session=%s -> skipped (no training data)", session_id)

        if session_done:
            self._flush_pending_record(session_id, None)

            if self._replay_mode:
                # Replay mode: extract experience from whole session, then replay teacher on all turns
                self._safe_create_task(self._replay_session(session_id))
            else:
                # Normal mode: finalize last turn and submit
                self._finalize_last_turn(session_id)
                if self._mode not in (self.MODE_EXTRACT, self.MODE_DEPLOY):
                    self._maybe_submit_ready_samples(session_id, force_drop_without_next_state=True)

                # Save session trajectory (for deploy and extract modes)
                if self._deploy_save_dir:
                    self._save_session_trajectory(session_id)

                # Trigger experience extraction (online and extract modes)
                # In session-experience mode, extraction already happens per-turn; skip global accumulation
                if self._mode in (self.MODE_ONLINE, self.MODE_EXTRACT) and not self._session_experience_mode:
                    self._safe_create_task(self._extract_experience_from_session(session_id))

                # Clean up per-session experience
                if self._session_experience_mode:
                    with self._session_experience_lock:
                        self._session_experience.pop(session_id, None)
                    logger.info("[OpenClaw-OEL] session=%s per-session experience cleared", session_id)

            self._turn_counts.pop(session_id, None)
            logger.info("[OpenClaw-OEL] session=%s done -> cleaned up (mode=%s)", session_id, self._mode)

        output["session_id"] = session_id
        return {"response": output}

    # -----------------------------------------------------------------------
    # OEL: Trajectory saving (deploy/extract modes)
    # -----------------------------------------------------------------------

    def _save_session_trajectory(self, session_id: str):
        """Save a session's trajectory data to disk for later consolidation."""
        if not self._deploy_save_dir:
            return

        pending = self._pending_turn_data.get(session_id, {})
        if not pending:
            return

        trajectory = {
            "session_id": session_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "turns": [],
        }
        for turn_num in sorted(pending.keys()):
            td = pending[turn_num]
            trajectory["turns"].append({
                "turn_num": turn_num,
                "prompt_ids": td["prompt_ids"],
                "response_ids": td["response_ids"],
                "response_logprobs": td["response_logprobs"],
                "prompt_text": td["prompt_text"],
                "response_text": td["response_text"],
                "messages": td["messages"],
                "tools": td.get("tools"),
            })

        traj_path = os.path.join(self._deploy_save_dir, f"traj_{session_id}.json")
        try:
            with open(traj_path, "w", encoding="utf-8") as f:
                json.dump(trajectory, f, ensure_ascii=False, indent=2)
            logger.info(
                "[OpenClaw-OEL] saved trajectory session=%s turns=%d -> %s",
                session_id, len(trajectory["turns"]), traj_path,
            )
        except OSError as e:
            logger.warning("[OpenClaw-OEL] failed to save trajectory: %s", e)

    # -----------------------------------------------------------------------
    # Teacher evaluation (experience-augmented distillation)
    # -----------------------------------------------------------------------

    async def _teacher_evaluate(
        self, session_id: str, turn_num: int, turn_data: dict[str, Any], next_state: dict[str, Any]
    ) -> dict[str, Any]:
        """Compute teacher log-probs using experience-augmented prompts.

        In all modes, the teacher sees the same prompt as the student but with
        accumulated experience injected. This creates the KL target.
        """
        next_state_text = _flatten_message_content(next_state.get("content")) if next_state else ""
        next_state_role = next_state.get("role", "user") if next_state else "user"

        # Optional: PRM eval for monitoring
        eval_score = None
        if self._eval_mode and self._prm_enabled:
            eval_msgs = _build_prm_eval_prompt(turn_data["response_text"], next_state_text, next_state_role)
            if self._prm_tokenizer:
                eval_prompt = self._prm_tokenizer.apply_chat_template(
                    eval_msgs, tokenize=False, add_generation_prompt=True
                )
            else:
                eval_prompt = "\n".join(m["content"] for m in eval_msgs)
            async with self._teacher_lp_semaphore:
                eval_raw = await asyncio.gather(
                    *[self._query_prm_eval_once(eval_prompt, i) for i in range(self._prm_m)]
                )
            eval_score = _prm_eval_majority_vote(eval_raw)
            logger.info(
                "%s[OpenClaw-OEL] PRM eval session=%s turn=%d eval_score=%.1f%s",
                _CYAN, session_id, turn_num, eval_score, _RESET,
            )

        # Get experience (may be randomly sampled in multi-experience mode)
        experience = self.get_experience_for_turn(session_id=session_id)

        # Construct experience-augmented prompt
        enhanced_messages = self._inject_experience_to_messages(
            turn_data["messages"], experience
        )
        norm_enhanced = _normalize_messages_for_template(enhanced_messages)
        enhanced_prompt_text = self.tokenizer.apply_chat_template(
            norm_enhanced,
            tools=turn_data.get("tools"),
            tokenize=False,
            add_generation_prompt=True,
        )
        enhanced_full_text = enhanced_prompt_text + turn_data["response_text"]
        enhanced_ids = self.tokenizer(enhanced_full_text, add_special_tokens=False)["input_ids"]
        response_len = len(turn_data["response_ids"])

        # Compute teacher log-probs
        teacher_log_probs = await self._compute_teacher_log_probs(enhanced_ids, response_len)

        result: dict[str, Any] = {
            "accepted": True,
            "teacher_log_probs": teacher_log_probs,
            "hint": f"[OEL experience: {len(experience)} chars]",
            "votes": [],
            "eval_score": eval_score,
        }

        # Top-K teacher log-probs for distillation loss
        if self._use_topk_distillation:
            topk_lp, topk_idx = await self._compute_teacher_topk_logprobs(enhanced_ids, response_len)
            result["teacher_topk_log_probs"] = topk_lp
            result["teacher_topk_indices"] = topk_idx

        logger.info(
            "%s[OpenClaw-OEL] session=%s turn=%d experience_len=%d teacher_lp_len=%d%s",
            _CYAN, session_id, turn_num, len(experience), len(teacher_log_probs), _RESET,
        )
        return result

    def _fire_teacher_task(self, session_id: str, turn_num: int, turn_data: dict[str, Any], next_state: dict[str, Any]):
        """Fire async teacher evaluation for a turn."""
        # In extract/deploy modes, skip teacher evaluation (no training)
        if self._mode in (self.MODE_EXTRACT, self.MODE_DEPLOY):
            return

        if not self._prm_enabled:
            self._safe_create_task(self._submit_turn_sample_no_teacher(turn_data, session_id))
            return
        task = asyncio.create_task(self._teacher_evaluate(session_id, turn_num, turn_data, next_state))
        task.add_done_callback(self._task_done_cb)
        task.add_done_callback(lambda _t: self._maybe_submit_ready_samples(session_id))
        self._oel_tasks.setdefault(session_id, {})[turn_num] = task
        turn_data["has_next_state"] = True

    def _finalize_last_turn(self, session_id: str):
        """For the last turn in a session, fire teacher task even without next_state."""
        # In extract/deploy modes, skip teacher evaluation
        if self._mode in (self.MODE_EXTRACT, self.MODE_DEPLOY):
            return

        if not self._prm_enabled:
            return
        pending = self._pending_turn_data.get(session_id, {})
        for turn_num, td in list(pending.items()):
            if td.get("has_next_state"):
                continue
            task = asyncio.create_task(
                self._teacher_evaluate(session_id, turn_num, td, {"role": "user", "content": "[session ended]"})
            )
            task.add_done_callback(self._task_done_cb)
            task.add_done_callback(lambda _t: self._maybe_submit_ready_samples(session_id))
            self._oel_tasks.setdefault(session_id, {})[turn_num] = task
            td["has_next_state"] = True

    async def _format_and_extract_experience(self, conversation: list[dict]) -> str:
        """Format a conversation and extract experience items via PRM.

        Returns parsed experience text, or ``"No previous experience."`` if
        extraction fails or yields nothing new.
        """
        if not conversation or not self._prm_enabled:
            return "No previous experience."

        interaction_parts = []
        for i, turn in enumerate(conversation):
            interaction_parts.append(f"Turn {i+1}:")
            interaction_parts.append(f"  User: {turn['user'][:500]}")
            interaction_parts.append(f"  Assistant: {turn['assistant'][:500]}")
        latest_interaction = "\n".join(interaction_parts)

        with self._experience_lock:
            current_exp = self._experience_text

        exp_prompt = self._extraction_prompt_template.format(
            PREVIOUS_EXPERIENCE=current_exp if current_exp != "No previous experience." else "No previous experience.",
            LATEST_EXPERIENCE=latest_interaction,
        )

        exp_text = await self._generate_experience_text(exp_prompt)
        if not exp_text or "no new experience" in exp_text.lower():
            return "No previous experience."

        parsed = self._parse_experience_items(exp_text)
        return parsed if parsed else "No previous experience."

    async def _replay_session(self, session_id: str):
        """Replay mode: extract experience from the full session, then run teacher on every turn.

        Flow:
          1. Extract one experience from the entire conversation (synchronously await).
          2. Temporarily inject that experience so get_experience_for_turn returns it.
          3. For every pending turn, call _teacher_evaluate with the full-session experience.
          4. After all teacher tasks complete, submit samples and clean up.
        """
        pending = self._pending_turn_data.get(session_id, {})
        if not pending:
            logger.info("[OpenClaw-OEL] replay session=%s no pending turns", session_id)
            return

        turn_nums = sorted(pending.keys())
        logger.info(
            "%s[OpenClaw-OEL] REPLAY session=%s starting: %d turns%s",
            _MAGENTA, session_id, len(turn_nums), _RESET,
        )

        # Step 1: extract experience from complete conversation
        with self._session_conv_lock:
            conversation = self._session_conversations.get(session_id, [])

        experience = await self._format_and_extract_experience(conversation)
        if experience != "No previous experience.":
            logger.info(
                "%s[OpenClaw-OEL] REPLAY session=%s extracted experience: %d chars%s",
                _MAGENTA, session_id, len(experience), _RESET,
            )

        # Clean up conversation tracking
        with self._session_conv_lock:
            self._session_conversations.pop(session_id, None)

        # Step 2: temporarily store replay experience for this session
        # Override get_experience_for_turn by using _session_experience with replay key
        with self._session_experience_lock:
            self._session_experience[session_id] = experience

        # Step 3: fire teacher evaluation for all turns
        # Reconstruct next_state for each turn from stored data
        tasks = []
        for i, tn in enumerate(turn_nums):
            td = pending[tn]
            # next_state: saved during Phase 1, or "[session ended]" for last turn
            next_state = td.pop("_replay_next_state", None)
            if next_state is None:
                next_state = {"role": "user", "content": "[session ended]"}
            task = asyncio.create_task(self._teacher_evaluate(session_id, tn, td, next_state))
            task.add_done_callback(self._task_done_cb)
            self._oel_tasks.setdefault(session_id, {})[tn] = task
            td["has_next_state"] = True
            tasks.append(task)

        # Wait for all teacher evaluations to complete
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        # Step 4: submit all ready samples
        self._maybe_submit_ready_samples(session_id, force_drop_without_next_state=True)

        # Clean up replay experience
        with self._session_experience_lock:
            self._session_experience.pop(session_id, None)

        logger.info(
            "%s[OpenClaw-OEL] REPLAY session=%s completed: %d turns replayed%s",
            _MAGENTA, session_id, len(turn_nums), _RESET,
        )

    # -----------------------------------------------------------------------
    # Experience injection
    # -----------------------------------------------------------------------

    def _inject_experience_to_messages(self, messages: list[dict], experience: str) -> list[dict]:
        """Inject accumulated experience into the last user message."""
        if not experience or experience == "No previous experience.":
            return messages

        cloned = copy.deepcopy(messages)
        target_idx = None
        for i in range(len(cloned) - 1, -1, -1):
            if cloned[i].get("role") == "user":
                target_idx = i
                break
        if target_idx is None:
            return cloned

        content = _flatten_message_content(cloned[target_idx].get("content"))
        enhanced_content = EXPERIENCE_SOLVE_PROMPT_TEMPLATE.format(
            experience=experience, prompt=content
        )
        cloned[target_idx]["content"] = enhanced_content
        return cloned

    # -----------------------------------------------------------------------
    # Experience extraction from completed sessions
    # -----------------------------------------------------------------------

    def _record_conversation_turn(
        self, session_id: str, messages: list[dict], response_content: str, tool_calls: list
    ):
        """Record conversation turns for experience extraction later."""
        with self._session_conv_lock:
            if session_id not in self._session_conversations:
                self._session_conversations[session_id] = []
            if messages:
                last_user = None
                for msg in reversed(messages):
                    if msg.get("role") == "user":
                        last_user = _flatten_message_content(msg.get("content"))
                        break
                if last_user:
                    self._session_conversations[session_id].append(
                        {"user": last_user, "assistant": response_content or "[tool call]"}
                    )

    async def _extract_session_experience(self, session_id: str):
        """Extract experience from current session conversation and store per-session.

        Unlike _extract_experience_from_session which accumulates globally,
        this stores experience only for the current session. Each turn triggers
        a fresh extraction from all conversation so far in this session.
        The experience is cleared when the session ends.
        """
        with self._session_conv_lock:
            conversation = list(self._session_conversations.get(session_id, []))

        if not conversation:
            return

        # Format conversation as interaction text
        interaction_parts = []
        for i, turn in enumerate(conversation):
            interaction_parts.append(f"Turn {i+1}:")
            interaction_parts.append(f"  User: {turn['user'][:500]}")
            interaction_parts.append(f"  Assistant: {turn['assistant'][:500]}")
        latest_interaction = "\n".join(interaction_parts)

        # Get current per-session experience (for dedup in subsequent turns)
        with self._session_experience_lock:
            current_exp = self._session_experience.get(session_id, "No previous experience.")

        # Build experience extraction prompt
        exp_prompt = self._extraction_prompt_template.format(
            PREVIOUS_EXPERIENCE=current_exp,
            LATEST_EXPERIENCE=latest_interaction,
        )

        # Call PRM engine to generate experience
        exp_text = await self._generate_experience_text(exp_prompt)
        if not exp_text:
            return

        if "no new experience" in exp_text.lower():
            logger.info("[OpenClaw-OEL] session=%s (per-session) no new experience", session_id)
            return

        parsed_items = self._parse_experience_items(exp_text)
        if not parsed_items:
            logger.info("[OpenClaw-OEL] session=%s (per-session) no valid items parsed", session_id)
            return

        # Store per-session (append within session, but never cross-session)
        with self._session_experience_lock:
            prev = self._session_experience.get(session_id, "No previous experience.")
            if prev == "No previous experience.":
                self._session_experience[session_id] = parsed_items
            else:
                self._session_experience[session_id] = prev + "\n" + parsed_items
            self._session_experience[session_id] = self._truncate_experience(
                self._session_experience[session_id], self._experience_max_tokens
            )
            exp_snapshot = self._session_experience[session_id]

        new_items = len(parsed_items.strip().split("\n"))
        logger.info(
            "%s[OpenClaw-OEL] session=%s (per-session) extracted %d items, exp_len=%d chars%s",
            _MAGENTA, session_id, new_items, len(exp_snapshot), _RESET,
        )

    async def _extract_experience_from_session(self, session_id: str):
        """Extract experience items from a completed session and append to global experience."""
        with self._session_conv_lock:
            conversation = self._session_conversations.pop(session_id, [])

        if not conversation:
            logger.info("[OpenClaw-OEL] session=%s no conversation to extract experience from", session_id)
            return

        parsed_items = await self._format_and_extract_experience(conversation)
        if parsed_items == "No previous experience.":
            logger.info("[OpenClaw-OEL] session=%s experience extraction yielded nothing new", session_id)
            return

        # Update global experience
        with self._experience_lock:
            if self._no_accumulate:
                # No-accumulate mode: replace global experience with this session's extract only
                self._experience_text = parsed_items
                logger.info(
                    "[OpenClaw-OEL] session=%s no-accumulate: replaced global experience (%d chars)",
                    session_id, len(parsed_items),
                )
            else:
                # Default: append to global experience
                if self._experience_text == "No previous experience.":
                    self._experience_text = parsed_items
                else:
                    self._experience_text = self._experience_text + "\n" + parsed_items
            self._experience_text = self._truncate_experience(
                self._experience_text, self._experience_max_tokens
            )
            self._experience_session_count += 1
            new_items = len(parsed_items.strip().split("\n"))
            self._experience_item_count += new_items

        # Save experience to file
        exp_path = os.path.join(
            self._experience_save_dir,
            f"experience_session_{self._experience_session_count}.txt",
        )
        with self._experience_lock:
            exp_snapshot = self._experience_text
        try:
            with open(exp_path, "w", encoding="utf-8") as f:
                f.write(exp_snapshot)
        except OSError as e:
            logger.warning("[OpenClaw-OEL] failed to save experience: %s", e)

        logger.info(
            "%s[OpenClaw-OEL] session=%s extracted %d new items, total=%d items, exp_len=%d chars%s",
            _MAGENTA, session_id, new_items, self._experience_item_count,
            len(exp_snapshot), _RESET,
        )

    async def _generate_experience_text(self, prompt: str) -> str:
        """Use PRM engine to generate experience text."""
        if not self._prm_url:
            return ""

        if self._prm_tokenizer:
            msgs = [{"role": "user", "content": prompt}]
            text_prompt = self._prm_tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
        else:
            text_prompt = prompt

        payload = {
            "text": text_prompt,
            "sampling_params": {
                "temperature": 0.6,
                "top_p": 1.0,
                "top_k": -1,
                "max_new_tokens": 1024,
                "skip_special_tokens": False,
                "no_stop_trim": True,
                "spaces_between_special_tokens": False,
            },
            "return_logprob": False,
        }
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
                resp = await client.post(self._prm_url, json=payload)
                resp.raise_for_status()
                data = resp.json()
            raw = data.get("text", data) if isinstance(data, dict) else str(data)
            if isinstance(raw, list):
                raw = raw[0] if raw else ""
            text = str(raw)
            # Strip thinking tags
            text = re.sub(r"<think>[\s\S]*?</think>\s*", "", text).strip()
            return text
        except Exception as e:
            logger.warning("[OpenClaw-OEL] experience generation failed: %s", e)
            return ""

    @staticmethod
    def _parse_experience_items(text: str) -> str:
        """Parse '- EXPERIENCE ITEM: ...' lines from generated text."""
        marker = "- EXPERIENCE ITEM:"
        lines = text.split("\n")
        result_lines = []
        for line in lines:
            if marker in line:
                parts = line.split(marker)[1:]
                result_lines.extend([marker + p.rstrip() for p in parts if p.strip()])
        return "\n".join(result_lines)

    def _truncate_experience(self, experience: str, max_tokens: int) -> str:
        """Truncate experience to max_tokens, keeping the most recent items."""
        tokens = self.tokenizer.encode(experience, add_special_tokens=False)
        if len(tokens) <= max_tokens:
            return experience

        # Keep the last max_tokens tokens
        truncated_tokens = tokens[-max_tokens:]
        truncated = self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)

        # Try to start at a clean line boundary
        lines = truncated.split("\n")
        for i, line in enumerate(lines):
            if line.strip().startswith("- EXPERIENCE ITEM:"):
                return "\n".join(lines[i:])

        return truncated

    # -----------------------------------------------------------------------
    # Teacher log-prob computation
    # -----------------------------------------------------------------------

    async def _compute_teacher_log_probs(self, input_ids: list[int], response_len: int) -> list[float]:
        start_len = max(0, len(input_ids) - response_len)
        payload = {
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": 0,
                "skip_special_tokens": False,
            },
            "return_logprob": True,
            "logprob_start_len": start_len,
        }
        async with self._teacher_lp_semaphore:
            async with httpx.AsyncClient(timeout=None) as client:
                resp = await client.post(self._prm_url, json=payload)
                resp.raise_for_status()
                result = resp.json()

        meta = result.get("meta_info", {}) if isinstance(result, dict) else {}
        inp = meta.get("input_token_logprobs")
        if not isinstance(inp, list):
            return [0.0] * response_len

        all_lp = []
        for item in inp:
            if isinstance(item, (list, tuple)) and len(item) >= 1:
                val = item[0]
                all_lp.append(float(val) if val is not None else 0.0)
            elif isinstance(item, dict) and "logprob" in item:
                val = item["logprob"]
                all_lp.append(float(val) if val is not None else 0.0)
            else:
                all_lp.append(0.0)
        if len(all_lp) > 1:
            all_lp = all_lp[1:]
        if len(all_lp) >= response_len:
            return all_lp[-response_len:]
        return [0.0] * (response_len - len(all_lp)) + all_lp

    async def _compute_teacher_topk_logprobs(
        self, input_ids: list[int], response_len: int
    ) -> tuple[list[list[float]], list[list[int]]]:
        K = self._teacher_topk_request_size  # May be larger than distill_topk when topk_source=student
        K_out = K  # Return all requested logprobs
        start_len = max(0, len(input_ids) - response_len)
        payload = {
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": 0,
                "skip_special_tokens": False,
            },
            "return_logprob": True,
            "logprob_start_len": start_len,
            "top_logprobs_num": K,
        }
        async with self._teacher_lp_semaphore:
            async with httpx.AsyncClient(timeout=None) as client:
                resp = await client.post(self._prm_url, json=payload)
                resp.raise_for_status()
                result = resp.json()

        meta = result.get("meta_info", {}) if isinstance(result, dict) else {}
        inp_top = meta.get("input_top_logprobs")

        if not isinstance(inp_top, list):
            return [[0.0] * K] * response_len, [list(range(K))] * response_len

        all_logprobs: list[list[float]] = []
        all_indices: list[list[int]] = []
        for pos_data in inp_top:
            if isinstance(pos_data, (list, tuple)):
                row_lp = []
                row_idx = []
                for entry in pos_data:
                    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                        row_lp.append(float(entry[0]) if entry[0] is not None else 0.0)
                        row_idx.append(int(entry[1]))
                    elif isinstance(entry, dict):
                        row_lp.append(float(entry.get("logprob", 0.0)))
                        row_idx.append(int(entry.get("token_id", 0)))
                    else:
                        row_lp.append(0.0)
                        row_idx.append(0)
                while len(row_lp) < K:
                    row_lp.append(0.0)
                    row_idx.append(0)
                all_logprobs.append(row_lp[:K])
                all_indices.append(row_idx[:K])
            else:
                all_logprobs.append([0.0] * K)
                all_indices.append(list(range(K)))

        if len(all_logprobs) > 1:
            all_logprobs = all_logprobs[1:]
            all_indices = all_indices[1:]

        if len(all_logprobs) >= response_len:
            return all_logprobs[-response_len:], all_indices[-response_len:]
        pad_len = response_len - len(all_logprobs)
        return (
            [[0.0] * K] * pad_len + all_logprobs,
            [list(range(K))] * pad_len + all_indices,
        )

    # -----------------------------------------------------------------------
    # PRM eval queries (for monitoring only)
    # -----------------------------------------------------------------------

    async def _query_prm_eval_once(self, prompt_text: str, vote_id: int) -> int | None:
        if not self._prm_url:
            return None
        payload = {
            "text": prompt_text,
            "sampling_params": {
                "temperature": self._prm_temperature,
                "top_p": 1.0,
                "top_k": -1,
                "max_new_tokens": self._prm_max_tokens,
                "skip_special_tokens": False,
                "no_stop_trim": True,
                "spaces_between_special_tokens": False,
            },
            "return_logprob": False,
        }
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                resp = await client.post(self._prm_url, json=payload)
                resp.raise_for_status()
                data = resp.json()
            raw = data.get("text", data) if isinstance(data, dict) else str(data)
            if isinstance(raw, list):
                raw = raw[0] if raw else ""
            return _parse_prm_eval_score(str(raw))
        except Exception as e:
            logger.warning("[OpenClaw-OEL] PRM eval query failed (vote %d): %s", vote_id, e)
            return None

    # -----------------------------------------------------------------------
    # Sample submission
    # -----------------------------------------------------------------------

    def _maybe_submit_ready_samples(self, session_id: str, force_drop_without_next_state: bool = False):
        oel_tasks = self._oel_tasks.get(session_id, {})
        pending = self._pending_turn_data.get(session_id, {})
        for turn_num in sorted(list(pending.keys())):
            td = pending[turn_num]
            task = oel_tasks.get(turn_num)

            if task is None:
                if force_drop_without_next_state:
                    pending.pop(turn_num, None)
                    if self._eval_mode:
                        with self._eval_scores_lock:
                            self._eval_scores.append(0.0)
                    logger.info(
                        "[OpenClaw-OEL] dropped session=%s turn=%d (no teacher task)",
                        session_id, turn_num,
                    )
                continue
            if not task.done():
                continue

            pending.pop(turn_num, None)
            oel_tasks.pop(turn_num, None)
            try:
                result = task.result()
            except Exception as e:
                logger.warning("[OpenClaw-OEL] teacher task failed session=%s turn=%d: %s", session_id, turn_num, e)
                if self._eval_mode:
                    with self._eval_scores_lock:
                        self._eval_scores.append(0.0)
                continue

            if self._eval_mode:
                es = result.get("eval_score")
                if es is not None:
                    with self._eval_scores_lock:
                        self._eval_scores.append(es)

            if not result.get("accepted"):
                continue
            self._safe_create_task(self._submit_turn_sample(td, session_id, result))

    async def _submit_turn_sample(self, turn_data: dict[str, Any], session_id: str, teacher_result: dict[str, Any]):
        prompt_ids = turn_data["prompt_ids"]
        response_ids = turn_data["response_ids"]

        teacher_log_probs = teacher_result.get("teacher_log_probs") or []
        if len(teacher_log_probs) > len(response_ids):
            teacher_log_probs = teacher_log_probs[: len(response_ids)]
        elif len(teacher_log_probs) < len(response_ids):
            teacher_log_probs = teacher_log_probs + [0.0] * (len(response_ids) - len(teacher_log_probs))

        sample = Sample()
        sample.prompt = turn_data["prompt_text"]
        sample.response = turn_data["response_text"]
        sample.tokens = prompt_ids + response_ids
        sample.response_length = len(response_ids)
        sample.loss_mask = [1] * len(response_ids)
        sample.rollout_log_probs = turn_data["response_logprobs"]
        sample.teacher_log_probs = torch.tensor(teacher_log_probs, dtype=torch.float32)

        if self._use_topk_distillation:
            K = self._teacher_topk_request_size  # May be > distill_topk when topk_source=student
            topk_lp = teacher_result.get("teacher_topk_log_probs") or []
            topk_idx = teacher_result.get("teacher_topk_indices") or []
            if len(topk_lp) > len(response_ids):
                topk_lp = topk_lp[: len(response_ids)]
                topk_idx = topk_idx[: len(response_ids)]
            elif len(topk_lp) < len(response_ids):
                pad_len = len(response_ids) - len(topk_lp)
                topk_lp = [[0.0] * K] * pad_len + topk_lp
                topk_idx = [list(range(K))] * pad_len + topk_idx
            sample.teacher_topk_log_probs = torch.tensor(topk_lp, dtype=torch.float32)
            sample.teacher_topk_indices = torch.tensor(topk_idx, dtype=torch.long)

        sample.status = Sample.Status.COMPLETED
        sample.index = next(self._index_counter)
        sample.group_index = next(self._group_counter)
        sample.reward = {"score": 1.0}

        logger.info(
            "[OpenClaw-OEL] submitted sample session=%s index=%d prompt_len=%d response_len=%d",
            session_id, sample.index, len(prompt_ids), len(response_ids),
        )
        await asyncio.to_thread(self.output_queue.put, (sample.group_index, [sample]))

    async def _submit_turn_sample_no_teacher(self, turn_data: dict[str, Any], session_id: str):
        """Submit a sample without teacher log-probs (when PRM is not available)."""
        prompt_ids = turn_data["prompt_ids"]
        response_ids = turn_data["response_ids"]

        sample = Sample()
        sample.prompt = turn_data["prompt_text"]
        sample.response = turn_data["response_text"]
        sample.tokens = prompt_ids + response_ids
        sample.response_length = len(response_ids)
        sample.loss_mask = [1] * len(response_ids)
        sample.rollout_log_probs = turn_data["response_logprobs"]
        sample.teacher_log_probs = torch.tensor(turn_data["response_logprobs"], dtype=torch.float32)

        if self._use_topk_distillation:
            K = self._teacher_topk_request_size
            sample.teacher_topk_log_probs = torch.zeros(len(response_ids), K, dtype=torch.float32)
            sample.teacher_topk_indices = torch.zeros(len(response_ids), K, dtype=torch.long)

        sample.status = Sample.Status.COMPLETED
        sample.index = next(self._index_counter)
        sample.group_index = next(self._group_counter)
        sample.reward = {"score": 1.0}

        logger.info(
            "[OpenClaw-OEL] submitted no-teacher sample session=%s index=%d",
            session_id, sample.index,
        )
        await asyncio.to_thread(self.output_queue.put, (sample.group_index, [sample]))

    # -----------------------------------------------------------------------
    # Recording and utilities
    # -----------------------------------------------------------------------

    def _buffer_record(self, session_id, turn_num, messages, prompt_text, response_text, tool_calls):
        if not self._record_file:
            return
        self._pending_records[session_id] = {
            "session_id": session_id,
            "turn": turn_num,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "messages": messages,
            "prompt_text": prompt_text,
            "response_text": response_text,
            "tool_calls": tool_calls or None,
        }

    def _flush_pending_record(self, session_id, next_state):
        rec = self._pending_records.pop(session_id, None)
        if rec is None:
            return
        rec["next_state"] = next_state
        if self._record_file:
            try:
                with open(self._record_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except OSError as e:
                logger.warning("[OpenClaw-OEL] failed to write record: %s", e)

    def _cleanup_session(self, session_id: str):
        """Clean up all state for a session."""
        self._turn_counts.pop(session_id, None)
        self._pending_turn_data.pop(session_id, None)
        self._pending_records.pop(session_id, None)
        self._oel_tasks.pop(session_id, None)
        with self._session_conv_lock:
            self._session_conversations.pop(session_id, None)

    def get_experience_stats(self) -> dict[str, float]:
        """Return experience statistics for logging."""
        with self._experience_lock:
            exp_len = len(self._experience_text)
            session_count = self._experience_session_count
            item_count = self._experience_item_count
        pool_size = 0
        if self._multi_experience:
            with self._experience_pool_lock:
                pool_size = len(self._experience_pool)
        if session_count == 0 and pool_size == 0:
            return {}
        stats = {
            "rollout/experience_chars": float(exp_len),
            "rollout/experience_sessions": float(session_count),
            "rollout/experience_items": float(item_count),
        }
        if pool_size > 0:
            stats["rollout/experience_pool_size"] = float(pool_size)
        return stats

    def drain_eval_scores(self) -> list[float]:
        with self._eval_scores_lock:
            scores = list(self._eval_scores)
            self._eval_scores.clear()
            return scores

    def reset_eval_scores(self):
        with self._eval_scores_lock:
            self._eval_scores.clear()

    def purge_record_files(self):
        for path, label in [
            (self._record_file, "record"),
            (self._prm_record_file, "PRM record"),
        ]:
            if not path:
                continue
            try:
                open(path, "w").close()
                logger.info("[OpenClaw-OEL] %s file purged: %s", label, path)
            except OSError as e:
                logger.warning("[OpenClaw-OEL] failed to purge %s file: %s", label, e)

    def _safe_create_task(self, coro):
        task = asyncio.create_task(coro)
        task.add_done_callback(self._task_done_cb)

    @staticmethod
    def _task_done_cb(task: asyncio.Task):
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error("[OpenClaw-OEL] background task failed: %s", exc, exc_info=exc)

    async def _stream_response(self, result: dict[str, Any]):
        payload = result["response"]
        choice = payload.get("choices", [{}])[0]
        message = choice.get("message", {})
        delta = {"role": "assistant", "content": message.get("content", "") or ""}
        if message.get("tool_calls"):
            delta["tool_calls"] = message["tool_calls"]
        chunk_base = {
            "id": payload.get("id", ""),
            "object": "chat.completion.chunk",
            "created": payload.get("created", int(time.time())),
            "model": payload.get("model", ""),
            "session_id": payload.get("session_id", ""),
        }
        first = {**chunk_base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
        final = {
            **chunk_base,
            "choices": [{"index": 0, "delta": {}, "finish_reason": choice.get("finish_reason", "stop")}],
        }
        yield f"data: {json.dumps(first, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    # -----------------------------------------------------------------------
    # Server lifecycle
    # -----------------------------------------------------------------------

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level="info")
        self._server = uvicorn.Server(config=config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        self._readiness_thread = threading.Thread(target=self._wait_for_sglang_ready, daemon=True)
        self._readiness_thread.start()

    def _wait_for_sglang_ready(self):
        while True:
            try:
                r = httpx.get(self.sglang_health_url, timeout=5)
                if r.status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(3)
        logger.info("[OpenClaw-OEL] policy server ready")

        if self._prm_enabled and self._prm_url:
            prm_health = self._prm_url.rsplit("/", 1)[0] + "/health"
            while True:
                try:
                    r = httpx.get(prm_health, timeout=5)
                    if r.status_code == 200:
                        break
                except Exception:
                    pass
                time.sleep(3)
            logger.info("[OpenClaw-OEL] PRM/teacher server ready")

        time.sleep(8)
        banner = (
            f"\n{'=' * 70}\n"
            f"  [OpenClaw-OEL] model is ready (mode={self._mode})\n"
            f"  proxy {self.host}:{self.port} -> SGLang {self.args.sglang_router_ip}:{self.args.sglang_router_port}\n"
            f"  PRM/teacher {self._prm_url} (m={self._prm_m})\n"
            f"  experience max tokens: {self._experience_max_tokens}\n"
            f"  no-accumulate (replace per session): {self._no_accumulate}\n"
            f"  multi-experience: {self._multi_experience}\n"
            f"  deploy save dir: {self._deploy_save_dir or '(none)'}\n"
            f"{'=' * 70}\n"
        )
        logger.info(f"{_GREEN}{banner}{_RESET}")

    def stop(self):
        if self._server is not None:
            self._server.should_exit = True
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
