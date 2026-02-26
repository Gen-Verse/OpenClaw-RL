import asyncio
import collections
import json
import logging
import os
import queue
import re
import threading
import time
from itertools import count
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from slime.utils.processing_utils import load_tokenizer
from slime.utils.types import Sample

_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_CYAN = "\033[36m"
_RESET = "\033[0m"
logger = logging.getLogger(__name__)

_BOXED_RE = re.compile(r"\\boxed\{([-+]?\d)\}")

_NON_STANDARD_BODY_KEYS = {"session_id", "session_done", "turn_type"}


def _flatten_message_content(content):
    """Extract plain text from multimodal content lists."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
        return " ".join(parts) if parts else ""
    return str(content) if content is not None else ""


def _normalize_messages_for_template(messages: list[dict]) -> list[dict]:
    """Make messages compatible with the chat template.

    - developer → system (templates only know 'system')
    - multimodal content lists → plain text strings
    """
    out = []
    for msg in messages:
        m = dict(msg)
        if m.get("role") == "developer":
            m["role"] = "system"
        raw = m.get("content")
        if not isinstance(raw, str) and raw is not None:
            m["content"] = _flatten_message_content(raw)
        out.append(m)
    return out


def _extract_logprobs_from_chat_response(choice: dict[str, Any]) -> list[float]:
    logprobs_obj = choice.get("logprobs")
    if not isinstance(logprobs_obj, dict):
        return []
    content = logprobs_obj.get("content")
    if not isinstance(content, list):
        return []
    return [float(item.get("logprob", 0.0)) for item in content if isinstance(item, dict)]


def _build_prm_judge_prompt(response_text: str, next_state_text: str) -> list[dict]:
    system = (
        "You are a process reward model (PRM) evaluating an AI assistant.\n"
        "You will see the assistant's output and the subsequent user reply or environment feedback.\n"
        "Judge the quality of the assistant's output based on the feedback.\n"
        "The user may express satisfaction, dissatisfaction, or give no clear signal.\n"
        "If the user shows no clear attitude, infer from their follow-up whether the response was helpful.\n"
        "Think step-by-step, then give your final score inside \\boxed{}.\n"
        "Valid scores: \\boxed{1} (good), \\boxed{-1} (bad), \\boxed{0} (neutral or uncertain)."
    )
    user = (
        f"## Assistant output\n{response_text}\n\n"
        f"## Next state (user reply / environment feedback)\n{next_state_text}\n\n"
        "Now evaluate the assistant's output. Think, then output \\boxed{1}, \\boxed{-1}, or \\boxed{0}."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _parse_prm_score(text: str) -> int | None:
    matches = _BOXED_RE.findall(text)
    if not matches:
        return None
    val = int(matches[-1])
    if val in (1, -1, 0):
        return val
    return None


def _majority_vote(scores: list[int | None]) -> float:
    valid = [s for s in scores if s is not None]
    if not valid:
        return 0.0
    counter = collections.Counter(valid)
    top = counter.most_common(1)[0]
    if list(counter.values()).count(top[1]) > 1:
        return 0.0
    return float(top[0])


async def reward_func(args, sample_or_samples, **kwargs):
    if isinstance(sample_or_samples, list):
        return [{"score": s.reward.get("score", 0.0) if isinstance(s.reward, dict) else 0.0}
                for s in sample_or_samples]
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


class OpenClawAPIServer:
    """Proxy between OpenClaw and SGLang for RL training data collection.

    OpenClaw sends ``X-Session-Id`` and ``X-Turn-Type`` headers with every
    request.  The proxy simply forwards to SGLang, and when ``turn_type``
    is ``"main"`` it tokenises the full prompt+response and submits a
    training sample.  Side tasks (``turn_type != "main"``) are forwarded
    but produce no training data.
    """

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
        self.served_model_name = os.getenv("SERVED_MODEL_NAME", "qwen3-8b")

        self._index_counter = count(0)
        self._group_counter = count(0)
        self._turn_counts: dict[str, int] = {}
        self._pending_records: dict[str, dict] = {}
        self._prm_tasks: dict[str, dict[int, asyncio.Task]] = {}  # session → {turn → task}
        self._pending_turn_data: dict[str, dict[int, dict]] = {}  # session → {turn → turn_data}
        self._session_effective: dict[str, int] = {}  # session → count of samples with loss_mask=[1]

        self._prm_enabled = getattr(args, "prm_enable", False)
        self._prm_m = int(os.getenv("PRM_M", getattr(args, "prm_m", 3)))
        self._prm_temperature = float(getattr(args, "prm_temperature", 0.6))
        self._prm_max_tokens = int(getattr(args, "prm_max_new_tokens", 4096))
        prm_ip = getattr(args, "prm_router_ip", None)
        prm_port = getattr(args, "prm_router_port", None)
        self._prm_url = f"http://{prm_ip}:{prm_port}/generate" if prm_ip and prm_port else ""
        self._prm_tokenizer = None
        if self._prm_enabled:
            prm_path = getattr(args, "prm_model_path", None) or args.hf_checkpoint
            self._prm_tokenizer = load_tokenizer(prm_path, trust_remote_code=True)
            logger.info("[OpenClaw] PRM enabled: url=%s m=%d", self._prm_url, self._prm_m)

        self._record_file = os.getenv("OPENCLAW_RECORD_FILE", "") if os.getenv("OPENCLAW_RECORD_ENABLED", "0") == "1" else ""
        if self._record_file:
            os.makedirs(os.path.dirname(self._record_file), exist_ok=True)
            open(self._record_file, "w").close()
            logger.info("[OpenClaw] record file initialized (cleared): %s", self._record_file)

        self._prm_record_file = os.getenv("OPENCLAW_PRM_RECORD_FILE", "")
        if not self._prm_record_file and self._record_file and self._prm_enabled:
            base, ext = os.path.splitext(self._record_file)
            self._prm_record_file = f"{base}_prm{ext}"
        if self._prm_record_file:
            os.makedirs(os.path.dirname(self._prm_record_file), exist_ok=True)
            open(self._prm_record_file, "w").close()
            logger.info("[OpenClaw] PRM record file initialized (cleared): %s", self._prm_record_file)

        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self.app = self._build_app()

    # ------------------------------------------------------------------ app
    def _build_app(self) -> FastAPI:
        app = FastAPI(title="OpenClaw SLIME Proxy")
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
            owner: OpenClawAPIServer = request.app.state.owner
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
                body, session_id=session_id, turn_type=turn_type, session_done=session_done,
            )
            if stream:
                return StreamingResponse(owner._stream_response(result), media_type="text/event-stream")
            return JSONResponse(content=result["response"])

        return app

    async def _check_auth(self, authorization: str | None):
        if not self.expected_api_key:
            return
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="missing bearer token")
        token = authorization.split(" ", 1)[1].strip()
        if token != self.expected_api_key:
            raise HTTPException(status_code=401, detail="invalid api key")

    # ---------------------------------------------------- record file
    def _flush_pending_record(self, session_id: str, next_state):
        """Write out the buffered record for *session_id* with its next_state and fire PRM."""
        rec = self._pending_records.pop(session_id, None)
        if rec is None:
            return
        rec["next_state"] = next_state
        if next_state:
            ns_role = next_state.get("role", "?")
            ns_content = _flatten_message_content(next_state.get("content"))
            logger.info(
                f"{_GREEN}[OpenClaw] session={session_id} turn={rec['turn']} "
                f"next_state role={ns_role} len={len(ns_content)}: "
                f"{ns_content[:200]}{_RESET}"
            )
            self._fire_prm_scoring(session_id, rec["turn"], rec["response_text"], next_state)
        if self._record_file:
            try:
                with open(self._record_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except OSError as e:
                logger.warning("[OpenClaw] failed to write record: %s", e)

    def _buffer_record(self, session_id: str, turn_num: int, messages: list,
                       prompt_text: str, response_text: str, tool_calls: list):
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

    def _append_prm_record(self, session_id: str, turn_num: int,
                           score: float, votes: list, representative: str):
        if not self._prm_record_file:
            return
        rec: dict[str, Any] = {
            "session_id": session_id,
            "turn": turn_num,
            "score": score,
            "votes": votes,
        }
        if score != 0.0 and representative:
            rec["representative_eval"] = representative
        try:
            with open(self._prm_record_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError as e:
            logger.warning("[OpenClaw] failed to write PRM record: %s", e)

    # ---------------------------------------------------- record purge
    def purge_record_files(self):
        """Clear all record JSONL files. Called when training starts."""
        for path, label in [
            (self._record_file, "record"),
            (self._prm_record_file, "PRM record"),
        ]:
            if not path:
                continue
            try:
                open(path, "w").close()
                logger.info("[OpenClaw] %s file purged: %s", label, path)
            except OSError as e:
                logger.warning("[OpenClaw] failed to purge %s file: %s", label, e)

    # ---------------------------------------------------- PRM scoring
    async def _query_prm_once(self, judge_prompt: str, vote_id: int) -> tuple[int | None, str]:
        if not self._prm_url:
            return None, ""
        payload = {
            "text": judge_prompt,
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
            return _parse_prm_score(str(raw)), str(raw)
        except Exception as e:
            logger.warning("[OpenClaw] PRM query failed (vote %d): %s", vote_id, e)
            return None, ""

    async def _prm_evaluate(self, session_id: str, turn_num: int,
                            response_text: str, next_state) -> dict:
        ns_text = _flatten_message_content(next_state.get("content")) if next_state else ""
        msgs = _build_prm_judge_prompt(response_text, ns_text)
        if self._prm_tokenizer:
            judge_prompt = self._prm_tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
            )
        else:
            judge_prompt = "\n".join(m["content"] for m in msgs)

        results = await asyncio.gather(
            *[self._query_prm_once(judge_prompt, i) for i in range(self._prm_m)]
        )
        scores = [r[0] for r in results]
        final = _majority_vote(scores)

        representative = ""
        if final != 0.0:
            for s, text in results:
                if s is not None and s == int(final):
                    representative = text
                    break

        votes_display = [s if s is not None else "fail" for s in scores]
        logger.info(
            f"{_CYAN}[OpenClaw] PRM session={session_id} turn={turn_num} "
            f"votes={votes_display} → score={final}{_RESET}"
        )
        self._append_prm_record(session_id, turn_num, final, votes_display, representative)
        return {"score": final, "votes": votes_display, "representative_eval": representative}

    def _fire_prm_scoring(self, session_id: str, turn_num: int,
                          response_text: str, next_state):
        if not self._prm_enabled or not next_state:
            return
        task = asyncio.create_task(
            self._prm_evaluate(session_id, turn_num, response_text, next_state)
        )
        task.add_done_callback(self._task_done_cb)
        task.add_done_callback(lambda _t: self._maybe_submit_ready_samples(session_id))
        self._prm_tasks.setdefault(session_id, {})[turn_num] = task
        td = self._pending_turn_data.get(session_id, {}).get(turn_num)
        if td is not None:
            td["has_next_state"] = True

    # ---------------------------------------------------- request handling
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

        async with httpx.AsyncClient(timeout=None) as client:
            sglang_resp = await client.post(self.sglang_chat_url, json=forward_body)
            if sglang_resp.status_code != 200:
                logger.error("[OpenClaw] SGLang returned %d: %s", sglang_resp.status_code, sglang_resp.text[:1000])
                sglang_resp.raise_for_status()
            try:
                output = sglang_resp.json()
            except Exception:
                logger.error("[OpenClaw] SGLang non-JSON body: %s", sglang_resp.text[:1000])
                raise

        choice = output.get("choices", [{}])[0]
        assistant_msg = choice.get("message", {})
        tool_calls = assistant_msg.get("tool_calls") or []

        content = assistant_msg.get("content") or ""
        reasoning = assistant_msg.get("reasoning_content") or ""
        logger.info(
            f"{_YELLOW}[OpenClaw] [{turn_type}] session={session_id} "
            f"prompt_msgs={len(messages)}{_RESET}"
        )
        logger.info(
            f"{_RED}[OpenClaw] [{turn_type}] session={session_id} "
            f"thinking={len(reasoning)} chars, response:\n{content}{_RESET}"
        )
        if tool_calls:
            logger.info("[OpenClaw] tool_calls: %s", json.dumps(tool_calls, ensure_ascii=False)[:500])

        if turn_type == "main":
            if session_id in self._pending_records and messages:
                self._flush_pending_record(session_id, messages[-1])

            response_msg = dict(assistant_msg)
            if response_msg.get("content") is None:
                response_msg["content"] = ""

            norm_msgs = _normalize_messages_for_template(messages)
            norm_resp = _normalize_messages_for_template([response_msg])[0]
            full_norm = norm_msgs + [norm_resp]

            prompt_text = self.tokenizer.apply_chat_template(
                norm_msgs, tools=tools, tokenize=False, add_generation_prompt=True,
            )
            full_text = self.tokenizer.apply_chat_template(
                full_norm, tools=tools, tokenize=False, add_generation_prompt=False,
            )

            if full_text.startswith(prompt_text):
                response_text = full_text[len(prompt_text):]
            else:
                logger.warning("[OpenClaw] prompt_text is not a prefix of full_text, using full_text as response")
                response_text = full_text

            prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            response_ids = self.tokenizer(response_text, add_special_tokens=False)["input_ids"]

            if not response_ids and not response_text.strip():
                logger.info("[OpenClaw] MAIN session=%s → empty response, skipping", session_id)
                output["session_id"] = session_id
                return {"response": output}

            response_logprobs = _extract_logprobs_from_chat_response(choice)
            if len(response_logprobs) > len(response_ids):
                response_logprobs = response_logprobs[: len(response_ids)]
            elif len(response_logprobs) < len(response_ids):
                response_logprobs = response_logprobs + [0.0] * (len(response_ids) - len(response_logprobs))

            turn_data = {
                "prompt_ids": prompt_ids,
                "response_ids": response_ids,
                "response_logprobs": response_logprobs,
                "prompt_text": prompt_text,
                "response_text": response_text,
            }

            self._turn_counts[session_id] = self._turn_counts.get(session_id, 0) + 1
            turn_num = self._turn_counts[session_id]

            logger.info(
                "[OpenClaw] MAIN session=%s turn=%d prompt_tokens=%d response_tokens=%d",
                session_id, turn_num, len(prompt_ids), len(response_ids),
            )
            self._buffer_record(session_id, turn_num, messages, prompt_text, response_text, tool_calls)
            self._pending_turn_data.setdefault(session_id, {})[turn_num] = turn_data
            self._maybe_submit_ready_samples(session_id)
        else:
            logger.info("[OpenClaw] SIDE session=%s → skipped (no training data)", session_id)

        if session_done:
            self._flush_pending_record(session_id, None)
            self._maybe_submit_ready_samples(session_id, force_no_prm=True)
            eff = self._session_effective.pop(session_id, 0)
            self._turn_counts.pop(session_id, None)
            logger.info("[OpenClaw] session=%s done → cleaned up (effective_samples=%d)", session_id, eff)

        output["session_id"] = session_id
        return {"response": output}

    # ------------------------------------------------- sample submission
    def _maybe_submit_ready_samples(self, session_id: str, force_no_prm: bool = False):
        """Submit turns whose PRM is done (or PRM not needed).

        force_no_prm: also submit turns that have no PRM task yet (used at
        session end for the last turn which will never get a next_state).
        """
        prm_tasks = self._prm_tasks.get(session_id, {})
        pending = self._pending_turn_data.get(session_id, {})
        for turn_num in sorted(list(pending.keys())):
            task = prm_tasks.get(turn_num)
            if not self._prm_enabled:
                pass  # no PRM → submit immediately
            elif task is not None and not task.done():
                continue  # PRM still running
            elif task is None and not force_no_prm:
                continue  # waiting for next_state to fire PRM
            turn_data = pending.pop(turn_num)
            prm_result = None
            if task is not None and task.done():
                try:
                    prm_result = task.result()
                except Exception:
                    pass
                prm_tasks.pop(turn_num, None)
            self._safe_create_task(
                self._submit_turn_sample(turn_data, session_id, prm_result)
            )

    async def _submit_turn_sample(self, turn_data: dict[str, Any], session_id: str,
                                  prm_result: dict | None):
        prompt_ids = turn_data["prompt_ids"]
        response_ids = turn_data["response_ids"]

        has_next_state = turn_data.get("has_next_state", False)
        if prm_result:
            score = prm_result["score"]
        else:
            score = 0.0

        exclude = not has_next_state or score == 0.0
        # Guarantee at least one sample per session contributes to training.
        # If this session has produced zero effective samples so far and this
        # turn has a next_state (i.e. it was PRM-evaluated, just scored 0),
        # promote it to effective.
        if exclude and has_next_state and self._session_effective.get(session_id, 0) == 0:
            exclude = False
            logger.info("[OpenClaw] promoting session=%s turn with score=0 → loss_mask=1 (at-least-one guarantee)", session_id)

        sample = Sample()
        sample.prompt = turn_data["prompt_text"]
        sample.response = turn_data["response_text"]
        sample.tokens = prompt_ids + response_ids
        sample.response_length = len(response_ids)
        sample.loss_mask = [0] * len(response_ids) if exclude else [1] * len(response_ids)
        sample.rollout_log_probs = turn_data["response_logprobs"]
        sample.status = Sample.Status.COMPLETED
        sample.index = next(self._index_counter)
        sample.group_index = next(self._group_counter)
        sample.reward = {"score": score}

        if not exclude:
            self._session_effective[session_id] = self._session_effective.get(session_id, 0) + 1

        logger.info(
            "[OpenClaw] submitted sample session=%s index=%d score=%.1f exclude=%s prompt_len=%d response_len=%d",
            session_id, sample.index, score, exclude, len(prompt_ids), len(response_ids),
        )
        await asyncio.to_thread(self.output_queue.put, (sample.group_index, [sample]))

    def _safe_create_task(self, coro):
        task = asyncio.create_task(coro)
        task.add_done_callback(self._task_done_cb)

    @staticmethod
    def _task_done_cb(task: asyncio.Task):
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error("[OpenClaw] background task failed: %s", exc, exc_info=exc)

    # ----------------------------------------------------------- streaming
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
        final = {**chunk_base, "choices": [{"index": 0, "delta": {}, "finish_reason": choice.get("finish_reason", "stop")}]}
        yield f"data: {json.dumps(first, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    # ------------------------------------------------------------- lifecycle
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
        logger.info("[OpenClaw] policy server ready")

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
            logger.info("[OpenClaw] PRM server ready")

        time.sleep(8)
        prm_line = ""
        if self._prm_enabled:
            prm_line = f"\n  PRM enabled: {self._prm_url} (m={self._prm_m})"
        banner = (
            f"\n{'=' * 70}\n"
            f"  [OpenClaw] your model is fired up and ready to roll\n"
            f"  proxy {self.host}:{self.port} -> SGLang "
            f"{self.args.sglang_router_ip}:{self.args.sglang_router_port}"
            f"{prm_line}\n"
            f"{'=' * 70}\n"
        )
        logger.info(f"{_GREEN}{banner}{_RESET}")

    def stop(self):
        if self._server is not None:
            self._server.should_exit = True
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
