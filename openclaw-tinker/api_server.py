"""FastAPI proxy for OpenClaw training data collection via Tinker.

Contains a shared base class and three method-specific subclasses:
  - OpenClawRLServer:      Binary RL with PRM scoring
  - OpenClawOPDServer:     On-Policy Distillation (hint judge + teacher logprobs)
  - OpenClawCombineServer: Combined OPD + RL with three-way dispatch

All methods share:
  - Tinker SamplingClient forwarding
  - OpenAI-compatible chat completion proxy
  - Session/turn tracking, record buffering, streaming
  - Eval score collection for W&B logging
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import re
import threading
import time
from itertools import count
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from config import TinkerConfig
from data_formatter import TrainingSample

logger = logging.getLogger(__name__)

_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_CYAN = "\033[36m"
_RESET = "\033[0m"

_NON_STANDARD_BODY_KEYS = {"session_id", "session_done", "turn_type"}


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _flatten_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        return " ".join(parts) if parts else ""
    return str(content) if content is not None else ""


def _normalize_messages(messages: list[dict]) -> list[dict]:
    """Normalize messages for the chat template (developer -> system, flatten content)."""
    out = []
    for msg in messages:
        m = dict(msg)
        if m.get("role") == "developer":
            m["role"] = "system"
        raw = m.get("content")
        if not isinstance(raw, str) and raw is not None:
            m["content"] = _flatten_content(raw)
        out.append(m)
    return out


def _extract_logprobs(choice: dict[str, Any]) -> list[float]:
    lp_obj = choice.get("logprobs")
    if not isinstance(lp_obj, dict):
        return []
    content = lp_obj.get("content")
    if not isinstance(content, list):
        return []
    return [float(item.get("logprob", 0.0)) for item in content if isinstance(item, dict)]


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_KIMI_TC_RE = re.compile(
    r"<\|tool_call_begin\|>\s*([a-zA-Z0-9_.-]+)(?::\d+)?\s*"
    r"<\|tool_call_argument_begin\|>\s*(\{.*?\})\s*<\|tool_call_end\|>",
    re.DOTALL,
)
_QWEN_TC_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def _extract_tool_calls(text: str) -> tuple[str, list[dict]]:
    """Parse tool-call tags from assistant text into OpenAI-style tool_calls."""
    if not text:
        return "", []
    tool_calls = []
    for i, m in enumerate(_KIMI_TC_RE.finditer(text)):
        raw_name = (m.group(1) or "").strip()
        args_raw = (m.group(2) or "{}").strip()
        try:
            args_str = json.dumps(json.loads(args_raw), ensure_ascii=False)
        except Exception:
            args_str = args_raw
        tool_calls.append({
            "id": f"call_{i}", "type": "function",
            "function": {"name": raw_name or "unknown_tool", "arguments": args_str},
        })
    for i, m in enumerate(_QWEN_TC_RE.finditer(text), start=len(tool_calls)):
        try:
            payload = json.loads(m.group(1).strip())
        except Exception:
            continue
        name = payload.get("name") or payload.get("function", {}).get("name") or "unknown_tool"
        args = payload.get("arguments") or payload.get("function", {}).get("arguments") or {}
        if not isinstance(args, str):
            args = json.dumps(args, ensure_ascii=False)
        tool_calls.append({
            "id": f"call_{i}", "type": "function",
            "function": {"name": str(name), "arguments": args},
        })
    clean = _THINK_RE.sub("", text)
    clean = clean.replace("</think>", "")
    clean = re.sub(r"<\|tool_call_begin\|>.*?<\|tool_call_end\|>", "", clean, flags=re.DOTALL)
    clean = re.sub(r"<\|tool_calls_section_begin\|>.*?<\|tool_calls_section_end\|>", "", clean, flags=re.DOTALL)
    clean = _QWEN_TC_RE.sub("", clean)
    return clean.strip(), tool_calls


# ===========================================================================
# Base Server
# ===========================================================================

class _BaseServer:
    """Shared infrastructure for all three OpenClaw Tinker proxy servers."""

    _TITLE = "OpenClaw Tinker Proxy"
    _SCORE_FILE = "scores.jsonl"

    def __init__(
        self,
        config: TinkerConfig,
        output_queue: queue.Queue,
        submission_enabled: threading.Event,
        sampling_client=None,
    ):
        self.config = config
        self.output_queue = output_queue
        self.submission_enabled = submission_enabled
        self._sampling_client = sampling_client

        self._index_counter = count(0)
        self._group_counter = count(0)
        self._turn_counts: dict[str, int] = {}
        self._pending_turn_data: dict[str, dict[int, dict]] = {}
        self._prm_tasks: dict[str, dict[int, asyncio.Task]] = {}
        self._pending_records: dict[str, dict] = {}

        self._eval_scores: list[float] = []
        self._eval_scores_lock = threading.Lock()

        os.makedirs(config.record_dir, exist_ok=True)
        self._record_file = os.path.join(config.record_dir, "conversations.jsonl")
        self._prm_record_file = os.path.join(config.record_dir, self._SCORE_FILE)
        open(self._record_file, "w").close()
        open(self._prm_record_file, "w").close()

        self._tokenizer = self._load_tokenizer()
        self.app = self._build_app()
        self._server: Optional[uvicorn.Server] = None
        self._thread: Optional[threading.Thread] = None

    def _load_tokenizer(self):
        try:
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained(self.config.model_name, trust_remote_code=True)
        except Exception as e:
            logger.error("[Server] FAILED to load tokenizer: %s", e, exc_info=True)
            return None

    # ------------------------------------------------------------------ app
    def _build_app(self) -> FastAPI:
        app = FastAPI(title=self._TITLE)
        app.state.owner = self

        @app.get("/healthz")
        async def healthz():
            return {"ok": True}

        @app.post("/v1/chat/completions")
        async def chat_completions(
            request: Request,
            authorization: Optional[str] = Header(default=None),
            x_session_id: Optional[str] = Header(default=None),
            x_turn_type: Optional[str] = Header(default=None),
            x_session_done: Optional[str] = Header(default=None),
        ):
            owner: _BaseServer = request.app.state.owner
            await owner._check_auth(authorization)
            if not owner.submission_enabled.is_set():
                resumed = await asyncio.to_thread(owner.submission_enabled.wait, 300.0)
                if not resumed:
                    raise HTTPException(status_code=503, detail="submission paused (timeout)")

            body = await request.json()
            session_id = x_session_id or body.get("session_id") or "unknown"
            turn_type = (x_turn_type or body.get("turn_type") or owner.config.default_turn_type).strip().lower()
            session_done = (
                (x_session_done and x_session_done.strip().lower() in {"1", "true", "yes", "on"})
                or str(body.get("session_done", "")).strip().lower() in {"1", "true", "yes", "on"}
            )
            stream = bool(body.get("stream", False))
            result = await owner._handle_request(body, session_id, turn_type, session_done)
            if stream:
                return StreamingResponse(owner._stream_response(result), media_type="text/event-stream")
            return JSONResponse(content=result["response"])

        return app

    async def _check_auth(self, authorization: Optional[str]):
        if not self.config.api_key:
            return
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="missing bearer token")
        if authorization.split(" ", 1)[1].strip() != self.config.api_key:
            raise HTTPException(status_code=401, detail="invalid api key")

    # -------------------------------------------------------------- Tinker
    async def _forward_to_tinker(self, body: dict[str, Any]) -> dict[str, Any]:
        """Forward inference request to Tinker SamplingClient."""
        import tinker

        if self._sampling_client is None:
            raise HTTPException(status_code=503, detail="no sampling client available")
        if self._tokenizer is None:
            raise HTTPException(status_code=503, detail="no tokenizer available")

        messages = body.get("messages", [])
        norm_msgs = _normalize_messages(messages)
        tools = body.get("tools")
        temperature = float(body.get("temperature", 0.6))
        max_tokens = int(body.get("max_tokens") or 2048)
        stop = body.get("stop")

        prompt_text = self._tokenizer.apply_chat_template(
            norm_msgs, tools=tools, tokenize=False, add_generation_prompt=True,
        )
        prompt_ids = self._tokenizer.encode(prompt_text, add_special_tokens=False)

        chunk = tinker.EncodedTextChunk(tokens=list(prompt_ids), type="encoded_text")
        model_input = tinker.ModelInput(chunks=[chunk])

        sp_kwargs = dict(temperature=temperature, max_tokens=max_tokens, top_k=50, top_p=0.95)
        if stop is not None:
            sp_kwargs["stop"] = stop
        sampling_params = tinker.SamplingParams(**sp_kwargs)

        response = await self._sampling_client.sample_async(
            prompt=model_input, num_samples=1, sampling_params=sampling_params,
            include_prompt_logprobs=False, topk_prompt_logprobs=0,
        )

        seq = response.sequences[0]
        raw_response_tokens = list(seq.tokens)
        raw_response_logprobs = [float(lp) for lp in (seq.logprobs or [])]

        response_text = self._tokenizer.decode(seq.tokens, skip_special_tokens=True)
        normalized_text, parsed_tool_calls = _extract_tool_calls(response_text)

        lp_content = [{"token": "", "logprob": lp, "top_logprobs": []} for lp in raw_response_logprobs]
        assistant_message: dict[str, Any] = {"role": "assistant", "content": normalized_text}
        if parsed_tool_calls:
            assistant_message["tool_calls"] = parsed_tool_calls

        return {
            "id": f"chatcmpl-tinker-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model", self.config.served_model_name),
            "choices": [{
                "index": 0,
                "message": assistant_message,
                "finish_reason": "tool_calls" if parsed_tool_calls else (seq.stop_reason or "stop"),
                "logprobs": {"content": lp_content},
            }],
            "usage": {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(raw_response_tokens),
                "total_tokens": len(prompt_ids) + len(raw_response_tokens),
            },
            # Raw Tinker data for training — strictly aligned with each other
            "_raw_prompt_ids": list(prompt_ids),
            "_raw_response_tokens": raw_response_tokens,
            "_raw_response_logprobs": raw_response_logprobs,
        }

    # ------------------------------------------------------------ records
    def _buffer_record(self, session_id: str, turn_num: int, messages: list,
                       prompt_text: str, response_text: str, tool_calls: list):
        self._pending_records[session_id] = {
            "session_id": session_id, "turn": turn_num,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "prompt_text": prompt_text, "response_text": response_text,
            "tool_calls": tool_calls or None,
        }

    def _flush_pending_record(self, session_id: str, next_state):
        rec = self._pending_records.pop(session_id, None)
        if rec is None:
            return
        rec["next_state"] = next_state
        if self._record_file:
            try:
                with open(self._record_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except OSError:
                pass

    def _append_score_record(self, record: dict):
        if not self._prm_record_file:
            return
        try:
            with open(self._prm_record_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def purge_record_files(self):
        for path in [self._record_file, self._prm_record_file]:
            if path:
                try:
                    open(path, "w").close()
                except OSError:
                    pass

    def drain_eval_scores(self) -> list[float]:
        with self._eval_scores_lock:
            scores = list(self._eval_scores)
            self._eval_scores.clear()
            return scores

    def reset_eval_scores(self):
        with self._eval_scores_lock:
            self._eval_scores.clear()

    # ----------------------------------------------------------- request (abstract)
    async def _handle_request(self, body: dict, session_id: str,
                              turn_type: str, session_done: bool) -> dict:
        raise NotImplementedError

    # -------------------------------------------------------- tokenize helpers
    def _tokenize_turn(self, messages, assistant_msg, tools, choice, output=None):
        """Shared tokenization logic for a main-line turn.

        Uses raw tokens/logprobs from Tinker (via output["_raw_*"]) to guarantee
        strict alignment between response_ids and response_logprobs.
        Falls back to re-tokenization only if raw data is unavailable.
        """
        norm_msgs = _normalize_messages(messages)
        prompt_text = self._tokenizer.apply_chat_template(
            norm_msgs, tools=tools, tokenize=False, add_generation_prompt=True,
        )

        # --- Use raw Tinker tokens (guaranteed aligned with logprobs) ---
        if output is not None and "_raw_response_tokens" in output:
            prompt_ids = output["_raw_prompt_ids"]
            response_ids = output["_raw_response_tokens"]
            response_logprobs = output["_raw_response_logprobs"]

            if len(response_logprobs) != len(response_ids):
                logger.error(
                    "[Server] CRITICAL: raw logprobs len=%d != raw tokens len=%d, "
                    "padding/truncating but this indicates a Tinker SDK bug",
                    len(response_logprobs), len(response_ids),
                )
                if len(response_logprobs) > len(response_ids):
                    response_logprobs = response_logprobs[:len(response_ids)]
                else:
                    response_logprobs = response_logprobs + [0.0] * (len(response_ids) - len(response_logprobs))

            response_text = self._tokenizer.decode(response_ids, skip_special_tokens=True)
            return prompt_ids, response_ids, response_logprobs, prompt_text, response_text

        # --- Fallback: re-tokenize (legacy path, logprob alignment NOT guaranteed) ---
        logger.warning(
            "[Server] _tokenize_turn: raw tokens unavailable, falling back to "
            "re-tokenization — logprob alignment is NOT guaranteed"
        )
        response_msg = dict(assistant_msg)
        if response_msg.get("content") is None:
            response_msg["content"] = ""

        norm_resp = _normalize_messages([response_msg])[0]
        full_norm = norm_msgs + [norm_resp]

        full_text = self._tokenizer.apply_chat_template(
            full_norm, tools=tools, tokenize=False, add_generation_prompt=False,
        )
        response_text = full_text[len(prompt_text):] if full_text.startswith(prompt_text) else full_text
        prompt_ids = self._tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        response_ids = self._tokenizer(response_text, add_special_tokens=False)["input_ids"]

        response_logprobs = _extract_logprobs(choice)
        if len(response_logprobs) > len(response_ids):
            response_logprobs = response_logprobs[:len(response_ids)]
        elif len(response_logprobs) < len(response_ids):
            response_logprobs += [0.0] * (len(response_ids) - len(response_logprobs))

        return prompt_ids, response_ids, response_logprobs, prompt_text, response_text

    # --------------------------------------------------------- streaming
    async def _stream_response(self, result: dict):
        payload = result["response"]
        choice = payload.get("choices", [{}])[0]
        message = choice.get("message", {})
        delta = {"role": "assistant", "content": message.get("content", "") or ""}
        if message.get("tool_calls"):
            delta["tool_calls"] = message["tool_calls"]
        base = {
            "id": payload.get("id", ""), "object": "chat.completion.chunk",
            "created": payload.get("created", int(time.time())),
            "model": payload.get("model", ""),
            "session_id": payload.get("session_id", ""),
        }
        yield f"data: {json.dumps({**base, 'choices': [{'index': 0, 'delta': delta, 'finish_reason': None}]}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({**base, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': choice.get('finish_reason', 'stop')}]}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    # --------------------------------------------------------- lifecycle
    def update_sampling_client(self, client):
        self._sampling_client = client
        logger.info("[Server] sampling client updated")

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        cfg = uvicorn.Config(self.app, host=self.config.proxy_host,
                             port=self.config.proxy_port, log_level="info")
        self._server = uvicorn.Server(cfg)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        threading.Thread(target=self._print_ready, daemon=True).start()

    def _print_ready(self):
        time.sleep(3)
        banner = (
            f"\n{'=' * 60}\n"
            f"  {self._TITLE} ready\n"
            f"  {self.config.proxy_host}:{self.config.proxy_port} -> Tinker cloud\n"
            f"  Method: {self.config.method}\n"
            f"  PRM/Teacher: Tinker SamplingClient (m={self.config.prm_m})\n"
            f"{'=' * 60}\n"
        )
        logger.info("%s%s%s", _GREEN, banner, _RESET)

    def stop(self):
        if self._server is not None:
            self._server.should_exit = True
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

    def _safe_create_task(self, coro):
        task = asyncio.create_task(coro)
        task.add_done_callback(self._task_done_cb)

    @staticmethod
    def _task_done_cb(task: asyncio.Task):
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error("[Server] background task failed: %s", exc, exc_info=exc)


# ===========================================================================
# RL Server
# ===========================================================================

class OpenClawRLServer(_BaseServer):
    """Proxy for Binary RL training with PRM scoring."""

    _TITLE = "OpenClaw-RL Tinker Proxy"
    _SCORE_FILE = "prm_scores.jsonl"

    def __init__(self, config, output_queue, submission_enabled,
                 sampling_client=None, prm_scorer=None):
        super().__init__(config, output_queue, submission_enabled, sampling_client)
        self.prm_scorer = prm_scorer
        self._session_effective: dict[str, int] = {}

    def _flush_pending_record(self, session_id: str, next_state):
        rec = self._pending_records.pop(session_id, None)
        if rec is None:
            return
        rec["next_state"] = next_state
        if next_state:
            ns_text = _flatten_content(next_state.get("content"))
            ns_role = next_state.get("role", "user")
            logger.info("%s[Server] session=%s turn=%d next_state role=%s len=%d%s",
                        _GREEN, session_id, rec["turn"], ns_role, len(ns_text), _RESET)
            self._fire_prm_scoring(session_id, rec["turn"], rec["response_text"], next_state)
        if self._record_file:
            try:
                with open(self._record_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except OSError:
                pass

    def _fire_prm_scoring(self, session_id, turn_num, response_text, next_state):
        if not self.prm_scorer or not next_state:
            return
        ns_text = _flatten_content(next_state.get("content"))
        ns_role = next_state.get("role", "user")
        task = asyncio.create_task(
            self.prm_scorer.evaluate(response_text, ns_text, ns_role, session_id, turn_num)
        )
        task.add_done_callback(self._task_done_cb)
        task.add_done_callback(lambda _t: self._maybe_submit_ready_samples(session_id))
        self._prm_tasks.setdefault(session_id, {})[turn_num] = task
        td = self._pending_turn_data.get(session_id, {}).get(turn_num)
        if td is not None:
            td["has_next_state"] = True

    async def _handle_request(self, body, session_id, turn_type, session_done):
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
            forward_body["model"] = self.config.served_model_name

        output = await self._forward_to_tinker(forward_body)
        choice = output.get("choices", [{}])[0]
        assistant_msg = choice.get("message", {})
        tool_calls = assistant_msg.get("tool_calls") or []
        content = assistant_msg.get("content") or ""
        reasoning = assistant_msg.get("reasoning_content") or ""

        logger.info("%s[Server] [%s] session=%s prompt_msgs=%d%s",
                    _YELLOW, turn_type, session_id, len(messages), _RESET)
        logger.info("%s[Server] [%s] session=%s thinking=%d response:\n%s%s",
                    _RED, turn_type, session_id, len(reasoning), content[:500], _RESET)

        if turn_type == "main":
            if session_id in self._pending_records and messages:
                self._flush_pending_record(session_id, messages[-1])

            prompt_ids, response_ids, response_logprobs, prompt_text, response_text = \
                self._tokenize_turn(messages, assistant_msg, tools, choice, output=output)

            if not response_ids and not response_text.strip():
                output["session_id"] = session_id
                return {"response": output}

            self._turn_counts[session_id] = self._turn_counts.get(session_id, 0) + 1
            turn_num = self._turn_counts[session_id]
            turn_data = {
                "prompt_ids": prompt_ids, "response_ids": response_ids,
                "response_logprobs": response_logprobs,
                "prompt_text": prompt_text, "response_text": response_text,
            }
            self._pending_turn_data.setdefault(session_id, {})[turn_num] = turn_data
            self._buffer_record(session_id, turn_num, messages, prompt_text, response_text, tool_calls)
            logger.info("[Server] MAIN session=%s turn=%d prompt=%d response=%d",
                        session_id, turn_num, len(prompt_ids), len(response_ids))
            self._maybe_submit_ready_samples(session_id)
        else:
            logger.info("[Server] SIDE session=%s -> skipped", session_id)

        if session_done:
            self._flush_pending_record(session_id, None)
            self._maybe_submit_ready_samples(session_id, force_no_prm=True)
            eff = self._session_effective.pop(session_id, 0)
            self._turn_counts.pop(session_id, None)
            logger.info("[Server] session=%s done (effective=%d)", session_id, eff)

        output["session_id"] = session_id
        return {"response": output}

    def _maybe_submit_ready_samples(self, session_id, force_no_prm=False):
        prm_tasks = self._prm_tasks.get(session_id, {})
        pending = self._pending_turn_data.get(session_id, {})
        for turn_num in sorted(list(pending.keys())):
            task = prm_tasks.get(turn_num)
            if self.prm_scorer:
                if task is not None and not task.done():
                    continue
                if task is None and not force_no_prm:
                    continue

            turn_data = pending.pop(turn_num)
            prm_result = None
            if task is not None and task.done():
                try:
                    prm_result = task.result()
                except Exception:
                    pass
                prm_tasks.pop(turn_num, None)
            self._safe_create_task(self._submit_turn_sample(turn_data, session_id, prm_result))

    async def _submit_turn_sample(self, turn_data, session_id, prm_result):
        prompt_ids = turn_data["prompt_ids"]
        response_ids = turn_data["response_ids"]
        has_next_state = turn_data.get("has_next_state", False)
        score = prm_result["score"] if prm_result else 0.0

        with self._eval_scores_lock:
            self._eval_scores.append(score)

        exclude = not has_next_state or score == 0.0
        # At-least-one guarantee
        if exclude and has_next_state and self._session_effective.get(session_id, 0) == 0:
            exclude = False
            logger.info("[Server] promoting session=%s score=0 -> loss_mask=1 (at-least-one)", session_id)

        loss_mask = [0] * len(response_ids) if exclude else [1] * len(response_ids)
        sample = TrainingSample(
            session_id=session_id, turn_num=self._turn_counts.get(session_id, 0),
            prompt_tokens=prompt_ids, response_tokens=response_ids,
            response_logprobs=turn_data["response_logprobs"],
            loss_mask=loss_mask, reward=score,
            prompt_text=turn_data.get("prompt_text", ""),
            response_text=turn_data.get("response_text", ""),
        )

        if not exclude:
            self._session_effective[session_id] = self._session_effective.get(session_id, 0) + 1

        if prm_result:
            self._append_score_record({
                "session_id": session_id, "turn": sample.turn_num,
                "score": score, "votes": prm_result.get("votes", []),
                "representative": prm_result.get("representative", ""),
            })

        index = next(self._index_counter)
        group_index = next(self._group_counter)
        logger.info("[Server] submitted session=%s idx=%d score=%.1f exclude=%s",
                    session_id, index, score, exclude)
        await asyncio.to_thread(self.output_queue.put, (group_index, [sample]))


# ===========================================================================
# OPD Server
# ===========================================================================

class OpenClawOPDServer(_BaseServer):
    """Proxy for On-Policy Distillation (OPD) training."""

    _TITLE = "OpenClaw-OPD Tinker Proxy"
    _SCORE_FILE = "opd_scores.jsonl"

    def __init__(self, config, output_queue, submission_enabled,
                 sampling_client=None, opd_scorer=None):
        super().__init__(config, output_queue, submission_enabled, sampling_client)
        self.opd_scorer = opd_scorer

    def _fire_opd_task(self, session_id, turn_num, turn_data, next_state):
        if not self.opd_scorer or not next_state:
            return
        task = asyncio.create_task(
            self.opd_scorer.evaluate(
                response_text=turn_data["response_text"],
                next_state_text=_flatten_content(next_state.get("content")),
                next_state_role=next_state.get("role", "user"),
                turn_data=turn_data, tokenizer=self._tokenizer,
                normalize_fn=_normalize_messages,
                session_id=session_id, turn_num=turn_num,
            )
        )
        task.add_done_callback(self._task_done_cb)
        task.add_done_callback(lambda _t: self._maybe_submit_ready_samples(session_id))
        self._prm_tasks.setdefault(session_id, {})[turn_num] = task
        turn_data["has_next_state"] = True

    async def _handle_request(self, body, session_id, turn_type, session_done):
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
            forward_body["model"] = self.config.served_model_name

        output = await self._forward_to_tinker(forward_body)
        choice = output.get("choices", [{}])[0]
        assistant_msg = choice.get("message", {})
        tool_calls = assistant_msg.get("tool_calls") or []
        content = assistant_msg.get("content") or ""
        reasoning = assistant_msg.get("reasoning_content") or ""

        logger.info("%s[Server] [%s] session=%s prompt_msgs=%d%s",
                    _YELLOW, turn_type, session_id, len(messages), _RESET)
        logger.info("%s[Server] [%s] session=%s thinking=%d response:\n%s%s",
                    _RED, turn_type, session_id, len(reasoning), content[:500], _RESET)

        if turn_type == "main":
            prev_turn_num = self._turn_counts.get(session_id, 0)

            if prev_turn_num > 0 and messages:
                self._flush_pending_record(session_id, messages[-1])
                prev_td = self._pending_turn_data.get(session_id, {}).get(prev_turn_num)
                if prev_td is not None:
                    self._fire_opd_task(session_id, prev_turn_num, prev_td, messages[-1])

            prompt_ids, response_ids, response_logprobs, prompt_text, response_text = \
                self._tokenize_turn(messages, assistant_msg, tools, choice, output=output)

            if not response_ids and not response_text.strip():
                output["session_id"] = session_id
                return {"response": output}

            self._turn_counts[session_id] = prev_turn_num + 1
            turn_num = self._turn_counts[session_id]
            turn_data = {
                "prompt_ids": prompt_ids, "response_ids": response_ids,
                "response_logprobs": response_logprobs,
                "prompt_text": prompt_text, "response_text": response_text,
                "messages": messages, "tools": tools,
                "has_next_state": False,
            }
            self._pending_turn_data.setdefault(session_id, {})[turn_num] = turn_data
            self._buffer_record(session_id, turn_num, messages, prompt_text, response_text, tool_calls)
            logger.info("[Server] MAIN session=%s turn=%d prompt=%d response=%d",
                        session_id, turn_num, len(prompt_ids), len(response_ids))
            self._maybe_submit_ready_samples(session_id)
        else:
            logger.info("[Server] SIDE session=%s -> skipped", session_id)

        if session_done:
            self._flush_pending_record(session_id, None)
            self._maybe_submit_ready_samples(session_id, force_drop=True)
            self._turn_counts.pop(session_id, None)
            logger.info("[Server] session=%s done", session_id)

        output["session_id"] = session_id
        return {"response": output}

    def _maybe_submit_ready_samples(self, session_id, force_drop=False):
        prm_tasks = self._prm_tasks.get(session_id, {})
        pending = self._pending_turn_data.get(session_id, {})
        for turn_num in sorted(list(pending.keys())):
            td = pending[turn_num]
            task = prm_tasks.get(turn_num)

            if task is None:
                if force_drop:
                    pending.pop(turn_num, None)
                    if self.config.eval_mode:
                        with self._eval_scores_lock:
                            self._eval_scores.append(0.0)
                    logger.info("[Server] dropped session=%s turn=%d (no next_state)", session_id, turn_num)
                continue
            if not task.done():
                continue

            pending.pop(turn_num, None)
            prm_tasks.pop(turn_num, None)
            try:
                opd_result = task.result()
            except Exception as e:
                logger.error("[Server] OPD task FAILED session=%s turn=%d: %s", session_id, turn_num, e, exc_info=True)
                if self.config.eval_mode:
                    with self._eval_scores_lock:
                        self._eval_scores.append(0.0)
                continue

            if self.config.eval_mode:
                es = opd_result.get("eval_score")
                if es is not None:
                    with self._eval_scores_lock:
                        self._eval_scores.append(es)

            if not opd_result.get("accepted"):
                self._append_score_record({
                    "session_id": session_id, "turn": turn_num,
                    "accepted": False, "hint": "",
                    "hint_raw": opd_result.get("hint_raw", ""),
                    "eval_raw": opd_result.get("eval_raw", ""),
                })
                continue

            hint = opd_result.get("hint", "")
            logger.info("[Server] OPD hint session=%s turn=%d:\n%s", session_id, turn_num, hint)
            self._append_score_record({
                "session_id": session_id, "turn": turn_num,
                "accepted": True, "hint": hint,
                "hint_len": len(hint),
                "hint_raw": opd_result.get("hint_raw", ""),
                "eval_raw": opd_result.get("eval_raw", ""),
            })
            self._safe_create_task(self._submit_turn_sample(td, session_id, opd_result))

    async def _submit_turn_sample(self, turn_data, session_id, opd_result):
        prompt_ids = turn_data["prompt_ids"]
        response_ids = turn_data["response_ids"]

        teacher_lps = opd_result.get("teacher_log_probs") or []
        if len(teacher_lps) > len(response_ids):
            teacher_lps = teacher_lps[:len(response_ids)]
        elif len(teacher_lps) < len(response_ids):
            teacher_lps = teacher_lps + [0.0] * (len(response_ids) - len(teacher_lps))

        sample = TrainingSample(
            session_id=session_id, turn_num=self._turn_counts.get(session_id, 0),
            prompt_tokens=prompt_ids, response_tokens=response_ids,
            response_logprobs=turn_data["response_logprobs"],
            loss_mask=[1] * len(response_ids), reward=1.0,
            prompt_text=turn_data.get("prompt_text", ""),
            response_text=turn_data.get("response_text", ""),
            teacher_logprobs=teacher_lps,
        )

        index = next(self._index_counter)
        group_index = next(self._group_counter)
        logger.info("[Server] submitted OPD session=%s idx=%d hint_len=%d",
                    session_id, index, len(opd_result.get("hint", "")))
        await asyncio.to_thread(self.output_queue.put, (group_index, [sample]))


# ===========================================================================
# Combined (OPD + RL) Server
# ===========================================================================

class OpenClawCombineServer(_BaseServer):
    """Proxy for Combined OPD + RL training with three-way dispatch.

    Dispatch rules:
      - hint accepted AND eval +/-1 -> OPD+RL combined sample
      - hint accepted, eval neutral -> OPD-only sample (reward=0)
      - no hint, eval +/-1          -> RL-only sample (teacher=student)
      - no hint, eval neutral       -> nothing
    """

    _TITLE = "OpenClaw-Combine Tinker Proxy"
    _SCORE_FILE = "combine_scores.jsonl"

    def __init__(self, config, output_queue, submission_enabled,
                 sampling_client=None, scorer=None):
        super().__init__(config, output_queue, submission_enabled, sampling_client)
        self.scorer = scorer

    def _print_ready(self):
        time.sleep(3)
        banner = (
            f"\n{'=' * 60}\n"
            f"  {self._TITLE} ready\n"
            f"  {self.config.proxy_host}:{self.config.proxy_port} -> Tinker cloud\n"
            f"  Method: combine\n"
            f"  PRM/Teacher: Tinker SamplingClient (m={self.config.prm_m})\n"
            f"  Weights: w_opd={self.config.w_opd} w_rl={self.config.w_rl}\n"
            f"{'=' * 60}\n"
        )
        logger.info("%s%s%s", _GREEN, banner, _RESET)

    def _fire_evaluation_task(self, session_id, turn_num, turn_data, next_state):
        if not self.scorer or not next_state:
            return
        task = asyncio.create_task(
            self.scorer.evaluate(
                response_text=turn_data["response_text"],
                next_state_text=_flatten_content(next_state.get("content")),
                next_state_role=next_state.get("role", "user"),
                turn_data=turn_data, tokenizer=self._tokenizer,
                normalize_fn=_normalize_messages,
                session_id=session_id, turn_num=turn_num,
            )
        )
        task.add_done_callback(self._task_done_cb)
        task.add_done_callback(lambda _t: self._maybe_submit_ready_samples(session_id))
        self._prm_tasks.setdefault(session_id, {})[turn_num] = task
        turn_data["has_next_state"] = True

    async def _handle_request(self, body, session_id, turn_type, session_done):
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
            forward_body["model"] = self.config.served_model_name

        output = await self._forward_to_tinker(forward_body)
        choice = output.get("choices", [{}])[0]
        assistant_msg = choice.get("message", {})
        tool_calls = assistant_msg.get("tool_calls") or []
        content = assistant_msg.get("content") or ""
        reasoning = assistant_msg.get("reasoning_content") or ""

        logger.info("%s[Server] [%s] session=%s prompt_msgs=%d%s",
                    _YELLOW, turn_type, session_id, len(messages), _RESET)
        logger.info("%s[Server] [%s] session=%s thinking=%d response:\n%s%s",
                    _RED, turn_type, session_id, len(reasoning), content[:500], _RESET)

        if turn_type == "main":
            prev_turn_num = self._turn_counts.get(session_id, 0)

            if prev_turn_num > 0 and messages:
                self._flush_pending_record(session_id, messages[-1])
                prev_td = self._pending_turn_data.get(session_id, {}).get(prev_turn_num)
                if prev_td is not None:
                    self._fire_evaluation_task(session_id, prev_turn_num, prev_td, messages[-1])

            prompt_ids, response_ids, response_logprobs, prompt_text, response_text = \
                self._tokenize_turn(messages, assistant_msg, tools, choice, output=output)

            if not response_ids and not response_text.strip():
                output["session_id"] = session_id
                return {"response": output}

            self._turn_counts[session_id] = prev_turn_num + 1
            turn_num = self._turn_counts[session_id]
            turn_data = {
                "prompt_ids": prompt_ids, "response_ids": response_ids,
                "response_logprobs": response_logprobs,
                "prompt_text": prompt_text, "response_text": response_text,
                "messages": messages, "tools": tools,
                "has_next_state": False,
            }
            self._pending_turn_data.setdefault(session_id, {})[turn_num] = turn_data
            self._buffer_record(session_id, turn_num, messages, prompt_text, response_text, tool_calls)
            logger.info("[Server] MAIN session=%s turn=%d prompt=%d response=%d",
                        session_id, turn_num, len(prompt_ids), len(response_ids))
            self._maybe_submit_ready_samples(session_id)
        else:
            logger.info("[Server] SIDE session=%s -> skipped", session_id)

        if session_done:
            self._flush_pending_record(session_id, None)
            self._maybe_submit_ready_samples(session_id, force_drop=True)
            self._turn_counts.pop(session_id, None)
            logger.info("[Server] session=%s done", session_id)

        output["session_id"] = session_id
        return {"response": output}

    @staticmethod
    def _is_valid_rl_score(score) -> bool:
        return score in (1, -1, 1.0, -1.0)

    def _maybe_submit_ready_samples(self, session_id, force_drop=False):
        prm_tasks = self._prm_tasks.get(session_id, {})
        pending = self._pending_turn_data.get(session_id, {})
        for turn_num in sorted(list(pending.keys())):
            td = pending[turn_num]
            task = prm_tasks.get(turn_num)

            if task is None:
                if force_drop:
                    pending.pop(turn_num, None)
                    with self._eval_scores_lock:
                        self._eval_scores.append(0.0)
                    logger.info("[Server] dropped session=%s turn=%d (no next_state)", session_id, turn_num)
                continue
            if not task.done():
                continue

            pending.pop(turn_num, None)
            prm_tasks.pop(turn_num, None)
            try:
                result = task.result()
            except Exception as e:
                logger.error("[Server] evaluation FAILED session=%s turn=%d: %s", session_id, turn_num, e, exc_info=True)
                with self._eval_scores_lock:
                    self._eval_scores.append(0.0)
                continue

            eval_score = result.get("eval_score")
            if eval_score is not None:
                with self._eval_scores_lock:
                    self._eval_scores.append(eval_score)

            opd_accepted = result.get("accepted")
            has_valid_rl = self._is_valid_rl_score(eval_score)

            hint = result.get("hint", "")
            hint_raw = result.get("hint_raw", "")
            eval_raw = result.get("eval_raw", "")

            if opd_accepted and has_valid_rl:
                logger.info("[Server] OPD+RL hint session=%s turn=%d:\n%s", session_id, turn_num, hint)
                self._safe_create_task(
                    self._submit_opd_sample(td, session_id, result, reward=float(eval_score))
                )
                self._append_score_record({
                    "session_id": session_id, "turn": turn_num,
                    "type": "opd+rl", "eval_score": eval_score,
                    "hint": hint, "hint_len": len(hint),
                    "hint_raw": hint_raw, "eval_raw": eval_raw,
                })
            elif opd_accepted:
                logger.info("[Server] OPD hint session=%s turn=%d:\n%s", session_id, turn_num, hint)
                self._safe_create_task(
                    self._submit_opd_sample(td, session_id, result, reward=0.0)
                )
                self._append_score_record({
                    "session_id": session_id, "turn": turn_num,
                    "type": "opd", "eval_score": eval_score,
                    "hint": hint, "hint_len": len(hint),
                    "hint_raw": hint_raw, "eval_raw": eval_raw,
                })
            elif has_valid_rl:
                self._safe_create_task(
                    self._submit_rl_sample(td, session_id, float(eval_score))
                )
                self._append_score_record({
                    "session_id": session_id, "turn": turn_num,
                    "type": "rl", "eval_score": eval_score,
                    "eval_raw": eval_raw,
                })
            else:
                logger.info("[Server] no signal session=%s turn=%d", session_id, turn_num)

    async def _submit_opd_sample(self, turn_data, session_id, opd_result, reward=0.0):
        prompt_ids = turn_data["prompt_ids"]
        response_ids = turn_data["response_ids"]

        teacher_lps = opd_result.get("teacher_log_probs") or []
        if len(teacher_lps) > len(response_ids):
            teacher_lps = teacher_lps[:len(response_ids)]
        elif len(teacher_lps) < len(response_ids):
            teacher_lps = teacher_lps + [0.0] * (len(response_ids) - len(teacher_lps))

        tag = "OPD+RL" if reward != 0.0 else "OPD"
        sample = TrainingSample(
            session_id=session_id, turn_num=self._turn_counts.get(session_id, 0),
            prompt_tokens=prompt_ids, response_tokens=response_ids,
            response_logprobs=turn_data["response_logprobs"],
            loss_mask=[1] * len(response_ids), reward=reward,
            prompt_text=turn_data.get("prompt_text", ""),
            response_text=turn_data.get("response_text", ""),
            teacher_logprobs=teacher_lps, sample_type=tag.lower(),
        )

        index = next(self._index_counter)
        group_index = next(self._group_counter)
        logger.info("[Server] submitted %s session=%s idx=%d reward=%.1f hint_len=%d",
                    tag, session_id, index, reward, len(opd_result.get("hint", "")))
        await asyncio.to_thread(self.output_queue.put, (group_index, [sample]))

    async def _submit_rl_sample(self, turn_data, session_id, eval_score):
        prompt_ids = turn_data["prompt_ids"]
        response_ids = turn_data["response_ids"]
        response_logprobs = turn_data["response_logprobs"]

        if len(response_logprobs) > len(response_ids):
            response_logprobs = response_logprobs[:len(response_ids)]
        elif len(response_logprobs) < len(response_ids):
            response_logprobs = response_logprobs + [0.0] * (len(response_ids) - len(response_logprobs))

        sample = TrainingSample(
            session_id=session_id, turn_num=self._turn_counts.get(session_id, 0),
            prompt_tokens=prompt_ids, response_tokens=response_ids,
            response_logprobs=response_logprobs,
            loss_mask=[1] * len(response_ids), reward=eval_score,
            prompt_text=turn_data.get("prompt_text", ""),
            response_text=turn_data.get("response_text", ""),
            teacher_logprobs=list(response_logprobs),  # teacher = student -> OPD advantage = 0
            sample_type="rl",
        )

        index = next(self._index_counter)
        group_index = next(self._group_counter)
        logger.info("[Server] submitted RL session=%s idx=%d score=%.1f",
                    session_id, index, eval_score)
        await asyncio.to_thread(self.output_queue.put, (group_index, [sample]))
