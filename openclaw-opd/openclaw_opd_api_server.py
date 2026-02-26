import asyncio
import copy
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
_RESET = "\033[0m"
logger = logging.getLogger(__name__)

_BOXED_RE = re.compile(r"\\boxed\{([-+]?\d)\}")
_HINT_RE = re.compile(r"\[HINT_START\](.*?)\[HINT_END\]", re.DOTALL)

_NON_STANDARD_BODY_KEYS = {"session_id", "session_done", "turn_type"}


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


def _build_hint_judge_messages(response_text: str, next_state_text: str) -> list[dict]:
    system = (
        "You are a process reward model used for hindsight hint extraction.\n"
        "You are given:\n"
        "1) The assistant response at turn t.\n"
        "2) The next state at turn t+1 (user reply or environment feedback).\n"
        "Your goal is to decide whether the next state reveals useful hindsight information\n"
        "that could have helped improve the assistant response at turn t.\n\n"
        "Output format rules (strict):\n"
        "- You MUST include exactly one final decision token: \\boxed{1} or \\boxed{-1}.\n"
        "- If and only if decision is \\boxed{1}, provide a concise, information-dense hint in 1-3 sentences,\n"
        "  wrapped between [HINT_START] and [HINT_END].\n"
        "- If decision is \\boxed{-1}, do not provide a hint block.\n"
        "- Hint must be concrete and actionable for improving the previous response."
    )
    user = (
        f"## Assistant response (turn t)\n{response_text}\n\n"
        f"## Next state (turn t+1)\n{next_state_text}\n\n"
        "Now output your decision and (if positive) the hint in the required format."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _parse_judge_result(text: str) -> tuple[int | None, str]:
    boxed = _BOXED_RE.findall(text)
    score = int(boxed[-1]) if boxed else None
    if score not in (1, -1):
        score = None
    hint_matches = _HINT_RE.findall(text)
    hint = hint_matches[-1].strip() if hint_matches else ""
    return score, hint


def _select_best_hint(votes: list[dict[str, Any]]) -> dict[str, Any] | None:
    good = [
        v for v in votes
        if v.get("score") == 1 and isinstance(v.get("hint"), str) and len(v["hint"].strip()) > 10
    ]
    if not good:
        return None
    return max(good, key=lambda v: len(v["hint"].strip()))


def _append_hint_to_messages(messages: list[dict], hint: str) -> list[dict]:
    cloned = copy.deepcopy(messages)
    if not cloned:
        return [{"role": "user", "content": f"[user's hint / instruction]\n{hint}"}]

    target_idx = None
    for i in range(len(cloned) - 1, -1, -1):
        if cloned[i].get("role") == "user":
            target_idx = i
            break
    if target_idx is None:
        target_idx = len(cloned) - 1

    content = _flatten_message_content(cloned[target_idx].get("content"))
    suffix = f"\n\n[user's hint / instruction]\n{hint.strip()}"
    cloned[target_idx]["content"] = (content + suffix).strip()
    return cloned


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


class OpenClawOPDAPIServer:
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
        self._prm_tasks: dict[str, dict[int, asyncio.Task]] = {}
        self._pending_records: dict[str, dict[str, Any]] = {}

        self._prm_enabled = getattr(args, "prm_enable", False)
        self._prm_m = int(os.getenv("PRM_M", getattr(args, "prm_m", 3)))
        self._prm_temperature = float(getattr(args, "prm_temperature", 0.6))
        self._prm_max_tokens = int(getattr(args, "prm_max_new_tokens", 4096))
        self._teacher_lp_max_concurrency = int(os.getenv("OPENCLAW_OPD_TEACHER_LP_MAX_CONCURRENCY", "3"))
        self._teacher_lp_semaphore = asyncio.Semaphore(max(1, self._teacher_lp_max_concurrency))
        prm_ip = getattr(args, "prm_router_ip", None)
        prm_port = getattr(args, "prm_router_port", None)
        self._prm_url = f"http://{prm_ip}:{prm_port}/generate" if prm_ip and prm_port else ""
        self._prm_tokenizer = None
        if self._prm_enabled:
            prm_path = getattr(args, "prm_model_path", None) or args.hf_checkpoint
            self._prm_tokenizer = load_tokenizer(prm_path, trust_remote_code=True)
            logger.info("[OpenClaw-OPD] PRM enabled: url=%s m=%d", self._prm_url, self._prm_m)

        self._record_file = os.getenv("OPENCLAW_RECORD_FILE", "") if os.getenv("OPENCLAW_RECORD_ENABLED", "0") == "1" else ""
        if self._record_file:
            os.makedirs(os.path.dirname(self._record_file), exist_ok=True)
            open(self._record_file, "w").close()
            logger.info("[OpenClaw-OPD] record file initialized (cleared): %s", self._record_file)

        self._prm_record_file = os.getenv("OPENCLAW_PRM_RECORD_FILE", "")
        if not self._prm_record_file and self._record_file and self._prm_enabled:
            base, ext = os.path.splitext(self._record_file)
            self._prm_record_file = f"{base}_prm{ext}"
        if self._prm_record_file:
            os.makedirs(os.path.dirname(self._prm_record_file), exist_ok=True)
            open(self._prm_record_file, "w").close()
            logger.info("[OpenClaw-OPD] PRM record file initialized (cleared): %s", self._prm_record_file)

        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self.app = self._build_app()

    def _build_app(self) -> FastAPI:
        app = FastAPI(title="OpenClaw OPD Proxy")
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
            owner: OpenClawOPDAPIServer = request.app.state.owner
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

        return app

    async def _check_auth(self, authorization: str | None):
        if not self.expected_api_key:
            return
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="missing bearer token")
        token = authorization.split(" ", 1)[1].strip()
        if token != self.expected_api_key:
            raise HTTPException(status_code=401, detail="invalid api key")

    def _buffer_record(
        self,
        session_id: str,
        turn_num: int,
        messages: list[dict[str, Any]],
        prompt_text: str,
        response_text: str,
        tool_calls: list[dict[str, Any]],
    ):
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

    def _flush_pending_record(self, session_id: str, next_state: dict[str, Any] | None):
        rec = self._pending_records.pop(session_id, None)
        if rec is None:
            return
        rec["next_state"] = next_state
        if self._record_file:
            try:
                with open(self._record_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except OSError as e:
                logger.warning("[OpenClaw-OPD] failed to write record: %s", e)

    def _append_prm_record(self, record: dict[str, Any]):
        if not self._prm_record_file:
            return
        try:
            with open(self._prm_record_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as e:
            logger.warning("[OpenClaw-OPD] failed to write PRM record: %s", e)

    async def _query_judge_once(self, judge_prompt: str, vote_id: int) -> dict[str, Any]:
        if not self._prm_url:
            return {"vote_id": vote_id, "score": None, "hint": "", "raw": ""}
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
            raw = str(raw)
            score, hint = _parse_judge_result(raw)
            return {"vote_id": vote_id, "score": score, "hint": hint, "raw": raw}
        except Exception as e:
            logger.warning("[OpenClaw-OPD] judge query failed (vote %d): %s", vote_id, e)
            return {"vote_id": vote_id, "score": None, "hint": "", "raw": ""}

    async def _compute_teacher_log_probs(self, input_ids: list[int], response_len: int) -> list[float]:
        # Only request logprobs for the response suffix to reduce peak memory.
        # This mirrors OPD need (token-level target on response tokens) and avoids
        # full-sequence logprob materialization on long contexts.
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

    async def _opd_evaluate(self, session_id: str, turn_num: int, turn_data: dict[str, Any], next_state: dict[str, Any]) -> dict[str, Any]:
        next_state_text = _flatten_message_content(next_state.get("content")) if next_state else ""
        judge_msgs = _build_hint_judge_messages(turn_data["response_text"], next_state_text)
        if self._prm_tokenizer:
            judge_prompt = self._prm_tokenizer.apply_chat_template(judge_msgs, tokenize=False, add_generation_prompt=True)
        else:
            judge_prompt = "\n".join(m["content"] for m in judge_msgs)

        votes = await asyncio.gather(*[self._query_judge_once(judge_prompt, i) for i in range(self._prm_m)])
        selected = _select_best_hint(votes)
        votes_display = [v.get("score", "fail") for v in votes]

        if selected is None:
            logger.info(
                "%s[OpenClaw-OPD] session=%s turn=%d no valid hint (votes=%s), sample dropped%s",
                _CYAN,
                session_id,
                turn_num,
                votes_display,
                _RESET,
            )
            self._append_prm_record(
                {
                    "session_id": session_id,
                    "turn": turn_num,
                    "accepted": False,
                    "hint": "",
                    "votes": votes,
                }
            )
            return {"accepted": False, "teacher_log_probs": None, "hint": "", "votes": votes}

        hint = selected["hint"].strip()
        enhanced_messages = _append_hint_to_messages(turn_data["messages"], hint)
        norm_enhanced = _normalize_messages_for_template(enhanced_messages)
        enhanced_prompt_text = self.tokenizer.apply_chat_template(
            norm_enhanced,
            tools=turn_data.get("tools"),
            tokenize=False,
            add_generation_prompt=True,
        )

        enhanced_full_text = enhanced_prompt_text + turn_data["response_text"]
        enhanced_ids = self.tokenizer(enhanced_full_text, add_special_tokens=False)["input_ids"]
        teacher_log_probs = await self._compute_teacher_log_probs(enhanced_ids, len(turn_data["response_ids"]))

        logger.info(
            "%s[OpenClaw-OPD] session=%s turn=%d accepted hint_len=%d votes=%s%s",
            _CYAN,
            session_id,
            turn_num,
            len(hint),
            votes_display,
            _RESET,
        )
        self._append_prm_record(
            {
                "session_id": session_id,
                "turn": turn_num,
                "accepted": True,
                "hint": hint,
                "hint_len": len(hint),
                "votes": votes,
                "teacher_logprob_len": len(teacher_log_probs),
            }
        )
        return {"accepted": True, "teacher_log_probs": teacher_log_probs, "hint": hint, "votes": votes}

    def _fire_opd_task(self, session_id: str, turn_num: int, turn_data: dict[str, Any], next_state: dict[str, Any]):
        if not self._prm_enabled or not next_state:
            return
        task = asyncio.create_task(self._opd_evaluate(session_id, turn_num, turn_data, next_state))
        task.add_done_callback(self._task_done_cb)
        task.add_done_callback(lambda _t: self._maybe_submit_ready_samples(session_id))
        self._prm_tasks.setdefault(session_id, {})[turn_num] = task
        turn_data["has_next_state"] = True

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
                logger.error("[OpenClaw-OPD] SGLang returned %d: %s", sglang_resp.status_code, sglang_resp.text[:1000])
                sglang_resp.raise_for_status()
            output = sglang_resp.json()

        choice = output.get("choices", [{}])[0]
        assistant_msg = choice.get("message", {})
        tool_calls = assistant_msg.get("tool_calls") or []
        content = assistant_msg.get("content") or ""
        reasoning = assistant_msg.get("reasoning_content") or ""
        logger.info(
            "%s[OpenClaw-OPD] [%s] session=%s prompt_msgs=%d%s",
            _YELLOW,
            turn_type,
            session_id,
            len(messages),
            _RESET,
        )
        logger.info(
            "%s[OpenClaw-OPD] [%s] session=%s thinking=%d chars, response:\n%s%s",
            _RED,
            turn_type,
            session_id,
            len(reasoning),
            content,
            _RESET,
        )
        if tool_calls:
            logger.info("[OpenClaw-OPD] tool_calls: %s", str(tool_calls)[:500])

        if turn_type == "main":
            prev_turn_num = self._turn_counts.get(session_id, 0)
            if prev_turn_num > 0 and messages:
                self._flush_pending_record(session_id, messages[-1])
                prev_turn_data = self._pending_turn_data.get(session_id, {}).get(prev_turn_num)
                if prev_turn_data is not None:
                    self._fire_opd_task(session_id, prev_turn_num, prev_turn_data, messages[-1])

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
                logger.info("[OpenClaw-OPD] MAIN session=%s -> empty response, skipping", session_id)
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
            logger.info(
                "[OpenClaw-OPD] MAIN session=%s turn=%d prompt_tokens=%d response_tokens=%d",
                session_id,
                turn_num,
                len(prompt_ids),
                len(response_ids),
            )
            self._maybe_submit_ready_samples(session_id)
        else:
            logger.info("[OpenClaw-OPD] SIDE session=%s -> skipped (no training data)", session_id)

        if session_done:
            self._flush_pending_record(session_id, None)
            self._maybe_submit_ready_samples(session_id, force_drop_without_next_state=True)
            self._turn_counts.pop(session_id, None)
            logger.info("[OpenClaw-OPD] session=%s done -> cleaned up", session_id)

        output["session_id"] = session_id
        return {"response": output}

    def _maybe_submit_ready_samples(self, session_id: str, force_drop_without_next_state: bool = False):
        prm_tasks = self._prm_tasks.get(session_id, {})
        pending = self._pending_turn_data.get(session_id, {})
        for turn_num in sorted(list(pending.keys())):
            td = pending[turn_num]
            task = prm_tasks.get(turn_num)

            if task is None:
                if force_drop_without_next_state:
                    pending.pop(turn_num, None)
                    logger.info(
                        "[OpenClaw-OPD] dropped session=%s turn=%d (no next_state -> no hint teacher)",
                        session_id,
                        turn_num,
                    )
                continue
            if not task.done():
                continue

            pending.pop(turn_num, None)
            prm_tasks.pop(turn_num, None)
            try:
                opd_result = task.result()
            except Exception as e:
                logger.warning("[OpenClaw-OPD] opd task failed session=%s turn=%d: %s", session_id, turn_num, e)
                continue

            if not opd_result.get("accepted"):
                continue
            self._safe_create_task(self._submit_turn_sample(td, session_id, opd_result))

    async def _submit_turn_sample(self, turn_data: dict[str, Any], session_id: str, opd_result: dict[str, Any]):
        prompt_ids = turn_data["prompt_ids"]
        response_ids = turn_data["response_ids"]
        teacher_log_probs = opd_result.get("teacher_log_probs") or []
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
        sample.status = Sample.Status.COMPLETED
        sample.index = next(self._index_counter)
        sample.group_index = next(self._group_counter)
        sample.reward = {"score": 1.0}

        logger.info(
            "[OpenClaw-OPD] submitted sample session=%s index=%d prompt_len=%d response_len=%d hint_len=%d",
            session_id,
            sample.index,
            len(prompt_ids),
            len(response_ids),
            len(opd_result.get("hint", "")),
        )
        await asyncio.to_thread(self.output_queue.put, (sample.group_index, [sample]))

    def purge_record_files(self):
        for path, label in [
            (self._record_file, "record"),
            (self._prm_record_file, "PRM record"),
        ]:
            if not path:
                continue
            try:
                open(path, "w").close()
                logger.info("[OpenClaw-OPD] %s file purged: %s", label, path)
            except OSError as e:
                logger.warning("[OpenClaw-OPD] failed to purge %s file: %s", label, e)

    def _safe_create_task(self, coro):
        task = asyncio.create_task(coro)
        task.add_done_callback(self._task_done_cb)

    @staticmethod
    def _task_done_cb(task: asyncio.Task):
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error("[OpenClaw-OPD] background task failed: %s", exc, exc_info=exc)

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
        logger.info("[OpenClaw-OPD] policy server ready")

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
            logger.info("[OpenClaw-OPD] PRM/teacher server ready")

        time.sleep(8)
        banner = (
            f"\n{'=' * 70}\n"
            f"  [OpenClaw-OPD] model is ready\n"
            f"  proxy {self.host}:{self.port} -> SGLang {self.args.sglang_router_ip}:{self.args.sglang_router_port}\n"
            f"  PRM/teacher {self._prm_url} (m={self._prm_m})\n"
            f"{'=' * 70}\n"
        )
        logger.info(f"{_GREEN}{banner}{_RESET}")

    def stop(self):
        if self._server is not None:
            self._server.should_exit = True
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
