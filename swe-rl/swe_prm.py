"""SWE PRM (Process Reward Model) for step-wise evaluation.

Evaluates each step of the SWE agent trajectory using a separate LLM
judge. Each step is scored +1 (good) or -1 (bad) via m-voting, providing
per-step reward signal for step-wise GRPO advantage.

Used by generate_with_swe_remote.py when --prm-enable is set.
"""

import asyncio
import copy
import logging
import os
import re
import time
from typing import Any, Dict, List, Tuple

from slime.utils.http_utils import post
from slime.utils.processing_utils import load_tokenizer

logger = logging.getLogger(__name__)

_ANSI_RED = "\033[91m"
_ANSI_RESET = "\033[0m"


# ---------------------------------------------------------------------------
# PRM prompt templates
# ---------------------------------------------------------------------------

SWE_PRM_SYSTEM_PROMPT = """
You are a strict evaluator for a software engineering agent that fixes GitHub issues.
You are given:
1) The issue description (problem statement).
2) The agent's recent action history.
3) The agent's most recent step (THOUGHT + bash command) to evaluate.
4) The execution result of that command (returncode + stdout/stderr).
""".strip()

SWE_PRM_INSTRUCTION_TEMPLATE = """\
## Issue Description
{problem_statement}

## Recent History ({n_history} steps)
{history_summary}

## Current Step to Evaluate (step {step_num})

Agent's full response:
{policy_response}

Execution result (returncode={returncode}):
{command_output}
"""

SWE_PRM_SCORING_TEMPLATE = """\
Evaluate ONLY the single most recent step above.

Assign a score of +1 if ALL of the following are true:
- The command executed without unexpected errors (returncode=0, or expected non-zero like grep with no match);
- The step is clearly relevant to diagnosing or fixing the stated issue;
- The output provides useful information OR the edit makes a logically correct change toward fixing the bug.

Otherwise assign a score of -1, for example if:
- The command fails with an unexpected error (wrong path, syntax error, missing tool);
- The step is irrelevant to the issue (wrong file, wrong concept, unnecessary exploration);
- The agent is going in circles (repeating a previously failed command or approach);
- The edit introduces an obvious bug or does not address the actual issue;
- The agent is wasting steps (e.g. reading files already fully examined).

Think carefully step by step, then provide your reasoning and put the final score in \\boxed{}.
"""


# ---------------------------------------------------------------------------
# Score extraction (shared with GUI PRM)
# ---------------------------------------------------------------------------

_PRM_BOXED_PATTERN = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", re.DOTALL)
_PRM_STRICT_NUMBER_PATTERN = re.compile(r"^\s*([-+]?\d+(?:\.\d+)?)\s*$")


def _extract_prm_sign_from_text(text: str) -> int:
    if not text:
        return 0
    match = _PRM_BOXED_PATTERN.search(text)
    if not match:
        return 0
    boxed_content = match.group(1).strip()
    strict_number_match = _PRM_STRICT_NUMBER_PATTERN.fullmatch(boxed_content)
    if not strict_number_match:
        return 0
    try:
        value = float(strict_number_match.group(1))
    except ValueError:
        return 0
    if abs(value - 1.0) < 1e-9:
        return 1
    if abs(value + 1.0) < 1e-9:
        return -1
    return 0


# ---------------------------------------------------------------------------
# SweRewardAgent
# ---------------------------------------------------------------------------

class SweRewardAgent:
    def __init__(
        self,
        max_history_steps: int = 8,
        max_problem_len: int = 8000,
        max_output_len: int = 4000,
        max_history_output_len: int = 1000,
        skip_submit: bool = True,
        tokenizer: Any | None = None,
    ):
        self.max_history_steps = max(1, int(max_history_steps))
        self.max_problem_len = int(max_problem_len)
        self.max_output_len = int(max_output_len)
        self.max_history_output_len = int(max_history_output_len)
        self.skip_submit = skip_submit
        self.tokenizer = tokenizer
        self.reward_trajectory: list[dict[str, Any]] = []
        self._prm_semaphore: asyncio.Semaphore | None = None
        self._formatter_ready = False

    # -- formatter lazy-init ---------------------------------------------------

    def _ensure_formatter(self, args: Any) -> None:
        if self._formatter_ready:
            return
        self._formatter_ready = True
        prm_model_path = getattr(args, "prm_model_path", None)
        if not prm_model_path:
            return
        if self.tokenizer is None:
            self.tokenizer = load_tokenizer(prm_model_path, trust_remote_code=True)

    # -- concurrency -----------------------------------------------------------

    def _get_prm_semaphore(self, args: Any) -> asyncio.Semaphore:
        if self._prm_semaphore is None:
            prm_num_gpus = max(1, int(getattr(args, "prm_num_gpus", 1)))
            prm_num_gpus_per_engine = max(1, int(getattr(args, "prm_num_gpus_per_engine", 1)))
            engine_count = max(1, prm_num_gpus // prm_num_gpus_per_engine)
            base_concurrency = max(1, int(getattr(args, "sglang_server_concurrency", 512)) * engine_count)
            cap = int(getattr(args, "prm_max_concurrency", os.getenv("SWE_PRM_MAX_CONCURRENCY", "32")))
            self._prm_semaphore = asyncio.Semaphore(min(base_concurrency, cap) if cap > 0 else base_concurrency)
        return self._prm_semaphore

    # -- output formatting -----------------------------------------------------

    @staticmethod
    def _format_output(d: dict, max_chars: int = 4000) -> str:
        """De-duplicate and combine output_head + output_tail."""
        output_len = d.get("output_len", 0)
        head = d.get("output_head", "")
        tail = d.get("output_tail", "")

        if output_len <= 2000:
            text = head
        elif output_len <= 4000:
            overlap = 4000 - output_len
            text = head + tail[overlap:]
        else:
            gap = output_len - 4000
            text = f"{head}\n... ({gap} characters omitted) ...\n{tail}"

        if len(text) > max_chars:
            text = text[:max_chars] + f"\n... (truncated to {max_chars} chars)"
        return text

    # -- prompt construction ---------------------------------------------------

    def _build_reward_messages(
        self,
        *,
        problem_statement: str,
        step_debug: list[dict],
        policy_response: str,
        step_index: int,
    ) -> List[Dict[str, Any]]:
        step_index = int(step_index)
        rstart = max(0, step_index - self.max_history_steps)

        history_lines: list[str] = []
        for i in range(rstart, step_index):
            if i >= len(step_debug):
                break
            d = step_debug[i]
            output_preview = self._format_output(d, max_chars=self.max_history_output_len)
            history_lines.append(
                f"Step {i + 1}: $ {d.get('action', '?')}\n"
                f"  returncode={d.get('returncode', '?')}\n"
                f"  output:\n{output_preview}"
            )
        history_summary = "\n\n".join(history_lines) if history_lines else "None (this is the first step)"

        cur = step_debug[step_index] if step_index < len(step_debug) else {}
        command_output = self._format_output(cur, max_chars=self.max_output_len)

        problem_text = problem_statement
        if len(problem_text) > self.max_problem_len:
            problem_text = problem_text[: self.max_problem_len] + "\n... (truncated)"

        user_content = SWE_PRM_INSTRUCTION_TEMPLATE.format(
            problem_statement=problem_text,
            n_history=len(history_lines),
            history_summary=history_summary,
            step_num=step_index + 1,
            policy_response=policy_response,
            returncode=cur.get("returncode", "?"),
            command_output=command_output,
        ) + "\n" + SWE_PRM_SCORING_TEMPLATE

        return [
            {"role": "system", "content": SWE_PRM_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    # -- SGLang payload --------------------------------------------------------

    def _build_payload(
        self,
        *,
        args: Any,
        messages: List[Dict[str, Any]],
        vote_id: int,
    ) -> Tuple[Dict[str, Any], int]:
        if self.tokenizer is None:
            raise RuntimeError("PRM tokenizer is not initialized.")

        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        sampling_params = {
            "temperature": float(getattr(args, "prm_temperature", 1.0) or 1.0),
            "top_p": 1.0,
            "top_k": -1,
            "max_new_tokens": int(getattr(args, "prm_max_new_tokens", 4096) or 4096),
            "skip_special_tokens": False,
            "no_stop_trim": True,
            "spaces_between_special_tokens": False,
            "sampling_seed": int(getattr(args, "rollout_seed", 42)) * 1000 + int(vote_id),
        }

        input_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        payload: Dict[str, Any] = {
            "input_ids": input_ids,
            "sampling_params": sampling_params,
            "return_logprob": True,
        }
        return payload, len(input_ids)

    # -- decode ----------------------------------------------------------------

    def _decode_text(self, output: Any) -> str:
        if not isinstance(output, dict):
            return "" if output is None else str(output)
        meta = output.get("meta_info", {})
        if isinstance(meta, dict) and self.tokenizer is not None:
            otlp = meta.get("output_token_logprobs")
            if isinstance(otlp, list) and otlp:
                output_tokens = [item[1] for item in otlp]
                return self.tokenizer.decode(output_tokens)
        text = output.get("text")
        return text if isinstance(text, str) else ""

    # -- single PRM query ------------------------------------------------------

    async def _query_prm_once(
        self,
        args: Any,
        messages_for_prm: List[Dict[str, Any]],
        step_index: int,
        vote_id: int,
    ) -> Dict[str, Any]:
        prm_router_ip = getattr(args, "prm_router_ip", None)
        prm_router_port = getattr(args, "prm_router_port", None)
        if not prm_router_ip or not prm_router_port:
            return {"score": 0, "latency_ms": 0, "raw_text": "", "ok": False}

        payload, in_len = await asyncio.to_thread(
            self._build_payload,
            args=args,
            messages=messages_for_prm,
            vote_id=vote_id,
        )

        start = time.perf_counter()
        try:
            output = await post(
                f"http://{prm_router_ip}:{prm_router_port}/generate",
                payload,
                max_retries=30,
            )
        except Exception:
            return {"score": 0, "latency_ms": int((time.perf_counter() - start) * 1000), "raw_text": "", "ok": False}

        text = self._decode_text(output)
        logger.info(
            f"{_ANSI_RED}[SWE-PRM]{_ANSI_RESET} step=%s vote=%s in_len=%s output:\n%r",
            int(step_index), int(vote_id), in_len, text,
        )

        return {
            "score": _extract_prm_sign_from_text(text),
            "latency_ms": int((time.perf_counter() - start) * 1000),
            "raw_text": text,
            "ok": True,
        }

    # -- m-voting --------------------------------------------------------------

    async def _prm_vote(
        self,
        args: Any,
        messages_for_prm: List[Dict[str, Any]],
        step_index: int,
        m: int,
    ) -> Dict[str, Any]:
        semaphore = self._get_prm_semaphore(args)

        async def _single(vote_id: int) -> Dict[str, Any]:
            async with semaphore:
                return await self._query_prm_once(args, messages_for_prm, step_index, vote_id)

        votes = await asyncio.gather(*[_single(i) for i in range(max(1, m))])
        scores = [v["score"] for v in votes]
        valid_scores = [int(s) for s in scores if int(s) in (-1, 1)]
        return {
            "scores": scores,
            "valid_scores": valid_scores,
            "valid_vote_count": len(valid_scores),
            "mean_score": (sum(valid_scores) / len(valid_scores)) if valid_scores else 0.0,
            "votes": votes,
        }

    # -- public API: judge one step --------------------------------------------

    async def judge_step(
        self,
        args: Any,
        *,
        problem_statement: str,
        step_debug: list[dict],
        policy_response: str,
        step_index: int,
    ) -> Dict[str, Any]:
        if not getattr(args, "prm_enable", False):
            return {"status": "disabled", "step_index": step_index, "scores": [0], "mean_score": 0.0, "votes": []}
        if not getattr(args, "prm_router_ip", None) or not getattr(args, "prm_router_port", None):
            return {"status": "disabled_no_router", "step_index": step_index, "scores": [0], "mean_score": 0.0, "votes": []}

        self._ensure_formatter(args)

        reward_messages = self._build_reward_messages(
            problem_statement=problem_statement,
            step_debug=step_debug,
            policy_response=policy_response,
            step_index=int(step_index),
        )

        out = await self._prm_vote(
            args=args,
            messages_for_prm=reward_messages,
            step_index=int(step_index),
            m=max(1, int(getattr(args, "prm_m", 3))),
        )

        out["status"] = "ok"
        out["step_index"] = int(step_index)

        self.reward_trajectory.append({
            "step_index": int(step_index),
            "reward_messages": copy.deepcopy(reward_messages),
            "prm_scores": out.get("scores", []),
            "prm_mean_score": out.get("mean_score", 0.0),
        })
        return out

    # -- public API: async dispatch --------------------------------------------

    def submit_step_judge(
        self,
        args: Any,
        *,
        problem_statement: str,
        step_debug: list[dict],
        policy_response: str,
        step_index: int,
    ) -> asyncio.Task:
        return asyncio.create_task(
            self.judge_step(
                args=args,
                problem_statement=problem_statement,
                step_debug=step_debug,
                policy_response=policy_response,
                step_index=step_index,
            )
        )

    # -- public API: collect all results ---------------------------------------

    async def collect_step_results(
        self,
        pending_tasks: list[tuple[int, asyncio.Task]],
    ) -> tuple[list[float], list[dict[str, Any]]]:
        if not pending_tasks:
            return [], []
        done = await asyncio.gather(*[task for _, task in pending_tasks], return_exceptions=True)
        step_details: list[dict[str, Any]] = []
        for (step_idx, _), result in zip(pending_tasks, done, strict=False):
            if isinstance(result, Exception):
                logger.warning(f"[SWE-PRM] step {step_idx} exception: {result}")
                step_details.append(
                    {"status": "exception", "step_index": step_idx, "scores": [0], "mean_score": 0.0, "votes": []}
                )
                continue
            if not isinstance(result, dict):
                step_details.append(
                    {"status": "invalid_result", "step_index": step_idx, "scores": [0], "mean_score": 0.0, "votes": []}
                )
                continue
            result["step_index"] = int(step_idx)
            step_details.append(result)

        step_details.sort(key=lambda x: x.get("step_index", 10**9))
        step_scores = [float(item.get("mean_score", 0.0)) for item in step_details]
        return step_scores, step_details
