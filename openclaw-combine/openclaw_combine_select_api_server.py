"""Multi-candidate variant of OpenClawCombineAPIServer.

Drop-in replacement that, instead of selecting the SINGLE longest accepted
hint and shipping ONE ``teacher_tokens``, KEEPS ALL accepted hints
(deduped, shortest-first, capped at ``OPENCLAW_TOPK_MAX_CAND``),
materializes a ``teacher_tokens`` candidate per hint, and ships them as
``sample.teacher_tokens_candidates`` (a ``list[list[int]]``).

Slime's multi-candidate teacher infra detects ``teacher_tokens_candidates``
automatically (see ``slime/backends/megatron_utils/actor.py::
compute_prm_teacher_log_probs`` and ``_gather_at_indices_multi_cand``)
and runs the K-loop, producing ``prm_teacher_topk_log_probs_cand``,
``prm_teacher_topk_indices_cand``, and (under student subset mode)
``prm_teacher_native_topk_indices_cand`` -- exactly what the
``openclaw_topk_select_loss`` kernel consumes.

Architectural notes
-------------------
* Forces the megatron PRM-teacher path. The inference-side
  ``_compute_teacher_log_probs`` / ``_compute_teacher_topk_logprobs``
  computed by the parent are NOT used by the topk select kernel and
  are skipped here for efficiency. The launcher must export
  ``OPENCLAW_COMBINE_OPD_TEACHER_SOURCE=megatron`` and pass
  ``--prm-teacher-load <ckpt>``.

* Single-CoT semantics per Sample: each turn fires
  ``_submit_turn_sample`` (or ``_submit_rl_turn_sample``) -> ONE
  Sample per session-turn (the dynamic-history training paradigm; each
  step is its own training datum).  ``step_token_spans`` is
  intentionally absent from the Sample; ``sequence_optimal`` naturally
  collapses to one ``k*`` per Sample which IS the canonical per-PRM-step
  ``k*`` under dynamic-history rollout.

* RL-only turns (no hint accepted, eval ±1) ship
  ``teacher_tokens_candidates = [prompt_ids + response_ids]``
  (degenerate ``K_i=1``, un-enhanced).  Slime's K-loop cyclically
  reuses this single candidate up to ``K_max`` identical forwards.
  The loss kernel handles ``K_i=1`` correctly (``k*=0`` always for
  those samples).

* Hint ordering: candidates are sorted SHORTEST-first by ``len(hint)``,
  matching the retool / hint_opd_exp convention.  This makes
  ``--hint-selection shortest`` correspond to ``k*=0`` deterministically.

Knobs
-----
``OPENCLAW_TOPK_MAX_CAND``
    Max ``K_i`` candidate hints to keep per turn.  Default 3 (matches the
    retool / hint_opd_exp ``--hint-m`` default of 3 wired through the
    launcher).
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import torch

from openclaw_combine_api_server import OpenClawCombineAPIServer
from openclaw_opd_api_server import (
    _append_hint_to_messages,
    _build_hint_judge_messages,
    _build_prm_eval_prompt,
    _flatten_message_content,
    _normalize_messages_for_template,
    _prm_eval_majority_vote,
)
from slime.utils.types import Sample

logger = logging.getLogger(__name__)

_CYAN = "\033[96m"
_RESET = "\033[0m"


def _max_cand() -> int:
    return max(1, int(os.environ.get("OPENCLAW_TOPK_MAX_CAND", "3")))


class OpenClawCombineSelectAPIServer(OpenClawCombineAPIServer):
    """Multi-candidate hint variant of ``OpenClawCombineAPIServer``.

    The three-case dispatch table from the parent (OPD-only / OPD+RL /
    RL-only) is preserved; only the OPD evaluation step changes:

      * Parent: ``_select_best_hint(votes)`` -> single hint -> single
        ``teacher_tokens``.
      * Here  : keep ALL accepted votes (deduped, shortest-first,
        capped at ``OPENCLAW_TOPK_MAX_CAND``) -> list of ``teacher_tokens``
        candidates -> ``sample.teacher_tokens_candidates``.
    """

    async def _opd_evaluate(
        self,
        session_id: str,
        turn_num: int,
        turn_data: dict[str, Any],
        next_state: dict[str, Any],
    ) -> dict[str, Any]:
        next_state_text = (
            _flatten_message_content(next_state.get("content")) if next_state else ""
        )
        next_state_role = next_state.get("role", "user") if next_state else "user"
        judge_msgs = _build_hint_judge_messages(
            turn_data["response_text"], next_state_text, next_state_role,
        )
        if self._prm_tokenizer:
            judge_prompt = self._prm_tokenizer.apply_chat_template(
                judge_msgs, tokenize=False, add_generation_prompt=True,
            )
        else:
            judge_prompt = "\n".join(m["content"] for m in judge_msgs)

        votes = await asyncio.gather(
            *[self._query_judge_once(judge_prompt, i) for i in range(self._prm_m)]
        )

        # PRM eval branch (unchanged from parent).
        if self._eval_mode:
            eval_msgs = _build_prm_eval_prompt(
                turn_data["response_text"], next_state_text, next_state_role,
            )
            if self._prm_tokenizer:
                eval_prompt = self._prm_tokenizer.apply_chat_template(
                    eval_msgs, tokenize=False, add_generation_prompt=True,
                )
            else:
                eval_prompt = "\n".join(m["content"] for m in eval_msgs)
            async with self._teacher_lp_semaphore:
                eval_raw = await asyncio.gather(
                    *[self._query_prm_eval_once(eval_prompt, i) for i in range(self._prm_m)]
                )
            eval_score = _prm_eval_majority_vote(eval_raw)
            logger.info(
                "%s[OpenClaw-Combine-Select] PRM eval session=%s turn=%d "
                "eval_votes=%s -> eval_score=%.1f%s",
                _CYAN, session_id, turn_num,
                [s if s is not None else "fail" for s in eval_raw],
                eval_score, _RESET,
            )
        else:
            eval_score = None

        # Multi-candidate selection: keep ALL accepted hints (score==1,
        # len>10), dedupe by hint text, sort shortest-first, cap at
        # OPENCLAW_TOPK_MAX_CAND. This matches the retool / hint_opd_exp
        # convention so --hint-selection shortest -> k*=0.
        accepted: list[dict[str, Any]] = []
        seen_hints: set[str] = set()
        for v in votes:
            if v.get("score") != 1:
                continue
            hint_raw = v.get("hint")
            if not isinstance(hint_raw, str):
                continue
            hint = hint_raw.strip()
            if len(hint) <= 10:
                continue
            if hint in seen_hints:
                continue
            seen_hints.add(hint)
            accepted.append({**v, "hint": hint})
        accepted.sort(key=lambda v: len(v["hint"]))
        accepted = accepted[: _max_cand()]
        votes_display = [v.get("score", "fail") for v in votes]

        if not accepted:
            logger.info(
                "%s[OpenClaw-Combine-Select] session=%s turn=%d no valid "
                "hint (votes=%s), sample dropped%s",
                _CYAN, session_id, turn_num, votes_display, _RESET,
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
            return {
                "accepted": False,
                "teacher_tokens_candidates": None,
                "hint": "",
                "hints": [],
                "votes": votes,
                "eval_score": eval_score,
            }

        # Materialize teacher_tokens_candidates: one enhanced_ids per
        # surviving hint. The teacher LP / topk are computed by the
        # Megatron PRM-teacher pass downstream; we DO NOT call
        # _compute_teacher_log_probs here.
        candidates: list[list[int]] = []
        hints: list[str] = []
        for v in accepted:
            hint = v["hint"]
            enhanced_messages = _append_hint_to_messages(turn_data["messages"], hint)
            norm_enhanced = _normalize_messages_for_template(enhanced_messages)
            enhanced_prompt_text = self.tokenizer.apply_chat_template(
                norm_enhanced,
                tools=turn_data.get("tools"),
                tokenize=False,
                add_generation_prompt=True,
            )
            enhanced_full_text = enhanced_prompt_text + turn_data["response_text"]
            enhanced_ids = self.tokenizer(
                enhanced_full_text, add_special_tokens=False,
            )["input_ids"]
            candidates.append(enhanced_ids)
            hints.append(hint)

        logger.info(
            "%s[OpenClaw-Combine-Select] session=%s turn=%d accepted "
            "K_i=%d hint_lens=%s votes=%s%s",
            _CYAN, session_id, turn_num,
            len(candidates),
            [len(h) for h in hints],
            votes_display,
            _RESET,
        )
        self._append_prm_record(
            {
                "session_id": session_id,
                "turn": turn_num,
                "accepted": True,
                "K_i": len(candidates),
                "hints": hints,
                "hint_lens": [len(h) for h in hints],
                "votes": votes,
                "teacher_tokens_lens": [len(c) for c in candidates],
            }
        )
        return {
            "accepted": True,
            "teacher_tokens_candidates": candidates,
            "hint": hints[0],   # for log-line back-compat with parent
            "hints": hints,
            "votes": votes,
            "eval_score": eval_score,
        }

    async def _submit_turn_sample(
        self,
        turn_data: dict[str, Any],
        session_id: str,
        opd_result: dict[str, Any],
        reward: float = 0.0,
    ):
        prompt_ids = turn_data["prompt_ids"]
        response_ids = turn_data["response_ids"]

        candidates = opd_result.get("teacher_tokens_candidates") or []
        if not candidates:
            # _maybe_submit_ready_samples shouldn't dispatch here when
            # no hint was accepted; guard anyway with a degenerate
            # K_i=1 (un-enhanced) so the multi-cand pipeline always
            # has a valid candidate list.
            candidates = [list(prompt_ids) + list(response_ids)]

        sample = Sample()
        sample.prompt = turn_data["prompt_text"]
        sample.response = turn_data["response_text"]
        sample.tokens = list(prompt_ids) + list(response_ids)
        sample.response_length = len(response_ids)
        sample.loss_mask = [1] * len(response_ids)
        sample.rollout_log_probs = turn_data["response_logprobs"]

        # teacher_log_probs is consumed only by the legacy combine_loss /
        # hint_opd_loss kernels (token-level teacher signal). Our topk
        # select kernel ignores it but slime's data pipeline still
        # concatenates it -- ship zeros of the right length.
        sample.teacher_log_probs = torch.zeros(len(response_ids), dtype=torch.float32)

        # Multi-candidate teacher tokens (the slime K-loop hook).
        sample.teacher_tokens_candidates = candidates
        # Back-compat: legacy code paths (and the K_i=1 degenerate case)
        # still read teacher_tokens; keep it pointing at candidate 0.
        sample.teacher_tokens = candidates[0]

        sample.status = Sample.Status.COMPLETED
        sample.index = next(self._index_counter)
        sample.group_index = next(self._group_counter)
        sample.reward = {"score": reward}

        tag = "OPD+RL" if reward != 0.0 else "OPD"
        logger.info(
            "[OpenClaw-Combine-Select] submitted %s sample session=%s "
            "index=%d reward=%.1f prompt_len=%d response_len=%d K_i=%d "
            "hint_lens=%s",
            tag, session_id, sample.index, reward,
            len(prompt_ids), len(response_ids),
            len(candidates),
            [len(h) for h in opd_result.get("hints") or [opd_result.get("hint", "")]],
        )
        await asyncio.to_thread(self.output_queue.put, (sample.group_index, [sample]))

    async def _submit_rl_turn_sample(
        self, turn_data: dict, session_id: str, eval_score: float,
    ):
        prompt_ids = turn_data["prompt_ids"]
        response_ids = turn_data["response_ids"]
        response_logprobs = turn_data["response_logprobs"]

        if len(response_logprobs) > len(response_ids):
            response_logprobs = response_logprobs[: len(response_ids)]
        elif len(response_logprobs) < len(response_ids):
            response_logprobs = response_logprobs + [0.0] * (
                len(response_ids) - len(response_logprobs)
            )

        sample = Sample()
        sample.prompt = turn_data["prompt_text"]
        sample.response = turn_data["response_text"]
        sample.tokens = list(prompt_ids) + list(response_ids)
        sample.response_length = len(response_ids)
        sample.loss_mask = [1] * len(response_ids)
        sample.rollout_log_probs = response_logprobs

        # No real teacher signal for RL-only samples -- ship zeros
        # (consumed by legacy kernels but ignored by topk select).
        sample.teacher_log_probs = torch.zeros(len(response_ids), dtype=torch.float32)

        # Degenerate K_i=1 candidate so the multi-cand pipeline always
        # has SOMETHING to gather. The teacher just forwards on the
        # un-enhanced sequence; the loss kernel handles K_i=1 with
        # k*=0 by construction.
        un_enhanced = list(prompt_ids) + list(response_ids)
        sample.teacher_tokens_candidates = [un_enhanced]
        sample.teacher_tokens = un_enhanced

        sample.status = Sample.Status.COMPLETED
        sample.index = next(self._index_counter)
        sample.group_index = next(self._group_counter)
        sample.reward = {"score": float(eval_score)}

        logger.info(
            "[OpenClaw-Combine-Select] submitted RL sample session=%s "
            "index=%d score=%.1f prompt_len=%d response_len=%d K_i=1",
            session_id, sample.index, float(eval_score),
            len(prompt_ids), len(response_ids),
        )
        await asyncio.to_thread(self.output_queue.put, (sample.group_index, [sample]))
