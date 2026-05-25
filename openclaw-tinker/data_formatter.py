"""Convert training samples to Tinker Datum format.

Supports all three methods:

  RL / OPD (sample_to_datum):
    advantage = scalar GRPO reward + (teacher_lp - student_lp) if teacher logprobs present
    Used via: batch_to_datums(batch, advantages)

  Combined (sample_to_datum_combined):
    combined_adv = w_opd * (teacher_lp - student_lp) + w_rl * reward
    Used via: batch_to_datums_combined(batch, w_opd, w_rl)

Tinker Datum convention:
  model_input   - input tokens (all but the last token of the full sequence)
  loss_fn_inputs:
    target_tokens - full sequence left-shifted by 1
    logprobs      - prompt positions = 0.0, response positions = sampled logprob
    advantages    - prompt = 0.0, response = advantage * loss_mask
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class TrainingSample:
    """One training example collected from the API proxy."""

    session_id: str
    turn_num: int
    prompt_tokens: list[int]
    response_tokens: list[int]
    response_logprobs: list[float]
    loss_mask: list[int]
    reward: float
    prompt_text: str = ""
    response_text: str = ""
    teacher_logprobs: Optional[list[float]] = None
    sample_type: str = ""  # "opd+rl", "opd", "rl", or "" (for pure RL method)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fit(lst: list[float], length: int) -> list[float]:
    if len(lst) > length:
        return lst[:length]
    if len(lst) < length:
        return lst + [0.0] * (length - len(lst))
    return lst


def _sanitize(lst: list[float], name: str, session_id: str, turn_num: int) -> list[float]:
    bad = [i for i, v in enumerate(lst) if not math.isfinite(v)]
    if bad:
        logger.warning(
            "[DataFormatter] non-finite %s at %d positions for session=%s turn=%d",
            name, len(bad), session_id, turn_num,
        )
        for i in bad:
            lst[i] = 0.0
    return lst


def _build_datum(all_tokens: list[int], logprobs: list[float], advantages: list[float],
                 session_id: str, turn_num: int):
    """Build a Tinker Datum from prepared sequences."""
    import tinker
    from tinker import TensorData

    T = len(all_tokens) - 1
    if T <= 0:
        raise ValueError(
            f"Empty sequence: session={session_id} turn={turn_num} tokens={len(all_tokens)}"
        )

    target_tokens = all_tokens[1:]
    logprobs = _fit(logprobs, T)
    advantages = _fit(advantages, T)
    logprobs = _sanitize(logprobs, "logprobs", session_id, turn_num)
    advantages = _sanitize(advantages, "advantages", session_id, turn_num)

    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(all_tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": TensorData.from_torch(torch.tensor(target_tokens, dtype=torch.long)),
            "logprobs": TensorData.from_torch(torch.tensor(logprobs, dtype=torch.float32)),
            "advantages": TensorData.from_torch(torch.tensor(advantages, dtype=torch.float32)),
        },
    )


# ---------------------------------------------------------------------------
# RL / OPD datum conversion
# ---------------------------------------------------------------------------

def sample_to_datum(sample: TrainingSample, advantage: float, max_tokens: int = 0):
    """Convert one sample + scalar advantage into a Tinker Datum (RL / OPD).

    For OPD samples with teacher_logprobs, the advantage is augmented with
    per-token distillation signal: (teacher_lp - student_lp).
    This matches Slime's --advantage-estimator on_policy_distillation where
    advantage = teacher_logp - old_logp (raw, no coefficient).
    """
    prompt_tokens = sample.prompt_tokens
    # Truncate prompt from left if total exceeds max_tokens (keep response intact)
    if max_tokens > 0:
        total = len(prompt_tokens) + len(sample.response_tokens)
        if total > max_tokens:
            keep = max(1, max_tokens - len(sample.response_tokens))
            trimmed = len(prompt_tokens) - keep
            prompt_tokens = prompt_tokens[-keep:]
            logger.info(
                "[DataFormatter] truncated prompt: %d -> %d tokens (session=%s turn=%d)",
                trimmed + keep, keep, sample.session_id, sample.turn_num,
            )
    prompt_len = len(prompt_tokens)
    all_tokens = prompt_tokens + sample.response_tokens

    logprobs = [0.0] * (prompt_len - 1) + list(sample.response_logprobs)
    resp_advantages = [advantage * float(m) for m in sample.loss_mask]

    # OPD: add per-token distillation advantage (teacher_lp - student_lp)
    if sample.teacher_logprobs is not None:
        for i in range(min(len(resp_advantages), len(sample.teacher_logprobs))):
            student_lp = sample.response_logprobs[i] if i < len(sample.response_logprobs) else 0.0
            teacher_lp = sample.teacher_logprobs[i]
            resp_advantages[i] += (teacher_lp - student_lp) * float(sample.loss_mask[i])

    advantages = [0.0] * (prompt_len - 1) + resp_advantages
    return _build_datum(all_tokens, logprobs, advantages, sample.session_id, sample.turn_num)


def batch_to_datums(batch: list[TrainingSample], advantages: list[float],
                    max_tokens: int = 0) -> list:
    """Convert a batch of samples + per-sample scalar advantages to Tinker Datums."""
    datums = []
    for sample, adv in zip(batch, advantages):
        try:
            datums.append(sample_to_datum(sample, adv, max_tokens=max_tokens))
        except Exception as e:
            logger.error(
                "[DataFormatter] FAILED to convert session=%s turn=%d: %s",
                sample.session_id, sample.turn_num, e, exc_info=True,
            )
    return datums


# ---------------------------------------------------------------------------
# Combined datum conversion
# ---------------------------------------------------------------------------

def sample_to_datum_combined(
    sample: TrainingSample,
    w_opd: float = 1.0,
    w_rl: float = 1.0,
    max_tokens: int = 0,
):
    """Convert one sample into a Tinker Datum with combined OPD+RL advantages.

    combined_adv_i = w_opd * (teacher_lp_i - student_lp_i) + w_rl * reward

    Matches Slime's combine_loss.py:
        combined_advantages = w_opd * teacher_advantages + w_rl * grpo_advantages
    where teacher_advantages = teacher_logp - old_logp (token-level, raw)
    and   grpo_advantages   = reward broadcast (scalar)
    """
    prompt_tokens = sample.prompt_tokens
    if max_tokens > 0:
        total = len(prompt_tokens) + len(sample.response_tokens)
        if total > max_tokens:
            keep = max(1, max_tokens - len(sample.response_tokens))
            prompt_tokens = prompt_tokens[-keep:]
    prompt_len = len(prompt_tokens)
    all_tokens = prompt_tokens + sample.response_tokens

    logprobs = [0.0] * (prompt_len - 1) + list(sample.response_logprobs)

    resp_advantages = []
    for i in range(len(sample.response_tokens)):
        mask = float(sample.loss_mask[i]) if i < len(sample.loss_mask) else 0.0

        # RL component: broadcast scalar reward
        rl_adv = w_rl * sample.reward * mask

        # OPD component: per-token (teacher_lp - student_lp)
        opd_adv = 0.0
        if sample.teacher_logprobs is not None and i < len(sample.teacher_logprobs):
            student_lp = sample.response_logprobs[i] if i < len(sample.response_logprobs) else 0.0
            teacher_lp = sample.teacher_logprobs[i]
            opd_adv = w_opd * (teacher_lp - student_lp) * mask

        resp_advantages.append(rl_adv + opd_adv)

    advantages = [0.0] * (prompt_len - 1) + resp_advantages
    return _build_datum(all_tokens, logprobs, advantages, sample.session_id, sample.turn_num)


def batch_to_datums_combined(
    batch: list[TrainingSample],
    w_opd: float = 1.0,
    w_rl: float = 1.0,
    max_tokens: int = 0,
) -> list:
    """Convert a batch of samples to Tinker Datums with combined advantages."""
    datums = []
    for sample in batch:
        try:
            datums.append(sample_to_datum_combined(
                sample, w_opd=w_opd, w_rl=w_rl, max_tokens=max_tokens,
            ))
        except Exception as e:
            logger.error(
                "[DataFormatter] FAILED to convert combined session=%s turn=%d: %s",
                sample.session_id, sample.turn_num, e, exc_info=True,
            )
    return datums


# ---------------------------------------------------------------------------
# GRPO advantages
# ---------------------------------------------------------------------------

def compute_grpo_advantages(batch: list[TrainingSample]) -> list[float]:
    """GRPO-style advantage: broadcast reward as advantage (no normalization).

    Matches OpenClaw-RL's --disable-rewards-normalization behavior.
    """
    return [s.reward for s in batch]
