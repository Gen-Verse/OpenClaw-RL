"""OEL Full KL(student || teacher) distillation loss with top-K approximation.

Implements context distillation: the teacher sees experience-augmented prompts,
the student sees bare prompts. We minimize KL(student || teacher) over the
top-K vocabulary tokens plus a tail bin.

Top-K selection source (controlled by env OPENCLAW_OEL_TOPK_SOURCE):
  - "teacher" (default): top-K tokens selected from teacher's distribution
  - "student": top-K tokens selected from student's distribution

Supports both Megatron and FSDP backends:
  - Megatron: uses fused_vocab_parallel_cross_entropy + tensor-parallel group
  - FSDP: uses pure-PyTorch F.log_softmax (no tensor parallelism)

Usage:
    --loss-type custom_loss
    --custom-loss-function-path oel_distillation_loss.oel_distillation_loss_function
    --distill-topk 50
    --disable-compute-advantages-and-returns

    # Select top-K from student distribution:
    export OPENCLAW_OEL_TOPK_SOURCE=student

Reference: OEL (arXiv 2603.16856), OPCD (arXiv 2602.12275), SDFT (arXiv 2601.19897)
"""

from __future__ import annotations

import os
from argparse import Namespace
from typing import Callable, Iterator

import torch
import torch.nn.functional as F

# --- Backend detection: Megatron vs FSDP ---
try:
    from megatron.core import mpu
    from slime.backends.megatron_utils.loss import get_responses
    from slime.utils.ppo_utils import compute_log_probs as _megatron_compute_log_probs

    _USE_MEGATRON = True
except ImportError:
    _USE_MEGATRON = False


# --- FSDP-compatible replacements ---

def _fsdp_get_responses(
    logits: torch.Tensor,
    total_lengths: list[int],
    response_lengths: list[int],
    unconcat_tokens: list[torch.Tensor],
    temperature: float = 1.0,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield (logits_chunk[R,V], tokens_chunk[R]) per sample -- FSDP version."""
    logits_2d = logits.squeeze(0)  # [T, V]
    end = 0
    for total_len, resp_len, tokens in zip(total_lengths, response_lengths, unconcat_tokens):
        end += total_len
        start = end - resp_len
        logits_chunk = logits_2d[start - 1 : end - 1]  # autoregressive shift
        tokens_chunk = tokens[-resp_len:]
        if temperature != 1.0:
            logits_chunk = logits_chunk / temperature
        yield logits_chunk, tokens_chunk


def _fsdp_compute_log_probs(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """Compute log P(token | context) without Megatron fused kernel."""
    return F.log_softmax(logits, dim=-1).gather(-1, tokens.unsqueeze(-1))


def _gather_teacher_at_student_topk(
    teacher_topk_logps: torch.Tensor,
    teacher_topk_indices: torch.Tensor,
    student_topk_indices: torch.Tensor,
    K: int,
) -> torch.Tensor:
    """Gather teacher log-probs at student's top-K token positions.

    The teacher's top-N (N >= K, typically N=1000 when topk_source=student)
    is used to look up teacher log-probs at student's top-K positions.
    For tokens in student's top-K that appear in teacher's top-N, we use
    the exact teacher logprob. For tokens NOT in teacher's top-N, we
    approximate with the teacher's tail probability.

    Args:
        teacher_topk_logps:   [R, N] teacher's top-N log-probs (N may be >> K)
        teacher_topk_indices: [R, N] teacher's top-N token indices
        student_topk_indices: [R, K] student's top-K token indices
        K: number of student's top-K tokens

    Returns:
        [R, K] teacher log-probs at student's top-K positions
    """
    R = student_topk_indices.shape[0]
    N = teacher_topk_indices.shape[1]  # teacher's top-N (may be >> K)
    device = student_topk_indices.device

    # Compute teacher tail: log(1 - sum(exp(teacher_topN_logps)))
    teacher_log_sum = torch.logsumexp(teacher_topk_logps, dim=-1, keepdim=True)  # [R, 1]
    teacher_log_sum = torch.clamp(teacher_log_sum, max=-1e-7)
    teacher_tail_total = torch.log(-torch.expm1(teacher_log_sum))  # [R, 1]
    # Each non-top-N token gets uniform share of remaining tail mass
    VOCAB_SIZE_APPROX = 151936  # Qwen vocab size
    teacher_tail_per_token = teacher_tail_total - torch.log(
        torch.tensor(max(VOCAB_SIZE_APPROX - N, 1), dtype=torch.float32, device=device)
    )  # [R, 1]

    # Start with tail approximation for all positions
    result = teacher_tail_per_token.expand(R, K).clone()  # [R, K]

    # For each student top-K token, check if it's in teacher's top-N
    # student_topk_indices: [R, K], teacher_topk_indices: [R, N]
    # Expand for broadcasting: [R, K, 1] vs [R, 1, N]
    s_exp = student_topk_indices.unsqueeze(2)  # [R, K, 1]
    t_exp = teacher_topk_indices.unsqueeze(1)  # [R, 1, N]
    match = (s_exp == t_exp)  # [R, K, N]

    # For each student position, find if any teacher position matches
    has_match = match.any(dim=2)  # [R, K]

    # Where matched, get the teacher logprob at that position
    match_teacher_idx = match.float().argmax(dim=2)  # [R, K]
    matched_teacher_logps = teacher_topk_logps.gather(1, match_teacher_idx)  # [R, K]

    # Fill in matched positions with exact teacher logprobs
    result = torch.where(has_match, matched_teacher_logps, result)

    return result


def oel_distillation_loss_function(
    args: Namespace,
    batch: dict,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute OEL full KL(student || teacher) with top-K + tail trick.

    Mathematics:
        KL(student || teacher) = sum_x P_s(x) * [log P_s(x) - log P_t(x)]

    Top-K approximation:
        Only compute over teacher's top-K tokens + a tail bin capturing the
        remaining probability mass: tail = log(1 - sum(exp(topk_logprobs))).

    Reads ``teacher_topk_log_probs`` ([T, K]) and ``teacher_topk_indices``
    ([T, K]) from the batch --- these are populated by the OEL API server using
    experience-augmented prompts as the teacher context.
    """
    teacher_topk_logprobs = batch["teacher_topk_log_probs"]
    teacher_topk_indices = batch["teacher_topk_indices"]
    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]

    K = args.distill_topk

    # Top-K source: "teacher" (default) or "student"
    topk_source = os.getenv("OPENCLAW_OEL_TOPK_SOURCE", "teacher").strip().lower()

    # Select backend-appropriate functions
    if _USE_MEGATRON:
        tp_group = mpu.get_tensor_model_parallel_group()
        max_seq_lens = batch.get("max_seq_lens", None)
        responses_iter = get_responses(
            logits,
            args=args,
            unconcat_tokens=batch["unconcat_tokens"],
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            max_seq_lens=max_seq_lens,
        )

        def compute_lp(logits_chunk, tokens_col):
            return _megatron_compute_log_probs(logits_chunk, tokens_col, tp_group)
    else:
        temperature = getattr(args, "rollout_temperature", 1.0)
        responses_iter = _fsdp_get_responses(
            logits,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            unconcat_tokens=batch["unconcat_tokens"],
            temperature=temperature,
        )

        def compute_lp(logits_chunk, tokens_col):
            return _fsdp_compute_log_probs(logits_chunk, tokens_col)

    all_student_topk_logps = []
    all_teacher_topk_logps = []
    for i, (logits_chunk, tokens_chunk) in enumerate(responses_iter):
        t_logps = teacher_topk_logprobs[i]
        t_indices = teacher_topk_indices[i]

        if not t_logps.is_cuda:
            t_logps = t_logps.to(device=logits_chunk.device)
        if not t_indices.is_cuda:
            t_indices = t_indices.to(device=logits_chunk.device)

        if topk_source == "student":
            # --- Student top-K: select top-K from student's distribution ---
            if _USE_MEGATRON:
                # Megatron: compute full student log-probs via repeated calls,
                # then select top-K from student; gather teacher at those positions
                # For efficiency, fall back to FSDP-style log_softmax when possible
                student_full_logps = F.log_softmax(logits_chunk.float(), dim=-1)
                student_topk_logps_i, student_topk_indices = torch.topk(
                    student_full_logps, K, dim=-1
                )
                # Gather teacher log-probs at student's top-K positions
                # teacher_topk_logprobs[i] has shape [R, K] from teacher's top-K —
                # we need teacher's logprobs at student's top-K positions instead.
                # Recompute teacher log-probs over the full simplex from the teacher's
                # top-K using log-sum-exp, then gather at student's positions.
                # However, we only have teacher's top-K, not full distribution.
                # So we approximate: gather from teacher's full distribution if available,
                # otherwise use the available top-K data with zero-fill for missing tokens.
                # Best approach: compute teacher log-probs at student's top-K from
                # the teacher_topk data (K tokens + tail).
                teacher_topk_for_student = _gather_teacher_at_student_topk(
                    t_logps, t_indices, student_topk_indices, K
                )
            else:
                # FSDP: single log_softmax, then topk from student
                student_full_logps = F.log_softmax(logits_chunk, dim=-1)  # [R, V]
                student_topk_logps_i, student_topk_indices = torch.topk(
                    student_full_logps, K, dim=-1
                )
                teacher_topk_for_student = _gather_teacher_at_student_topk(
                    t_logps, t_indices, student_topk_indices, K
                )

            all_student_topk_logps.append(student_topk_logps_i)
            all_teacher_topk_logps.append(teacher_topk_for_student)

        else:
            # --- Teacher top-K (default): select top-K from teacher's distribution ---
            # Gather student log-probs at teacher's top-K positions
            # Use single log_softmax + gather for all K (memory efficient)
            log_probs = F.log_softmax(logits_chunk, dim=-1)  # [R, V]
            student_topk_logps = log_probs.gather(-1, t_indices[:, :K])  # [R, K]
            del log_probs  # free immediately

            all_student_topk_logps.append(student_topk_logps)
            all_teacher_topk_logps.append(t_logps)

    # Concatenate all samples
    student_topk = torch.cat(all_student_topk_logps, dim=0)  # [total_tokens, K]
    teacher_topk = torch.cat(all_teacher_topk_logps, dim=0)  # [total_tokens, K]

    # Tail trick: log(1 - sum_of_topk_probs) for the remaining vocab mass
    student_log_s = torch.logsumexp(student_topk, dim=-1, keepdim=True)
    student_log_s = torch.clamp(student_log_s, max=-1e-7)
    student_tail = torch.log(-torch.expm1(student_log_s))

    teacher_log_s = torch.logsumexp(teacher_topk, dim=-1, keepdim=True)
    teacher_log_s = torch.clamp(teacher_log_s, max=-1e-7)
    teacher_tail = torch.log(-torch.expm1(teacher_log_s))

    # Build K+1 simplex: [topk_logprobs, tail_logprob]
    student_with_tail = torch.cat([student_topk, student_tail], dim=-1)
    teacher_with_tail = torch.cat([teacher_topk, teacher_tail], dim=-1)

    # KL(student || teacher): F.kl_div expects (input=log_Q, target=log_P)
    # where it computes sum P * (log P - log Q) when log_target=True
    # Here input=teacher (log Q), target=student (log P) -> KL(student || teacher)
    per_token_kl = F.kl_div(
        teacher_with_tail,
        student_with_tail,
        reduction="none",
        log_target=True,
    ).sum(dim=-1)

    kl_loss = sum_of_sample_mean(per_token_kl)

    loss = kl_loss
    entropy_loss = torch.tensor(0.0, device=logits.device)
    if args.entropy_coef != 0.0:
        student_probs = torch.exp(student_with_tail)
        entropy = -(student_probs * student_with_tail).sum(dim=-1)
        entropy_loss = sum_of_sample_mean(entropy)
        loss = loss - args.entropy_coef * entropy_loss

    # Prevent unused-parameter error when no tokens
    if per_token_kl.numel() == 0:
        loss = loss + 0 * logits.sum()

    # Monitoring metrics
    reported_loss = {
        "loss": loss.clone().detach(),
        "kl_loss": kl_loss.clone().detach(),
        "entropy_loss": entropy_loss.clone().detach(),
        "topk_source": torch.tensor(0.0 if topk_source == "teacher" else 1.0, device=logits.device),
    }

    if per_token_kl.numel() > 0:
        reported_loss["kl_mean"] = per_token_kl.mean().detach()
        reported_loss["kl_max"] = per_token_kl.max().detach()
    else:
        reported_loss["kl_mean"] = torch.tensor(0.0, device=logits.device)
        reported_loss["kl_max"] = torch.tensor(0.0, device=logits.device)

    return loss, reported_loss
