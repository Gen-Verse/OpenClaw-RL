"""Combined GRPO + top-K On-Policy-Distillation loss (verl-aligned).

```
loss = w_rl * grpo_pg_loss + w_opd * topk_opd_loss
```

with weights driven by env vars (defaults: ``w_rl=0``, ``w_opd=1``).

Top-K OPD design (aligned with verl's ``compute_topk_opd_loss``)
----------------------------------------------------------------
For each response position ``t`` and each candidate ``v in S_t``:

* Subset selection (``--distill-subset-mode``):
    - ``student``: S_t = top-K(pi_old)
    - ``teacher``: S_t = top-K(pi_T)  (slime ships this aligned to the student
      via the prm_teacher gather pass)
    - ``overlap``: S_t = top-K(pi_old) ∩ top-K(pi_T); per-token |S_t| varies
      and rows with empty intersection are masked out.

  The subset has TWO purposes ONLY:
    (1) per-token filter for which (t, v) pairs enter the loss;
    (2) the support over which the IS weight ``w_v`` sums to 1.
  The subset is NEVER used to renormalize log-probabilities.

* Three GLOBAL log-probs per (t, v in S_t):
    ``ell_old(v) = log pi_old(v|s_t)``    (detached, from old_actor forward)
    ``ell_cur(v) = log pi_theta(v|s_t)``  (autograd-connected, current actor)
    ``ell_T(v)   = log pi_T(v|s_t)``      (detached, from teacher forward)

* Detached IS weight (subset-normalized -- the ONLY use of subset normalization):
    ``w_v = softmax(ell_old over S_t)``,    sum_{v in S_t} w_v = 1.

* Detached advantage (GLOBAL log-ratio, weighted by w_v):
    ``A_v = (ell_T(v) - ell_old(v)) * w_v``.
  Sign: A_v > 0 iff teacher places more GLOBAL mass on v than (old) student.

* PPO ratio (GLOBAL, gradient through ell_cur only):
    ``rho_v = exp(ell_cur(v) - ell_old(v))``.
  Under on-policy training rho_v == 1 by value, but its gradient
  ``∇_θ ell_cur(v)`` is non-zero.

* PPO clipped surrogate per (t, v):
    ``L_v = max(-A_v * rho_v, -A_v * clip(rho_v, 1-eps_lo, 1+eps_hi))``.

* Per-token aggregation: SUM over S_t (NOT mean -- since sum_v w_v = 1, the
  sum is naturally bounded to O(|ell_T - ell_old|) and concentrates gradient
  pressure on the head of S_t):
    ``L_t = sum_{v in S_t} L_v``.

* Trajectory aggregation: ``sum_of_sample_mean`` (matches slime conventions).

Why GLOBAL ratios and GLOBAL advantage (vs subset-renormalized)
---------------------------------------------------------------
A subset-renormalized ratio ``q_bar_new / q_bar_old`` is invariant to any
global rescaling of the student's distribution that preserves within-S_t
ordering. The student can satisfy "match teacher on subset" by collapsing
``pi_theta(S_t)`` to near zero -- nothing in the surrogate prevents it.
Using the GLOBAL ratio ``pi_theta(v) / pi_old(v)`` keeps the IS correction
honest: any global mass leaving S_t shrinks rho_v proportionally.

Likewise, subset-renormalizing the teacher's log-probs would replace
``log pi_T(v)`` with ``log pi_T(v | v in S_t)``, which discards the
absolute confidence of the teacher (and the magnitude of disagreement
when the teacher prefers some v* outside S_t).

Memory note
-----------
Computing the GLOBAL log-prob ``ell_cur(v) = raw[v] - global_lse(raw)``
with autograd materializes the local softmax of the student's logits in
backward (one full-vocab pass per sample). This is the unavoidable cost
of a faithful global PPO ratio.
"""

from __future__ import annotations

import os
from argparse import Namespace
from collections.abc import Callable

import torch
import torch.distributed as dist
from megatron.core import mpu

from slime.backends.megatron_utils.loss import get_log_probs_and_entropy, get_responses
from slime.utils.ppo_utils import compute_approx_kl, compute_policy_loss


# ---------------------------------------------------------------------------
# TP-aware gather of raw logits at given GLOBAL vocab indices (autograd).
# ---------------------------------------------------------------------------


class _VocabParallelGatherRawLogits(torch.autograd.Function):
    """Single-pass raw-logit gather at K global vocab indices, TP-sharded.

    Forward:
        logits:     ``[R, V_local]`` (vocab dim is TP-sharded).
        idx_global: ``[R, K]`` global vocab ids in ``[0, V)``.

        Returns ``raw ∈ [R, K]`` where ``raw[r,k] = logits_global[r, idx[r,k]]``,
        reconstructed across TP via a single all_reduce(SUM) on the masked
        per-rank gather (off-shard entries contribute 0).

    Backward:
        ``∂L/∂logits[r,v] = sum_k g[r,k] · 1{v == idx[r,k] && in_shard}``
        Implemented via scatter_add on the rank's local logits region.

    Memory: saves only ``[R, K]`` index data + the ``in_shard`` mask; the
    raw-logit slice itself is rederived from the saved ``logits`` ref via
    a tiny gather. For training-time use the activation cost is O(R*K)
    instead of O(R*V_local).
    """

    @staticmethod
    def forward(ctx, logits, idx_global, tp_group, tp_world, tp_rank):
        V_local = logits.size(-1)
        shard_lo = tp_rank * V_local

        in_shard = (idx_global >= shard_lo) & (idx_global < shard_lo + V_local)
        idx_local = (idx_global - shard_lo).clamp(min=0, max=V_local - 1)
        gathered = torch.gather(logits, dim=-1, index=idx_local)  # [R, K]
        gathered_masked = torch.where(in_shard, gathered, torch.zeros_like(gathered))
        if tp_world > 1:
            dist.all_reduce(gathered_masked, op=dist.ReduceOp.SUM, group=tp_group)

        ctx.save_for_backward(idx_local, in_shard)
        ctx.logits_shape = logits.shape
        ctx.logits_dtype = logits.dtype
        ctx.logits_device = logits.device
        ctx.tp_world = tp_world
        return gathered_masked  # [R, K]

    @staticmethod
    def backward(ctx, grad_out):
        idx_local, in_shard = ctx.saved_tensors
        # Mask the incoming grad to only this rank's shard, then scatter into the
        # rank-local logits gradient. Off-shard contributions are zeroed.
        masked_grad = torch.where(in_shard, grad_out, torch.zeros_like(grad_out))
        grad_input = torch.zeros(
            ctx.logits_shape, dtype=ctx.logits_dtype, device=ctx.logits_device
        )
        grad_input.scatter_add_(dim=-1, index=idx_local, src=masked_grad.to(ctx.logits_dtype))
        return grad_input, None, None, None, None


def _gather_student_raw_logits_at_indices(
    logits_chunk: torch.Tensor,
    indices: torch.Tensor,
    tp_group,
) -> torch.Tensor:
    """Gather raw global logits at ``indices`` (autograd-connected)."""
    tp_world = dist.get_world_size(group=tp_group) if dist.is_initialized() else 1
    tp_rank = dist.get_rank(group=tp_group) if dist.is_initialized() else 0
    return _VocabParallelGatherRawLogits.apply(
        logits_chunk, indices, tp_group, tp_world, tp_rank
    )


# ---------------------------------------------------------------------------
# TP-aware global LSE over the full vocab (autograd).
# ---------------------------------------------------------------------------


class _VocabParallelGlobalLSE(torch.autograd.Function):
    """Numerically-stable global LSE over the FULL vocab dim, TP-aware.

    Forward:
        logits:    ``[R, V_local]``  (vocab dim is TP-sharded).
        Returns ``lse ∈ [R, 1]`` where
            ``lse[r] = log sum_{v=0..V-1} exp(logits_global[r, v])``.

    Backward:
        ``∂L/∂logits[r, v] = grad_out[r, 0] * softmax_global(r, v)``.
        Locally, ``softmax_global(r, v) = exp(logits[r, v] - lse[r])`` for
        v in this rank's shard. Each TP rank scatters its own local-shard
        softmax, which together reconstruct the global softmax gradient.

    Memory: backward materializes ``[R, V_local]`` softmax probabilities.
    This is unavoidable for a faithful global ratio.
    """

    @staticmethod
    def forward(ctx, logits, tp_group, tp_world):
        logits_f = logits.float()
        row_max = logits_f.max(dim=-1, keepdim=True).values  # [R, 1]
        if tp_world > 1:
            dist.all_reduce(row_max, op=dist.ReduceOp.MAX, group=tp_group)
        shifted = logits_f - row_max
        sum_exp = shifted.exp().sum(dim=-1, keepdim=True)  # [R, 1] local
        if tp_world > 1:
            dist.all_reduce(sum_exp, op=dist.ReduceOp.SUM, group=tp_group)
        lse = row_max + sum_exp.clamp_min(1e-30).log()  # [R, 1]

        ctx.save_for_backward(logits, lse)
        return lse

    @staticmethod
    def backward(ctx, grad_out):
        # grad_out: [R, 1].   d(lse)/d(logits[v]) = global softmax(v).
        # Local-shard softmax = exp(logits - global_lse). Multiplying by
        # grad_out broadcasts [R, 1] across the local vocab dim.
        logits, lse = ctx.saved_tensors
        softmax_local = (logits.float() - lse).exp()  # [R, V_local]
        grad_in = grad_out * softmax_local
        return grad_in.to(logits.dtype), None, None


def _vocab_parallel_global_lse(
    logits_chunk: torch.Tensor,
    tp_group,
) -> torch.Tensor:
    """Global LSE over the full vocab, TP-aware (autograd-connected).

    Returns ``[R, 1]`` in float32. Use to convert raw logits to global
    log-probabilities: ``log pi(v) = logits[v] - global_lse``.
    """
    tp_world = dist.get_world_size(group=tp_group) if dist.is_initialized() else 1
    return _VocabParallelGlobalLSE.apply(logits_chunk, tp_group, tp_world)


# ---------------------------------------------------------------------------
# Knobs
# ---------------------------------------------------------------------------


def _w_rl() -> float:
    return float(os.environ.get("HINT_OPD_W_RL", "0.0"))


def _w_opd() -> float:
    return float(os.environ.get("HINT_OPD_W_OPD", "1.0"))


def _eps_clip_lo(args: Namespace) -> float:
    v = os.environ.get("HINT_OPD_PPO_CLIP_EPS_LO", "")
    return float(v) if v else float(args.eps_clip)


def _eps_clip_hi(args: Namespace) -> float:
    v = os.environ.get("HINT_OPD_PPO_CLIP_EPS_HI", "")
    return float(v) if v else float(args.eps_clip_high)


def _adv_diff_clip() -> float | None:
    """Magnitude clamp on the per-candidate teacher-vs-old log-ratio.

    Applied as ``diff = (ell_T - ell_old).clamp(-t, t)`` before forming
    ``A_v = diff * w_v``. Bounds advantage magnitude when the teacher and
    old student disagree wildly on a candidate (common early in training,
    on rare-token candidates, or when distillation crosses tokenizers).

    Env var: ``HINT_OPD_ADV_DIFF_CLIP``.
    Returns:
        positive float t -> clamp diff to [-t, t]
        None (env var unset/empty -> defaults to 2.0)
        non-positive (env var <= 0) -> disabled, returns None
    """
    raw = os.environ.get("HINT_OPD_ADV_DIFF_CLIP", "")
    if raw == "":
        return 2.0
    val = float(raw)
    return val if val > 0.0 else None


# ---------------------------------------------------------------------------
# verl-aligned PPO surrogate over S_t for one sample.
# ---------------------------------------------------------------------------


_NEG_INF = float("-inf")
# verl-style numerical guard on the log-ratio before exp(). Prevents
# exp(huge) NaNs on rare tail candidates early in training.
_PPO_KL_CLAMP = 20.0


def _local_lse(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Numerically-stable lse over the True positions of ``mask``.

    Both tensors are ``[R, K]``. Returns ``[R, 1]``. If the entire row's
    mask is all-False we return 0 (the row will be masked out downstream).
    """
    masked_vals = torch.where(mask, values, values.new_full((), _NEG_INF))
    row_max = masked_vals.max(dim=-1, keepdim=True).values
    # If a row is all -inf (no valid k), replace its max with 0 so the
    # log/exp stays finite; the row's loss will be zeroed by row_valid.
    row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
    shifted = masked_vals - row_max
    exped = torch.where(mask, shifted.exp(), torch.zeros_like(shifted))
    sum_exp = exped.sum(dim=-1, keepdim=True)
    sum_exp = torch.clamp(sum_exp, min=1e-30)
    return row_max + sum_exp.log()


def _local_softmax(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Softmax over the True positions of ``mask``.

    Both tensors are ``[R, K]``. Returns ``[R, K]``: off-mask entries are 0
    and rows where the mask is all-False return all-zero (the row is
    masked out downstream by ``row_valid``).

    Used for the IS weight ``w_v = softmax(ell_old over S_t)``.
    """
    lse = _local_lse(values, mask)  # [R, 1]
    out = (values - lse).exp()
    out = torch.where(mask, out, torch.zeros_like(out))
    return out


def _opd_one_sample(
    logits_chunk: torch.Tensor,
    *,
    student_indices: torch.Tensor,    # [R, Kq] long, GLOBAL vocab ids
    student_old_lp: torch.Tensor,     # [R, Kq] log pi_old(v) for v in student_indices
    teacher_indices: torch.Tensor,    # [R, Kp] long
    teacher_lp: torch.Tensor,         # [R, Kp] log pi_T(v) for v in teacher_indices
    eps_lo: float,
    eps_hi: float,
    diff_clip: float | None,
    tp_group,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """verl-aligned top-K OPD surrogate for one sample.

    Pipeline (notation matches the file docstring):
        1. Build ``S_t = student_indices ∩ teacher_indices``, expressed as
           a [R, Kq] boolean mask aligned with ``student_indices``. For the
           ``student`` / ``teacher`` modes Kq == Kp and the two index lists
           are identical, so the mask is all-True.
        2. Pull GLOBAL teacher log-probs ``ell_T(v)`` at ``student_indices``
           via per-row index match. (The teacher gather pass already gave
           us global log pi_T values; we just reorder them.)
        3. Compute GLOBAL student-current log-probs:
              ``ell_cur(v) = raw_logits(v) - global_lse(raw_logits)``
           via the autograd-connected gather + global LSE helpers. Both
           halves contribute gradient to the student's logits.
        4. Detached IS weight on S_t:
              ``w_v = softmax(ell_old over S_t)``,    sum_v w_v = 1.
        5. Detached advantage (with optional magnitude clamp on the diff):
              ``diff_v = clamp(ell_T(v) - ell_old(v), -t, +t)`` (t = diff_clip,
              skipped when diff_clip is None),
              ``A_v   = diff_v * w_v``.
        6. PPO surrogate using the GLOBAL log-ratio:
              ``ppo_kl = ell_old - ell_cur``  (clamped for numerical safety),
              ``L_v    = max(-A_v * rho_v, -A_v * clip(rho_v, 1-eps_lo, 1+eps_hi))``
           via ``compute_policy_loss(ppo_kl, A)``.
        7. Per-token aggregation: SUM over S_t (NOT mean -- the IS weight
           already normalises within S_t, so summing concentrates gradient
           pressure on the high-w_v candidates without dividing it away).

    Returns:
        per_token_pg     [R]  OPD surrogate per token (sum over S_t)
        per_token_clip   [R]  fraction of S_t entries that hit the PPO clip
        per_token_diff   [R]  w-weighted |ell_T - ell_old| per token (monitor)
        row_valid        [R]  bool, True iff |S_t| >= 1
    """
    R = student_indices.size(0)
    if R == 0:
        z = student_indices.new_zeros((0,), dtype=torch.float32)
        b = student_indices.new_zeros((0,), dtype=torch.bool)
        return z, z, z, b

    # 1) Build the S_t mask on the student-indices axis.
    if torch.equal(student_indices, teacher_indices):
        # Fast path: student/teacher modes ship identical index sets.
        sub_mask = torch.ones_like(student_indices, dtype=torch.bool)
        eq = None  # not needed
    else:
        # eq: [R, Kq, Kp] booleans, at most one True per (r, k).
        eq = student_indices.unsqueeze(-1) == teacher_indices.unsqueeze(-2)
        sub_mask = eq.any(dim=-1)  # [R, Kq]
    row_valid = sub_mask.any(dim=-1)  # [R]
    mask_f = sub_mask.float()         # [R, Kq]

    # 2) Align teacher GLOBAL log-probs to student_indices ordering.
    if eq is None:
        teacher_lp_aligned = teacher_lp.float()
    else:
        # Weighted sum picks the unique matched teacher_lp per (r, k);
        # off-mask entries collapse to 0 (zeroed out by mask_f anyway).
        eq_f = eq.float()
        teacher_lp_aligned = (eq_f * teacher_lp.unsqueeze(-2).float()).sum(dim=-1)

    # 3) GLOBAL student-current log-probs at student_indices (autograd).
    #    raw_at_K - global_lse(raw)  =  log pi_theta(v).
    student_new_raw = _gather_student_raw_logits_at_indices(
        logits_chunk, student_indices, tp_group
    ).float()                                                      # [R, Kq]
    global_lse = _vocab_parallel_global_lse(logits_chunk, tp_group)  # [R, 1]
    ell_cur = student_new_raw - global_lse                         # [R, Kq]

    ell_old = student_old_lp.float()                               # [R, Kq]
    ell_T = teacher_lp_aligned                                     # [R, Kq]

    # 4) Detached IS weight w_v = softmax(ell_old | S_t).
    #    Built from ell_old alone, so the diff_clip below does NOT change w.
    w = _local_softmax(ell_old, sub_mask).detach()                 # [R, Kq]

    # 5) Detached advantage A_v = (ell_T - ell_old) * w_v.
    #    Optionally clamp the per-candidate teacher-vs-old log-ratio to
    #    [-diff_clip, +diff_clip]. Bounds advantage magnitude when the
    #    teacher and old student disagree wildly on a candidate (rare-token
    #    candidates, early training, cross-tokenizer distillation).
    diff = (ell_T - ell_old).detach()                              # [R, Kq]
    if diff_clip is not None:
        diff = diff.clamp(min=-diff_clip, max=diff_clip)
    advantage = (diff * w).detach()                                # [R, Kq]

    # 6) PPO surrogate with the GLOBAL log-ratio.
    #    compute_policy_loss expects ppo_kl = log(p_old / p_new) and returns
    #    -min(rho*A, clip(rho)*A), exactly our L_v.
    ppo_kl = (ell_old - ell_cur).clamp(min=-_PPO_KL_CLAMP, max=_PPO_KL_CLAMP)
    pg, clip = compute_policy_loss(ppo_kl, advantage, eps_lo, eps_hi)

    # 7) SUM over S_t (mask out non-subset entries).
    per_token_pg = (pg * mask_f).sum(dim=-1)                       # [R]
    # clipfrac: fraction of S_t entries that were clipped (not summed).
    n_per_token = mask_f.sum(dim=-1).clamp(min=1.0)
    per_token_clip = (clip * mask_f).sum(dim=-1) / n_per_token     # [R]
    # diff monitor: w-weighted |ell_T - ell_old| -- matches the loss's
    # importance weighting so it's a calibrated "mean teacher gap" signal.
    per_token_diff = (diff.abs() * w).sum(dim=-1)                  # [R]

    # Zero rows with empty subset so they don't pollute the trajectory mean.
    row_valid_f = row_valid.float()
    return (
        per_token_pg * row_valid_f,
        per_token_clip * row_valid_f,
        per_token_diff * row_valid_f,
        row_valid,
    )


# ---------------------------------------------------------------------------
# Slime entry point
# ---------------------------------------------------------------------------


def hint_opd_loss_function(
    args: Namespace,
    batch: dict,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Slime ``--custom-loss-function-path`` entry point."""
    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]
    max_seq_lens = batch.get("max_seq_lens", None)

    w_rl = _w_rl()
    w_opd = _w_opd()
    eps_lo = _eps_clip_lo(args)
    eps_hi = _eps_clip_hi(args)
    diff_clip = _adv_diff_clip()
    need_entropy_for_loss = args.entropy_coef != 0.0

    # ---- forward: per-token student log-probs (and optional entropy) ----
    _, log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=need_entropy_for_loss,
        max_seq_lens=max_seq_lens,
    )
    new_log_probs_per_sample = log_probs_and_entropy["log_probs"]
    new_log_probs = torch.cat(new_log_probs_per_sample, dim=0)

    # Hard guarantee: every old log-prob we use is from the Megatron actor
    # forward (batch["log_probs"]), NEVER from SGLang rollout.
    assert not getattr(args, "use_rollout_logprobs", False), (
        "hint_opd loss requires --use-rollout-logprobs to be unset so old-policy "
        "log-probs come from the Megatron old_actor forward."
    )

    # ----------------- GRPO branch (token-level PG, weight w_rl) -----------------
    grpo_pg_loss = torch.zeros((), device=logits.device, dtype=torch.float32)
    grpo_pg_clipfrac = torch.zeros((), device=logits.device, dtype=torch.float32)
    ppo_kl_mean_sampled = torch.zeros((), device=logits.device, dtype=torch.float32)
    if w_rl != 0.0:
        old_log_probs = torch.cat(batch["log_probs"], dim=0)
        ppo_kl_sampled = old_log_probs - new_log_probs
        grpo_advantages = torch.cat(batch["advantages"], dim=0)
        pg_loss_tokens, pg_clipfrac_tokens = compute_policy_loss(
            ppo_kl_sampled, grpo_advantages, eps_lo, eps_hi
        )
        grpo_pg_loss = sum_of_sample_mean(pg_loss_tokens)
        grpo_pg_clipfrac = sum_of_sample_mean(pg_clipfrac_tokens)
        ppo_kl_mean_sampled = sum_of_sample_mean(ppo_kl_sampled)

    # ----------------- Top-K OPD branch (verl-aligned, weight w_opd) -----------------
    opd_loss = torch.zeros((), device=logits.device, dtype=torch.float32)
    opd_clipfrac_scalar = torch.zeros((), device=logits.device, dtype=torch.float32)
    teacher_student_logp_diff_mean: torch.Tensor | None = None
    subset_size_mean: torch.Tensor | None = None

    student_topk_lp = batch.get("topk_log_probs")
    student_topk_idx = batch.get("topk_indices")
    teacher_topk_lp = batch.get("prm_teacher_topk_log_probs")
    teacher_topk_idx = batch.get("prm_teacher_topk_indices")

    have_student_side = (
        student_topk_lp is not None
        and student_topk_idx is not None
        and len(student_topk_lp) > 0
        and len(student_topk_idx) > 0
    )
    have_teacher_side = (
        teacher_topk_lp is not None
        and teacher_topk_idx is not None
        and len(teacher_topk_lp) > 0
        and len(teacher_topk_idx) > 0
    )

    if w_opd != 0.0:
        if not (have_student_side and have_teacher_side):
            raise RuntimeError(
                "hint_opd_loss requires both ('topk_log_probs', 'topk_indices') "
                "and ('prm_teacher_topk_log_probs', 'prm_teacher_topk_indices') in "
                "the batch. Confirm --distill-topk > 0, --prm-teacher-load is set, "
                "and the slime old_actor pass runs inside emit_topk_logprobs() so "
                "student-side top-K data is shipped."
            )

        tp_group = mpu.get_tensor_model_parallel_group()
        all_pg = []
        all_clip = []
        all_diff = []
        all_size = []
        for i, (logits_chunk, _tokens_chunk) in enumerate(
            get_responses(
                logits,
                args=args,
                unconcat_tokens=batch["unconcat_tokens"],
                total_lengths=total_lengths,
                response_lengths=response_lengths,
                max_seq_lens=max_seq_lens,
            )
        ):
            s_idx = student_topk_idx[i].to(device=logits_chunk.device, dtype=torch.long)
            s_lp = student_topk_lp[i].to(device=logits_chunk.device, dtype=torch.float32)
            t_idx = teacher_topk_idx[i].to(device=logits_chunk.device, dtype=torch.long)
            t_lp = teacher_topk_lp[i].to(device=logits_chunk.device, dtype=torch.float32)
            assert s_idx.size(0) == logits_chunk.size(0), (
                f"student topk size mismatch: s_idx[0]={s_idx.size(0)} "
                f"vs logits_chunk[0]={logits_chunk.size(0)}"
            )
            assert t_idx.size(0) == logits_chunk.size(0), (
                f"teacher topk size mismatch: t_idx[0]={t_idx.size(0)} "
                f"vs logits_chunk[0]={logits_chunk.size(0)}"
            )

            pg_t, clip_t, diff_t, valid_t = _opd_one_sample(
                logits_chunk,
                student_indices=s_idx,
                student_old_lp=s_lp,
                teacher_indices=t_idx,
                teacher_lp=t_lp,
                eps_lo=eps_lo,
                eps_hi=eps_hi,
                diff_clip=diff_clip,
                tp_group=tp_group,
            )
            all_pg.append(pg_t)
            all_clip.append(clip_t)
            all_diff.append(diff_t)
            # Per-token subset size for monitoring.
            if torch.equal(s_idx, t_idx):
                size_t = s_idx.new_full((s_idx.size(0),), s_idx.size(-1), dtype=torch.float32)
            else:
                eq = s_idx.unsqueeze(-1) == t_idx.unsqueeze(-2)
                size_t = eq.any(dim=-1).float().sum(dim=-1)
            all_size.append(size_t * valid_t.float())

        opd_pg_tokens = torch.cat(all_pg, dim=0)
        opd_clip_tokens = torch.cat(all_clip, dim=0)
        opd_diff_tokens = torch.cat(all_diff, dim=0)
        opd_size_tokens = torch.cat(all_size, dim=0)
        opd_loss = sum_of_sample_mean(opd_pg_tokens)
        opd_clipfrac_scalar = sum_of_sample_mean(opd_clip_tokens)
        teacher_student_logp_diff_mean = sum_of_sample_mean(opd_diff_tokens)
        subset_size_mean = sum_of_sample_mean(opd_size_tokens)

    # ----------------- entropy term -----------------
    if need_entropy_for_loss:
        entropy = torch.cat(log_probs_and_entropy["entropy"], dim=0)
        entropy_loss = sum_of_sample_mean(entropy)
    else:
        with torch.no_grad():
            _, ent_data = get_log_probs_and_entropy(
                logits,
                args=args,
                unconcat_tokens=batch["unconcat_tokens"],
                total_lengths=total_lengths,
                response_lengths=response_lengths,
                with_entropy=True,
                max_seq_lens=max_seq_lens,
            )
            entropy_loss = sum_of_sample_mean(torch.cat(ent_data["entropy"], dim=0))

    # ----------------- combine -----------------
    loss = w_rl * grpo_pg_loss + w_opd * opd_loss - args.entropy_coef * entropy_loss

    # ----------------- optional KL-to-ref regulariser -----------------
    kl_loss = torch.tensor(0.0, device=logits.device)
    if args.use_kl_loss and batch.get("ref_log_probs") is not None:
        ref_log_probs = torch.cat(batch["ref_log_probs"], dim=0)
        kl = compute_approx_kl(
            new_log_probs, ref_log_probs, kl_loss_type=args.kl_loss_type,
        )
        kl_loss = sum_of_sample_mean(kl)
        loss = loss + args.kl_loss_coef * kl_loss

    if new_log_probs.numel() == 0:
        loss = loss + 0 * logits.sum()

    # ----------------- monitoring -----------------
    train_rollout_logprob_abs_diff = None
    if "rollout_log_probs" in batch and batch["rollout_log_probs"]:
        rollout_lp = torch.cat(batch["rollout_log_probs"], dim=0)
        train_rollout_logprob_abs_diff = sum_of_sample_mean(
            (new_log_probs.detach() - rollout_lp).abs()
        )

    reported: dict[str, torch.Tensor] = {
        "loss": loss.clone().detach(),
        "grpo_pg_loss": grpo_pg_loss.clone().detach(),
        "opd_loss": opd_loss.clone().detach(),
        "entropy_loss": entropy_loss.clone().detach(),
        "grpo_pg_clipfrac": grpo_pg_clipfrac.clone().detach(),
        "opd_pg_clipfrac": opd_clipfrac_scalar.clone().detach(),
        "ppo_kl_sampled": ppo_kl_mean_sampled.clone().detach(),
        "w_rl": torch.tensor(w_rl, device=loss.device),
        "w_opd": torch.tensor(w_opd, device=loss.device),
    }
    if teacher_student_logp_diff_mean is not None:
        reported["opd_teacher_student_logp_topk_abs_mean"] = (
            teacher_student_logp_diff_mean.clone().detach()
        )
    if subset_size_mean is not None:
        reported["opd_subset_size"] = subset_size_mean.clone().detach()
    if train_rollout_logprob_abs_diff is not None:
        reported["train_rollout_logprob_abs_diff"] = (
            train_rollout_logprob_abs_diff.clone().detach()
        )
    if args.use_kl_loss:
        reported["kl_loss"] = kl_loss.clone().detach()

    return loss, reported
