"""Combined GRPO + top-K On-Policy-Distillation loss for openclaw-combine,
with K-candidate teacher supervision selection (9-cell matrix).

Drop-in alternative to ``combine_loss.combine_loss_function`` for the
``openclaw_combine_select_rollout.generate_rollout_openclaw_combine_select``
rollout path.

Supports all 9 combinations of ``--distill-subset-mode`` × ``--hint-selection``:

============================== =================  ==============  =====================
``--hint-selection``           ``student``        ``overlap``     ``teacher``
============================== =================  ==============  =====================
``shortest``                   in-kernel k*=0     in-kernel k*=0  legacy (actor-side k*=0)
``token_optimal``              in-kernel k*(t)    in-kernel k*(t) legacy (actor-side k*(t))
``sequence_optimal``           in-kernel k*       in-kernel k*    legacy (actor-side k*)
============================== =================  ==============  =====================

For openclaw-combine, ``sequence_optimal`` is per-Sample. This is the
canonical per-PRM-step k* under the dynamic-history training paradigm:
``_submit_turn_sample`` fires ONCE per session-turn, so each Sample IS
one PRM step (rather than a multi-step trajectory rolled into one
Sample as in retool). The kernel's "no step_token_spans → per-sample"
fallback fires here because there's nothing to step over inside a
Sample; the step granularity has already been materialized at the
Sample level by the rollout.

Equations (consistent with hint_opd_select_loss; see that file's docstring
for full derivation):

  S^q_t = top-K(pi_old)         (student top-K vocab at t)
  S^p_{t,k} = top-K(pi_T,k)     (teacher candidate k's top-K vocab at t)
  O[k, t] = | S^q_t ∩ S^p_{t,k} |       (overlap selection signal)

  k*(t)  = 0                                          if shortest
         = argmax_k O[k, t]                           if token_optimal
         = argmax_k Σ_t O[k, t]                       if sequence_optimal
                                                      (per-Sample = per-PRM-step
                                                      under openclaw dynamic-history)

  S_t    = S^q_t                                      if subset_mode=student
         = S^p_{t, k*(t)}                             if subset_mode=teacher
         = S^q_t ∩ S^p_{t, k*(t)}                     if subset_mode=overlap

  w_v       = softmax_{v∈S_t}( ell_old(v) )           (IS weight, detached)
  diff_v    = clip( ell_T,k*(v) - ell_old(v),
                    -OPENCLAW_TOPK_ADV_DIFF_CLIP,
                    +OPENCLAW_TOPK_ADV_DIFF_CLIP )    (detached)
  A_v       = diff_v * w_v                             (detached)
  rho_v     = exp( ell_cur(v) - ell_old(v) )           (GLOBAL ratio, gradient)
  L_v       = max( -A_v * rho_v, -A_v * clip(rho_v, 1-eps, 1+eps_hi) )
  L^OPD_t   = Σ_{v ∈ S_t} L_v                          (sum, NOT mean)

GRPO branch (per-token):
  L^GRPO_t  = max( -A^grpo_t * rho^seq_t,
                   -A^grpo_t * clip(rho^seq_t, 1-eps, 1+eps_hi) )

Combined per-token (the combine semantics):
  L^combine_t = w_RL * L^GRPO_t + w_OPD * L^OPD_t

Trajectory aggregation: slime ``sum_of_sample_mean`` over response tokens.

Environment variables (all OPENCLAW_TOPK_* prefix; defaults shown):
    OPENCLAW_TOPK_W_RL              weight on GRPO PG     (default 1.0)
    OPENCLAW_TOPK_W_OPD             weight on top-K OPD   (default 1.0)
    OPENCLAW_TOPK_ADV_DIFF_CLIP     clamp on (ell_T - ell_old)  (default 2.0)
    OPENCLAW_TOPK_PPO_CLIP_EPS_LO   override args.eps_clip       (optional)
    OPENCLAW_TOPK_PPO_CLIP_EPS_HI   override args.eps_clip_high  (optional)

Defaults differ from hint_opd's (w_rl=0.0, w_opd=1.0): combine has BOTH
signals on every Sample (RL+OPD, OPD-only, or RL-only depending on
which branches accepted; see openclaw_combine_api_server's three-case
table), so we leave both weights at 1.0 to match the existing combine
launcher's defaults.
"""

from __future__ import annotations

import os
from argparse import Namespace
from collections.abc import Callable

import torch
from megatron.core import mpu

# Reuse the verl-aligned per-sample OPD surrogate AND the multi-cand
# selection helpers from the hint_opt_exp implementation. These are
# pure functions (no env-var coupling) so importing them is safe and
# keeps the math identical to retool / hint_opd_exp.
from hint_opd_loss import _opd_one_sample
from hint_opd_select_loss import (
    _gather_along_K,
    _overlap_count_per_token,
    _select_k_star_per_token,
)
from slime.backends.megatron_utils.loss import get_log_probs_and_entropy, get_responses
from slime.utils.ppo_utils import compute_approx_kl, compute_policy_loss


_PPO_KL_CLAMP = 20.0


def _w_rl() -> float:
    return float(os.environ.get("OPENCLAW_TOPK_W_RL", "1.0"))


def _w_opd() -> float:
    return float(os.environ.get("OPENCLAW_TOPK_W_OPD", "1.0"))


def _eps_clip_lo(args: Namespace) -> float:
    v = os.environ.get("OPENCLAW_TOPK_PPO_CLIP_EPS_LO", "")
    return float(v) if v else float(args.eps_clip)


def _eps_clip_hi(args: Namespace) -> float:
    v = os.environ.get("OPENCLAW_TOPK_PPO_CLIP_EPS_HI", "")
    return float(v) if v else float(args.eps_clip_high)


def _adv_diff_clip() -> float | None:
    raw = os.environ.get("OPENCLAW_TOPK_ADV_DIFF_CLIP", "")
    if raw == "":
        return 2.0
    val = float(raw)
    return val if val > 0.0 else None


def openclaw_topk_select_loss_function(
    args: Namespace,
    batch: dict,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """``--custom-loss-function-path`` entry point for openclaw_combine_select.

    Dispatches on (``args.distill_subset_mode``, ``args.hint_selection``)
    to cover all 9 cells:

      * ``subset_mode == "teacher"``: actor-side ``train_actor`` already
        collapsed the cand teacher tensors into single-cand keys
        (``prm_teacher_topk_log_probs``, ``prm_teacher_topk_indices``)
        AFTER selecting k* and re-gathering student-old log-probs at the
        chosen S^p. We consume those single-cand keys directly.
      * ``subset_mode in {student, overlap}``: read the cand keys
        directly and slice at k*(t) in-kernel. Under ``student`` the
        teacher cand log-probs are GATHERED at S^q (constant indices
        across k); the per-(k, t) selection signal travels in
        ``prm_teacher_native_topk_indices_cand``.
    """
    hint_selection = getattr(args, "hint_selection", "shortest")
    subset_mode = getattr(args, "distill_subset_mode", "student")
    if hint_selection not in ("shortest", "token_optimal", "sequence_optimal"):
        raise ValueError(
            f"Unknown --hint-selection: {hint_selection!r}. Expected one of "
            "'shortest', 'token_optimal', 'sequence_optimal'."
        )
    if subset_mode not in ("student", "teacher", "overlap"):
        raise ValueError(
            f"Unknown --distill-subset-mode: {subset_mode!r}. Expected one of "
            "'student', 'teacher', 'overlap'."
        )

    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]
    max_seq_lens = batch.get("max_seq_lens", None)

    w_rl = _w_rl()
    w_opd = _w_opd()
    eps_lo = _eps_clip_lo(args)
    eps_hi = _eps_clip_hi(args)
    diff_clip = _adv_diff_clip()
    entropy_coef = float(getattr(args, "entropy_coef", 0.0) or 0.0)
    need_entropy_for_loss = entropy_coef != 0.0

    _, log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=need_entropy_for_loss,
        max_seq_lens=max_seq_lens,
    )
    new_log_probs = torch.cat(log_probs_and_entropy["log_probs"], dim=0)

    # Hard guarantee: old log-probs come from the Megatron old_actor
    # forward, NOT from rollout. Aligns with the OPD math (we need the
    # full-vocab old logits, which the rollout SGLang engine doesn't
    # emit).
    assert not getattr(args, "use_rollout_logprobs", False), (
        "openclaw_topk_select_loss requires --use-rollout-logprobs to be "
        "unset so old-policy log-probs come from the Megatron old_actor."
    )

    grpo_pg_loss = torch.zeros((), device=logits.device, dtype=torch.float32)
    grpo_pg_clipfrac = torch.zeros((), device=logits.device, dtype=torch.float32)
    ppo_kl_mean_sampled = torch.zeros((), device=logits.device, dtype=torch.float32)
    if w_rl != 0.0:
        old_log_probs = torch.cat(batch["log_probs"], dim=0)
        # Match the OPD branch's verl-style guard before compute_policy_loss()
        # exponentiates the log-ratio. Without this, a deterministic outlier
        # batch can produce inf * 0 -> NaN for OPD-only / zero-advantage tokens.
        ppo_kl_sampled = (old_log_probs - new_log_probs).clamp(
            min=-_PPO_KL_CLAMP,
            max=_PPO_KL_CLAMP,
        )
        rl_advantages = torch.cat(batch["advantages"], dim=0)
        pg_loss_tokens, pg_clipfrac_tokens = compute_policy_loss(
            ppo_kl_sampled, rl_advantages, eps_lo, eps_hi
        )
        grpo_pg_loss = sum_of_sample_mean(pg_loss_tokens)
        grpo_pg_clipfrac = sum_of_sample_mean(pg_clipfrac_tokens)
        ppo_kl_mean_sampled = sum_of_sample_mean(ppo_kl_sampled)

    opd_loss = torch.zeros((), device=logits.device, dtype=torch.float32)
    opd_clipfrac_scalar = torch.zeros((), device=logits.device, dtype=torch.float32)
    teacher_student_logp_diff_mean: torch.Tensor | None = None
    subset_size_mean: torch.Tensor | None = None
    sel_overlap_mean: torch.Tensor | None = None
    sel_k_star_mean: torch.Tensor | None = None

    student_topk_lp = batch.get("topk_log_probs")
    student_topk_idx = batch.get("topk_indices")

    have_student = (
        student_topk_lp is not None
        and student_topk_idx is not None
        and len(student_topk_lp) > 0
        and len(student_topk_idx) > 0
    )

    # The teacher tensors are EITHER cand-suffixed (multi-cand path,
    # subset in {student, overlap}) OR plain single-cand (actor-side
    # collapsed, subset == teacher). Pick which side to read once.
    use_cand_keys = subset_mode != "teacher"
    if use_cand_keys:
        teacher_topk_lp_any = batch.get("prm_teacher_topk_log_probs_cand")
        teacher_topk_idx_any = batch.get("prm_teacher_topk_indices_cand")
        teacher_native_idx_cand = batch.get("prm_teacher_native_topk_indices_cand")
    else:
        teacher_topk_lp_any = batch.get("prm_teacher_topk_log_probs")
        teacher_topk_idx_any = batch.get("prm_teacher_topk_indices")
        teacher_native_idx_cand = None

    have_teacher = (
        teacher_topk_lp_any is not None
        and teacher_topk_idx_any is not None
        and len(teacher_topk_lp_any) > 0
        and len(teacher_topk_idx_any) > 0
    )

    step_spans_per_sample = batch.get("step_wise_step_token_spans")

    if w_opd != 0.0:
        if not (have_student and have_teacher):
            cand_suffix = "_cand" if use_cand_keys else ""
            raise RuntimeError(
                "openclaw_topk_select_loss requires both student top-K "
                "(topk_log_probs / topk_indices) and the teacher top-K "
                f"(prm_teacher_topk_log_probs{cand_suffix} / "
                f"prm_teacher_topk_indices{cand_suffix}) in the batch. "
                "Confirm --distill-topk > 0, --hint-m > 0, "
                "OPENCLAW_COMBINE_OPD_TEACHER_SOURCE=megatron, --prm-teacher-load "
                "set, and the rollout function path is "
                "openclaw_combine_select_rollout."
                "generate_rollout_openclaw_combine_select."
            )
        # Selection signal sanity for student/{token,sequence}_optimal.
        if (
            subset_mode == "student"
            and hint_selection != "shortest"
            and (
                teacher_native_idx_cand is None
                or len(teacher_native_idx_cand) == 0
            )
        ):
            raise RuntimeError(
                "subset_mode=student with --hint-selection in "
                "{token_optimal, sequence_optimal} requires "
                "prm_teacher_native_topk_indices_cand (the per-candidate "
                "selection signal) in the batch. Confirm slime's "
                "gather_at_indices multi-cand path is engaged. "
                "(shortest does not need it because k*=0 always.)"
            )

        tp_group = mpu.get_tensor_model_parallel_group()
        all_pg = []
        all_clip = []
        all_diff = []
        all_size = []
        all_overlap_sel = []
        all_k_star = []

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
            R = logits_chunk.size(0)
            assert s_idx.dim() == 2 and s_idx.size(0) == R, (
                f"student topk shape mismatch: s_idx={tuple(s_idx.shape)} "
                f"vs R={R}"
            )

            t_lp_any_i = teacher_topk_lp_any[i].to(
                device=logits_chunk.device, dtype=torch.float32
            )
            t_idx_any_i = teacher_topk_idx_any[i].to(
                device=logits_chunk.device, dtype=torch.long
            )

            if use_cand_keys:
                # Multi-candidate path (subset in {student, overlap}):
                # tensors are [K, R, K_p].
                assert t_idx_any_i.dim() == 3 and t_idx_any_i.size(1) == R, (
                    f"teacher topk_cand shape mismatch: t_idx_cand="
                    f"{tuple(t_idx_any_i.shape)} vs R={R}; expected [K, R, K_p]."
                )
                assert t_lp_any_i.shape == t_idx_any_i.shape, (
                    f"teacher logp/idx cand shape mismatch: "
                    f"lp={tuple(t_lp_any_i.shape)} idx={tuple(t_idx_any_i.shape)}"
                )

                # Selection signal (only computed when needed).
                if hint_selection == "shortest":
                    overlap_kr = None
                else:
                    if subset_mode == "student":
                        # Cand log-probs are at S^q (constant across k);
                        # use each candidate's NATIVE top-K for the
                        # overlap signal instead.
                        sel_idx_src = teacher_native_idx_cand[i].to(
                            device=logits_chunk.device, dtype=torch.long
                        )
                        assert sel_idx_src.shape[1] == R, (
                            f"native_topk shape mismatch: native="
                            f"{tuple(sel_idx_src.shape)} vs R={R}; "
                            "expected [K, R, K_p]."
                        )
                    else:  # overlap
                        sel_idx_src = t_idx_any_i
                    overlap_kr = _overlap_count_per_token(s_idx, sel_idx_src)

                spans_i = (
                    step_spans_per_sample[i]
                    if step_spans_per_sample is not None
                    and i < len(step_spans_per_sample)
                    else None
                )
                k_star_per_token = _select_k_star_per_token(
                    overlap_kr,
                    hint_selection=hint_selection,
                    step_token_spans=spans_i,
                    R=R,
                    device=logits_chunk.device,
                )

                # Slice cand tensors at k*(t).
                t_lp_sel = _gather_along_K(t_lp_any_i, k_star_per_token)
                if subset_mode == "student":
                    # Force kernel's S^p to S^q (cand indices are already
                    # constant across k under student subset; this keeps
                    # the contract explicit).
                    t_idx_sel = s_idx
                else:  # overlap
                    t_idx_sel = _gather_along_K(t_idx_any_i, k_star_per_token)

                if overlap_kr is None:
                    # ``shortest``: report selected-overlap as 0 so the
                    # wandb panel is well-defined and visibly distinct
                    # from the optimal modes (where it ≈ K_q for student).
                    overlap_sel_per_token = torch.zeros(
                        R, device=logits_chunk.device, dtype=torch.float32
                    )
                else:
                    row_idx = torch.arange(R, device=overlap_kr.device)
                    overlap_sel_per_token = overlap_kr[k_star_per_token, row_idx].float()
            else:
                # Legacy single-cand path (subset == teacher): the actor
                # has already done k*-selection AND re-gathered
                # student-old log-probs at the chosen S^p. We consume
                # the collapsed [R, K_p] tensors directly.
                assert t_idx_any_i.dim() == 2 and t_idx_any_i.size(0) == R, (
                    f"teacher topk shape mismatch: t_idx="
                    f"{tuple(t_idx_any_i.shape)} vs R={R}; expected [R, K_p]."
                )
                assert t_lp_any_i.shape == t_idx_any_i.shape
                t_idx_sel = t_idx_any_i
                t_lp_sel = t_lp_any_i
                overlap_sel_per_token = torch.zeros(
                    R, device=logits_chunk.device, dtype=torch.float32
                )
                k_star_per_token = torch.zeros(
                    R, device=logits_chunk.device, dtype=torch.long
                )

            pg_t, clip_t, diff_t, valid_t = _opd_one_sample(
                logits_chunk,
                student_indices=s_idx,
                student_old_lp=s_lp,
                teacher_indices=t_idx_sel,
                teacher_lp=t_lp_sel,
                eps_lo=eps_lo,
                eps_hi=eps_hi,
                diff_clip=diff_clip,
                tp_group=tp_group,
            )
            all_pg.append(pg_t)
            all_clip.append(clip_t)
            all_diff.append(diff_t)
            if torch.equal(s_idx, t_idx_sel):
                size_t = s_idx.new_full(
                    (s_idx.size(0),), s_idx.size(-1), dtype=torch.float32
                )
            else:
                eq = s_idx.unsqueeze(-1) == t_idx_sel.unsqueeze(-2)
                size_t = eq.any(dim=-1).float().sum(dim=-1)
            all_size.append(size_t * valid_t.float())
            all_overlap_sel.append(overlap_sel_per_token * valid_t.float())
            all_k_star.append(k_star_per_token.float() * valid_t.float())

        opd_pg_tokens = torch.cat(all_pg, dim=0)
        opd_clip_tokens = torch.cat(all_clip, dim=0)
        opd_diff_tokens = torch.cat(all_diff, dim=0)
        opd_size_tokens = torch.cat(all_size, dim=0)
        opd_overlap_sel_tokens = torch.cat(all_overlap_sel, dim=0)
        opd_k_star_tokens = torch.cat(all_k_star, dim=0)
        opd_loss = sum_of_sample_mean(opd_pg_tokens)
        opd_clipfrac_scalar = sum_of_sample_mean(opd_clip_tokens)
        teacher_student_logp_diff_mean = sum_of_sample_mean(opd_diff_tokens)
        subset_size_mean = sum_of_sample_mean(opd_size_tokens)
        sel_overlap_mean = sum_of_sample_mean(opd_overlap_sel_tokens)
        sel_k_star_mean = sum_of_sample_mean(opd_k_star_tokens)

    if need_entropy_for_loss:
        entropy = torch.cat(log_probs_and_entropy["entropy"], dim=0)
        entropy_loss = sum_of_sample_mean(entropy)
    else:
        entropy_loss = torch.zeros((), device=logits.device, dtype=torch.float32)

    loss = w_rl * grpo_pg_loss + w_opd * opd_loss
    if entropy_coef != 0.0:
        loss = loss - entropy_coef * entropy_loss

    kl_loss = torch.tensor(0.0, device=logits.device)
    kl_loss_coef = float(getattr(args, "kl_loss_coef", 0.0) or 0.0)
    if args.use_kl_loss and batch.get("ref_log_probs") is not None and kl_loss_coef != 0.0:
        ref_log_probs = torch.cat(batch["ref_log_probs"], dim=0)
        kl = compute_approx_kl(
            new_log_probs, ref_log_probs, kl_loss_type=args.kl_loss_type,
        )
        kl_loss = sum_of_sample_mean(kl)
        loss = loss + kl_loss_coef * kl_loss

    if new_log_probs.numel() == 0:
        loss = loss + 0 * logits.sum()

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
    if sel_overlap_mean is not None:
        reported["sel_overlap_at_k_star"] = sel_overlap_mean.clone().detach()
    if sel_k_star_mean is not None:
        reported["sel_k_star_mean"] = sel_k_star_mean.clone().detach()
    if train_rollout_logprob_abs_diff is not None:
        reported["train_rollout_logprob_abs_diff"] = (
            train_rollout_logprob_abs_diff.clone().detach()
        )
    if args.use_kl_loss:
        reported["kl_loss"] = kl_loss.clone().detach()

    # Embed selection-mode / subset-mode tags as constant ints for wandb
    # grouping without needing the run config.
    mode_id = {"shortest": 0, "token_optimal": 1, "sequence_optimal": 2}[hint_selection]
    subset_id = {"student": 0, "overlap": 1, "teacher": 2}[subset_mode]
    reported["hint_selection_mode_id"] = torch.tensor(mode_id, device=loss.device)
    reported["distill_subset_mode_id"] = torch.tensor(subset_id, device=loss.device)

    return loss, reported
