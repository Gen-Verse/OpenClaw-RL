"""Hint-OPD top-K loss with K-candidate teacher supervision selection.

Drop-in alternative to ``hint_opd_loss.hint_opd_loss_function`` for the
``hint_opd_hint_rollout_select`` path (``--hint-m K``, ``--hint-selection``).

Supports all 9 combinations of ``--hint-selection`` × ``--distill-subset-mode``:

============================== =================  ============  ==============
``--hint-selection``           ``student``        ``overlap``   ``teacher``
============================== =================  ============  ==============
``shortest``                   in-kernel k*=0     in-kernel k*=0  legacy (actor-side k*=0)
``token_optimal``              in-kernel k*(t)    in-kernel k*(t) legacy (actor-side k*(t))
``sequence_optimal``           in-kernel k*       in-kernel k*    legacy (actor-side k*)
============================== =================  ============  ==============

For ``hint_opt_exp`` ``sequence_optimal`` is per-sample (one ``k*`` for all
response tokens of a prompt) -- the rollout module ships a single CoT per
sample with no PRM-step structure (``step_wise_step_token_spans`` is
absent), so the kernel's "sequence = whole response" fallback fires.
This is the natural granularity here: the prompt has one response, the
response IS the sequence. Contrast with the retool variant
(``hybrid_stepwise_topk_opd_select_loss``) where ``step_token_spans``
IS shipped and ``sequence_optimal`` is per-PRM-step.

Selection score (all three subset modes):

    O[k, t] = | S^q_t ∩ S^p_{t,k} |

  * ``token_optimal``    : k*(t) = argmax_k O[k, t]
  * ``sequence_optimal`` : k*    = argmax_k Σ_t O[k, t] (one k* per sample,
                           broadcast over all response tokens). This is
                           the natural granularity for the hint_opt_exp
                           single-CoT pipeline (no tool-calling steps).
                           If a future caller starts shipping
                           ``step_token_spans`` metadata the kernel will
                           transparently switch to per-step k* via the
                           helper ``_select_k_star_per_token``.

Subset-mode semantics (orthogonal to selection):

  * ``--distill-subset-mode student`` : ``S_t = S^q_t``. Teacher log-probs
    arrive in ``prm_teacher_topk_log_probs_cand`` GATHERED at the
    student's top-K (the actor-side multi-cand gather pass populates
    this). Indices on S^q are constant across k; the per-(k, t)
    selection signal travels in
    ``prm_teacher_native_topk_indices_cand`` (each candidate's own
    native top-K). The loss feeds ``_opd_one_sample`` with
    ``teacher_indices = student_topk_idx`` so its subset matches S^q.

  * ``--distill-subset-mode overlap`` : ``S_t = S^q_t ∩ S^p_{t, k*(t)}``.
    Teacher log-probs / indices arrive at the candidate's native top-K
    in the ``_cand`` keys. The kernel computes the overlap mask
    internally inside ``_opd_one_sample``.

  * ``--distill-subset-mode teacher`` : ``S_t = S^p_{t, k*(t)}``. The
    actor-side ``train_actor`` does selection AND the extra student-old
    re-gather at the chosen S^p, then collapses the cand tensors into
    legacy single-cand keys. The kernel here simply delegates to the
    legacy single-cand ``hint_opd_loss_function`` for that case.

The GRPO branch (``w_rl * grpo_pg``) and the KL-to-ref / entropy
branches are imported verbatim from ``hint_opd_loss``; only the OPD
branch differs.

Environment variables (same as the topk-OPD baseline)
-----------------------------------------------------
    HINT_OPD_W_RL          weight on GRPO PG    (default 0.0)
    HINT_OPD_W_OPD         weight on top-K OPD  (default 1.0)
    HINT_OPD_PPO_CLIP_EPS_LO  override args.eps_clip      (optional)
    HINT_OPD_PPO_CLIP_EPS_HI  override args.eps_clip_high (optional)
    HINT_OPD_ADV_DIFF_CLIP    clamp on (ell_T - ell_old)  (default 2.0)
"""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Callable

import torch
from megatron.core import mpu

from hint_opd_loss import (
    _adv_diff_clip,
    _eps_clip_hi,
    _eps_clip_lo,
    _opd_one_sample,
    _w_opd,
    _w_rl,
    hint_opd_loss_function,
)
from slime.backends.megatron_utils.loss import get_log_probs_and_entropy, get_responses
from slime.utils.ppo_utils import compute_approx_kl, compute_policy_loss


# ---------------------------------------------------------------------------
# Per-candidate / student top-K overlap on the response-token axis.
# ---------------------------------------------------------------------------


def _overlap_count_per_token(
    student_idx: torch.Tensor,   # [R, K_q]
    teacher_idx: torch.Tensor,   # [K, R, K_p]
) -> torch.Tensor:
    """``O[k, t] = | S^q_t ∩ S^p_{t,k} |`` via broadcasted equality.

    Memory: ``[K, R, K_q, K_p]`` boolean intermediate. With K_q=K_p=20 and
    R≤8192, K≤8 this is ~25 MB per sample — small.
    """
    eq = student_idx.unsqueeze(0).unsqueeze(-1) == teacher_idx.unsqueeze(-2)
    return eq.any(dim=-1).sum(dim=-1).to(torch.long)  # [K, R]


def _select_k_star_per_token(
    overlap_kr: torch.Tensor | None,                # [K, R] or None for shortest
    *,
    hint_selection: str,
    step_token_spans: list[list[int]] | None,       # per-sample list of [t0, t1]
    R: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute ``k*(t) ∈ [0, K)`` per response token.

    * ``shortest``         : ``k* = 0`` for every token (the rollout module
      orders candidates shortest-first, so candidate 0 is the shortest hint
      that survived the min-token / dedup filter). ``overlap_kr`` may be
      ``None`` since no overlap signal is required.
    * ``token_optimal``    : argmax over candidates per token.
    * ``sequence_optimal`` : argmax over candidates per PRM step
      (broadcast over tokens in the step). When ``step_token_spans`` is
      missing/empty -- the case for the hint_opt_exp single-CoT pipeline
      where the "sequence" IS the whole response -- this collapses to
      per-sample argmax (one ``k*`` for every token in the response).
    """
    if hint_selection == "shortest":
        return torch.zeros(R, dtype=torch.long, device=device)
    assert overlap_kr is not None, (
        "_select_k_star_per_token: overlap_kr is required for "
        f"hint_selection={hint_selection!r}."
    )
    K, R_kr = overlap_kr.shape
    assert R_kr == R, f"overlap_kr R mismatch: {R_kr} vs {R}"
    if K == 1:
        return torch.zeros(R, dtype=torch.long, device=device)
    if hint_selection == "token_optimal":
        return overlap_kr.argmax(dim=0)  # [R]
    if hint_selection == "sequence_optimal":
        if not step_token_spans:
            k_star_scalar = int(overlap_kr.sum(dim=-1).argmax().item())
            return torch.full((R,), k_star_scalar, dtype=torch.long, device=device)
        out = torch.zeros(R, dtype=torch.long, device=device)
        for span in step_token_spans:
            t0, t1 = int(span[0]), int(span[1])
            t0 = max(0, min(t0, R))
            t1 = max(t0, min(t1, R))
            if t1 == t0:
                continue
            seg_score = overlap_kr[:, t0:t1].sum(dim=-1)  # [K]
            out[t0:t1] = int(seg_score.argmax().item())
        return out
    raise ValueError(
        f"Unknown --hint-selection: {hint_selection!r}. Expected "
        "'shortest' / 'token_optimal' / 'sequence_optimal'."
    )


def _gather_along_K(
    cand_tensor: torch.Tensor,        # [K, R, *]
    k_star_per_token: torch.Tensor,   # [R]
) -> torch.Tensor:
    """Slice ``cand_tensor[k*(t), t, ...]`` per token. Returns ``[R, *]``."""
    R = cand_tensor.size(1)
    trailing = cand_tensor.shape[2:]
    expand_shape = (1, R, *trailing)
    view_shape = (1, R) + tuple(1 for _ in trailing)
    gather_idx = k_star_per_token.view(view_shape).expand(expand_shape)
    return torch.gather(cand_tensor, dim=0, index=gather_idx).squeeze(0)


# ---------------------------------------------------------------------------
# Slime entry point
# ---------------------------------------------------------------------------


def hint_opd_select_loss_function(
    args: Namespace,
    batch: dict,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """``--custom-loss-function-path`` entry point for hint_opd_hint_select.

    Dispatches on ``args.hint_selection`` AND ``args.distill_subset_mode``.
    All 9 combinations are supported:

    * ``subset_mode == "teacher"`` (any ``hint_selection``): the actor-side
      ``train_actor`` already collapsed the cand teacher tensors into
      legacy single-cand keys -- selection + extra student-old re-gather
      happen there. ``_select_teacher_cand_per_sample`` short-circuits
      ``shortest`` to ``k* = 0``. We delegate to the legacy single-cand
      ``hint_opd_loss_function`` for all three teacher cells.

    * ``subset_mode in {student, overlap}`` (any ``hint_selection``):
      runs the in-kernel selection branch below. Under multi-cand the
      ``train_actor`` does NOT collapse the cand keys (only teacher mode
      does), so the legacy single-cand kernel CANNOT be used here -- the
      ``_cand`` keys are present but the legacy ``prm_teacher_topk_*``
      keys are absent. The in-kernel branch consumes the ``_cand`` keys
      directly and slices at ``k*(t)`` (=0 for ``shortest``).
    """
    hint_selection = getattr(args, "hint_selection", "shortest")
    subset_mode = getattr(args, "distill_subset_mode", "student")
    if hint_selection not in ("shortest", "token_optimal", "sequence_optimal"):
        raise ValueError(
            f"Unknown --hint-selection: {hint_selection!r}. Expected one of "
            "'shortest', 'token_optimal', 'sequence_optimal'."
        )
    if subset_mode == "teacher":
        return hint_opd_loss_function(args, batch, logits, sum_of_sample_mean)

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
    new_log_probs = torch.cat(log_probs_and_entropy["log_probs"], dim=0)

    assert not getattr(args, "use_rollout_logprobs", False), (
        "hint_opd_select_loss requires --use-rollout-logprobs to be unset so "
        "old-policy log-probs come from the Megatron old_actor."
    )

    # ----------------- GRPO branch (token PG, weight w_rl) -----------------
    grpo_pg_loss = torch.zeros((), device=logits.device, dtype=torch.float32)
    grpo_pg_clipfrac = torch.zeros((), device=logits.device, dtype=torch.float32)
    ppo_kl_mean_sampled = torch.zeros((), device=logits.device, dtype=torch.float32)
    if w_rl != 0.0:
        old_log_probs = torch.cat(batch["log_probs"], dim=0)
        ppo_kl_sampled = old_log_probs - new_log_probs
        rl_advantages = torch.cat(batch["advantages"], dim=0)
        pg_loss_tokens, pg_clipfrac_tokens = compute_policy_loss(
            ppo_kl_sampled, rl_advantages, eps_lo, eps_hi
        )
        grpo_pg_loss = sum_of_sample_mean(pg_loss_tokens)
        grpo_pg_clipfrac = sum_of_sample_mean(pg_clipfrac_tokens)
        ppo_kl_mean_sampled = sum_of_sample_mean(ppo_kl_sampled)

    # ----------------- K-candidate top-K OPD branch -----------------
    opd_loss = torch.zeros((), device=logits.device, dtype=torch.float32)
    opd_clipfrac_scalar = torch.zeros((), device=logits.device, dtype=torch.float32)
    teacher_student_logp_diff_mean: torch.Tensor | None = None
    subset_size_mean: torch.Tensor | None = None
    sel_overlap_mean: torch.Tensor | None = None
    sel_k_star_mean: torch.Tensor | None = None

    student_topk_lp = batch.get("topk_log_probs")
    student_topk_idx = batch.get("topk_indices")
    teacher_topk_lp_cand = batch.get("prm_teacher_topk_log_probs_cand")
    teacher_topk_idx_cand = batch.get("prm_teacher_topk_indices_cand")
    teacher_native_idx_cand = batch.get("prm_teacher_native_topk_indices_cand")
    step_spans_per_sample = batch.get("step_wise_step_token_spans")

    have_student = (
        student_topk_lp is not None
        and student_topk_idx is not None
        and len(student_topk_lp) > 0
        and len(student_topk_idx) > 0
    )
    have_teacher_cand = (
        teacher_topk_lp_cand is not None
        and teacher_topk_idx_cand is not None
        and len(teacher_topk_lp_cand) > 0
        and len(teacher_topk_idx_cand) > 0
    )

    if w_opd != 0.0:
        if not (have_student and have_teacher_cand):
            raise RuntimeError(
                "hint_opd_select_loss requires both student top-K "
                "(topk_log_probs / topk_indices) and the candidate-axis "
                "teacher top-K (prm_teacher_topk_log_probs_cand / "
                "prm_teacher_topk_indices_cand) in the batch. Confirm "
                "--distill-topk > 0, --hint-m > 0, --distill-subset-mode "
                "in {student, overlap, teacher}, and that the rollout "
                "function path is "
                "hint_opd_hint_rollout_select.generate_rollout_with_hint_select."
            )
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
            t_idx_cand = teacher_topk_idx_cand[i].to(
                device=logits_chunk.device, dtype=torch.long
            )
            t_lp_cand = teacher_topk_lp_cand[i].to(
                device=logits_chunk.device, dtype=torch.float32
            )

            R = logits_chunk.size(0)
            assert s_idx.dim() == 2 and s_idx.size(0) == R, (
                f"student topk shape mismatch: s_idx={tuple(s_idx.shape)} "
                f"vs R={R}"
            )
            assert t_idx_cand.dim() == 3 and t_idx_cand.size(1) == R, (
                f"teacher topk_cand shape mismatch: t_idx_cand="
                f"{tuple(t_idx_cand.shape)} vs R={R}; expected [K, R, K_p]."
            )
            assert t_lp_cand.shape == t_idx_cand.shape, (
                f"teacher logp/idx cand shape mismatch: "
                f"lp={tuple(t_lp_cand.shape)} idx={tuple(t_idx_cand.shape)}"
            )

            # Selection signal: only computed when actually needed.
            #   * ``shortest``       : k*=0 always, no overlap signal needed.
            #   * ``student`` mode   : t_idx_cand is a constant copy of S^q
            #                          across k, so overlap with it would be
            #                          trivially K_q; use each candidate's
            #                          NATIVE top-K (emitted by slime under
            #                          ``emit_native_topk_indices``).
            #   * ``overlap`` mode   : use the candidate's own top-K
            #                          (= t_idx_cand under this subset mode).
            if hint_selection == "shortest":
                overlap_kr = None
            else:
                if subset_mode == "student":
                    sel_idx_src = teacher_native_idx_cand[i].to(
                        device=logits_chunk.device, dtype=torch.long
                    )
                    assert sel_idx_src.shape[1] == R, (
                        f"native_topk shape mismatch: native="
                        f"{tuple(sel_idx_src.shape)} vs R={R}; "
                        "expected [K, R, K_p]."
                    )
                else:
                    sel_idx_src = t_idx_cand
                overlap_kr = _overlap_count_per_token(s_idx, sel_idx_src)  # [K, R]

            spans_i = (
                step_spans_per_sample[i]
                if step_spans_per_sample is not None and i < len(step_spans_per_sample)
                else None
            )
            k_star_per_token = _select_k_star_per_token(
                overlap_kr,
                hint_selection=hint_selection,
                step_token_spans=spans_i,
                R=R,
                device=logits_chunk.device,
            )

            # Slice cand tensors at k*(t) per token. Under student mode the
            # log-probs are at S^q (constant indices across k); under overlap
            # they are at the candidate's own top-K.
            t_lp_sel = _gather_along_K(t_lp_cand, k_star_per_token)
            if subset_mode == "student":
                # Force the kernel's S^p to S^q so the loss subset is S^q.
                # (Indices in t_idx_cand are already constant across k, but
                # this keeps the contract explicit.)
                t_idx_sel = s_idx
            else:
                t_idx_sel = _gather_along_K(t_idx_cand, k_star_per_token)

            if overlap_kr is None:
                # ``shortest``: report selected-overlap as 0 so the wandb
                # panel is well-defined and visibly distinct from the
                # optimal modes (where it's ~K_q for student mode).
                overlap_sel_per_token = torch.zeros(
                    R, device=logits_chunk.device, dtype=torch.float32
                )
            else:
                row_idx = torch.arange(R, device=overlap_kr.device)
                overlap_sel_per_token = overlap_kr[k_star_per_token, row_idx].float()

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

    # Reported-metric naming matches `hint_opd_loss_function` so wandb
    # panels created for the single-cand baseline keep working.
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

    # Embed selection-mode / subset-mode tags (constant ints) so the wandb
    # dashboard can group runs without needing the run config.
    mode_id = {"shortest": 0, "token_optimal": 1, "sequence_optimal": 2}[hint_selection]
    subset_id = {"student": 0, "overlap": 1, "teacher": 2}.get(subset_mode, -1)
    reported["hint_selection_mode_id"] = torch.tensor(mode_id, device=loss.device)
    reported["distill_subset_mode_id"] = torch.tensor(subset_id, device=loss.device)

    return loss, reported
