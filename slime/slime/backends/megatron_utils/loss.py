import logging
from argparse import Namespace
from collections.abc import Callable, Iterator
from typing import Any

import torch
import torch.distributed as dist
from megatron.core import mpu
from torch.utils.checkpoint import checkpoint

from slime.utils.distributed_utils import distributed_masked_whiten
from slime.utils.misc import load_function

logger = logging.getLogger(__name__)

# Module-level toggle. When True, get_log_probs_and_entropy additionally
# returns "topk_log_probs" / "topk_indices" (used by the prm_teacher actor
# path for distillation). Defaults to False so that the regular old_actor
# / ref / actor compute_log_prob passes are unchanged.
_EMIT_TOPK_LOGPROBS: bool = False


class emit_topk_logprobs:
    """Context manager: opt the wrapped compute_log_prob call into emitting
    per-token top-K log-probabilities and global vocab indices.

    Only the prm_teacher actor uses this; the regular actor/old_actor/ref
    forwards do NOT enter this context, so their output dicts are byte-
    identical to before.
    """

    def __enter__(self):
        global _EMIT_TOPK_LOGPROBS
        self._prev = _EMIT_TOPK_LOGPROBS
        _EMIT_TOPK_LOGPROBS = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _EMIT_TOPK_LOGPROBS
        _EMIT_TOPK_LOGPROBS = self._prev
        return False


from slime.utils.ppo_utils import (
    calculate_log_probs_and_entropy,
    compute_approx_kl,
    compute_gspo_kl,
    compute_opsm_mask,
    compute_policy_loss,
    get_advantages_and_returns_batch,
    get_grpo_returns,
    get_reinforce_plus_plus_baseline_advantages,
    get_reinforce_plus_plus_returns,
)
from slime.utils.types import RolloutBatch

from .cp_utils import (
    all_gather_with_cp,
    get_logits_and_tokens_offset_with_cp,
    get_sum_of_sample_mean,
    slice_log_prob_with_cp,
)


def get_responses(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    max_seq_lens: list[int] | None = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield response-aligned `(logits_chunk, tokens_chunk)` pairs per sample.

    After squeezing batch dimension and applying temperature scaling, this
    function extracts the logits and tokens corresponding to response segments
    for each sample.  When context parallelism is disabled, it slices directly
    from the concatenated sequence.  With context parallelism enabled, it
    handles split sequences across ranks.

    Logits may arrive in any floating-point dtype (fp32, bf16, fp16).  The
    yielded chunks preserve the original dtype — no fp32 up-cast is performed
    here, so GPU memory stays bounded by the micro-batch logits tensor.
    Downstream consumers (``fused_vocab_parallel_cross_entropy``, etc.) handle
    mixed-precision arithmetic internally.

    Args:
        logits: Model outputs with shape ``[1, T, V]`` (policy) or
            ``[1, T, 1]`` (value).
        args: Configuration containing ``rollout_temperature`` for scaling.
        unconcat_tokens: List of token tensors (prompt+response) per sample.
        total_lengths: Total sequence lengths (prompt+response) per sample.
        response_lengths: Response segment lengths per sample.

    Yields:
        Tuple of ``(logits_chunk, tokens_chunk)`` where ``logits_chunk`` is
        shape ``[R, V]`` (policy) or ``[R, 1]`` (value) and ``tokens_chunk``
        is shape ``[R]`` (1-D int64), both aligned to response tokens for one
        sample.
    """
    qkv_format = args.qkv_format

    assert len(logits.shape) == 3, f"{logits.shape}"

    logits_gib = logits.nelement() * logits.element_size() / (1 << 30)
    if logits_gib > 2.0:
        logger.warning(
            "get_responses: large logits tensor %s dtype=%s (%.2f GiB), "
            "num_samples=%d, sum(total_lengths)=%d, sum(response_lengths)=%d",
            list(logits.shape),
            logits.dtype,
            logits_gib,
            len(unconcat_tokens),
            sum(total_lengths),
            sum(response_lengths),
        )

    if qkv_format == "thd":
        assert logits.size(0) == 1, f"{logits.shape}"
        logits = logits.squeeze(0)
    else:
        assert max_seq_lens is not None
        logits = logits.view(-1, logits.size(-1))

    apply_temp = args.rollout_temperature != 1.0

    cp_size = mpu.get_context_parallel_world_size()
    end = 0
    for i, (tokens, total_length, response_length) in enumerate(
        zip(unconcat_tokens, total_lengths, response_lengths, strict=False)
    ):
        max_seq_len = max_seq_lens[i] if max_seq_lens is not None else None

        if cp_size == 1:
            if qkv_format == "bshd":
                end = max_seq_len * i + total_length
                start = end - response_length
            else:
                end += total_length
                start = end - response_length
            logits_chunk = logits[start - 1 : end - 1]
            tokens_chunk = tokens[-response_length:]
        else:
            # TODO: this is super ugly... do better abstraction.
            chunk_size, chunks_offset, logits_offset, tokens_offset = get_logits_and_tokens_offset_with_cp(
                total_length, response_length, qkv_format, max_seq_len
            )

            logits_0, logits_1 = logits[end : end + chunk_size], logits[end + chunk_size : end + 2 * chunk_size]
            end += 2 * chunk_size

            logits_0 = logits_0[logits_offset[0][0] - chunks_offset[0][0] : logits_offset[0][1] - chunks_offset[0][0]]
            tokens_0 = tokens[tokens_offset[0][0] : tokens_offset[0][1]]

            logits_1 = logits_1[logits_offset[1][0] - chunks_offset[1][0] : logits_offset[1][1] - chunks_offset[1][0]]
            tokens_1 = tokens[tokens_offset[1][0] : tokens_offset[1][1]]

            assert logits_0.size(0) == tokens_0.size(0), f"{logits_0.size(0)} vs {tokens_0.size(0)}"
            assert logits_1.size(0) == tokens_1.size(0), f"{logits_1.size(0)} vs {tokens_1.size(0)}"

            logits_chunk = torch.cat([logits_0, logits_1], dim=0)
            tokens_chunk = torch.cat([tokens_0, tokens_1], dim=0)

        if apply_temp:
            logits_chunk = logits_chunk / args.rollout_temperature
        yield logits_chunk, tokens_chunk


def get_log_probs_and_entropy(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    with_entropy: bool = False,
    non_loss_data: bool = True,
    max_seq_lens: list[int] | None = None,
) -> dict[str, list[torch.Tensor]]:
    """Compute per-token log-probabilities (and optionally entropy) on responses.

    For each sample, extracts response-aligned logits and tokens, then computes
    log-probabilities via softmax across the tensor-parallel group. Log-probs
    are squeezed from `[R, 1]` to `[R]`. Entropy values are always appended
    (even when `with_entropy=False`), but only included in the result dict
    when requested.

    Args:
        logits: Policy logits with shape `[1, T, V]`.
        args: Configuration (temperature applied in `get_responses`).
        unconcat_tokens: List of token tensors per sample.
        total_lengths: Total sequence lengths per sample.
        response_lengths: Response segment lengths per sample.
        with_entropy: If True, include "entropy" key in result.
        non_loss_data: Unused; kept for API compatibility.

    Returns:
        Dict with key "log_probs" mapping to a list of `[R]` tensors per
        sample. If `with_entropy` is True, also includes "entropy" key with
        a list of `[R]` tensors.
    """
    assert non_loss_data
    tp_group = mpu.get_tensor_model_parallel_group()
    # Top-K is opt-in via emit_topk_logprobs() context manager. By default
    # the regular old_actor / ref / actor compute_log_prob passes do NOT
    # emit top-K data, so their outputs are unchanged. Only the prm_teacher
    # path opts in (see actor.compute_prm_teacher_log_probs).
    distill_topk = int(getattr(args, "distill_topk", 0) or 0) if _EMIT_TOPK_LOGPROBS else 0
    log_probs_list = []
    entropy_list = []
    topk_log_probs_list = []
    topk_indices_list = []
    for logits_chunk, tokens_chunk in get_responses(
        logits,
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        max_seq_lens=max_seq_lens,
    ):
        log_prob, entropy = calculate_log_probs_and_entropy(
            logits_chunk,
            tokens_chunk,
            tp_group,
            with_entropy=with_entropy,
            chunk_size=args.log_probs_chunk_size,
        )

        log_probs_list.append(log_prob.squeeze(-1))
        entropy_list.append(entropy)

        if distill_topk > 0:
            topk_lp, topk_idx = _vocab_parallel_topk_log_probs(logits_chunk, distill_topk, tp_group)
            topk_log_probs_list.append(topk_lp)
            topk_indices_list.append(topk_idx)

    res = {
        "log_probs": log_probs_list,
    }
    if with_entropy:
        res["entropy"] = entropy_list
    if distill_topk > 0:
        res["topk_log_probs"] = topk_log_probs_list
        res["topk_indices"] = topk_indices_list
    return torch.empty((0,), device=logits.device), res


def _vocab_parallel_topk_log_probs(
    logits: torch.Tensor,
    K: int,
    tp_group,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Global top-K log-probs and (global vocab) indices from TP-sharded logits.

    Args:
        logits: ``[seq, vocab_local]`` (vocab_local = vocab // tp_world).
        K: global top-K to keep.
        tp_group: TP process group.

    Returns:
        ``(topk_log_probs [seq, K], topk_indices [seq, K])`` with indices in
        the GLOBAL vocab space, so any TP rank can re-gather log-probs at
        them with ``compute_log_probs``.
    """
    if K <= 0 or logits.numel() == 0:
        return (
            logits.new_zeros((logits.size(0), 0)),
            torch.zeros((logits.size(0), 0), dtype=torch.long, device=logits.device),
        )

    tp_world = dist.get_world_size(group=tp_group) if dist.is_initialized() else 1
    tp_rank = dist.get_rank(group=tp_group) if dist.is_initialized() else 0
    vocab_local = logits.size(-1)

    # Global log-softmax across TP-sharded vocab (kept in fp32 for safety;
    # logits may arrive as bf16/fp16).
    work = logits.float()
    local_max = work.max(dim=-1, keepdim=True).values
    if tp_world > 1:
        dist.all_reduce(local_max, op=dist.ReduceOp.MAX, group=tp_group)
    shifted = work - local_max
    sum_exp = shifted.exp().sum(dim=-1, keepdim=True)
    if tp_world > 1:
        dist.all_reduce(sum_exp, op=dist.ReduceOp.SUM, group=tp_group)
    log_probs_local = shifted - sum_exp.log()

    K_local = min(K, vocab_local)
    local_lp, local_idx_local = log_probs_local.topk(K_local, dim=-1)
    local_idx_global = local_idx_local + tp_rank * vocab_local

    if tp_world == 1:
        return local_lp, local_idx_global

    # Gather every shard's local top-K, then global top-K from K_local*tp_world
    # candidates. Exact: any global top-K token is in some shard's top-K_local.
    gathered_lp = [torch.zeros_like(local_lp) for _ in range(tp_world)]
    gathered_idx = [torch.zeros_like(local_idx_global) for _ in range(tp_world)]
    dist.all_gather(gathered_lp, local_lp, group=tp_group)
    dist.all_gather(gathered_idx, local_idx_global, group=tp_group)
    cat_lp = torch.cat(gathered_lp, dim=-1)
    cat_idx = torch.cat(gathered_idx, dim=-1)
    K_eff = min(K, cat_lp.size(-1))
    final_lp, sort_pos = cat_lp.topk(K_eff, dim=-1)
    final_idx = torch.gather(cat_idx, -1, sort_pos)
    return final_lp, final_idx


# Per-call indices payload for the gather-at-indices forward path. A list
# of [R_i, K] long tensors (one per sample), GLOBAL vocab ids. Set just
# before calling forward_only(gather_log_probs_at_indices, ...) and read
# inside that function. Module-level (not threadlocal) because Megatron's
# pipeline runner is single-threaded per actor process.
_GATHER_AT_INDICES_PAYLOAD: list[torch.Tensor] | None = None
_GATHER_AT_INDICES_CURSOR: int = 0

# When >0, gather_log_probs_at_indices ALSO computes the teacher's native
# top-K indices on the same logits chunk (no extra forward pass) and
# returns them under "topk_native_indices". This lets the multi-candidate
# student-mode K-loop carry the candidate's own top-K to the loss as a
# selection signal alongside the gathered log-probs at the student's S^q.
# Default 0 leaves the existing single-candidate path byte-identical.
_EMIT_NATIVE_TOPK_K: int = 0


class emit_native_topk_indices:
    """Context manager: ALSO emit teacher's native top-K indices in the
    gather-at-indices forward.

    Args:
        K: native top-K width to compute on each logits chunk; if ``K<=0``
           the augmentation is disabled.

    Used by the multi-candidate path in ``--distill-subset-mode student``:
    the teacher's log-probs are gathered at the student's top-K (so values
    at S^q are valid for the loss), and the candidate's own native top-K
    indices ride alongside as a side channel for the per-token / per-step
    overlap selection in the loss.
    """

    def __init__(self, K: int):
        self._K = int(K)

    def __enter__(self):
        global _EMIT_NATIVE_TOPK_K
        self._prev = _EMIT_NATIVE_TOPK_K
        _EMIT_NATIVE_TOPK_K = max(0, self._K)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _EMIT_NATIVE_TOPK_K
        _EMIT_NATIVE_TOPK_K = self._prev
        return False


class set_gather_at_indices:
    """Context manager: provide a list[Tensor] of GLOBAL vocab ids per sample.

    The next ``len(payload)`` calls to ``gather_log_probs_at_indices`` (one
    per response chunk in pipeline order) consume one tensor each. Used to
    drive the extra "gather at S_t" forwards needed by ``--distill-subset-
    mode={student,teacher}`` distillation.
    """

    def __init__(self, payload: list[torch.Tensor]):
        self._payload = payload

    def __enter__(self):
        global _GATHER_AT_INDICES_PAYLOAD, _GATHER_AT_INDICES_CURSOR
        self._prev_payload = _GATHER_AT_INDICES_PAYLOAD
        self._prev_cursor = _GATHER_AT_INDICES_CURSOR
        _GATHER_AT_INDICES_PAYLOAD = self._payload
        _GATHER_AT_INDICES_CURSOR = 0
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _GATHER_AT_INDICES_PAYLOAD, _GATHER_AT_INDICES_CURSOR
        _GATHER_AT_INDICES_PAYLOAD = self._prev_payload
        _GATHER_AT_INDICES_CURSOR = self._prev_cursor
        return False


def _vocab_parallel_log_probs_at_indices(
    logits: torch.Tensor,
    indices: torch.Tensor,
    tp_group,
) -> torch.Tensor:
    """No-grad: GLOBAL log-probs at given GLOBAL vocab indices, TP-sharded.

    Args:
        logits:  ``[seq, V_local]``.
        indices: ``[seq, K]`` global vocab ids in ``[0, V)``.

    Returns:
        ``[seq, K]`` fp32 ``log softmax(logits_global)[indices]``.
    """
    if logits.numel() == 0 or indices.numel() == 0:
        return logits.new_zeros((logits.size(0), indices.size(-1) if indices.dim() > 1 else 0), dtype=torch.float32)

    tp_world = dist.get_world_size(group=tp_group) if dist.is_initialized() else 1
    tp_rank = dist.get_rank(group=tp_group) if dist.is_initialized() else 0
    V_local = logits.size(-1)
    shard_lo = tp_rank * V_local

    work = logits.float()
    local_max = work.max(dim=-1, keepdim=True).values
    if tp_world > 1:
        dist.all_reduce(local_max, op=dist.ReduceOp.MAX, group=tp_group)
    shifted = work - local_max
    sum_exp = shifted.exp().sum(dim=-1, keepdim=True)
    if tp_world > 1:
        dist.all_reduce(sum_exp, op=dist.ReduceOp.SUM, group=tp_group)
    log_sum_exp = sum_exp.log()  # [seq, 1] fp32

    in_shard = (indices >= shard_lo) & (indices < shard_lo + V_local)
    idx_local = (indices - shard_lo).clamp(min=0, max=V_local - 1)
    gathered = torch.gather(work, dim=-1, index=idx_local)  # [seq, K] fp32
    gathered_masked = torch.where(in_shard, gathered, torch.zeros_like(gathered))
    if tp_world > 1:
        dist.all_reduce(gathered_masked, op=dist.ReduceOp.SUM, group=tp_group)
    return gathered_masked - log_sum_exp


def gather_log_probs_at_indices(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    with_entropy: bool = False,
    non_loss_data: bool = True,
    max_seq_lens: list[int] | None = None,
) -> tuple[torch.Tensor, dict[str, list[torch.Tensor]]]:
    """No-grad: gather GLOBAL log-probs at indices supplied via ``set_gather_at_indices``.

    Mirrors ``get_log_probs_and_entropy``'s signature so it slots into the
    same ``forward_only`` codepath. Returns a dict with two keys:

    * ``topk_log_probs``: list of ``[R_i, K]`` fp32 log-probs per sample.
    * ``topk_indices``:   list of ``[R_i, K]`` long indices per sample
      (echoed from the payload for downstream consumers).

    When ``emit_native_topk_indices(K)`` is active, additionally returns:

    * ``topk_native_indices``: list of ``[R_i, K]`` long indices per sample
      holding the model's *own* global top-K indices on the same logits
      chunk. Computed in the same forward (no extra pass).
    """
    assert non_loss_data
    global _GATHER_AT_INDICES_PAYLOAD, _GATHER_AT_INDICES_CURSOR
    payload = _GATHER_AT_INDICES_PAYLOAD
    if payload is None:
        raise RuntimeError(
            "gather_log_probs_at_indices called outside set_gather_at_indices() context."
        )
    tp_group = mpu.get_tensor_model_parallel_group()
    native_K = int(_EMIT_NATIVE_TOPK_K)
    log_probs_list: list[torch.Tensor] = []
    indices_list: list[torch.Tensor] = []
    native_indices_list: list[torch.Tensor] = []
    for logits_chunk, _tokens_chunk in get_responses(
        logits,
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        max_seq_lens=max_seq_lens,
    ):
        if _GATHER_AT_INDICES_CURSOR >= len(payload):
            raise RuntimeError(
                f"gather_log_probs_at_indices: payload exhausted at cursor "
                f"{_GATHER_AT_INDICES_CURSOR} (payload len={len(payload)})."
            )
        idx_cpu = payload[_GATHER_AT_INDICES_CURSOR]
        _GATHER_AT_INDICES_CURSOR += 1
        idx = idx_cpu.to(device=logits_chunk.device, dtype=torch.long)
        if idx.dim() == 1:
            idx = idx.unsqueeze(-1)
        if idx.size(0) != logits_chunk.size(0):
            raise RuntimeError(
                f"gather_log_probs_at_indices: chunk has {logits_chunk.size(0)} "
                f"tokens but payload has {idx.size(0)} index rows."
            )
        lp = _vocab_parallel_log_probs_at_indices(logits_chunk, idx, tp_group)
        log_probs_list.append(lp)
        indices_list.append(idx)
        if native_K > 0:
            _, native_idx = _vocab_parallel_topk_log_probs(
                logits_chunk, native_K, tp_group
            )
            native_indices_list.append(native_idx)
    out = {
        "topk_log_probs": log_probs_list,
        "topk_indices": indices_list,
    }
    if native_K > 0:
        out["topk_native_indices"] = native_indices_list
    return torch.empty((0,), device=logits.device), out


def get_values(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    with_entropy: bool = False,
    non_loss_data: bool = True,
    max_seq_lens: list[int] | None = None,
) -> dict[str, list[torch.Tensor]]:
    """Extract per-token value predictions over response tokens.

    For each sample, extracts response-aligned chunks from the value head
    output and squeezes the final dimension from `[R, 1]` to `[R]`.

    Args:
        logits: Value head output with shape `[1, T, 1]`.
        args: Configuration (passed to `get_responses` which uses
            `rollout_temperature` even though values don't need temperature).
        unconcat_tokens: List of token tensors per sample.
        total_lengths: Total sequence lengths per sample.
        response_lengths: Response segment lengths per sample.
        with_entropy: Unused; kept for signature compatibility.
        non_loss_data: Unused; kept for signature compatibility.

    Returns:
        Dict with key "values" mapping to a list of `[R]` value tensors
        per sample.
    """
    value_list = []
    for logits_chunk, _ in get_responses(
        logits,
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        max_seq_lens=max_seq_lens,
    ):
        assert logits_chunk.size(-1) == 1, f"{logits_chunk.shape}"
        value_list.append(logits_chunk.squeeze(-1))

    return torch.empty((0,), device=logits.device), {
        "values": value_list,
    }


def compute_advantages_and_returns(args: Namespace, rollout_data: RolloutBatch) -> None:
    """Compute advantages and returns in-place based on `args.advantage_estimator`.

    This function extracts rewards, log-probs, values, and masks from
    `rollout_data`, computes KL divergences, then applies the chosen advantage
    estimator. Supported methods: "grpo", "gspo", "step_wise", "ppo",
    "reinforce_plus_plus", and "reinforce_plus_plus_baseline". When
    `args.normalize_advantages` is
    True, advantages are whitened across the data-parallel group using masked
    statistics.

    Early returns if both `log_probs` and `values` are None (intermediate
    pipeline stages).

    Args:
        args: Configuration specifying estimator type, KL coefficient,
            normalization settings, and other hyperparameters.
        rollout_data: Dict containing input lists ("log_probs", "ref_log_probs",
            "rewards", "values", "response_lengths", "loss_masks",
            "total_lengths"). Modified in-place to add "advantages" and
            "returns" keys, each mapping to lists of tensors per sample.
    """
    log_probs: list[torch.Tensor] = rollout_data.get("rollout_log_probs" if args.use_rollout_logprobs else "log_probs")
    ref_log_probs: list[torch.Tensor] = rollout_data.get("ref_log_probs")
    rewards: list[float] = rollout_data.get("rewards")
    values: None | list[torch.Tensor] = rollout_data.get("values")
    response_lengths: list[int] = rollout_data.get("response_lengths")
    loss_masks: list[torch.Tensor] = rollout_data.get("loss_masks")
    total_lengths: list[int] = rollout_data.get("total_lengths")
    max_seq_lens: list[int] | None = rollout_data.get("max_seq_lens", None)

    # return when not the last pp stage.
    if not mpu.is_pipeline_last_stage():
        return

    # loss_masks live on CPU (lazy-loading optimisation).  We need GPU copies
    # for the advantage / KL / normalisation math below.  The original CPU
    # tensors in rollout_data["loss_masks"] are NOT modified.
    if loss_masks and isinstance(loss_masks[0], torch.Tensor) and not loss_masks[0].is_cuda:
        _gpu = torch.cuda.current_device()
        loss_masks = [m.to(device=_gpu) for m in loss_masks]

    if args.kl_coef == 0 or not log_probs:
        # when kl_coef is 0, we won't compute ref_log_prob
        xs = log_probs if log_probs is not None else values
        kl = [torch.zeros_like(x, dtype=torch.float32, device=x.device) for x in xs]
    else:
        kl = [
            compute_approx_kl(
                log_probs[i],
                ref_log_probs[i],
                kl_loss_type=args.kl_loss_type,
            )
            for i in range(len(log_probs))
        ]

    if args.advantage_estimator in ["grpo", "gspo"]:
        # Reward normalization is handled in rollout.py for clarity.
        rewards = torch.tensor(rewards, dtype=torch.float32, device=kl[0].device)
        returns = get_grpo_returns(rewards, kl)
        # TODO: is the copy necessary?
        advantages = [r for r in returns]

    elif args.advantage_estimator == "step_wise":
        # Step-wise rewards are pre-normalized in rollout.py; here we only
        # project each step scalar onto its token span.
        step_rewards_per_sample = rollout_data.get("step_wise_step_rewards")
        step_spans_per_sample = rollout_data.get("step_wise_step_token_spans")
        step_indices_per_sample = rollout_data.get("step_wise_step_indices")

        if step_rewards_per_sample is None or step_spans_per_sample is None or step_indices_per_sample is None:
            raise ValueError(
                "step_wise advantage requires rollout_data keys: "
                "step_wise_step_rewards, step_wise_step_token_spans, step_wise_step_indices"
            )

        advantages = []
        returns = []
        num_samples = len(response_lengths)
        for i in range(num_samples):
            response_len = int(response_lengths[i])
            total_len = int(total_lengths[i])
            max_seq_len = max_seq_lens[i] if max_seq_lens is not None else None

            # Build full response-space advantage first, then slice with CP.
            full_adv = torch.zeros(response_len, dtype=torch.float32, device=kl[0].device)
            full_mask = loss_masks[i].to(device=full_adv.device, dtype=full_adv.dtype)

            step_rewards_i = step_rewards_per_sample[i] or []
            step_spans_i = step_spans_per_sample[i] or []
            step_indices_i = (
                (step_indices_per_sample[i] or [])
                if step_indices_per_sample is not None
                else list(range(min(len(step_rewards_i), len(step_spans_i))))
            )
            aligned_len = min(len(step_rewards_i), len(step_spans_i), len(step_indices_i))

            for pos in range(aligned_len):
                span = step_spans_i[pos]
                if not isinstance(span, (list, tuple)) or len(span) != 2:
                    continue
                start, end = int(span[0]), int(span[1])
                if not (0 <= start < end <= response_len):
                    continue

                # Broadcast the same step-level value to all tokens in this step span.
                full_adv[start:end] = float(step_rewards_i[pos])

            # Ensure non-trainable positions remain zero.
            full_adv = full_adv * full_mask

            # Convert full response-space vector to local CP chunk.
            local_adv = slice_log_prob_with_cp(
                full_adv,
                total_len,
                response_len,
                args.qkv_format,
                max_seq_len,
            )
            if isinstance(local_adv, list):
                local_adv = torch.tensor(local_adv, dtype=full_adv.dtype, device=full_adv.device)

            # GRPO-style training path expects returns; for this estimator we set
            # returns == advantages.
            advantages.append(local_adv)
            returns.append(local_adv.clone())

    elif args.advantage_estimator == "ppo":
        if values is None:
            raise ValueError("ppo advantage estimator requires rollout_data['values'], but got None.")

        old_rewards = rewards
        rewards = []
        kl_coef = -args.kl_coef
        cp_rank = mpu.get_context_parallel_rank()
        for reward, k in zip(old_rewards, kl, strict=False):
            k *= kl_coef
            if cp_rank == 0:
                k[-1] += reward
            rewards.append(k)
        advantages, returns = get_advantages_and_returns_batch(
            total_lengths, response_lengths, values, rewards, args.gamma, args.lambd
        )

    elif args.advantage_estimator == "reinforce_plus_plus":
        rewards = torch.tensor(rewards, dtype=torch.float32, device=kl[0].device)
        returns = get_reinforce_plus_plus_returns(
            rewards=rewards,
            kl=kl,
            loss_masks=loss_masks,
            response_lengths=response_lengths,
            total_lengths=total_lengths,
            kl_coef=args.kl_coef,
            gamma=args.gamma,
        )
        advantages = [r for r in returns]

    elif args.advantage_estimator == "reinforce_plus_plus_baseline":
        rewards = torch.tensor(rewards, dtype=torch.float32, device=kl[0].device)
        advantages = get_reinforce_plus_plus_baseline_advantages(
            rewards=rewards,
            kl=kl,
            loss_masks=loss_masks,
            kl_coef=args.kl_coef,
        )
        returns = advantages

    elif args.advantage_estimator == "on_policy_distillation":
        student_log_probs = log_probs
        teacher_log_probs = rollout_data.get("teacher_log_probs")
        response_lengths = rollout_data.get("response_lengths")
        device = student_log_probs[0].device
        teacher_log_probs = [t_log_prob.to(device=device) for t_log_prob in teacher_log_probs]
        teacher_log_probs = [
            t_log_prob[-response_length:]
            for t_log_prob, response_length in zip(teacher_log_probs, response_lengths, strict=False)
        ]
        advantages = [
            teacher_log_prob - student_log_prob
            for teacher_log_prob, student_log_prob in zip(teacher_log_probs, student_log_probs, strict=False)
        ]
        returns = advantages

    else:
        raise NotImplementedError(f"advantage_estimator {args.advantage_estimator} is not supported. ")

    # TODO: OpenRLHF always does advantages normalization but veRL doesn't seem to do it.
    if args.normalize_advantages and args.advantage_estimator != "step_wise":
        all_advs = torch.cat(advantages)
        cp_size = mpu.get_context_parallel_world_size()
        if cp_size == 1:
            all_masks = torch.cat(loss_masks)
        else:
            mask_chunks = []
            for i in range(len(advantages)):
                total_len = total_lengths[i]
                response_len = response_lengths[i]
                prompt_len = total_len - response_len
                max_seq_len = max_seq_lens[i] if max_seq_lens is not None else None

                _, _, _, token_offsets = get_logits_and_tokens_offset_with_cp(
                    total_len, response_len, args.qkv_format, max_seq_len
                )

                # Convert global offsets to response-space offsets
                s0, e0 = token_offsets[0]
                s1, e1 = token_offsets[1]
                res_s0, res_e0 = max(0, s0 - prompt_len), max(0, e0 - prompt_len)
                res_s1, res_e1 = max(0, s1 - prompt_len), max(0, e1 - prompt_len)

                local_mask_parts = []
                full_mask = loss_masks[i]
                if res_e0 > res_s0:
                    local_mask_parts.append(full_mask[res_s0:res_e0])
                if res_e1 > res_s1:
                    local_mask_parts.append(full_mask[res_s1:res_e1])

                # Concatenate the parts to form the final mask chunk for this rank and this sequence
                local_mask_chunk = (
                    torch.cat(local_mask_parts)
                    if local_mask_parts
                    else torch.tensor([], device=all_advs.device, dtype=full_mask.dtype)
                )
                mask_chunks.append(local_mask_chunk)

            all_masks = torch.cat(mask_chunks)

        if all_masks.numel() > 0:
            assert (
                all_advs.size() == all_masks.size()
            ), f"Shape mismatch before whitening: advantages {all_advs.size()}, masks {all_masks.size()}"
            dp_group = mpu.get_data_parallel_group()

            whitened_advs_flat = distributed_masked_whiten(
                all_advs,
                all_masks,
                process_group=dp_group,
                shift_mean=True,
            )
            chunk_lengths = [chunk.size(0) for chunk in advantages]
            advantages = list(torch.split(whitened_advs_flat, chunk_lengths))

    rollout_data["advantages"] = advantages
    rollout_data["returns"] = returns


def vanilla_tis_function(
    args,
    *,
    pg_loss: torch.Tensor,
    train_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    **kwargs: Any,
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
    rollout_log_probs = torch.cat(rollout_log_probs, dim=0)
    old_log_probs = torch.cat(train_log_probs, dim=0)
    tis = torch.exp(old_log_probs - rollout_log_probs)
    tis_abs = (torch.exp(old_log_probs - rollout_log_probs) - 1).abs()
    tis_weights = torch.clamp(tis, min=args.tis_clip_low, max=args.tis_clip)
    tis_clipfrac = (tis_weights != tis).float()
    metrics = {
        "tis": tis.clone().detach(),
        "tis_clipfrac": tis_clipfrac.clone().detach(),
        "tis_abs": tis_abs.clone().detach(),
    }
    pg_loss = pg_loss * tis_weights
    return pg_loss, loss_masks, metrics


def icepop_function(
    args,
    *,
    pg_loss: torch.Tensor,
    train_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    **kwargs: Any,
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
    rollout_log_probs = torch.cat(rollout_log_probs, dim=0)
    old_log_probs = torch.cat(train_log_probs, dim=0)
    ice_ratio = torch.exp(old_log_probs - rollout_log_probs)
    ice_abs = (torch.exp(old_log_probs - rollout_log_probs) - 1).abs()
    ice_weight = torch.where(
        (ice_ratio >= args.tis_clip_low) & (ice_ratio <= args.tis_clip), ice_ratio, torch.zeros_like(ice_ratio)
    )
    ice_clipfrac = (ice_weight != ice_ratio).float()
    metrics = {
        "tis": ice_ratio.clone().detach(),
        "tis_clipfrac": ice_clipfrac.clone().detach(),
        "tis_abs": ice_abs.clone().detach(),
    }
    pg_loss = pg_loss * ice_weight
    return pg_loss, loss_masks, metrics


def policy_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute policy loss (PPO/GSPO) and metrics.

    Computes current log-probabilities and entropy from model logits, then
    calculates PPO-style clipped policy gradient loss. For GSPO, gathers
    full sequences via context-parallel all-gather before computing per-sample
    KL. Optionally applies TIS (Truncated Importance Sampling) correction and
    adds KL loss term if configured.

    Args:
        args: Configuration controlling advantage estimator, clipping thresholds,
            entropy/KL coefficients, and TIS settings.
        batch: Mini-batch containing "advantages", "log_probs" (old policy),
            "unconcat_tokens", "response_lengths", "total_lengths", "loss_masks",
            and optionally "ref_log_probs" and "rollout_log_probs".
        logits: Policy logits with shape `[1, T, V]`.
        sum_of_sample_mean: Reduction function that averages per-sample values.

    Returns:
        Tuple of `(loss, metrics)` where `loss` is a scalar tensor and `metrics`
        is a dict containing detached scalars: "loss", "pg_loss",
        "entropy_loss", "pg_clipfrac", "ppo_kl". Additional keys "kl_loss",
        "tis", "ois", "tis_clipfrac" are included when the respective features
        are enabled.
    """
    advantages = torch.cat(batch["advantages"], dim=0)
    old_log_probs = batch["rollout_log_probs"] if args.use_rollout_logprobs else batch["log_probs"]

    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]
    max_seq_lens = batch.get("max_seq_lens", None)

    need_entropy_for_loss = args.entropy_coef != 0.0
    _, log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=need_entropy_for_loss,
        max_seq_lens=max_seq_lens,
    )

    log_probs = log_probs_and_entropy["log_probs"]

    # Pre-gather log probs if needed by OPSM or GSPO to avoid duplicate gathering
    need_full_log_probs = args.use_opsm or args.advantage_estimator == "gspo"

    full_log_probs = None
    full_old_log_probs = None
    if need_full_log_probs:
        full_log_probs = [
            all_gather_with_cp(log_prob, total_length, response_length)
            for log_prob, total_length, response_length in zip(
                log_probs, total_lengths, response_lengths, strict=False
            )
        ]
        full_old_log_probs = [
            all_gather_with_cp(old_log_prob, total_length, response_length)
            for old_log_prob, total_length, response_length in zip(
                old_log_probs, total_lengths, response_lengths, strict=False
            )
        ]

    # Compute OPSM mask if enabled
    if args.use_opsm:
        opsm_mask, opsm_clipfrac = compute_opsm_mask(
            args=args,
            full_log_probs=full_log_probs,
            full_old_log_probs=full_old_log_probs,
            advantages=batch["advantages"],
            loss_masks=batch["loss_masks"],
        )

    # Compute KL divergence (GSPO uses sequence-level KL, others use per-token KL)
    if args.advantage_estimator == "gspo":
        ppo_kl = compute_gspo_kl(
            full_log_probs=full_log_probs,
            full_old_log_probs=full_old_log_probs,
            local_log_probs=log_probs,
            loss_masks=batch["loss_masks"],
        )
        old_log_probs = torch.cat(old_log_probs, dim=0)
        log_probs = torch.cat(log_probs, dim=0)
    else:
        old_log_probs = torch.cat(old_log_probs, dim=0)
        log_probs = torch.cat(log_probs, dim=0)
        ppo_kl = old_log_probs - log_probs

    pg_loss, pg_clipfrac = compute_policy_loss(ppo_kl, advantages, args.eps_clip, args.eps_clip_high)

    if args.use_opsm:
        pg_loss = pg_loss * opsm_mask

    # Apply off-policy correction using importance sampling if enabled
    if args.get_mismatch_metrics or args.use_tis:
        # NOTE:
        # `tis_func` may apply rejection-sampling style masking (RS) and return `modified_response_masks`.
        # We rebuild `sum_of_sample_mean` with those masks to correct denominators for loss/backprop.
        #
        # However, mismatch/TIS/RS metrics (e.g., "truncate_fraction") are often defined over the
        # *pre-RS* valid tokens. If we aggregate metrics with `modified_response_masks`, the rejected
        # tokens are excluded from the denominator and the metric can be artificially driven to 0.
        # Keep a copy of the original reducer (based on `batch["loss_masks"]`) for metric aggregation.
        sum_of_sample_mean_for_mismatch_metrics = sum_of_sample_mean

        assert "rollout_log_probs" in batch, "rollout_log_probs must be provided for TIS"

        ois = (-ppo_kl).exp()
        tis_kwargs = {
            "args": args,
            "pg_loss": pg_loss,
            "train_log_probs": batch["log_probs"],
            "rollout_log_probs": batch["rollout_log_probs"],
            "loss_masks": batch["loss_masks"],
            "total_lengths": total_lengths,
            "response_lengths": response_lengths,
        }

        if args.custom_tis_function_path is not None:
            tis_func = load_function(args.custom_tis_function_path)
        else:
            tis_func = vanilla_tis_function
        pg_loss, modified_response_masks, tis_metrics = tis_func(**tis_kwargs)

        # [decouple IS and rejection] Rebuild sum_of_sample_mean with modified_response_masks for denominator correction
        # modified_response_masks will be sliced with cp in get_sum_of_sample_mean
        sum_of_sample_mean = get_sum_of_sample_mean(
            total_lengths,
            response_lengths,
            modified_response_masks,
            args.calculate_per_token_loss,
            args.qkv_format,
            max_seq_lens,
        )

    # Determine pg_loss reducer: use custom if specified, otherwise default
    if getattr(args, "custom_pg_loss_reducer_function_path", None) is not None:
        custom_pg_loss_reducer_func = load_function(args.custom_pg_loss_reducer_function_path)
        # Determine which loss_masks to use for pg_loss reducer
        pg_loss_masks = modified_response_masks if (args.get_mismatch_metrics or args.use_tis) else batch["loss_masks"]
        pg_loss_reducer = custom_pg_loss_reducer_func(
            total_lengths, response_lengths, pg_loss_masks, args.calculate_per_token_loss
        )
    else:
        pg_loss_reducer = sum_of_sample_mean

    pg_loss = pg_loss_reducer(pg_loss)
    pg_clipfrac = sum_of_sample_mean(pg_clipfrac)
    ppo_kl = sum_of_sample_mean(ppo_kl)

    # entropy loss:
    # - when entropy contributes to the objective, compute with grad.
    # - when entropy_coef == 0, compute under no_grad only for monitoring.
    if need_entropy_for_loss:
        entropy = log_probs_and_entropy["entropy"]
        entropy = torch.cat(entropy, dim=0)
        entropy_loss = sum_of_sample_mean(entropy)
    else:
        with torch.no_grad():
            _, entropy_for_monitor = get_log_probs_and_entropy(
                logits,
                args=args,
                unconcat_tokens=batch["unconcat_tokens"],
                total_lengths=total_lengths,
                response_lengths=response_lengths,
                with_entropy=True,
                max_seq_lens=max_seq_lens,
            )
            entropy = torch.cat(entropy_for_monitor["entropy"], dim=0)
            entropy_loss = sum_of_sample_mean(entropy)

    loss = pg_loss - args.entropy_coef * entropy_loss

    if args.use_kl_loss:
        ref_log_probs = batch["ref_log_probs"]
        ref_log_probs = torch.cat(ref_log_probs, dim=0)
        importance_ratio = None
        if args.use_unbiased_kl:
            importance_ratio = torch.exp(log_probs - old_log_probs)
        kl = compute_approx_kl(
            log_probs,
            ref_log_probs,
            kl_loss_type=args.kl_loss_type,
            importance_ratio=importance_ratio,
        )
        kl_loss = sum_of_sample_mean(kl)

        loss = loss + args.kl_loss_coef * kl_loss

    # make sure the gradient could backprop correctly.
    if log_probs.numel() == 0:
        loss += 0 * logits.sum()

    train_rollout_logprob_abs_diff = None
    if "rollout_log_probs" in batch and batch["rollout_log_probs"]:
        rollout_log_probs = torch.cat(batch["rollout_log_probs"], dim=0)
        train_rollout_logprob_abs_diff = sum_of_sample_mean((old_log_probs - rollout_log_probs).abs())

    reported_loss = {
        "loss": loss.clone().detach(),
        "pg_loss": pg_loss.clone().detach(),
        "entropy_loss": entropy_loss.clone().detach(),
        "pg_clipfrac": pg_clipfrac.clone().detach(),
        "ppo_kl": ppo_kl.clone().detach(),
    }

    if train_rollout_logprob_abs_diff is not None:
        reported_loss["train_rollout_logprob_abs_diff"] = train_rollout_logprob_abs_diff.clone().detach()

    if args.use_kl_loss:
        reported_loss["kl_loss"] = kl_loss.clone().detach()

    if args.get_mismatch_metrics or args.use_tis:
        # Aggregate mismatch/TIS/RS related metrics with the *pre-RS* masks.
        # See comment above where `sum_of_sample_mean_for_mismatch_metrics` is defined.
        reported_loss["ois"] = sum_of_sample_mean_for_mismatch_metrics(ois).clone().detach()
        # Assume all metrics are already cloned and detached
        for metric_key, metric_value in tis_metrics.items():
            key_name = f"{metric_key}"
            reported_loss[key_name] = sum_of_sample_mean_for_mismatch_metrics(metric_value)

    if args.use_opsm:
        reported_loss["opsm_clipfrac"] = opsm_clipfrac

    return loss, reported_loss


def value_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute clipped value loss and metrics.

    Extracts current value predictions from `logits`, compares them against
    stored old values with clipping, and computes the maximum of clipped and
    unclipped squared errors (PPO-style value clipping).

    Args:
        args: Configuration containing `value_clip` threshold.
        batch: Mini-batch with "values" (old predictions), "returns",
            "unconcat_tokens", "total_lengths", and "response_lengths".
        logits: Value head output with shape `[1, T, 1]`.
        sum_of_sample_mean: Reduction function that averages per-sample values.

    Returns:
        Tuple of `(loss, metrics)` where `loss` is a scalar tensor and
        `metrics` contains detached scalars "value_loss" and "value_clipfrac".
    """
    old_values = torch.cat(batch["values"], dim=0)

    _, values = get_values(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        max_seq_lens=batch.get("max_seq_lens", None),
    )
    values = torch.cat([value.flatten() for value in values["values"]], dim=0)

    returns = torch.cat(batch["returns"], dim=0)

    values_clipfrac = torch.abs(values - old_values) > args.value_clip
    values_clipped = old_values + (values - old_values).clamp(-args.value_clip, args.value_clip)
    surr1 = (values_clipped - returns) ** 2
    surr2 = (values - returns) ** 2
    loss = torch.max(surr1, surr2)

    loss = sum_of_sample_mean(loss)
    values_clipfrac = sum_of_sample_mean(values_clipfrac.float())

    # make sure the gradient could backprop correctly.
    if values.numel() == 0:
        loss += 0 * values.sum()

    reported_loss = {
        "value_loss": loss.clone().detach(),
        "value_clipfrac": values_clipfrac.clone().detach(),
    }

    return loss, reported_loss


def sft_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute supervised fine-tuning loss over response tokens.

    Computes log-probabilities of the ground-truth tokens in the response
    segments and returns the negative log-likelihood as the loss.

    Args:
        args: Configuration (passed through to helpers).
        batch: Mini-batch with "unconcat_tokens", "response_lengths", and
            "total_lengths".
        logits: Policy logits with shape `[1, T, V]`.
        sum_of_sample_mean: Reduction function that averages per-sample values.

    Returns:
        Tuple of `(loss, metrics)` where `metrics` contains a single detached
        scalar "loss".
    """
    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]

    _, log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=False,
        max_seq_lens=batch.get("max_seq_lens", None),
    )

    log_probs = log_probs_and_entropy["log_probs"]
    log_probs = torch.cat(log_probs, dim=0)
    loss = -sum_of_sample_mean(log_probs)

    # make sure the gradient could backprop correctly.
    if log_probs.numel() == 0:
        loss += 0 * logits.sum()

    return (
        loss,
        {
            "loss": loss.clone().detach(),
        },
    )


def loss_function(
    args: Namespace,
    batch: RolloutBatch,
    num_microbatches: int,
    logits: torch.Tensor,
) -> tuple[torch.Tensor, int | torch.Tensor, dict[str, list[str] | torch.Tensor]]:
    """Dispatch to the configured loss and rescale for Megatron integration.

    Selects one of "policy_loss", "value_loss", "sft_loss", or a custom loss
    function based on `args.loss_type`, computes the loss and metrics, then
    rescales the loss by micro-batch and parallelism factors to integrate with
    Megatron's gradient accumulation.

    Args:
        args: Configuration specifying `loss_type`, `calculate_per_token_loss`,
            `global_batch_size`, and optionally `custom_loss_function_path`.
        batch: Mini-batch with "loss_masks", "response_lengths", and other
            keys required by the selected loss function.
        num_microbatches: Number of gradient accumulation steps.
        logits: Model outputs (policy or value head).

    Returns:
        Tuple of `(scaled_loss, normalizer, logging_dict)` where:
        - `scaled_loss` is the loss tensor (scalar) rescaled for Megatron.
        - `normalizer` is `num_tokens` (scalar tensor) if
          `args.calculate_per_token_loss` is True, else `1` (int).
        - `logging_dict` has keys "keys" (list of str metric names) and
          "values" (1D tensor: [count, metric1, metric2, ...]).
    """
    num_tokens = sum([torch.clamp_min(loss_mask.sum(), 1) for loss_mask in batch["loss_masks"]])
    num_samples = len(batch["response_lengths"])
    sum_of_sample_mean = get_sum_of_sample_mean(
        batch["total_lengths"],
        batch["response_lengths"],
        batch["loss_masks"],
        args.calculate_per_token_loss,
        args.qkv_format,
        batch.get("max_seq_lens", None),
    )

    match args.loss_type:
        case "policy_loss":
            func = policy_loss_function
        case "value_loss":
            func = value_loss_function
        case "sft_loss":
            func = sft_loss_function
        case "custom_loss":
            func = load_function(args.custom_loss_function_path)
        case _:
            raise ValueError(f"Unknown loss type: {args.loss_type}")

    if args.recompute_loss_function:
        loss, log = checkpoint(func, args, batch, logits, sum_of_sample_mean)
    else:
        loss, log = func(args, batch, logits, sum_of_sample_mean)

    # Here we need to divide by cp_size because to cancel the multiply in Megatron.
    global_batch_size = batch.get("dynamic_global_batch_size", args.global_batch_size)
    if not args.calculate_per_token_loss:
        loss = (
            loss * num_microbatches / global_batch_size * mpu.get_data_parallel_world_size(with_context_parallel=True)
        )
    else:
        loss = loss * mpu.get_context_parallel_world_size()

    return (
        loss,
        (num_tokens if args.calculate_per_token_loss else torch.tensor(1, device=logits.device)),
        {
            "keys": list(log.keys()),
            "values": torch.tensor(
                [
                    num_samples if not args.calculate_per_token_loss else num_tokens,
                ]
                + list(log.values()),
                device=logits.device,
            ),
        },
    )
