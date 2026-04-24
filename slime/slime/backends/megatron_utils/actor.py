import copy
import inspect
import logging
import os
import random
import socket
from argparse import Namespace
from contextlib import nullcontext

import ray
import torch
import torch.distributed as dist

from slime.utils.common import is_npu
if is_npu():
    import mindspeed.megatron_adaptor
    from mindspeed.megatron_adaptor import repatch
from megatron.core import mpu
from ray.actor import ActorHandle
from torch_memory_saver import torch_memory_saver
from transformers import AutoConfig, AutoTokenizer

from slime.ray.train_actor import TrainRayActor
from slime.utils import train_dump_utils
from slime.utils.data import process_rollout_data
from slime.utils.distributed_utils import get_gloo_group, init_process_group
from slime.utils.logging_utils import init_tracking
from slime.utils.memory_utils import clear_memory, print_memory
from slime.utils.misc import Box
from slime.utils.reloadable_process_group import destroy_process_groups, monkey_patch_torch_dist, reload_process_groups
from slime.utils.routing_replay import RoutingReplay
from slime.utils.timer import Timer, inverse_timer, timer, with_defer
from slime.utils.types import RolloutBatch

from ...utils.profile_utils import TrainProfiler
from ...utils.tensor_backper import TensorBackuper
from .checkpoint import load_checkpoint
from .cp_utils import slice_log_prob_with_cp, slice_with_cp
from .data import DataIterator, get_data_iterator, log_perf_data, log_rollout_data, sync_actor_critic_data
from .initialize import init, is_megatron_main_rank
from .loss import (
    compute_advantages_and_returns,
    emit_topk_logprobs,
    gather_log_probs_at_indices,
    get_log_probs_and_entropy,
    get_values,
    set_gather_at_indices,
)
from .model import forward_only, initialize_model_and_optimizer, save, train
from .update_weight.common import named_params_and_buffers
from .update_weight.update_weight_from_distributed import UpdateWeightFromDistributed
from .update_weight.update_weight_from_tensor import UpdateWeightFromTensor

logging.getLogger("megatron").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


# Architectural args copied from the PRM teacher's torch_dist `common.pt` onto the
# deepcopied `args` namespace when the teacher runs in raw mode. This is what lets
# the teacher have a different model size than the student's MODEL_ARGS (e.g. an
# 8B teacher next to a 4B student) without --megatron-to-hf-mode bridge.
#
# Scope is intentionally architecture-only: shape (decoder/embedding/RoPE/MoE/MLA/
# MTP), not training knobs (dropout, optimizer, parallelism, recompute, dtype,
# kernels). Parallelism overrides for the teacher are still done explicitly below.
# The transformer layer spec (--spec) is *not* overridden -- it must stay the
# student's spec, which works as long as student and teacher share an architecture
# family (e.g. Qwen3-4B / Qwen3-8B). Cross-family mixing isn't supported here.
_PRM_TEACHER_ARCH_FIELDS = (
    "num_layers",
    "hidden_size",
    "ffn_hidden_size",
    "num_attention_heads",
    "num_query_groups",
    "kv_channels",
    "max_position_embeddings",
    "qk_layernorm",
    "normalization",
    "norm_epsilon",
    "swiglu",
    "untie_embeddings_and_output_weights",
    "padded_vocab_size",
    "vocab_size",
    "add_bias_linear",
    "group_query_attention",
    "position_embedding_type",
    "rotary_percent",
    "rotary_base",
    "rotary_interleaved",
    "rotary_seq_len_interpolation_factor",
    "use_rope_scaling",
    "use_rotary_position_embeddings",
    "num_experts",
    "moe_ffn_hidden_size",
    "moe_router_topk",
    "mtp_num_layers",
    "multi_latent_attention",
)


def _read_megatron_common_args(load_path: str) -> Namespace | None:
    """Read the `args` Namespace from a Megatron torch_dist checkpoint's common.pt.

    Returns None if the path is not a Megatron checkpoint or common.pt is missing.
    Used by the prm_teacher init path to discover the teacher's architecture so a
    raw-mode teacher can have a different model size than the student.
    """
    from pathlib import Path

    import re as _re

    p = Path(load_path)
    common_pt = None
    iter_file = p / "latest_checkpointed_iteration.txt"
    if iter_file.is_file():
        tag = iter_file.read_text().strip()
        if tag == "release":
            common_pt = p / "release" / "common.pt"
        else:
            try:
                step = int(tag)
                common_pt = p / f"iter_{step:07d}" / "common.pt"
            except ValueError:
                common_pt = None
    elif _re.fullmatch(r"iter_\d{7}", p.name) and (p / "common.pt").is_file():
        common_pt = p / "common.pt"

    if common_pt is None or not common_pt.is_file():
        return None
    try:
        ckpt = torch.load(str(common_pt), map_location="cpu", weights_only=False)
    except Exception as exc:
        logger.warning("Failed to read teacher common.pt at %s: %s", common_pt, exc)
        return None
    if not isinstance(ckpt, dict):
        return None
    return ckpt.get("args")


def _offload_rollout_data_to_cpu(rollout_data: RolloutBatch) -> None:
    """Move per-sample GPU tensors in rollout_data back to CPU to free GPU memory.

    After compute_advantages_and_returns, various computed tensors (log_probs,
    ref_log_probs, advantages, returns, values, entropy, etc.) reside on GPU.
    This function moves them all to CPU so that the subsequent training phase
    can lazily load only the current micro-batch to GPU via get_batch().
    """
    moved_any = False
    for key in list(rollout_data.keys()):
        vals = rollout_data[key]
        if not isinstance(vals, list) or not vals:
            continue
        if isinstance(vals[0], torch.Tensor) and vals[0].is_cuda:
            rollout_data[key] = [v.to("cpu", non_blocking=True) for v in vals]
            moved_any = True
    if moved_any:
        if not is_npu():
            torch.cuda.synchronize()
        else:
            torch.npu.synchronize()


class MegatronTrainRayActor(TrainRayActor):
    @with_defer(lambda: Timer().start("train_wait"))
    def init(
        self,
        args: Namespace,
        role: str,
        with_ref: bool = False,
    ) -> int | None:
        monkey_patch_torch_dist()

        if role == "prm_teacher":
            import re
            from pathlib import Path

            args = copy.deepcopy(args)
            args.tensor_model_parallel_size = getattr(args, "prm_teacher_num_gpus", 1)
            args.pipeline_model_parallel_size = 1
            args.context_parallel_size = 1
            args.sequence_parallel = False
            args.expert_model_parallel_size = 1
            args.expert_tensor_parallel_size = 1
            args.pretrained_checkpoint = args.prm_teacher_load
            args.load = args.prm_teacher_load
            args.no_load_optim = True
            args.no_load_rng = True

            # Per-teacher bridge/raw mode is independent from the student's mode.
            # When --prm-teacher-megatron-to-hf-mode is unset, inherit from the
            # global --megatron-to-hf-mode for backward compatibility.
            _teacher_mode_override = getattr(args, "prm_teacher_megatron_to_hf_mode", None)
            if _teacher_mode_override is not None:
                args.megatron_to_hf_mode = _teacher_mode_override
            _teacher_mode = args.megatron_to_hf_mode

            _teacher_load = Path(args.prm_teacher_load)
            _teacher_is_megatron_ckpt = (_teacher_load / "latest_checkpointed_iteration.txt").is_file() or bool(
                re.fullmatch(r"iter_\d{7}", _teacher_load.name)
            )

            # hf_checkpoint resolution. Used for: (a) tokenizer/HF-config lookup
            # below, and (b) bridge model construction in model_provider when
            # _teacher_mode == "bridge".
            _teacher_hf_override = getattr(args, "prm_teacher_hf_checkpoint", None)
            if _teacher_hf_override is not None:
                args.hf_checkpoint = _teacher_hf_override
            elif not _teacher_is_megatron_ckpt:
                # Bridge-style HF dir as load path doubles as the HF source.
                args.hf_checkpoint = args.prm_teacher_load

            # Mode-specific consistency checks and architecture wiring.
            if _teacher_mode == "bridge":
                # Bridge teacher needs an HF directory it can build the model
                # from. Either prm_teacher_load is HF, or the user pointed
                # --prm-teacher-hf-checkpoint at one.
                _hf_dir_ok = Path(args.hf_checkpoint).is_dir() and (
                    Path(args.hf_checkpoint) / "config.json"
                ).is_file()
                assert _hf_dir_ok, (
                    f"prm_teacher in bridge mode needs an HF directory with config.json. "
                    f"Got hf_checkpoint={args.hf_checkpoint!r} (derived from "
                    f"--prm-teacher-hf-checkpoint or --prm-teacher-load). "
                    f"Pass --prm-teacher-hf-checkpoint when --prm-teacher-load is a "
                    f"torch_dist directory."
                )
            else:
                # Raw mode: prm_teacher_load MUST be a torch_dist directory and
                # we read its common.pt to discover the teacher's architecture.
                # This is what allows e.g. an 8B teacher next to a 4B student in
                # raw mode -- without it, the student's MODEL_ARGS would be used
                # to build the teacher and load_checkpoint would shape-mismatch.
                assert _teacher_is_megatron_ckpt, (
                    f"prm_teacher in raw mode requires a torch_dist directory at "
                    f"--prm-teacher-load (with latest_checkpointed_iteration.txt). "
                    f"Got {args.prm_teacher_load!r}. Either convert with "
                    f"tools/convert_hf_to_torch_dist.py, or pass "
                    f"--prm-teacher-megatron-to-hf-mode bridge."
                )
                _teacher_ckpt_args = _read_megatron_common_args(args.prm_teacher_load)
                if _teacher_ckpt_args is None:
                    logger.warning(
                        "prm_teacher: could not read common.pt from %s; "
                        "teacher will inherit student's architectural args, which "
                        "will shape-mismatch unless student and teacher are the "
                        "same model size.",
                        args.prm_teacher_load,
                    )
                else:
                    _changed = []
                    for _field in _PRM_TEACHER_ARCH_FIELDS:
                        if not hasattr(_teacher_ckpt_args, _field):
                            continue
                        _new = getattr(_teacher_ckpt_args, _field)
                        _old = getattr(args, _field, None)
                        if _new != _old:
                            setattr(args, _field, _new)
                            _changed.append((_field, _old, _new))
                    if _changed:
                        logger.info(
                            "prm_teacher: applied %d architectural override(s) from "
                            "teacher common.pt -> %s",
                            len(_changed),
                            ", ".join(f"{n}: {o}->{v}" for n, o, v in _changed),
                        )

            # --prm-teacher-rotary-base is an explicit user override and wins over
            # whatever common.pt or the student carried. Use case: actor and PRM
            # teacher with different rope_theta (e.g. actor is a long-context SFT
            # fine-tune with rope_theta=5e6 while the teacher is stock base with
            # rope_theta=1e6).
            prm_teacher_rotary_base = getattr(args, "prm_teacher_rotary_base", None)
            if prm_teacher_rotary_base is not None:
                args.rotary_base = prm_teacher_rotary_base

        super().init(args, role, with_ref)

        init(args)
        if is_npu():
            repatch(args)

        if is_megatron_main_rank():
            init_tracking(args, primary=False)

        self.prof = TrainProfiler(args)

        # read config and tokenizer serialized to prevent concurrent writing bug.
        for i in range(args.num_gpus_per_node):
            if i == dist.get_rank() % args.num_gpus_per_node:
                self.hf_config = AutoConfig.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
                self.tokenizer = AutoTokenizer.from_pretrained(self.args.hf_checkpoint, trust_remote_code=True)
            dist.barrier(group=get_gloo_group())

        self.train_parallel_config = {
            "dp_size": mpu.get_data_parallel_world_size(with_context_parallel=False),
        }
        dist.barrier(group=get_gloo_group())

        if args.offload_train:
            if (x := args.train_memory_margin_bytes) > 0:
                logger.info(f"Set torch_memory_saver.memory_margin_bytes to {x}")
                torch_memory_saver.memory_margin_bytes = x

        if self.args.debug_rollout_only:
            return 0

        if role == "critic":
            self.args.load = self.args.critic_load
            self.args.save = self.args.critic_save
            self.args.lr = self.args.critic_lr
            self.args.lr_warmup_iters = self.args.critic_lr_warmup_iters

        (self.model, self.optimizer, self.opt_param_scheduler, loaded_rollout_id) = initialize_model_and_optimizer(
            args, role
        )

        if role == "critic":
            if self.args.offload_train:
                self.sleep()
            return

        if role == "prm_teacher":
            clear_memory()
            return

        start_rollout_id = loaded_rollout_id + 1

        self.weights_backuper = TensorBackuper.create(
            source_getter=lambda: named_params_and_buffers(
                self.args,
                self.model,
                convert_to_global_name=args.megatron_to_hf_mode == "raw",
                translate_gpu_to_cpu=not self.args.enable_weights_backuper,
            ),
            single_tag=None if args.enable_weights_backuper else "actor",
        )
        self._active_model_tag: str | None = "actor"
        self.weights_backuper.backup("actor")

        if with_ref:
            self.load_other_checkpoint("ref", args.ref_load)

        if self.args.keep_old_actor:
            # Load old_actor checkpoint
            self.load_other_checkpoint("old_actor", args.load)
            # Create rollout_actor as a copy of current actor
            if args.update_weights_interval == 1:
                self.weights_backuper.backup("rollout_actor")

        if self.args.vocab_size is None:
            self.args.vocab_size = self.tokenizer.vocab_size

        update_weight_cls = UpdateWeightFromTensor if self.args.colocate else UpdateWeightFromDistributed
        self.weight_updater = update_weight_cls(
            self.args,
            self.model,
            weights_getter=lambda: self.weights_backuper.get("actor"),
            model_name=type(self.hf_config).__name__.lower() if self.args.model_name is None else self.args.model_name,
            quantization_config=getattr(self.hf_config, "quantization_config", None),
        )

        # empty cache after initialization
        clear_memory()

        if self.args.offload_train:
            # recover to actor in the end.
            self._switch_model("actor")
            self.sleep()

        self.rollout_engines = None

        self.rollout_data_postprocess = None
        if self.args.rollout_data_postprocess_path is not None:
            from slime.utils.misc import load_function

            self.rollout_data_postprocess = load_function(self.args.rollout_data_postprocess_path)

        self.prof.on_init_end()

        return start_rollout_id

    @timer
    def sleep(self) -> None:
        assert self.args.offload_train

        clear_memory(clear_host_memory=True)
        print_memory("before offload model")
        destroy_process_groups()

        torch_memory_saver.pause()

        print_memory("after offload model")

    @timer
    def wake_up(self) -> None:
        assert self.args.offload_train
        print_memory("before wake_up model")

        torch_memory_saver.resume()

        clear_memory()
        reload_process_groups()
        print_memory("after wake_up model")

    def _get_rollout_data(self, rollout_data_ref: Box) -> RolloutBatch:
        # Fetch data through ray on CPU, not sure if this will be performance bottleneck.
        # Both first pp stage and the last pp stage will receive the data.
        rollout_data = process_rollout_data(
            self.args,
            rollout_data_ref,
            mpu.get_data_parallel_rank(with_context_parallel=False),
            mpu.get_data_parallel_world_size(with_context_parallel=False),
        )
        # Keep all per-sample data on CPU.  They will be lazily moved to GPU
        # per micro-batch inside get_batch(), so GPU memory scales with
        # micro-batch size instead of total sample count.
        rollout_data["tokens"] = [
            torch.tensor(t, dtype=torch.long) for t in rollout_data["tokens"]
        ]
        if "teacher_tokens" in rollout_data:
            rollout_data["teacher_tokens"] = [
                torch.tensor(t, dtype=torch.long) for t in rollout_data["teacher_tokens"]
            ]
        rollout_data["loss_masks"] = [
            torch.tensor(t, dtype=torch.int) for t in rollout_data["loss_masks"]
        ]
        # multimodal_train_inputs: kept on CPU as-is (no .to(cuda))

        if self.args.qkv_format == "bshd":
            # TODO: micro-batch wise dynamic, possibly move to @data.py:get_data_iterator
            max_seq_len = max(rollout_data["total_lengths"])

            # pad to reduce memory fragmentation and maybe make the computation faster
            pad_size = mpu.get_tensor_model_parallel_world_size() * self.args.data_pad_size_multiplier
            max_seq_len = (max_seq_len + pad_size - 1) // pad_size * pad_size

            rollout_data["max_seq_lens"] = [max_seq_len] * len(rollout_data["tokens"])

        for key in ["rollout_log_probs", "teacher_log_probs"]:
            if key not in rollout_data:
                continue
            rollout_data[key] = [
                torch.tensor(
                    (
                        slice_log_prob_with_cp(
                            log_prob,
                            total_length,
                            response_length,
                            self.args.qkv_format,
                            rollout_data["max_seq_lens"][i] if self.args.qkv_format == "bshd" else None,
                        )
                    ),
                    dtype=torch.float32,
                )
                for i, (log_prob, total_length, response_length) in enumerate(
                    zip(
                        rollout_data[key],
                        rollout_data["total_lengths"],
                        rollout_data["response_lengths"],
                        strict=False,
                    )
                )
            ]
        if "teacher_topk_log_probs" in rollout_data:
            rollout_data["teacher_topk_log_probs"] = [
                torch.tensor(
                    slice_log_prob_with_cp(
                        lp,
                        total_length,
                        response_length,
                        self.args.qkv_format,
                        rollout_data["max_seq_lens"][i] if self.args.qkv_format == "bshd" else None,
                    ),
                    dtype=torch.float32,
                )
                for i, (lp, total_length, response_length) in enumerate(
                    zip(
                        rollout_data["teacher_topk_log_probs"],
                        rollout_data["total_lengths"],
                        rollout_data["response_lengths"],
                        strict=False,
                    )
                )
            ]
        if "teacher_topk_indices" in rollout_data:
            rollout_data["teacher_topk_indices"] = [
                torch.tensor(
                    slice_log_prob_with_cp(
                        indices,
                        total_length,
                        response_length,
                        self.args.qkv_format,
                        rollout_data["max_seq_lens"][i] if self.args.qkv_format == "bshd" else None,
                    ),
                    dtype=torch.long,
                )
                for i, (indices, total_length, response_length) in enumerate(
                    zip(
                        rollout_data["teacher_topk_indices"],
                        rollout_data["total_lengths"],
                        rollout_data["response_lengths"],
                        strict=False,
                    )
                )
            ]
        if "rollout_routed_experts" in rollout_data:
            rollout_data["rollout_routed_experts"] = [
                torch.from_numpy(r) for r in rollout_data["rollout_routed_experts"]
            ]
        return rollout_data

    def _switch_model(self, target_tag: str) -> None:
        if target_tag not in self.weights_backuper.backup_tags:
            raise ValueError(f"Cannot switch to unknown model tag: {target_tag}")
        self.weights_backuper.restore(target_tag)
        self._active_model_tag = target_tag

    def fill_routing_replay(self, data_iterator, num_microbatches, rollout_data):
        if "rollout_routed_experts" not in rollout_data:
            raise ValueError(
                "rollout_routed_experts is required in rollout_data when use_rollout_routing_replay is set."
            )

        from megatron.core.transformer.transformer_block import get_num_layers_to_build
        from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

        from slime.utils.routing_replay import RoutingReplay

        for iterator in data_iterator:
            iterator.reset()

        tp_rank = mpu.get_tensor_model_parallel_rank()
        tp_size = mpu.get_tensor_model_parallel_world_size()

        def pad_func(experts, pad):
            _, num_layers, topk = experts.shape
            pad = (
                torch.arange(
                    pad * num_layers * topk,
                    device=experts.device,
                    dtype=experts.dtype,
                ).reshape((pad, num_layers, topk))
                % self.args.num_experts
            )
            return torch.cat([experts, pad], dim=0)

        for _ in range(sum(num_microbatches)):
            batch = data_iterator[0].get_next(["rollout_routed_experts", "tokens"])
            rollout_routed_experts = batch["rollout_routed_experts"]
            tokens = batch["tokens"]
            assert len(rollout_routed_experts) == len(tokens)
            for a, b in zip(rollout_routed_experts, tokens, strict=False):
                assert a.shape[0] == b.shape[0] - 1, f"{a.shape}, {b.shape}"

            # We need to pad the experts to the last token. We won't calculate loss on this token so this should be fine.
            # TODO: fuse this padding with the following slice_with_cp to reduce memory copy.
            rollout_routed_experts = [pad_func(r, 1) for r in rollout_routed_experts]
            # TODO: maybe extract a common process function for here and get_batch?
            rollout_routed_experts = [slice_with_cp(r, pad_func) for r in rollout_routed_experts]
            rollout_routed_experts = torch.cat(rollout_routed_experts, dim=0)
            pad_size = mpu.get_tensor_model_parallel_world_size() * self.args.data_pad_size_multiplier
            pad = (pad_size - rollout_routed_experts.size(0) % pad_size) % pad_size
            if pad != 0:
                rollout_routed_experts = pad_func(rollout_routed_experts, pad)

            if self.args.sequence_parallel:
                seqlen = rollout_routed_experts.size(0)
                assert seqlen % tp_size == 0
                start, end = seqlen // tp_size * tp_rank, seqlen // tp_size * (tp_rank + 1)
                rollout_routed_experts = rollout_routed_experts[start:end]

            routing_replay_offset = 0
            for vp_stage, model in enumerate(self.model):
                config = model.module.config
                num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage)
                offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
                for layer_id in range(offset, offset + num_layers_to_build):
                    # skip dense layer
                    if isinstance(config.moe_layer_freq, int):
                        if layer_id % config.moe_layer_freq != 0:
                            continue
                    elif isinstance(config.moe_layer_freq, list):
                        assert len(config.moe_layer_freq) == config.num_layers
                        if config.moe_layer_freq[layer_id] == 0:
                            continue
                    layer_routed_experts = rollout_routed_experts[:, layer_id]
                    RoutingReplay.all_routing_replays[routing_replay_offset].record(layer_routed_experts)
                    routing_replay_offset += 1
            assert routing_replay_offset == len(RoutingReplay.all_routing_replays)

        del rollout_data["rollout_routed_experts"]

        for iterator in data_iterator:
            iterator.reset()

    def compute_log_prob(
        self,
        data_iterator: list[DataIterator],
        num_microbatches: list[int],
        store_prefix: str = "",
    ) -> dict[str, list[torch.Tensor]]:

        with timer(f"{store_prefix}log_probs"):
            return forward_only(
                get_log_probs_and_entropy,
                self.args,
                self.model,
                data_iterator,
                num_microbatches,
                store_prefix=store_prefix,
            )

    def set_prm_teacher_log_probs(self, prm_teacher_log_probs):
        """Receive PRM teacher log-probs from a separate PRM teacher actor group."""
        self._prm_teacher_log_probs = prm_teacher_log_probs

    def train(self, rollout_id: int, rollout_data_ref: Box) -> None:
        if self.args.offload_train:
            self.wake_up()

        with timer("data_preprocess"):
            rollout_data = self._get_rollout_data(rollout_data_ref)
            if self.args.debug_rollout_only:
                log_rollout_data(rollout_id, self.args, rollout_data)
                return

        if self.role == "critic":
            return self.train_critic(rollout_id, rollout_data)
        elif self.role == "prm_teacher":
            return self.compute_prm_teacher_log_probs(rollout_id, rollout_data)
        else:
            return self.train_actor(rollout_id, rollout_data)

    def compute_prm_teacher_log_probs(self, rollout_id: int, rollout_data: RolloutBatch):
        """Compute log-probs on the dedicated PRM teacher GPU using hint-enhanced tokens.

        Returns a dict ``{key_with_prm_teacher_prefix: list[Tensor]}`` so that
        any extra outputs produced by ``get_log_probs_and_entropy`` (e.g. the
        per-position top-K log-probs / indices when ``--distill-topk > 0``)
        also travel back to the actor.
        """
        teacher_tokens = rollout_data.get("teacher_tokens")
        if teacher_tokens is not None:
            rollout_data["tokens"] = teacher_tokens
            rollout_data["total_lengths"] = rollout_data["teacher_total_lengths"]
        data_iterator, num_microbatches = get_data_iterator(self.args, self.model, rollout_data)
        # Opt the underlying get_log_probs_and_entropy into also returning
        # per-token top-K log-probs / indices when --distill-topk > 0. The
        # regular old_actor / ref forwards do NOT enter this context, so
        # they remain byte-identical to before.
        with emit_topk_logprobs():
            result = self.compute_log_prob(data_iterator, num_microbatches, store_prefix="prm_teacher_")
        out: dict[str, list] = {}
        for key, val in result.items():
            if isinstance(val, list):
                out[key] = [v.cpu() if isinstance(v, torch.Tensor) else v for v in val]
            else:
                out[key] = val
        return out

    def compute_student_topk(self, rollout_id: int, rollout_data_ref: Box):
        """Run an old_actor forward and return ONLY the student top-K indices.

        Used by ``--distill-subset-mode student``: the outer training loop
        feeds these indices to the PRM teacher actor's ``gather_at_indices``
        so the teacher's log-probs are aligned to the student's subset.

        Returns a dict ``{"topk_indices": list[Tensor]}`` (CPU long tensors)
        on every actor rank; the outer loop reads only rank 0's payload.
        """
        if self.args.offload_train:
            self.wake_up()

        with timer("data_preprocess"):
            rollout_data = self._get_rollout_data(rollout_data_ref)
        data_iterator, num_microbatches = get_data_iterator(self.args, self.model, rollout_data)
        if self.args.keep_old_actor:
            self._switch_model("old_actor")
        else:
            self._switch_model("actor")
        with emit_topk_logprobs():
            result = self.compute_log_prob(
                data_iterator, num_microbatches, store_prefix=""
            )
        topk_indices = result.get("topk_indices", [])
        out_indices = [t.cpu() if isinstance(t, torch.Tensor) else t for t in topk_indices]
        for iterator in data_iterator:
            iterator.reset()
        return {"topk_indices": out_indices}

    def gather_at_indices(
        self, rollout_id: int, rollout_data_ref: Box, indices_per_sample: list
    ):
        """Run a PRM-teacher forward gathering log-probs at provided indices.

        Used by ``--distill-subset-mode student``. ``indices_per_sample`` is
        a list of ``[R_i, K]`` long tensors (CPU) -- the student top-K
        indices computed earlier on the actor side.

        Returns ``{"prm_teacher_topk_log_probs": [...], "prm_teacher_topk_indices": [...]}``.
        """
        if self.args.offload_train:
            self.wake_up()
        with timer("data_preprocess"):
            rollout_data = self._get_rollout_data(rollout_data_ref)
        teacher_tokens = rollout_data.get("teacher_tokens")
        if teacher_tokens is not None:
            rollout_data["tokens"] = teacher_tokens
            rollout_data["total_lengths"] = rollout_data["teacher_total_lengths"]
        data_iterator, num_microbatches = get_data_iterator(self.args, self.model, rollout_data)
        # The payload (`indices_per_sample`) arrives in ORIGINAL sample order
        # (forward_only re-orders results back to original order; see
        # model.py:`forward_only` post-loop reorder).  However, the teacher
        # forward consumes one chunk per sample in the teacher's microbatch
        # order, which differs whenever dynamic-batch packing is on.  Permute
        # the payload to the teacher's consumption order; the post-forward
        # reorder will put the gathered log-probs back to original order so
        # downstream code remains correct.
        teacher_mb_indices = getattr(data_iterator[0], "micro_batch_indices", None)
        if teacher_mb_indices is not None:
            flat_indices: list[int] = sum(teacher_mb_indices, [])
            if len(flat_indices) != len(indices_per_sample):
                raise RuntimeError(
                    f"gather_at_indices: payload size {len(indices_per_sample)} != "
                    f"teacher consumption order length {len(flat_indices)}"
                )
            payload_in_consumption_order = [indices_per_sample[i] for i in flat_indices]
        else:
            payload_in_consumption_order = list(indices_per_sample)
        with timer("prm_teacher_gather_at_indices_log_probs"):
            with set_gather_at_indices(payload_in_consumption_order):
                result = forward_only(
                    gather_log_probs_at_indices,
                    self.args,
                    self.model,
                    data_iterator,
                    num_microbatches,
                    store_prefix="prm_teacher_",
                )
        out: dict[str, list] = {}
        for key, val in result.items():
            if isinstance(val, list):
                out[key] = [v.cpu() if isinstance(v, torch.Tensor) else v for v in val]
            else:
                out[key] = val
        return out

    def train_critic(self, rollout_id: int, rollout_data: RolloutBatch) -> None:
        # Create data iterator for log_probs and train.
        data_iterator, num_microbatches = get_data_iterator(self.args, self.model, rollout_data)
        rollout_data.update(
            forward_only(
                get_values,
                self.args,
                self.model,
                data_iterator,
                num_microbatches,
            )
        )

        if rollout_id >= self.args.num_critic_only_steps:
            sync_actor_critic_data(self.args, rollout_data, self._actor_critic_groups)

        compute_advantages_and_returns(self.args, rollout_data)

        # Offload computed GPU tensors to CPU before training.
        _offload_rollout_data_to_cpu(rollout_data)
        clear_memory()

        self.args.loss_type = "value_loss"
        train(
            rollout_id,
            self.model,
            self.optimizer,
            self.opt_param_scheduler,
            data_iterator,
            num_microbatches,
        )

    def train_actor(self, rollout_id: int, rollout_data: RolloutBatch) -> None:
        # Create data iterator for log_probs and train.
        data_iterator, num_microbatches = get_data_iterator(self.args, self.model, rollout_data)

        if self.args.use_rollout_routing_replay:
            self.fill_routing_replay(data_iterator, num_microbatches, rollout_data)

        with inverse_timer("train_wait"), timer("train"):
            if self.args.compute_advantages_and_returns:
                if "ref" in self.weights_backuper.backup_tags:
                    if self.args.use_routing_replay:
                        os.environ["ROUTING_REPLAY_STAGE"] = "fallthrough"
                    self._switch_model("ref")
                    rollout_data.update(
                        self.compute_log_prob(
                            data_iterator,
                            num_microbatches,
                            store_prefix="ref_",
                        )
                    )
                if hasattr(self, "_prm_teacher_log_probs") and self._prm_teacher_log_probs is not None:
                    payload = self._prm_teacher_log_probs
                    if isinstance(payload, dict):
                        # New (multi-key) shape: {prm_teacher_log_probs: [...],
                        # prm_teacher_topk_log_probs: [...], ...}
                        for k, v in payload.items():
                            rollout_data[k] = v
                    else:
                        # Legacy (single list) shape.
                        rollout_data["prm_teacher_log_probs"] = payload
                    self._prm_teacher_log_probs = None
                self._switch_model("old_actor" if self.args.keep_old_actor else "actor")
                if not self.args.use_rollout_logprobs or self.args.get_mismatch_metrics:
                    if self.args.use_routing_replay:
                        if self.args.use_rollout_routing_replay:
                            os.environ["ROUTING_REPLAY_STAGE"] = "replay_forward"
                        else:
                            os.environ["ROUTING_REPLAY_STAGE"] = "record"
                    # Top-K OPD: wrap old_actor's compute_log_prob in
                    # emit_topk_logprobs() so it ALSO ships per-token
                    # student top-K (`topk_log_probs` / `topk_indices` in
                    # the result dict). Modes:
                    #   * student / overlap: keep these as student-side data.
                    #   * teacher: throw them away below and re-gather at the
                    #     teacher's top-K indices.
                    distill_topk = int(getattr(self.args, "distill_topk", 0) or 0)
                    subset_mode = getattr(self.args, "distill_subset_mode", "student")
                    # `teacher` mode re-gathers student-old at teacher's
                    # indices below, so the student top-K from the regular
                    # pass would be thrown away -- skip emit_topk to save
                    # the extra TP top-K scan.
                    use_emit = distill_topk > 0 and subset_mode in ("student", "overlap")
                    emit_ctx = emit_topk_logprobs() if use_emit else nullcontext()
                    with emit_ctx:
                        rollout_data.update(
                            self.compute_log_prob(
                                data_iterator,
                                num_microbatches,
                                store_prefix="",
                            )
                        )
                    if self.args.use_rollout_routing_replay:
                        RoutingReplay.clear_all_forward()

                    # Mode `teacher`: do an extra old_actor forward that
                    # gathers student-old log-probs at the TEACHER's top-K
                    # indices (already shipped via _prm_teacher_log_probs
                    # → rollout_data["prm_teacher_topk_indices"]). REPLACE
                    # the student-side topk_log_probs / topk_indices so the
                    # loss sees S^q == S^p == teacher top-K (mask all-True).
                    if distill_topk > 0 and subset_mode == "teacher":
                        teacher_idx = rollout_data.get("prm_teacher_topk_indices")
                        if teacher_idx is None or len(teacher_idx) == 0:
                            raise RuntimeError(
                                "subset_mode=teacher requires prm_teacher_topk_indices "
                                "in rollout_data. Did the prm_teacher pass run with "
                                "--distill-topk > 0?"
                            )
                        # Reset iterator: forward_only consumed it once.
                        for it in data_iterator:
                            it.reset()
                        with timer("old_actor_gather_at_teacher_topk"):
                            with set_gather_at_indices(teacher_idx):
                                gather_res = forward_only(
                                    gather_log_probs_at_indices,
                                    self.args,
                                    self.model,
                                    data_iterator,
                                    num_microbatches,
                                    store_prefix="",
                                )
                        # Replace student-side keys with the teacher-aligned ones.
                        rollout_data["topk_log_probs"] = gather_res["topk_log_probs"]
                        rollout_data["topk_indices"] = gather_res["topk_indices"]

                if self.args.use_critic:
                    sync_actor_critic_data(
                        self.args,
                        rollout_data,
                        self._actor_critic_groups,
                    )
                if self._active_model_tag != "actor":
                    self._switch_model("actor")

                # Calculate adv and returns. Need to performed before training (instead of on the fly),
                # because we may need normalize the whole rollout.
                compute_advantages_and_returns(self.args, rollout_data)

            # Move all computed per-sample GPU tensors (log_probs, ref_log_probs,
            # advantages, returns, entropy, values, etc.) back to CPU.
            # log_rollout_data() operates on CPU tensors for logging;
            # train() lazily loads each micro-batch to GPU via get_batch().
            _offload_rollout_data_to_cpu(rollout_data)
            clear_memory()

            if self.rollout_data_postprocess is not None:
                fn = self.rollout_data_postprocess
                try:
                    params = inspect.signature(fn).parameters
                except (TypeError, ValueError):
                    params = {}
                if len(params) >= 2:
                    fn(self.args, rollout_data)
                else:
                    fn(self.args)

            log_rollout_data(
                rollout_id,
                self.args,
                rollout_data,
            )

            # Train
            if self.args.use_routing_replay:
                os.environ["ROUTING_REPLAY_STAGE"] = "replay_backward"
            with timer("actor_train"):
                train(
                    rollout_id,
                    self.model,
                    self.optimizer,
                    self.opt_param_scheduler,
                    data_iterator,
                    num_microbatches,
                )

            self.prof.step(rollout_id=rollout_id)

        train_dump_utils.save_debug_train_data(self.args, rollout_id=rollout_id, rollout_data=rollout_data)

        if self.args.use_routing_replay:
            RoutingReplay.clear_all()

        # update the cpu actor weight to the latest model
        self.weights_backuper.backup("actor")

        # Update ref model if needed
        if (
            self.args.ref_update_interval is not None
            and (rollout_id + 1) % self.args.ref_update_interval == 0
            and "ref" in self.weights_backuper.backup_tags
        ):
            with timer("ref_model_update"):
                if is_megatron_main_rank():
                    logger.info(f"Updating ref model at rollout_id {rollout_id}")
                self.weights_backuper.backup("ref")

        log_perf_data(rollout_id, self.args)

    @timer
    def save_model(self, rollout_id: int, force_sync: bool = False) -> None:
        if self.args.debug_rollout_only:
            return

        # torch dist may trigger nccl communication during saving.
        if self.args.offload_train:
            reload_process_groups()

        if self.args.async_save:
            from megatron.training.async_utils import maybe_finalize_async_save

            maybe_finalize_async_save(blocking=True)

        save(rollout_id, self.model, self.optimizer, self.opt_param_scheduler)

        if force_sync and self.args.async_save:
            maybe_finalize_async_save(blocking=True)

        if self.args.save_hf is not None and self.role == "actor":
            from slime.backends.megatron_utils.model import save_hf_model

            save_hf_model(self.args, rollout_id, self.model)

        if self.args.offload_train:
            destroy_process_groups()

    @timer
    def update_weights(self) -> None:
        if self.args.debug_train_only or self.args.debug_rollout_only:
            return

        if self.args.use_fault_tolerance:
            if dist.get_rank() == 0:
                ray.get(self.rollout_manager.recover_rollout_engines.remote())
            dist.barrier(group=get_gloo_group())

        rollout_engines, rollout_engine_lock, num_new_engines = ray.get(
            self.rollout_manager.get_rollout_engines_and_lock.remote(include_prm=False)
        )

        if self.args.offload_train:
            reload_process_groups()

        if num_new_engines > 0:
            self.weight_updater.connect_rollout_engines(rollout_engines, rollout_engine_lock)
            dist.barrier(group=get_gloo_group())
            if dist.get_rank() == 0:
                ray.get(self.rollout_manager.clear_num_new_engines.remote())

        with torch_memory_saver.disable() if self.args.offload_train else nullcontext():
            print_memory("before update_weights")
            self.weight_updater.update_weights()
            print_memory("after update_weights")

            if self.args.ci_test and len(rollout_engines) > 0:
                engine = random.choice(rollout_engines)
                engine_version = ray.get(engine.get_weight_version.remote())
                if str(engine_version) != str(self.weight_updater.weight_version):
                    raise RuntimeError(
                        f"Weight version mismatch! Engine: {engine_version}, Updater: {self.weight_updater.weight_version}"
                    )

            if getattr(self.args, "keep_old_actor", False):
                if self.args.update_weights_interval == 1:
                    logger.info("updating model queue: rollout_actor -> old_actor, actor -> rollout_actor")
                    # Queue-style update: rollout_actor params -> old_actor, actor params -> rollout_actor
                    # First copy rollout_actor to old_actor
                    self.weights_backuper.copy(src_tag="rollout_actor", dst_tag="old_actor")
                    # Then copy current actor to rollout_actor
                    self.weights_backuper.backup("rollout_actor")
                else:
                    self.weights_backuper.backup("old_actor")

        if self.args.offload_train:
            destroy_process_groups()

    def load_other_checkpoint(self, model_tag: str, path: str) -> None:
        old_args = self.args.load, self.args.no_load_optim, self.args.no_load_rng, self.args.finetune
        self.args.load = path
        self.args.no_load_optim = True
        self.args.no_load_rng = True
        self.args.finetune = True

        if model_tag == "ref" and self.args.ref_ckpt_step is not None:
            old_ckpt_step = self.args.ckpt_step
            self.args.ckpt_step = self.args.ref_ckpt_step

        _, _ = load_checkpoint(
            self.model,
            None,
            None,
            checkpointing_context={},
            skip_load_to_model_and_opt=False,
        )
        self.args.load, self.args.no_load_optim, self.args.no_load_rng, self.args.finetune = old_args

        if model_tag == "ref" and self.args.ref_ckpt_step is not None:
            self.args.ckpt_step = old_ckpt_step

        self.weights_backuper.backup(model_tag)
        self._active_model_tag = model_tag

    def connect_actor_critic(
        self,
        actor_handle: ActorHandle | None = None,
        master_address: str | None = None,
        master_port: int | None = None,
    ) -> None:
        if self.role == "actor":
            master_address = ray.util.get_node_ip_address()
            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = sock.getsockname()[1]
            actor_handle.connect_actor_critic.remote(master_address=master_address, master_port=master_port)

        group_name = "actor_critic"
        world_size = 2
        if is_npu():
            backend = "hccl"
        else:
            backend = "nccl"
        self._actor_critic_groups = init_process_group(
            backend=backend,
            init_method=f"tcp://{master_address}:{master_port}",
            world_size=world_size,
            rank=0 if self.role == "actor" else 1,
            group_name=group_name,
        )
