from __future__ import annotations

import copy
import itertools
import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from megatron.core import mpu, tensor_parallel
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.common.embeddings.rope_utils import get_pos_emb_on_this_cp_rank
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import get_num_layers_to_build
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
from megatron.core.utils import deprecate_inference_params
from torch import Tensor

from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.transformer_block import Qwen3VLTransformerBlock
try:
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.utils import (
        get_rope_index as get_qwen3vl_rope_index,
    )
except ImportError:
    # Megatron-Bridge changed the exported rope helper name/signature across versions.
    # The local Qwen3.5 rope path below can handle multimodal batches without it.
    get_qwen3vl_rope_index = None
from megatron.bridge.models.qwen_vl.qwen3_vl_provider import Qwen3VLModelProvider
from megatron.bridge.utils.common_utils import hook_hf_module_setattr_for_tp_grad_sync
from slime_plugins.models.qwen3_5 import Qwen3_5GatedDeltaNet

from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

logger = logging.getLogger(__name__)
_QWEN35_BRIDGE_FORWARD_LOGGED = False
_QWEN35_SEQPAR_LOGITS_GATHER_LOGGED = False


def _should_log_qwen35_bridge_debug() -> bool:
    return os.getenv("SLIME_DEBUG_QWEN35_BRIDGE", "").lower() in {"1", "true", "yes", "on"}


def _should_gather_qwen35_sequence_parallel_logits() -> bool:
    return os.getenv("SLIME_QWEN35_GATHER_SEQPAR_LOGITS", "1").lower() in {"1", "true", "yes", "on"}


def _build_attention_mask_from_cu_seqlens(
    cu_seqlens: torch.Tensor,
    *,
    max_seq_len: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    attention_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.long, device=device)
    seq_lens = cu_seqlens[1 : batch_size + 1] - cu_seqlens[:batch_size]
    for i, seq_len in enumerate(seq_lens.tolist()):
        valid = min(int(seq_len), max_seq_len)
        attention_mask[i, :valid] = 1
    return attention_mask


def _thd_to_bshd(
    packed_values: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    batch_size: int,
) -> torch.Tensor:
    seq_lens = cu_seqlens[1 : batch_size + 1] - cu_seqlens[:batch_size]
    max_seq_len = int(seq_lens.max().item()) if batch_size > 0 else 0
    trailing_shape = packed_values.shape[2:]
    results = packed_values.new_zeros((batch_size, max_seq_len, *trailing_shape))
    for i, seq_len in enumerate(seq_lens.tolist()):
        results[i, :seq_len] = packed_values[0, cu_seqlens[i] : cu_seqlens[i] + seq_len]
    return results


def _bshd_to_thd(
    unpacked_values: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    batch_size: int,
) -> torch.Tensor:
    trailing_shape = unpacked_values.shape[2:]
    total_len = int(cu_seqlens[-1].item())
    results = unpacked_values.new_zeros((1, total_len, *trailing_shape))
    seq_lens = cu_seqlens[1 : batch_size + 1] - cu_seqlens[:batch_size]
    for i, seq_len in enumerate(seq_lens.tolist()):
        results[0, cu_seqlens[i] : cu_seqlens[i] + seq_len] = unpacked_values[i, :seq_len]
    return results


class Qwen35BridgeAttention(MegatronModule):
    def __init__(
        self,
        config,
        layer_number: int,
        hf_config,
        sequence_parallel: bool = False,
        cp_comm_type: str = "p2p",
        pg_collection=None,
    ):
        super().__init__(config=config)
        self.config = config
        self.layer_number = layer_number
        self.hf_layer_idx = layer_number - 1
        self.sequence_parallel = sequence_parallel
        self.hf_config = copy.deepcopy(hf_config)
        dtype_name = getattr(self.hf_config, "dtype", None)
        if isinstance(dtype_name, str):
            self.hf_config.dtype = getattr(torch, dtype_name, dtype_name)
        self.hf_config._attn_implementation = "flash_attention_2"
        self.linear_attn = Qwen3_5GatedDeltaNet(self.hf_config, self.hf_layer_idx)

        try:
            from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextRMSNorm

            self.input_layernorm = Qwen3NextRMSNorm(
                self.hf_config.hidden_size,
                eps=self.hf_config.rms_norm_eps,
            )
        except ImportError:
            from torch.nn import RMSNorm

            self.input_layernorm = RMSNorm(self.hf_config.hidden_size, eps=self.hf_config.rms_norm_eps)

    def hf_forward(self, hidden_states: torch.Tensor, packed_seq_params: PackedSeqParams) -> torch.Tensor:
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.linear_attn(
            hidden_states=hidden_states,
            cu_seqlens=packed_seq_params.cu_seqlens_q,
        )
        return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        key_value_states: torch.Tensor | None = None,
        inference_context: BaseInferenceContext | None = None,
        rotary_pos_emb: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None,
        rotary_pos_cos: torch.Tensor | None = None,
        rotary_pos_sin: torch.Tensor | None = None,
        rotary_pos_cos_sin: torch.Tensor | None = None,
        attention_bias: torch.Tensor | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        sequence_len_offset: int | None = None,
        *,
        inference_params: BaseInferenceContext | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        global _QWEN35_BRIDGE_FORWARD_LOGGED
        if _should_log_qwen35_bridge_debug() and not _QWEN35_BRIDGE_FORWARD_LOGGED:
            _QWEN35_BRIDGE_FORWARD_LOGGED = True
            logger.warning(
                "Qwen35 bridge debug: entered Qwen35BridgeAttention.forward "
                "file=%s layer_number=%s packed_seq=%s hidden_states_shape=%s",
                self.forward.__code__.co_filename,
                self.layer_number,
                packed_seq_params is not None,
                tuple(hidden_states.shape),
            )
        assert packed_seq_params is not None
        cu_seqlens = packed_seq_params.cu_seqlens_q

        if self.sequence_parallel:
            hidden_states = tensor_parallel.gather_from_sequence_parallel_region(
                hidden_states,
                group=mpu.get_tensor_model_parallel_group(),
            )

        if mpu.get_context_parallel_world_size() > 1:
            cp_size = mpu.get_context_parallel_world_size()
            hidden_states_list = dist.nn.all_gather(
                hidden_states,
                group=mpu.get_context_parallel_group(),
            )

            whole_hidden_states_list = []
            local_cu_seqlens = cu_seqlens // cp_size
            for i in range(len(cu_seqlens) - 1):
                seqlen = cu_seqlens[i + 1] - cu_seqlens[i]
                chunk_size = seqlen // 2 // cp_size
                whole_hidden_states_list.extend(
                    [
                        hidden_states_list[cp_rank][local_cu_seqlens[i] : local_cu_seqlens[i] + chunk_size]
                        for cp_rank in range(cp_size)
                    ]
                    + [
                        hidden_states_list[cp_rank][local_cu_seqlens[i] + chunk_size : local_cu_seqlens[i + 1]]
                        for cp_rank in range(cp_size)
                    ][::-1],
                )
            hidden_states = torch.cat(whole_hidden_states_list, dim=0)

        hidden_states = hidden_states.permute(1, 0, 2)
        output = self.hf_forward(hidden_states, packed_seq_params)
        bias = None
        output = output.permute(1, 0, 2)

        if mpu.get_context_parallel_world_size() > 1:
            cp_rank = mpu.get_context_parallel_rank()
            output_list = []
            for i in range(len(cu_seqlens) - 1):
                seqlen = cu_seqlens[i + 1] - cu_seqlens[i]
                chunk_size = seqlen // 2 // mpu.get_context_parallel_world_size()
                seq = output[cu_seqlens[i] : cu_seqlens[i + 1]]
                chunks = torch.chunk(seq, 2 * mpu.get_context_parallel_world_size(), dim=0)
                output_list.append(chunks[cp_rank])
                output_list.append(chunks[2 * mpu.get_context_parallel_world_size() - 1 - cp_rank])
            output = torch.cat(output_list, dim=0)

        if self.sequence_parallel:
            output = tensor_parallel.scatter_to_sequence_parallel_region(
                output,
                group=mpu.get_tensor_model_parallel_group(),
            )

        return output, bias


class Qwen35TextRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mrope_section = list(config.rope_parameters.get("mrope_section", [11, 11, 10]))
        inv_freq, _ = self.compute_default_rope_parameters(config)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @staticmethod
    def compute_default_rope_parameters(config) -> tuple[torch.Tensor, float]:
        base = config.rope_parameters["rope_theta"]
        partial_rotary_factor = config.rope_parameters.get("partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        dim = int(head_dim * partial_rotary_factor)
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64, device=None).float() / dim)
        )
        return inv_freq, 1.0

    @staticmethod
    def apply_interleaved_mrope(freqs: torch.Tensor, mrope_section: List[int]) -> torch.Tensor:
        freqs_t = freqs[0]
        for dim, offset in enumerate((1, 2), start=1):
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    def forward(
        self,
        position_ids: torch.Tensor,
        mrope_section: List[int] | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        cp_group: torch.distributed.ProcessGroup | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del packed_seq_params, kwargs
        if mrope_section is None:
            mrope_section = self.mrope_section
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(2, 3)
        freqs = self.apply_interleaved_mrope(freqs, mrope_section)
        emb = torch.cat((freqs, freqs), dim=-1)
        emb = emb[..., None, :].transpose(0, 1).contiguous()
        if cp_group is not None and dist.is_available() and dist.is_initialized():
            try:
                if dist.get_world_size(cp_group) > 1:
                    emb = get_pos_emb_on_this_cp_rank(emb, 0, cp_group)
            except Exception:
                # Bridge/runtime versions differ in when CP groups are initialized.
                # Ignore the hint rather than failing on single-CP jobs.
                pass
        return emb


class Qwen35VLGPTModel(GPTModel):
    def __init__(
        self,
        config,
        transformer_layer_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        position_embedding_type: str = "learned_absolute",
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        rope_scaling: bool = False,
        rope_scaling_factor: float = 8.0,
        scatter_embedding_sequence_parallel: bool = True,
        seq_len_interpolation_factor: Optional[float] = None,
        mtp_block_spec: Optional[ModuleSpec] = None,
        vp_stage: Optional[int] = None,
    ) -> None:
        super().__init__(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=fp16_lm_cross_entropy,
            parallel_output=parallel_output,
            share_embeddings_and_output_weights=share_embeddings_and_output_weights,
            position_embedding_type=position_embedding_type,
            rotary_percent=rotary_percent,
            rotary_base=rotary_base,
            rope_scaling=rope_scaling,
            rope_scaling_factor=rope_scaling_factor,
            scatter_embedding_sequence_parallel=scatter_embedding_sequence_parallel,
            seq_len_interpolation_factor=seq_len_interpolation_factor,
            mtp_block_spec=mtp_block_spec,
            vp_stage=vp_stage,
        )

        self.rotary_pos_emb = Qwen35TextRotaryEmbedding(config.hf_text_config)
        self.mrope_section = self.config.mrope_section
        assert self.mrope_section is not None, "mrope requires mrope_section in the transformer config"

        self.decoder = Qwen3VLTransformerBlock(
            config=self.config,
            spec=transformer_layer_spec,
            pre_process=self.pre_process,
            post_process=self.post_process,
            vp_stage=vp_stage,
        )

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        inference_context: BaseInferenceContext = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        runtime_gather_output: Optional[bool] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        loss_mask: Optional[Tensor] = None,
        mtp_kwargs: Optional[dict] = None,
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
    ) -> Tensor:
        global _QWEN35_SEQPAR_LOGITS_GATHER_LOGGED
        inference_context = deprecate_inference_params(inference_context, inference_params)

        preproc_output = self._preprocess(
            input_ids=input_ids,
            position_ids=position_ids,
            decoder_input=decoder_input,
            inference_context=inference_context,
            packed_seq_params=packed_seq_params,
        )
        decoder_input, rotary_pos_emb, rotary_pos_cos, rotary_pos_sin, sequence_len_offset = preproc_output[:5]

        hidden_states = self.decoder(
            hidden_states=decoder_input,
            attention_mask=attention_mask,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **(extra_block_kwargs or {}),
        )

        outputs = self._postprocess(
            hidden_states=hidden_states,
            input_ids=input_ids,
            position_ids=position_ids,
            labels=labels,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            mtp_in_postprocess=self.mtp_process,
            loss_mask=loss_mask,
            decoder_input=decoder_input,
            attention_mask=attention_mask,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            runtime_gather_output=runtime_gather_output,
            mtp_kwargs=mtp_kwargs,
            extra_block_kwargs=extra_block_kwargs,
            inference_context=inference_context,
        )

        should_gather_seqpar_logits = (
            labels is None
            and self.config.sequence_parallel
            and _should_gather_qwen35_sequence_parallel_logits()
            and isinstance(outputs, torch.Tensor)
            and outputs.ndim == 3
            and packed_seq_params is not None
            and packed_seq_params.qkv_format == "thd"
            and packed_seq_params.cu_seqlens_q is not None
        )
        if should_gather_seqpar_logits:
            expected_tokens = int(packed_seq_params.cu_seqlens_q[-1].item())
            local_tokens = int(outputs.size(1))
            tp_world_size = mpu.get_tensor_model_parallel_world_size()
            if local_tokens != expected_tokens and local_tokens * tp_world_size == expected_tokens:
                gathered_outputs = tensor_parallel.gather_from_sequence_parallel_region(
                    outputs.transpose(0, 1).contiguous(),
                    tensor_parallel_output_grad=False,
                    group=mpu.get_tensor_model_parallel_group(),
                )
                outputs = gathered_outputs.transpose(0, 1).contiguous()
                if _should_log_qwen35_bridge_debug() and not _QWEN35_SEQPAR_LOGITS_GATHER_LOGGED:
                    _QWEN35_SEQPAR_LOGITS_GATHER_LOGGED = True
                    logger.warning(
                        "Qwen35 bridge debug: gathered sequence-parallel logits "
                        "local_tokens=%s expected_tokens=%s tp_world_size=%s output_shape=%s",
                        local_tokens,
                        expected_tokens,
                        tp_world_size,
                        tuple(outputs.shape),
                    )

        return outputs

    def _postprocess(
        self,
        hidden_states,
        input_ids,
        position_ids,
        labels,
        rotary_pos_emb,
        rotary_pos_cos,
        rotary_pos_sin,
        mtp_in_postprocess=None,
        loss_mask=None,
        decoder_input=None,
        attention_mask=None,
        inference_params=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        runtime_gather_output=None,
        mtp_kwargs=None,
        extra_block_kwargs=None,
        inference_context=None,
        is_spec_decode=None,
    ):
        mtp_labels = None if mtp_kwargs is None else mtp_kwargs.get("mtp_labels", None)
        should_skip_mtp_loss = labels is None and mtp_labels is None and self.config.mtp_num_layers
        if not should_skip_mtp_loss:
            return super()._postprocess(
                hidden_states=hidden_states,
                input_ids=input_ids,
                position_ids=position_ids,
                labels=labels,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                mtp_in_postprocess=mtp_in_postprocess,
                loss_mask=loss_mask,
                decoder_input=decoder_input,
                attention_mask=attention_mask,
                inference_params=inference_params,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
                runtime_gather_output=runtime_gather_output,
                extra_block_kwargs=extra_block_kwargs,
                inference_context=inference_context,
                is_spec_decode=is_spec_decode,
            )

        original_mtp_num_layers = self.config.mtp_num_layers
        try:
            self.config.mtp_num_layers = None
            return super()._postprocess(
                hidden_states=hidden_states,
                input_ids=input_ids,
                position_ids=position_ids,
                labels=labels,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                mtp_in_postprocess=mtp_in_postprocess,
                loss_mask=loss_mask,
                decoder_input=decoder_input,
                attention_mask=attention_mask,
                inference_params=inference_params,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
                runtime_gather_output=runtime_gather_output,
                extra_block_kwargs=extra_block_kwargs,
                inference_context=inference_context,
                is_spec_decode=is_spec_decode,
            )
        finally:
            self.config.mtp_num_layers = original_mtp_num_layers


def get_qwen35_rope_index(
    spatial_merge_size: int,
    image_token_id: int,
    video_token_id: int,
    vision_start_token_id: int,
    input_ids: torch.LongTensor,
    mm_token_type_ids: torch.IntTensor | None = None,
    image_grid_thw: torch.LongTensor | None = None,
    video_grid_thw: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    packed_seq_params: PackedSeqParams | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if mm_token_type_ids is None:
        if get_qwen3vl_rope_index is None:
            raise RuntimeError(
                "Megatron-Bridge does not export qwen3_vl.get_rope_index in this environment, "
                "and mm_token_type_ids was not provided for the local Qwen3.5 rope fallback."
            )
        return get_qwen3vl_rope_index(
            spatial_merge_size=spatial_merge_size,
            image_token_id=image_token_id,
            video_token_id=video_token_id,
            vision_start_token_id=vision_start_token_id,
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
            packed_seq_params=packed_seq_params,
        )

    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1

    if packed_seq_params is not None and attention_mask is None:
        attention_mask = _build_attention_mask_from_cu_seqlens(
            packed_seq_params.cu_seqlens_q,
            max_seq_len=input_ids.shape[1],
            batch_size=mm_token_type_ids.shape[0],
            device=input_ids.device,
        )

    mrope_position_deltas = []
    position_ids = torch.zeros(
        3,
        input_ids.shape[0],
        input_ids.shape[1],
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    grid_iters = {
        1: iter(image_grid_thw) if image_grid_thw is not None else None,
        2: iter(video_grid_thw) if video_grid_thw is not None else None,
    }

    for batch_idx, current_input_ids in enumerate(input_ids):
        input_token_type = mm_token_type_ids[batch_idx]
        if attention_mask is not None:
            current_input_ids = current_input_ids[attention_mask[batch_idx].bool()]
            input_token_type = input_token_type[attention_mask[batch_idx].bool()]

        input_type_group = []
        for key, group in itertools.groupby(enumerate(input_token_type.tolist()), lambda x: x[1]):
            group = list(group)
            start_index = group[0][0]
            end_index = group[-1][0] + 1
            input_type_group.append((key, start_index, end_index))

        current_pos = 0
        llm_pos_ids_list = []
        for modality_type, start_idx, end_idx in input_type_group:
            if modality_type == 0:
                text_len = end_idx - start_idx
                llm_pos_ids_list.append(
                    torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1) + current_pos
                )
                current_pos += text_len
                continue

            grid_iter = grid_iters.get(modality_type)
            if grid_iter is None:
                raise ValueError(f"Missing grid_thw iterator for modality_type={modality_type}")
            grid_thw = next(grid_iter)
            llm_grid_t = grid_thw[0].item()
            llm_grid_h = grid_thw[1].item() // spatial_merge_size
            llm_grid_w = grid_thw[2].item() // spatial_merge_size

            text_len = end_idx - start_idx - (llm_grid_t * llm_grid_h * llm_grid_w)
            if text_len > 0:
                llm_pos_ids_list.append(
                    torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1) + current_pos
                )
                current_pos += text_len

            t_index = torch.arange(llm_grid_t, device=input_ids.device).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
            h_index = torch.arange(llm_grid_h, device=input_ids.device).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
            w_index = torch.arange(llm_grid_w, device=input_ids.device).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
            llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + current_pos)
            current_pos += max(llm_grid_h, llm_grid_w)

        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        if attention_mask is not None:
            position_ids[:, batch_idx, attention_mask[batch_idx].bool()] = llm_positions.to(position_ids.device)
        else:
            position_ids[:, batch_idx] = llm_positions.to(position_ids.device)
        mrope_position_deltas.append(llm_positions.max() + 1 - len(current_input_ids))

    mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
    return position_ids, mrope_position_deltas


class Qwen35VLModel(MegatronModule):
    def __init__(
        self,
        language_transformer_config,
        language_transformer_layer_spec: ModuleSpec,
        vision_transformer_config,
        parallel_output: bool = True,
        pre_process: bool = True,
        post_process: bool = True,
        add_encoder: bool = True,
        add_decoder: bool = True,
    ) -> None:
        super().__init__(config=language_transformer_config)

        self.pre_process = pre_process
        self.post_process = post_process
        self.add_encoder = add_encoder
        self.add_decoder = add_decoder
        self.encoder_hidden_state = None
        self.vision_model = None
        self.language_model = None
        self.image_token_id = language_transformer_config.image_token_id
        self.video_token_id = language_transformer_config.video_token_id
        self.vision_start_token_id = language_transformer_config.vision_start_token_id
        self.share_embeddings_and_output_weights = False

        deepstack_visual_indexes = getattr(vision_transformer_config, "deepstack_visual_indexes", [])
        assert not deepstack_visual_indexes, "Qwen3.5 local bridge does not support deepstack visual features"

        if self.pre_process:
            self.vision_model = Qwen3_5VisionModel._from_config(vision_transformer_config)
            hook_hf_module_setattr_for_tp_grad_sync(self.vision_model)
            if torch.cuda.is_available():
                self.vision_model = self.vision_model.to("cuda")

        self.language_model = Qwen35VLGPTModel(
            config=language_transformer_config,
            transformer_layer_spec=language_transformer_layer_spec,
            vocab_size=language_transformer_config.vocab_size,
            max_sequence_length=language_transformer_config.language_max_sequence_length,
            parallel_output=parallel_output,
            position_embedding_type="mrope",
            rotary_percent=language_transformer_config.rotary_percent,
            pre_process=self.pre_process,
            post_process=self.post_process,
            rotary_base=language_transformer_config.rotary_base,
            fp16_lm_cross_entropy=language_transformer_config.fp16_lm_cross_entropy,
            share_embeddings_and_output_weights=language_transformer_config.share_embeddings_and_output_weights,
            scatter_embedding_sequence_parallel=False,
        )
        self.share_embeddings_and_output_weights = self.language_model.share_embeddings_and_output_weights

    def shared_embedding_or_output_weight(self):
        if self.add_decoder:
            return self.language_model.shared_embedding_or_output_weight()
        return None

    def set_input_tensor(self, input_tensor) -> None:
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1, "input_tensor should only be length 1 for Qwen3.5 VL"

        if self.pre_process:
            self.encoder_hidden_state = input_tensor[0]
        else:
            self.language_model.set_input_tensor(input_tensor[0])

    def freeze(
        self,
        freeze_language_model: bool,
        freeze_vision_model: bool,
        freeze_vision_projection: bool,
    ):
        modules = []
        if freeze_language_model and self.language_model is not None:
            modules.append(self.language_model)
        if freeze_vision_model and self.vision_model is not None:
            if hasattr(self.vision_model, "patch_embed"):
                modules.append(self.vision_model.patch_embed)
            if hasattr(self.vision_model, "blocks"):
                modules.append(self.vision_model.blocks)
            if hasattr(self.vision_model, "pos_embed"):
                modules.append(self.vision_model.pos_embed)
            if hasattr(self.vision_model, "rotary_pos_emb"):
                modules.append(self.vision_model.rotary_pos_emb)
        if freeze_vision_projection and self.vision_model is not None and hasattr(self.vision_model, "merger"):
            modules.append(self.vision_model.merger)
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        loss_mask: torch.Tensor = None,
        inference_params: BaseInferenceContext = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        pixel_values: torch.Tensor = None,
        pixel_values_videos: torch.Tensor = None,
        image_grid_thw: torch.Tensor = None,
        video_grid_thw: torch.Tensor = None,
        image_input_mask: torch.Tensor = None,
        mm_token_type_ids: torch.Tensor = None,
        mtp_kwargs: Optional[dict] = None,
    ) -> torch.Tensor:
        assert pixel_values_videos is None and video_grid_thw is None, "Qwen3.5 local bridge does not support video"
        assert inference_params is None, "Qwen3.5 local bridge does not support inference"

        position_ids = None
        image_mask = None

        if self.pre_process:
            if image_grid_thw is not None:
                image_mask = image_input_mask
                if image_mask is None:
                    image_mask = (input_ids == self.image_token_id).contiguous()

            vision_embeds = None
            if image_grid_thw is not None and image_grid_thw.shape[0] > 0:
                vision_output = self.vision_model(
                    hidden_states=pixel_values,
                    grid_thw=image_grid_thw,
                )
                vision_embeds = vision_output.pooler_output

            combined_embeddings = self.language_model.embedding(
                input_ids=input_ids,
                position_ids=None,
            ).clone()

            if vision_embeds is not None:
                if image_mask is None:
                    raise ValueError("image_mask is required when image embeddings are present")
                if int(image_mask.sum().item()) != int(vision_embeds.shape[0]):
                    raise ValueError(
                        f"Image placeholder count ({int(image_mask.sum().item())}) does not match image embeds "
                        f"({int(vision_embeds.shape[0])})"
                    )
                combined_embeddings = combined_embeddings.transpose(0, 1).contiguous()
                combined_embeddings[image_mask] = vision_embeds.to(
                    device=combined_embeddings.device,
                    dtype=combined_embeddings.dtype,
                )
                combined_embeddings = combined_embeddings.transpose(0, 1).contiguous()

            if self.config.sequence_parallel:
                combined_embeddings = tensor_parallel.scatter_to_sequence_parallel_region(combined_embeddings)
                combined_embeddings = combined_embeddings.contiguous()
        else:
            combined_embeddings = None

        if mm_token_type_ids is not None and mm_token_type_ids.dim() == 1:
            mm_token_type_ids = mm_token_type_ids.unsqueeze(0)

        cu_seqlens_padded = None
        if packed_seq_params is not None:
            if packed_seq_params.cu_seqlens_q_padded is not None:
                cu_seqlens_padded = packed_seq_params.cu_seqlens_q_padded
            else:
                cu_seqlens_padded = packed_seq_params.cu_seqlens_q

        if position_ids is None:
            if mm_token_type_ids is None or cu_seqlens_padded is None:
                position_ids, _ = get_qwen35_rope_index(
                    self.config.spatial_merge_size,
                    self.image_token_id,
                    self.video_token_id,
                    self.vision_start_token_id,
                    input_ids,
                    mm_token_type_ids=mm_token_type_ids,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    attention_mask=attention_mask,
                    packed_seq_params=packed_seq_params,
                )
            else:
                batch_size = mm_token_type_ids.shape[0]
                input_ids_for_rope_index = _thd_to_bshd(
                    input_ids,
                    cu_seqlens_padded,
                    batch_size=batch_size,
                )
                position_ids, _ = get_qwen35_rope_index(
                    self.config.spatial_merge_size,
                    self.image_token_id,
                    self.video_token_id,
                    self.vision_start_token_id,
                    input_ids_for_rope_index,
                    mm_token_type_ids=mm_token_type_ids,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    attention_mask=None,
                    packed_seq_params=packed_seq_params,
                )
                position_ids = _bshd_to_thd(
                    position_ids.permute(1, 2, 0),
                    cu_seqlens_padded,
                    batch_size=batch_size,
                ).permute(2, 0, 1)

        return self.language_model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_input=combined_embeddings,
            packed_seq_params=packed_seq_params,
            loss_mask=loss_mask,
            mtp_kwargs=mtp_kwargs,
            extra_block_kwargs=extra_block_kwargs,
            visual_pos_masks=None,
            deepstack_visual_embeds=None,
        )


def _get_qwen35_layer_spec(provider: "Qwen35VLModelProvider", vp_stage=None):
    config = provider
    if not getattr(config, "num_experts", None):
        config.moe_layer_freq = [0] * config.num_layers

    kwargs = {"use_transformer_engine": True}
    if vp_stage is not None:
        kwargs["vp_stage"] = vp_stage
    transformer_layer_spec = get_gpt_decoder_block_spec(config, **kwargs)

    assert config.pipeline_model_parallel_layout is None, "pipeline layout override is not supported"

    num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage)
    offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
    layer_types = list(getattr(provider.hf_text_config, "layer_types", []))
    if not layer_types:
        interval = getattr(provider.hf_text_config, "full_attention_interval", 4)
        layer_types = [
            "full_attention" if (i + 1) % interval == 0 else "linear_attention"
            for i in range(provider.hf_text_config.num_hidden_layers)
        ]

    replaced_linear_layers = []
    for layer_id in range(num_layers_to_build):
        if layer_types[layer_id + offset] == "linear_attention":
            layer_specs = copy.deepcopy(transformer_layer_spec.layer_specs[layer_id])
            layer_specs.submodules.self_attention = ModuleSpec(
                module=Qwen35BridgeAttention,
                params={
                    "hf_config": provider.hf_text_config,
                    "sequence_parallel": provider.sequence_parallel,
                },
            )
            transformer_layer_spec.layer_specs[layer_id] = layer_specs
            replaced_linear_layers.append(layer_id + offset)

    if _should_log_qwen35_bridge_debug():
        logger.warning(
            "Qwen35 bridge debug: _get_qwen35_layer_spec file=%s vp_stage=%s "
            "offset=%s num_layers_to_build=%s layer_types_head=%s replaced_linear_layers=%s",
            _get_qwen35_layer_spec.__code__.co_filename,
            vp_stage,
            offset,
            num_layers_to_build,
            layer_types[:32],
            replaced_linear_layers,
        )

    return transformer_layer_spec


@dataclass
class Qwen35VLModelProvider(Qwen3VLModelProvider):
    hf_checkpoint_path: Optional[str] = None
    deepstack_visual_indexes: List[int] = field(default_factory=list)

    def provide(self, pre_process=None, post_process=None, vp_stage=None):
        language_transformer_layer_spec = _get_qwen35_layer_spec(self, vp_stage=vp_stage)

        model = Qwen35VLModel(
            language_transformer_config=self,
            language_transformer_layer_spec=language_transformer_layer_spec,
            vision_transformer_config=self.vision_config,
            pre_process=pre_process,
            post_process=post_process,
        )

        if self.freeze_language_model or self.freeze_vision_model or self.freeze_vision_projection:
            model.freeze(
                freeze_language_model=self.freeze_language_model,
                freeze_vision_model=self.freeze_vision_model,
                freeze_vision_projection=self.freeze_vision_projection,
            )

        return model

    def provide_language_model(self, pre_process=None, post_process=None, vp_stage=None):
        language_transformer_layer_spec = _get_qwen35_layer_spec(self, vp_stage=vp_stage)
        return Qwen35VLGPTModel(
            config=self,
            transformer_layer_spec=language_transformer_layer_spec,
            vocab_size=self.vocab_size,
            max_sequence_length=self.language_max_sequence_length,
            parallel_output=True,
            position_embedding_type="mrope",
            rotary_percent=self.rotary_percent,
            pre_process=pre_process,
            post_process=post_process,
            rotary_base=self.rotary_base,
            fp16_lm_cross_entropy=self.fp16_lm_cross_entropy,
            share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
            scatter_embedding_sequence_parallel=False,
            vp_stage=vp_stage,
        )
