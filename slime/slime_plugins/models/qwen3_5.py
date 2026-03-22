import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import get_num_layers_to_build
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
from transformers import AutoConfig

Qwen3_5Attention = None
Qwen3_5RMSNorm = None
FusedRMSNormGated = None
ShortConvolution = None
chunk_gated_delta_rule = None

try:
    from fla.modules import FusedRMSNormGated, ShortConvolution
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention, Qwen3_5RMSNorm
except ImportError:
    pass

from .hf_attention import HuggingfaceAttention


def _unwrap_text_config(hf_config):
    return getattr(hf_config, "text_config", hf_config)


class Qwen35GatedDeltaNet(nn.Module):
    """Varlen-compatible gated delta net for Qwen3.5 linear-attention layers."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.layer_norm_epsilon = config.rms_norm_eps

        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = ShortConvolution(
            hidden_size=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
        )

        self.in_proj_qkv = nn.Linear(self.hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        A = torch.empty(self.num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))

        self.norm = FusedRMSNormGated(
            self.head_v_dim,
            eps=self.layer_norm_epsilon,
            device=torch.cuda.current_device(),
            dtype=getattr(config, "dtype", None) or torch.get_current_dtype(),
        )
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

    def fix_query_key_value_ordering(self, mixed_qkv):
        new_tensor_shape_qkv = mixed_qkv.size()[:-1] + (
            self.num_k_heads,
            2 * self.head_k_dim + self.head_v_dim * self.num_v_heads // self.num_k_heads,
        )
        mixed_qkv = mixed_qkv.view(*new_tensor_shape_qkv)
        split_arg_list_qkv = [
            self.head_k_dim,
            self.head_k_dim,
            self.num_v_heads // self.num_k_heads * self.head_v_dim,
        ]
        query, key, value = torch.split(mixed_qkv, split_arg_list_qkv, dim=3)
        value = value.reshape(value.size(0), value.size(1), -1, self.head_v_dim)
        return query, key, value

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor = None,
    ):
        projected_qkv = self.in_proj_qkv(hidden_states)
        projected_z = self.in_proj_z(hidden_states)
        projected_b = self.in_proj_b(hidden_states)
        projected_a = self.in_proj_a(hidden_states)

        query, key, value = self.fix_query_key_value_ordering(projected_qkv)
        query, key, value = (x.reshape(x.shape[0], x.shape[1], -1) for x in (query, key, value))

        mixed_qkv = torch.cat((query, key, value), dim=-1)
        mixed_qkv, _ = self.conv1d(
            x=mixed_qkv,
            cu_seqlens=cu_seqlens,
        )

        query, key, value = torch.split(
            mixed_qkv,
            [
                self.key_dim,
                self.key_dim,
                self.value_dim,
            ],
            dim=-1,
        )
        query = query.reshape(query.shape[0], query.shape[1], -1, self.head_k_dim)
        key = key.reshape(key.shape[0], key.shape[1], -1, self.head_k_dim)
        value = value.reshape(value.shape[0], value.shape[1], -1, self.head_v_dim)

        gate = projected_z.reshape(projected_z.shape[0], projected_z.shape[1], -1, self.head_v_dim)
        beta = projected_b.sigmoid()
        alpha = projected_a
        g = -self.A_log.float().exp() * F.softplus(alpha.float() + self.dt_bias)

        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        core_attn_out, _ = chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )

        gate_shape = gate.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        gate = gate.reshape(-1, gate.shape[-1])
        core_attn_out = self.norm(core_attn_out, gate)
        core_attn_out = core_attn_out.reshape(gate_shape)
        core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1)

        return self.out_proj(core_attn_out)


class Attention(HuggingfaceAttention):
    def __init__(
        self,
        args,
        config,
        layer_number: int,
        cp_comm_type: str = "p2p",
        pg_collection=None,
    ):
        super().__init__(
            args,
            config,
            layer_number,
            cp_comm_type,
            pg_collection,
        )
        if (
            Qwen3_5Attention is None
            or Qwen3_5RMSNorm is None
            or FusedRMSNormGated is None
            or ShortConvolution is None
            or chunk_gated_delta_rule is None
        ):
            raise ImportError("Please install a transformers build with Qwen3.5 support.")

        self.hf_config = _unwrap_text_config(self.hf_config)
        self.linear_attn = Qwen35GatedDeltaNet(self.hf_config, self.hf_layer_idx)
        self.input_layernorm = Qwen3_5RMSNorm(self.hf_config.hidden_size, eps=self.hf_config.rms_norm_eps)

    def hf_forward(self, hidden_states, packed_seq_params):
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.linear_attn(
            hidden_states=hidden_states,
            cu_seqlens=packed_seq_params.cu_seqlens_q,
        )
        return hidden_states


def get_qwen3_5_spec(args, config, vp_stage):
    kwargs = {
        "use_transformer_engine": True,
    }
    if vp_stage is not None:
        kwargs["vp_stage"] = vp_stage
    transformer_layer_spec = get_gpt_decoder_block_spec(config, **kwargs)

    assert config.pipeline_model_parallel_layout is None, "not support this at the moment"

    num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage)
    offset = get_transformer_layer_offset(config, vp_stage=vp_stage)

    hf_config = _unwrap_text_config(AutoConfig.from_pretrained(args.hf_checkpoint, trust_remote_code=True))

    for layer_id in range(num_layers_to_build):
        if hf_config.layer_types[layer_id + offset] == "linear_attention":
            layer_specs = copy.deepcopy(transformer_layer_spec.layer_specs[layer_id])
            layer_specs.submodules.self_attention = ModuleSpec(
                module=Attention,
                params={"args": args},
            )
            transformer_layer_spec.layer_specs[layer_id] = layer_specs
    return transformer_layer_spec
