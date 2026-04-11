"""
Text-only Qwen3.5 bridge for megatron.bridge.

This is used only for the isolated text-only retool path. Multimodal runs keep
using the local VLM bridge registered in `qwen3_5.py`.
"""

from __future__ import annotations

import logging

import torch
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import AutoMapping, GatedMLPMapping, QKVMapping
from megatron.core.models.gpt import GPTModel

logger = logging.getLogger(__name__)

# Use a string so we don't need transformers to have Qwen3.5 at import time
_Qwen3_5HF = "Qwen3_5ForConditionalGeneration"


def _get_text_config(hf_config):
    """Unwrap text_config from VLM config if present."""
    return getattr(hf_config, "text_config", hf_config)


@MegatronModelBridge.register_bridge(source=_Qwen3_5HF, target=GPTModel)
class MegatronQwen35TextBridge(MegatronModelBridge):
    """Bridge between HuggingFace Qwen3.5 and Megatron GPTModel."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hf_pretrained = None

    def load_weights_hf_to_megatron(self, hf_pretrained, model):
        """Store hf_pretrained before calling parent's load method."""
        self.hf_pretrained = hf_pretrained
        return super().load_weights_hf_to_megatron(hf_pretrained, model)

    def provider_bridge(self, hf_pretrained):
        """Create a GPT ModelProvider from Qwen3.5 HF config."""
        from megatron.bridge.models.qwen.qwen_provider import Qwen3ModelProvider

        hf_config = hf_pretrained.config
        text_config = _get_text_config(hf_config)

        model_dtype = self.dtype_from_hf(text_config, default=torch.bfloat16)

        rope_params = getattr(text_config, "rope_parameters", {}) or {}
        rope_theta = rope_params.get("rope_theta", getattr(text_config, "rope_theta", 10000000))
        partial_rotary_factor = rope_params.get(
            "partial_rotary_factor", getattr(text_config, "partial_rotary_factor", 0.25)
        )

        provider = Qwen3ModelProvider(
            num_layers=text_config.num_hidden_layers,
            hidden_size=text_config.hidden_size,
            ffn_hidden_size=text_config.intermediate_size,
            num_attention_heads=text_config.num_attention_heads,
            num_query_groups=text_config.num_key_value_heads,
            kv_channels=getattr(text_config, "head_dim", 256),
            init_method_std=getattr(text_config, "initializer_range", 0.02),
            layernorm_epsilon=text_config.rms_norm_eps,
            gated_linear_unit=True,
            make_vocab_size_divisible_by=self.make_vocab_size_divisible_by(text_config.vocab_size),
            rotary_base=rope_theta,
            rotary_percent=partial_rotary_factor,
            share_embeddings_and_output_weights=getattr(text_config, "tie_word_embeddings", True),
            vocab_size=text_config.vocab_size,
            seq_length=getattr(text_config, "max_position_embeddings", 262144),
            fp16=(model_dtype == torch.float16),
            bf16=(model_dtype == torch.bfloat16),
            params_dtype=model_dtype,
            # Qwen3.5 specific
            qk_layernorm=True,
            attention_output_gate=True,
        )

        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Weight mappings from HF Qwen3.5 to Megatron format."""
        if self.hf_pretrained is None:
            raise RuntimeError(
                "hf_pretrained is not set. Ensure load_weights_hf_to_megatron() "
                "is called before mapping_registry()."
            )

        hf_config = self.hf_pretrained.config
        is_vlm = hasattr(hf_config, "text_config")
        pfx = "model.language_model" if is_vlm else "model"

        param_mappings = {
            f"embedding.word_embeddings.weight": f"{pfx}.embed_tokens.weight",
            f"output_layer.weight": "lm_head.weight",
            f"decoder.final_layernorm.weight": f"{pfx}.norm.weight",
            f"decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": f"{pfx}.layers.*.input_layernorm.weight",
            f"decoder.layers.*.input_layernorm.weight": f"{pfx}.layers.*.input_layernorm.weight",
            f"decoder.layers.*.self_attention.linear_proj.weight": f"{pfx}.layers.*.self_attn.o_proj.weight",
            f"decoder.layers.*.self_attention.q_layernorm.weight": f"{pfx}.layers.*.self_attn.q_norm.weight",
            f"decoder.layers.*.self_attention.k_layernorm.weight": f"{pfx}.layers.*.self_attn.k_norm.weight",
            f"decoder.layers.*.mlp.linear_fc1.layer_norm_weight": f"{pfx}.layers.*.post_attention_layernorm.weight",
            f"decoder.layers.*.pre_mlp_layernorm.weight": f"{pfx}.layers.*.post_attention_layernorm.weight",
            f"decoder.layers.*.mlp.linear_fc2.weight": f"{pfx}.layers.*.mlp.down_proj.weight",
        }

        linear_attn_weights = [
            "input_layernorm.weight",
            "linear_attn.A_log",
            "linear_attn.conv1d.weight",
            "linear_attn.dt_bias",
            "linear_attn.in_proj_a.weight",
            "linear_attn.in_proj_b.weight",
            "linear_attn.in_proj_qkv.weight",
            "linear_attn.in_proj_z.weight",
            "linear_attn.norm.weight",
            "linear_attn.out_proj.weight",
        ]
        for weight_name in linear_attn_weights:
            param_mappings[f"decoder.layers.*.self_attention.{weight_name}"] = f"{pfx}.layers.*.{weight_name}"

        mapping_list = [
            AutoMapping(megatron_param=megatron_param, hf_param=hf_param)
            for megatron_param, hf_param in param_mappings.items()
        ]

        mapping_list.append(
            QKVMapping(
                megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                q=f"{pfx}.layers.*.self_attn.q_proj.weight",
                k=f"{pfx}.layers.*.self_attn.k_proj.weight",
                v=f"{pfx}.layers.*.self_attn.v_proj.weight",
            )
        )

        mapping_list.append(
            GatedMLPMapping(
                megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
                gate=f"{pfx}.layers.*.mlp.gate_proj.weight",
                up=f"{pfx}.layers.*.mlp.up_proj.weight",
            )
        )

        return MegatronMappingRegistry(*mapping_list)
