from __future__ import annotations

import logging

import torch
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVMapping,
    ReplicatedMapping,
)

from slime_plugins.megatron_bridge.qwen35_vl import Qwen35VLModel, Qwen35VLModelProvider

logger = logging.getLogger(__name__)

try:
    from transformers import Qwen3_5ForConditionalGeneration
except ImportError:
    Qwen3_5ForConditionalGeneration = "Qwen3_5ForConditionalGeneration"


@MegatronModelBridge.register_bridge(source=Qwen3_5ForConditionalGeneration, target=Qwen35VLModel)
class MegatronQwen35VLBridge(MegatronModelBridge):
    def __init__(self):
        super().__init__()
        self.hf_pretrained = None

    def load_weights_hf_to_megatron(self, hf_pretrained, model):
        self.hf_pretrained = hf_pretrained
        return super().load_weights_hf_to_megatron(hf_pretrained, model)

    def provider_bridge(self, hf_pretrained) -> Qwen35VLModelProvider:
        hf_config = hf_pretrained.config
        text_config = hf_config.text_config
        vision_config = hf_config.vision_config
        provider_kwargs = self.hf_config_to_provider_kwargs(text_config)
        vision_config.torch_dtype = provider_kwargs.get("params_dtype", torch.float32)

        provider = Qwen35VLModelProvider(**provider_kwargs)

        # For VLMs, tie_word_embeddings lives on the top-level config, not text_config.
        provider.share_embeddings_and_output_weights = getattr(hf_config, "tie_word_embeddings", False)

        provider.normalization = "RMSNorm"
        provider.gated_linear_unit = True
        provider.add_qkv_bias = getattr(text_config, "attention_bias", False)
        provider.add_bias_linear = False
        provider.qk_layernorm = True
        provider.hidden_dropout = 0.0

        provider.layernorm_zero_centered_gamma = True
        provider.attention_output_gate = True
        provider.experimental_attention_variant = "gated_delta_net"
        provider.linear_attention_freq = getattr(text_config, "full_attention_interval", 4)
        provider.rotary_percent = getattr(text_config, "rope_parameters", {}).get("partial_rotary_factor", 0.25)

        provider.linear_conv_kernel_dim = getattr(text_config, "linear_conv_kernel_dim", 4)
        provider.linear_key_head_dim = getattr(text_config, "linear_key_head_dim", 128)
        provider.linear_value_head_dim = getattr(text_config, "linear_value_head_dim", 128)
        provider.linear_num_key_heads = getattr(text_config, "linear_num_key_heads", 16)
        provider.linear_num_value_heads = getattr(text_config, "linear_num_value_heads", 48)

        provider.position_embedding_type = "mrope"
        provider.vision_config = vision_config
        provider.hf_text_config = text_config
        provider.head_dim = getattr(text_config, "head_dim", 256)
        provider.bos_token_id = getattr(text_config, "bos_token_id", 248045)
        provider.eos_token_id = getattr(text_config, "eos_token_id", 248044)
        provider.vision_start_token_id = getattr(hf_config, "vision_start_token_id", 248053)
        provider.vision_end_token_id = getattr(hf_config, "vision_end_token_id", 248054)
        provider.image_token_id = getattr(hf_config, "image_token_id", 248056)
        provider.video_token_id = getattr(hf_config, "video_token_id", 248057)
        provider.mrope_section = getattr(text_config, "rope_scaling", {}).get("mrope_section", [11, 11, 10])

        if provider.mtp_num_layers:
            provider.mtp_loss_scaling_factor = 0.1

        provider.hf_checkpoint_path = getattr(hf_pretrained, "name_or_path", getattr(hf_config, "_name_or_path", None))
        provider.deepstack_visual_indexes = list(getattr(vision_config, "deepstack_visual_indexes", []))
        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        auto_param_mappings = {
            "language_model.embedding.word_embeddings.weight": "model.language_model.embed_tokens.weight",
            "language_model.output_layer.weight": "lm_head.weight",
            "language_model.decoder.final_layernorm.weight": "model.language_model.norm.weight",
            "language_model.decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.language_model.layers.*.post_attention_layernorm.weight",
            "language_model.decoder.layers.*.mlp.linear_fc2.weight": "model.language_model.layers.*.mlp.down_proj.weight",
            "language_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.language_model.layers.*.input_layernorm.weight",
            "language_model.decoder.layers.*.self_attention.q_layernorm.weight": "model.language_model.layers.*.self_attn.q_norm.weight",
            "language_model.decoder.layers.*.self_attention.k_layernorm.weight": "model.language_model.layers.*.self_attn.k_norm.weight",
            "language_model.decoder.layers.*.self_attention.linear_proj.weight": "model.language_model.layers.*.self_attn.o_proj.weight",
        }

        # These parameters belong to local HF-style submodules embedded in the Megatron model
        # (`Qwen35BridgeAttention.linear_attn` and `Qwen3_5VisionModel`), not TP-aware
        # Megatron layers. They should therefore be treated as replicated weights.
        replicated_param_mappings = {
            # Local bridge linear-attention layers use HF-style submodules, not Megatron GDN names.
            "language_model.decoder.layers.*.self_attention.input_layernorm.weight": "model.language_model.layers.*.input_layernorm.weight",
            "language_model.decoder.layers.*.self_attention.linear_attn.dt_bias": "model.language_model.layers.*.linear_attn.dt_bias",
            "language_model.decoder.layers.*.self_attention.linear_attn.A_log": "model.language_model.layers.*.linear_attn.A_log",
            "language_model.decoder.layers.*.self_attention.linear_attn.conv1d.weight": "model.language_model.layers.*.linear_attn.conv1d.weight",
            "language_model.decoder.layers.*.self_attention.linear_attn.in_proj_qkv.weight": "model.language_model.layers.*.linear_attn.in_proj_qkv.weight",
            "language_model.decoder.layers.*.self_attention.linear_attn.in_proj_z.weight": "model.language_model.layers.*.linear_attn.in_proj_z.weight",
            "language_model.decoder.layers.*.self_attention.linear_attn.in_proj_b.weight": "model.language_model.layers.*.linear_attn.in_proj_b.weight",
            "language_model.decoder.layers.*.self_attention.linear_attn.in_proj_a.weight": "model.language_model.layers.*.linear_attn.in_proj_a.weight",
            "language_model.decoder.layers.*.self_attention.linear_attn.norm.weight": "model.language_model.layers.*.linear_attn.norm.weight",
            "language_model.decoder.layers.*.self_attention.linear_attn.out_proj.weight": "model.language_model.layers.*.linear_attn.out_proj.weight",
            # Local bridge vision model is the HF module directly, so it exposes blocks.* and merger.norm.*.
            "vision_model.blocks.*.norm1.weight": "model.visual.blocks.*.norm1.weight",
            "vision_model.blocks.*.norm1.bias": "model.visual.blocks.*.norm1.bias",
            "vision_model.blocks.*.norm2.weight": "model.visual.blocks.*.norm2.weight",
            "vision_model.blocks.*.norm2.bias": "model.visual.blocks.*.norm2.bias",
            "vision_model.blocks.*.attn.qkv.weight": "model.visual.blocks.*.attn.qkv.weight",
            "vision_model.blocks.*.attn.qkv.bias": "model.visual.blocks.*.attn.qkv.bias",
            "vision_model.blocks.*.attn.proj.weight": "model.visual.blocks.*.attn.proj.weight",
            "vision_model.blocks.*.attn.proj.bias": "model.visual.blocks.*.attn.proj.bias",
            "vision_model.blocks.*.mlp.linear_fc1.weight": "model.visual.blocks.*.mlp.linear_fc1.weight",
            "vision_model.blocks.*.mlp.linear_fc1.bias": "model.visual.blocks.*.mlp.linear_fc1.bias",
            "vision_model.blocks.*.mlp.linear_fc2.weight": "model.visual.blocks.*.mlp.linear_fc2.weight",
            "vision_model.blocks.*.mlp.linear_fc2.bias": "model.visual.blocks.*.mlp.linear_fc2.bias",
            "vision_model.patch_embed.proj.weight": "model.visual.patch_embed.proj.weight",
            "vision_model.patch_embed.proj.bias": "model.visual.patch_embed.proj.bias",
            "vision_model.pos_embed.weight": "model.visual.pos_embed.weight",
            "vision_model.merger.norm.weight": "model.visual.merger.norm.weight",
            "vision_model.merger.norm.bias": "model.visual.merger.norm.bias",
            "vision_model.merger.linear_fc1.weight": "model.visual.merger.linear_fc1.weight",
            "vision_model.merger.linear_fc1.bias": "model.visual.merger.linear_fc1.bias",
            "vision_model.merger.linear_fc2.weight": "model.visual.merger.linear_fc2.weight",
            "vision_model.merger.linear_fc2.bias": "model.visual.merger.linear_fc2.bias",
        }

        mapping_list = [
            AutoMapping(megatron_param=megatron_param, hf_param=hf_param)
            for megatron_param, hf_param in auto_param_mappings.items()
        ]
        mapping_list.extend(
            [
                ReplicatedMapping(megatron_param=megatron_param, hf_param=hf_param)
                for megatron_param, hf_param in replicated_param_mappings.items()
            ]
        )
        mapping_list.extend(
            [
                QKVMapping(
                    megatron_param="language_model.decoder.layers.*.self_attention.linear_qkv.weight",
                    q="model.language_model.layers.*.self_attn.q_proj.weight",
                    k="model.language_model.layers.*.self_attn.k_proj.weight",
                    v="model.language_model.layers.*.self_attn.v_proj.weight",
                ),
                GatedMLPMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.linear_fc1.weight",
                    gate="model.language_model.layers.*.mlp.gate_proj.weight",
                    up="model.language_model.layers.*.mlp.up_proj.weight",
                ),
            ]
        )

        mtp_param_mappings = {
            "language_model.mtp.layers.0.eh_proj.weight": "mtp.fc.weight",
            "language_model.mtp.layers.0.enorm.weight": "mtp.pre_fc_norm_embedding.weight",
            "language_model.mtp.layers.0.hnorm.weight": "mtp.pre_fc_norm_hidden.weight",
            "language_model.mtp.layers.0.final_layernorm.weight": "mtp.norm.weight",
            "language_model.mtp.layers.0.mtp_model_layer.mlp.linear_fc1.layer_norm_weight": "mtp.layers.0.post_attention_layernorm.weight",
            "language_model.mtp.layers.0.mtp_model_layer.mlp.linear_fc2.weight": "mtp.layers.0.mlp.down_proj.weight",
            "language_model.mtp.layers.0.mtp_model_layer.self_attention.linear_qkv.layer_norm_weight": "mtp.layers.0.input_layernorm.weight",
            "language_model.mtp.layers.0.mtp_model_layer.self_attention.q_layernorm.weight": "mtp.layers.0.self_attn.q_norm.weight",
            "language_model.mtp.layers.0.mtp_model_layer.self_attention.k_layernorm.weight": "mtp.layers.0.self_attn.k_norm.weight",
            "language_model.mtp.layers.0.mtp_model_layer.self_attention.linear_proj.weight": "mtp.layers.0.self_attn.o_proj.weight",
        }
        for megatron_param, hf_param in mtp_param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        mapping_list.extend(
            [
                QKVMapping(
                    megatron_param="language_model.mtp.layers.*.mtp_model_layer.self_attention.linear_qkv.weight",
                    q="mtp.layers.*.self_attn.q_proj.weight",
                    k="mtp.layers.*.self_attn.k_proj.weight",
                    v="mtp.layers.*.self_attn.v_proj.weight",
                ),
                GatedMLPMapping(
                    megatron_param="language_model.mtp.layers.*.mtp_model_layer.mlp.linear_fc1.weight",
                    gate="mtp.layers.*.mlp.gate_proj.weight",
                    up="mtp.layers.*.mlp.up_proj.weight",
                ),
            ]
        )

        return MegatronMappingRegistry(*mapping_list)
