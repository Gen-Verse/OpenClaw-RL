from mbridge.core import register_model

from .qwen3_next import Qwen3NextBridge


@register_model("qwen3_5")
@register_model("qwen3_5_text")
class Qwen35Bridge(Qwen3NextBridge):
    _ATTENTION_MAPPING = (
        Qwen3NextBridge._ATTENTION_MAPPING
        | {
            f"self_attention.{weight_name}": ["model.layers.{layer_number}." + weight_name]
            for weight_name in [
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
        }
    )

    _MLP_MAPPING = {
        "mlp.linear_fc1.weight": [
            "model.layers.{layer_number}.mlp.gate_proj.weight",
            "model.layers.{layer_number}.mlp.up_proj.weight",
        ],
        "mlp.linear_fc1.layer_norm_weight": ["model.layers.{layer_number}.post_attention_layernorm.weight"],
        "mlp.linear_fc2.weight": ["model.layers.{layer_number}.mlp.down_proj.weight"],
    }

    def __init__(self, hf_config, *args, **kwargs):
        self.full_hf_config = hf_config
        self._hf_text_weight_prefix = "model.language_model" if hasattr(hf_config, "text_config") else None
        if hasattr(hf_config, "text_config"):
            hf_config = hf_config.text_config
        super().__init__(hf_config, *args, **kwargs)

    def _prefix_hf_weight_name(self, name: str) -> str:
        if self._hf_text_weight_prefix is None:
            return name
        if name.startswith("model."):
            return f"{self._hf_text_weight_prefix}.{name[len('model.'):]}"
        if name == "lm_head.weight":
            return f"{self._hf_text_weight_prefix}.lm_head.weight"
        return name

    def _adjust_mapping_for_shared_weights(self):
        if getattr(self.hf_config, "tie_word_embeddings", False):
            self._DIRECT_MAPPING["output_layer.weight"] = "model.embed_tokens.weight"

    def _get_hf_shared_weight_keys(self):
        if getattr(self.hf_config, "tie_word_embeddings", False):
            return [self._prefix_hf_weight_name("model.embed_tokens.weight")]
        return []

    def _weight_name_mapping_mcore_to_hf(self, mcore_weights_name: str) -> list[str]:
        return [
            self._prefix_hf_weight_name(name)
            for name in super()._weight_name_mapping_mcore_to_hf(mcore_weights_name)
        ]

    def _build_config(self):
        mtp_args = {}
        if hasattr(self.hf_config, "mtp_num_hidden_layers"):
            mtp_args["mtp_num_layers"] = self.hf_config.mtp_num_hidden_layers

        return self._build_base_config(
            use_cpu_initialization=False,
            persist_layer_norm=True,
            bias_activation_fusion=True,
            bias_dropout_fusion=True,
            qk_layernorm=True,
            attention_output_gate=True,
            rotary_interleaved=True,
            **mtp_args,
        )
