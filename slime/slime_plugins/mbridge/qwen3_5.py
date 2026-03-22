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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.full_hf_config = self.hf_config
        if hasattr(self.hf_config, "text_config"):
            self.hf_config = self.hf_config.text_config

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
