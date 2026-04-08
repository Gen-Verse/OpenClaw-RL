import re
import torch


_LINEAR_ATTN_DIRECT_TO_HF = {
    "input_layernorm.weight": "input_layernorm.weight",
    "A_log": "linear_attn.A_log",
    "conv1d.weight": "linear_attn.conv1d.weight",
    "dt_bias": "linear_attn.dt_bias",
    "in_proj_a.weight": "linear_attn.in_proj_a.weight",
    "in_proj_b.weight": "linear_attn.in_proj_b.weight",
    "in_proj_qkv.weight": "linear_attn.in_proj_qkv.weight",
    "in_proj_z.weight": "linear_attn.in_proj_z.weight",
    "norm.weight": "linear_attn.norm.weight",
    "out_norm.weight": "linear_attn.norm.weight",
    "out_proj.weight": "linear_attn.out_proj.weight",
}


_VISION_DIRECT_TO_HF = {
    "patch_embed.proj.weight": "model.visual.patch_embed.proj.weight",
    "patch_embed.proj.bias": "model.visual.patch_embed.proj.bias",
    "pos_embed.weight": "model.visual.pos_embed.weight",
    "merger.linear_fc1.weight": "model.visual.merger.linear_fc1.weight",
    "merger.linear_fc1.bias": "model.visual.merger.linear_fc1.bias",
    "merger.linear_fc2.weight": "model.visual.merger.linear_fc2.weight",
    "merger.linear_fc2.bias": "model.visual.merger.linear_fc2.bias",
    "merger.norm.weight": "model.visual.merger.norm.weight",
    "merger.norm.bias": "model.visual.merger.norm.bias",
    "merger.patch_norm.weight": "model.visual.merger.norm.weight",
    "merger.patch_norm.bias": "model.visual.merger.norm.bias",
}


def _convert_vision_model_to_hf(name, param):
    vision_name = name[len("module.module.vision_model.") :]
    if vision_name in _VISION_DIRECT_TO_HF:
        return [(_VISION_DIRECT_TO_HF[vision_name], param)]

    decoder_layers_pattern = r"decoder\.layers\.(\d+)\.(.+)"
    match = re.match(decoder_layers_pattern, vision_name)
    if match:
        layer_idx, rest = match.groups()
        base = f"model.visual.blocks.{layer_idx}"

        if rest == "self_attention.linear_proj.weight":
            return [(f"{base}.attn.proj.weight", param)]
        if rest == "self_attention.linear_proj.bias":
            return [(f"{base}.attn.proj.bias", param)]
        if rest == "self_attention.linear_qkv.weight":
            return [(f"{base}.attn.qkv.weight", param)]
        if rest == "self_attention.linear_qkv.bias":
            return [(f"{base}.attn.qkv.bias", param)]
        if rest in ("input_layernorm.weight", "self_attention.linear_qkv.layer_norm_weight"):
            return [(f"{base}.norm1.weight", param)]
        if rest in ("input_layernorm.bias", "self_attention.linear_qkv.layer_norm_bias"):
            return [(f"{base}.norm1.bias", param)]
        if rest == "mlp.linear_fc1.weight":
            return [(f"{base}.mlp.linear_fc1.weight", param)]
        if rest == "mlp.linear_fc1.bias":
            return [(f"{base}.mlp.linear_fc1.bias", param)]
        if rest == "mlp.linear_fc2.weight":
            return [(f"{base}.mlp.linear_fc2.weight", param)]
        if rest == "mlp.linear_fc2.bias":
            return [(f"{base}.mlp.linear_fc2.bias", param)]
        if rest in ("pre_mlp_layernorm.weight", "mlp.linear_fc1.layer_norm_weight"):
            return [(f"{base}.norm2.weight", param)]
        if rest in ("pre_mlp_layernorm.bias", "mlp.linear_fc1.layer_norm_bias"):
            return [(f"{base}.norm2.bias", param)]

    if vision_name.startswith("blocks.") or vision_name.startswith("patch_embed.") or vision_name.startswith("merger."):
        return [(f"model.visual.{vision_name}", param)]

    raise ValueError(f"Unknown vision parameter name: {name}")


def _convert_mtp_layer(args, name, param, layer_idx):
    if "enorm.weight" in name:
        return[("mtp.pre_fc_norm_embedding.weight", param)]
    if "hnorm.weight" in name:
        return [("mtp.pre_fc_norm_hidden.weight", param)]
    if "final_layernorm.weight" in name:
        return [("mtp.norm.weight", param)]
    if "eh_proj.weight" in name:
        if param.dim() < 2:
            raise ValueError(f"eh_proj weight expects 2D tensor, got {param.shape}")
        first_half, second_half = param.chunk(2, dim=1)
        new_param = torch.cat([second_half, first_half], dim=1)
        return[("mtp.fc.weight", new_param)]

    for inner_layer_name in ("transformer_layer", "mtp_model_layer"):
        mtp_prefix = f"mtp.layers.{layer_idx}.{inner_layer_name}"
        if mtp_prefix in name:
            proxy_name = name.replace(mtp_prefix, f"decoder.layers.{layer_idx}")
            mapped_params = convert_qwen3_5_to_hf(args, proxy_name, param)

            final_params = []
            for hf_name, tensor in mapped_params:
                target_prefix = f"mtp.layers.{layer_idx}"
                if f"model.language_model.layers.{layer_idx}" in hf_name:
                    new_hf_name = hf_name.replace(
                        f"model.language_model.layers.{layer_idx}", target_prefix
                    )
                    final_params.append((new_hf_name, tensor))
                else:
                    final_params.append((hf_name, tensor))
            return final_params

    return None


def _split_qkv_weight(args, param, head_dim, value_num_per_group):
    param = param.view(args.num_query_groups, -1, head_dim, args.hidden_size)
    q_param, k_param, v_param = torch.split(
        param, split_size_or_sections=[2 * value_num_per_group, 1, 1], dim=1
    )
    q_param = (
        q_param.reshape(args.num_query_groups, 2, value_num_per_group, head_dim, args.hidden_size)
        .transpose(1, 2)
        .reshape(-1, args.hidden_size)
    )
    k_param = k_param.reshape(-1, args.hidden_size)
    v_param = v_param.reshape(-1, args.hidden_size)
    return q_param, k_param, v_param


def _split_qkv_bias(args, param, head_dim, value_num_per_group):
    param = param.view(args.num_query_groups, -1)
    q_bias, k_bias, v_bias = torch.split(
        param,
        split_size_or_sections=[value_num_per_group * head_dim * 2, head_dim, head_dim],
        dim=1,
    )
    q_bias = q_bias.contiguous().flatten()
    k_bias = k_bias.contiguous().flatten()
    v_bias = v_bias.contiguous().flatten()
    return q_bias, k_bias, v_bias


def _split_linear_attn_in_proj_weight(args, param):
    key_dim = args.linear_num_key_heads * args.linear_key_head_dim
    value_dim = args.linear_num_value_heads * args.linear_value_head_dim
    num_v_heads = args.linear_num_value_heads

    split_sizes = [key_dim * 2 + value_dim, value_dim, num_v_heads, num_v_heads]
    in_proj_qkv, in_proj_z, in_proj_b, in_proj_a = torch.split(param, split_sizes, dim=0)
    return in_proj_qkv, in_proj_z, in_proj_b, in_proj_a


def convert_qwen3_5_to_hf(args, name, param):
    if name.startswith("module.module.language_model."):
        name = "module.module." + name[len("module.module.language_model.") :]

    while name.startswith("module.module.module."):
        name = name.replace("module.module.module.", "module.module.", 1)

    if name.startswith("module.module.vision_model."):
        return _convert_vision_model_to_hf(name, param)

    if "mtp.layers" in name:
        parts = name.split(".")
        try:
            layer_idx_loc = parts.index("layers") + 1
            layer_idx = parts[layer_idx_loc]
        except (ValueError, IndexError) as e:
            raise ValueError(f"Invalid MTP layer name format: {name}") from e

        result = _convert_mtp_layer(args, name, param, layer_idx)
        if result is not None:
            return result

    # 基础组件添加 model.language_model 前缀
    if name == "module.module.embedding.word_embeddings.weight":
        return[("model.language_model.embed_tokens.weight", param)]
    if name == "module.module.output_layer.weight":
        return [("lm_head.weight", param)]
    if name == "module.module.decoder.final_layernorm.weight":
        return[("model.language_model.norm.weight", param)]

    try:
        head_dim = args.kv_channels if args.kv_channels is not None else args.hidden_size // args.num_attention_heads
    except AttributeError:
        head_dim = args.hidden_size // args.num_attention_heads
    value_num_per_group = args.num_attention_heads // args.num_query_groups

    decoder_layers_pattern = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
    match = re.match(decoder_layers_pattern, name)
    if match:
        layer_idx, rest = match.groups()

        # 层级组件全部添加 model.language_model.layers 前缀
        if rest == "self_attention.linear_proj.weight":
            return[(f"model.language_model.layers.{layer_idx}.self_attn.o_proj.weight", param)]
        elif rest == "self_attention.linear_qkv.weight":
            q_param, k_param, v_param = _split_qkv_weight(args, param, head_dim, value_num_per_group)
            return[
                (f"model.language_model.layers.{layer_idx}.self_attn.q_proj.weight", q_param),
                (f"model.language_model.layers.{layer_idx}.self_attn.k_proj.weight", k_param),
                (f"model.language_model.layers.{layer_idx}.self_attn.v_proj.weight", v_param),
            ]
        elif rest == "self_attention.in_proj.weight":
            in_proj_qkv, in_proj_z, in_proj_b, in_proj_a = _split_linear_attn_in_proj_weight(args, param)
            return[
                (f"model.language_model.layers.{layer_idx}.linear_attn.in_proj_qkv.weight", in_proj_qkv),
                (f"model.language_model.layers.{layer_idx}.linear_attn.in_proj_z.weight", in_proj_z),
                (f"model.language_model.layers.{layer_idx}.linear_attn.in_proj_b.weight", in_proj_b),
                (f"model.language_model.layers.{layer_idx}.linear_attn.in_proj_a.weight", in_proj_a),
            ]
        elif rest == "self_attention.linear_qkv.bias":
            q_bias, k_bias, v_bias = _split_qkv_bias(args, param, head_dim, value_num_per_group)
            return[
                (f"model.language_model.layers.{layer_idx}.self_attn.q_proj.bias", q_bias),
                (f"model.language_model.layers.{layer_idx}.self_attn.k_proj.bias", k_bias),
                (f"model.language_model.layers.{layer_idx}.self_attn.v_proj.bias", v_bias),
            ]
        elif rest == "self_attention.in_proj.bias":
            return []
        elif rest == "mlp.linear_fc1.weight":
            gate_weight, up_weight = param.chunk(2, dim=0)
            return[
                (f"model.language_model.layers.{layer_idx}.mlp.gate_proj.weight", gate_weight),
                (f"model.language_model.layers.{layer_idx}.mlp.up_proj.weight", up_weight),
            ]
        elif rest == "mlp.linear_fc2.weight":
            return[(f"model.language_model.layers.{layer_idx}.mlp.down_proj.weight", param)]
        elif rest in ["self_attention.linear_qkv.layer_norm_weight", "self_attention.in_proj.layer_norm_weight"]:
            return[(f"model.language_model.layers.{layer_idx}.input_layernorm.weight", param)]
        elif rest == "self_attention.in_proj.layer_norm_bias":
            return []
        elif rest == "self_attention.out_norm.weight":
            return [(f"model.language_model.layers.{layer_idx}.linear_attn.norm.weight", param)]
        elif rest == "self_attention.out_norm.bias":
            return []
        elif rest == "mlp.linear_fc1.layer_norm_weight":
            return[(f"model.language_model.layers.{layer_idx}.post_attention_layernorm.weight", param)]
        elif rest == "pre_mlp_layernorm.weight":
            return[(f"model.language_model.layers.{layer_idx}.post_attention_layernorm.weight", param)]
        elif rest == "self_attention.q_layernorm.weight":
            return[(f"model.language_model.layers.{layer_idx}.self_attn.q_norm.weight", param)]
        elif rest == "self_attention.k_layernorm.weight":
            return[(f"model.language_model.layers.{layer_idx}.self_attn.k_norm.weight", param)]
        elif rest.startswith("self_attention."):
            attn_rest = rest[len("self_attention.") :]
            if attn_rest in [
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
            ]:
                return[(f"model.language_model.layers.{layer_idx}.{attn_rest}", param)]
            if attn_rest in _LINEAR_ATTN_DIRECT_TO_HF:
                hf_suffix = _LINEAR_ATTN_DIRECT_TO_HF[attn_rest]
                return[(f"model.language_model.layers.{layer_idx}.{hf_suffix}", param)]

    raise ValueError(f"Unknown parameter name: {name}")
