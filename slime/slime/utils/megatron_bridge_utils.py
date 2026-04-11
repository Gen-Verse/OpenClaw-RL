from contextlib import contextmanager
import itertools
import json
import os
from pathlib import Path

try:
    from megatron.core.utils import unwrap_model
except ImportError:
    unwrap_model = None


@contextmanager
def patch_megatron_model(model):
    unwrapped_model = unwrap_model(model)[0]
    model_config = unwrapped_model.config
    attribute_was_added = False
    if not hasattr(model_config, "share_embeddings_and_output_weights"):
        model_config.share_embeddings_and_output_weights = unwrapped_model.share_embeddings_and_output_weights
        attribute_was_added = True

    try:
        yield
    finally:
        if attribute_was_added:
            delattr(model_config, "share_embeddings_and_output_weights")


def maybe_dump_bridge_runtime_layout(bridge, megatron_model):
    dump_path = os.getenv("SLIME_DEBUG_QWEN35_DUMP_PATH")
    if not dump_path:
        return None

    from megatron.bridge.models.conversion.model_bridge import _megatron_local_name_to_global
    from megatron.bridge.models.conversion.utils import get_module_and_param_from_name, persistent_buffers

    model_chunks = megatron_model if isinstance(megatron_model, list) else [megatron_model]
    unwrapped_model = unwrap_model(model_chunks)[0]
    model_config = unwrapped_model.config
    mapping_registry = bridge.mapping_registry()

    rows = []
    for vp_stage, model_chunk in enumerate(model_chunks):
        for local_name, _ in itertools.chain(model_chunk.named_parameters(), persistent_buffers(model_chunk)):
            if "_extra_state" in local_name or bridge._is_adapter_param_name(local_name):
                continue

            local_name = bridge._unwrap_name(local_name)
            global_name = _megatron_local_name_to_global(model_chunks, model_config, local_name, vp_stage)
            local_module, local_weights = get_module_and_param_from_name(model_chunks, local_name, vp_stage)
            mapping = mapping_registry.megatron_to_hf_lookup(bridge._get_lora_unwrapped_name(global_name))

            rows.append(
                {
                    "vp_stage": vp_stage,
                    "global_param_name": global_name,
                    "local_param_name": local_name,
                    "owner_module_class": type(local_module).__name__ if local_module is not None else None,
                    "owner_module_module": type(local_module).__module__ if local_module is not None else None,
                    "param_shape": list(local_weights.shape) if local_weights is not None else None,
                    "mapping_class": type(mapping).__name__ if mapping is not None else None,
                    "hf_param": getattr(mapping, "hf_param", None) if mapping is not None else None,
                }
            )

    output_path = Path(dump_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, ensure_ascii=True, indent=2))
    return str(output_path)


def _use_text_only_qwen35_bridge() -> bool:
    return os.getenv("SLIME_QWEN35_TEXT_ONLY_BRIDGE", "").lower() in {"1", "true", "yes", "on"}


def build_bridge_for_hf_checkpoint(hf_checkpoint: str, *, load_weights: bool):
    import slime_plugins.megatron_bridge  # noqa: F401
    from transformers import AutoConfig
    from megatron.bridge import AutoBridge

    hf_config = AutoConfig.from_pretrained(hf_checkpoint, trust_remote_code=True)
    model_type = getattr(hf_config, "model_type", None)

    if not _use_text_only_qwen35_bridge() and model_type == "qwen3_5" and hasattr(hf_config, "text_config"):
        from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
        from slime_plugins.megatron_bridge.qwen3_5 import MegatronQwen35VLBridge

        bridge = MegatronQwen35VLBridge()
        hf_pretrained = PreTrainedCausalLM.from_pretrained(
            hf_checkpoint,
            trust_remote_code=True,
        )
        return bridge, hf_pretrained, True

    auto_bridge = AutoBridge.from_hf_pretrained(hf_checkpoint, trust_remote_code=True)
    return auto_bridge, None, False
