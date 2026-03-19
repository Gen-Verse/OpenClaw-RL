import hashlib
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path

from transformers import AutoConfig

logger = logging.getLogger(__name__)


def is_qwen35_model_path(model_path: str) -> bool:
    try:
        hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    except Exception:
        return False
    return getattr(hf_config, "model_type", None) in {"qwen3_5", "qwen3_5_text", "qwen3_5_moe_text"}


def maybe_prepare_qwen35_text_model(model_path: str, *, language_only: bool) -> str:
    if not language_only:
        return model_path

    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if getattr(hf_config, "model_type", None) != "qwen3_5" or not hasattr(hf_config, "text_config"):
        return model_path

    target_dir = _get_shadow_model_dir(model_path)
    config_path = target_dir / "config.json"
    if config_path.exists():
        return str(target_dir)

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=target_dir.name + ".", dir=target_dir.parent))
    try:
        _populate_shadow_model_dir(source_dir=Path(model_path), target_dir=temp_dir, hf_config=hf_config)
        os.replace(temp_dir, target_dir)
    except FileExistsError:
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    logger.info("Prepared Qwen3.5 text-only shadow model at %s", target_dir)
    return str(target_dir)


def patch_sglang_qwen35() -> None:
    from sglang.srt.models import registry as registry_module
    from sglang.srt.models import qwen3_5 as qwen3_5_model

    entry_classes = [
        qwen3_5_model.Qwen3_5MoeForConditionalGeneration,
        qwen3_5_model.Qwen3_5ForConditionalGeneration,
        qwen3_5_model.Qwen3_5MoeForCausalLM,
        qwen3_5_model.Qwen3_5ForCausalLM,
    ]
    deduped = []
    seen = set()
    for cls in entry_classes:
        if cls not in seen:
            deduped.append(cls)
            seen.add(cls)
    qwen3_5_model.EntryClass = deduped

    def _get_model_config_for_expert_location(cls, config):
        text_config = getattr(config, "text_config", config)
        num_experts = getattr(text_config, "num_experts", None)
        if not num_experts:
            return None
        return qwen3_5_model.ModelConfigForExpertLocation(
            num_layers=text_config.num_hidden_layers,
            num_logical_experts=num_experts,
            num_groups=None,
        )

    for cls in [
        qwen3_5_model.Qwen3_5ForCausalLM,
        qwen3_5_model.Qwen3_5MoeForCausalLM,
        qwen3_5_model.Qwen3_5ForConditionalGeneration,
        qwen3_5_model.Qwen3_5MoeForConditionalGeneration,
    ]:
        cls.get_model_config_for_expert_location = classmethod(_get_model_config_for_expert_location)

    registry_module.import_model_classes.cache_clear()
    for cls in deduped:
        registry_module.ModelRegistry.models[cls.__name__] = cls


def _get_shadow_model_dir(model_path: str) -> Path:
    source = Path(model_path).resolve()
    source_hash = hashlib.sha256(str(source).encode("utf-8")).hexdigest()[:16]
    cache_root = os.environ.get("SLIME_SGLANG_MODEL_CACHE_DIR")
    if cache_root:
        base_dir = Path(cache_root)
    else:
        base_dir = Path(tempfile.gettempdir()) / "slime-sglang-models"
    return base_dir / f"qwen3_5_text_v4_{source_hash}"


def _populate_shadow_model_dir(source_dir: Path, target_dir: Path, hf_config) -> None:
    for entry in source_dir.iterdir():
        if entry.name == "config.json":
            continue
        (target_dir / entry.name).symlink_to(entry)

    text_config = hf_config.text_config
    text_config.architectures = ["Qwen3_5ForCausalLM"]
    text_config.model_type = "qwen3_5_text"
    text_config._name_or_path = str(source_dir)
    config_dict = text_config.to_dict()
    config_dict["architectures"] = ["Qwen3_5ForCausalLM"]
    config_dict["model_type"] = "qwen3_5_text"
    if "layer_types" in config_dict:
        normalized_layer_types = [
            "attention" if layer_type == "full_attention" else layer_type
            for layer_type in config_dict["layer_types"]
        ]
        config_dict["layers_block_type"] = normalized_layer_types
    if "rope_theta" not in config_dict:
        rope_theta = None
        if isinstance(config_dict.get("rope_parameters"), dict):
            rope_theta = config_dict["rope_parameters"].get("rope_theta")
        if rope_theta is None and isinstance(config_dict.get("rope_scaling"), dict):
            rope_theta = config_dict["rope_scaling"].get("rope_theta")
        if rope_theta is not None:
            config_dict["rope_theta"] = rope_theta
    config_path = target_dir / "config.json"
    config_path.write_text(json.dumps(config_dict, indent=2, sort_keys=True) + "\n")
