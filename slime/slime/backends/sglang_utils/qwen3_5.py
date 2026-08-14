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


_qwen35_patched = False


def patch_sglang_qwen35() -> None:
    global _qwen35_patched
    if _qwen35_patched:
        return
    _qwen35_patched = True
    import torch
    import torch.nn as nn
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig

    from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape, mamba2_state_dtype
    from sglang.srt.configs.update_config import adjust_tp_num_heads_if_necessary
    from sglang.srt.distributed import get_pp_group
    from sglang.srt.layers.logits_processor import LogitsProcessor
    from sglang.srt.layers.utils.common import PPMissingLayer
    from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
    from sglang.srt.layers.dp_attention import get_attention_tp_size
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.model_loader.weight_utils import default_weight_loader
    from sglang.srt.models import registry as registry_module
    from sglang.srt.models import qwen3_5 as qwen3_5_model
    from sglang.srt.server_args import get_global_server_args
    from sglang.srt.utils import add_prefix
    from sglang.srt.utils import is_cpu

    original_qwen35_dense_cls = qwen3_5_model.Qwen3_5ForCausalLM

    class PatchedQwen35ForCausalLM(nn.Module):
        def __init__(
            self,
            config: Qwen3_5TextConfig,
            quant_config=None,
            prefix: str = "",
        ) -> None:
            super().__init__()
            self.pp_group = get_pp_group()
            self.config = config
            self.quant_config = quant_config
            self.model = original_qwen35_dense_cls(
                config=config,
                quant_config=quant_config,
                prefix=add_prefix("model", prefix),
            )

            if self.pp_group.is_last_rank:
                if self.pp_group.world_size == 1 and config.tie_word_embeddings:
                    self.lm_head = self.model.embed_tokens
                else:
                    self.lm_head = ParallelLMHead(
                        config.vocab_size,
                        config.hidden_size,
                        quant_config=quant_config,
                        org_num_embeddings=config.vocab_size,
                        prefix=add_prefix("lm_head", prefix),
                        use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
                    )
            else:
                self.lm_head = PPMissingLayer()

            self.logits_processor = LogitsProcessor(config)
            self.capture_aux_hidden_states = False

        def get_input_embeddings(self):
            return self.model.get_input_embeddings()

        @property
        def start_layer(self):
            return self.model.layers.start_layer

        @property
        def end_layer(self):
            return self.model.layers.end_layer

        @torch.no_grad()
        def forward(
            self,
            input_ids,
            positions,
            forward_batch,
            input_embeds=None,
            pp_proxy_tensors=None,
            input_deepstack_embeds=None,
            **kwargs,
        ):
            hidden_states = self.model(
                input_ids=input_ids,
                positions=positions,
                forward_batch=forward_batch,
                input_embeds=input_embeds,
                pp_proxy_tensors=pp_proxy_tensors,
                input_deepstack_embeds=input_deepstack_embeds,
            )

            aux_hidden_states = None
            if self.capture_aux_hidden_states:
                hidden_states, aux_hidden_states = hidden_states

            if self.pp_group.is_last_rank:
                return self.logits_processor(
                    input_ids,
                    hidden_states,
                    self.lm_head,
                    forward_batch,
                    aux_hidden_states,
                )
            return hidden_states

        def load_weights(self, weights):
            stacked_params_mapping = [
                ("qkv_proj", "q_proj", "q"),
                ("qkv_proj", "k_proj", "k"),
                ("qkv_proj", "v_proj", "v"),
                ("gate_up_proj", "gate_proj", 0),
                ("gate_up_proj", "up_proj", 1),
            ]

            params_dict = dict(self.named_parameters(remove_duplicate=False))
            for name, loaded_weight in weights:
                if "rotary_emb.inv_freq" in name or "mtp" in name or "visual" in name:
                    continue
                if "language_model" in name:
                    name = name.replace(r"model.language_model.", r"model.")
                # SGLang's Qwen3.5 full-attention layer registers `qkv_proj`,
                # `o_proj`, `q_norm`, `k_norm` etc. as direct attributes of the
                # decoder layer (not under a `self_attn` submodule). Megatron
                # emits HF-style names with `.self_attn.` so we must strip it
                # to match the actual params_dict tree. Linear-attention layers
                # use `.linear_attn.` and are unaffected.
                if ".self_attn." in name:
                    name = name.replace(".self_attn", "")

                if name == "model.embed_tokens.weight":
                    if self.pp_group.is_last_rank and self.config.tie_word_embeddings:
                        lm_head_weight = params_dict.get("lm_head.weight")
                        if lm_head_weight is not None:
                            weight_loader = getattr(lm_head_weight, "weight_loader", default_weight_loader)
                            weight_loader(lm_head_weight, loaded_weight)

                for param_name, weight_name, shard_id in stacked_params_mapping:
                    if weight_name not in name:
                        continue
                    if "mlp.experts" in name:
                        continue
                    mapped_name = name.replace(weight_name, param_name)
                    if mapped_name.endswith(".bias") and mapped_name not in params_dict:
                        continue
                    if mapped_name not in params_dict:
                        continue
                    param = params_dict[mapped_name]
                    weight_loader = getattr(param, "weight_loader")
                    weight_loader(param, loaded_weight, shard_id)
                    break
                else:
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if name not in params_dict:
                        logger.warning("Parameter %s not found in params_dict", name)
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)

        def get_embed_and_head(self):
            return self.model.embed_tokens.weight, self.lm_head.weight

        def set_embed_and_head(self, embed, head):
            del self.model.embed_tokens.weight
            del self.lm_head.weight
            self.model.embed_tokens.weight = embed
            self.lm_head.weight = head
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    PatchedQwen35ForCausalLM.__name__ = "Qwen3_5ForCausalLM"
    qwen3_5_model.Qwen3_5ForCausalLM = PatchedQwen35ForCausalLM

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

    _is_cpu = is_cpu()

    def _layers_block_type(self):
        layer_types = getattr(self, "layer_types", None) or []
        return [
            "attention" if layer_type == "full_attention" else layer_type
            for layer_type in layer_types
        ]

    def _linear_layer_ids(self):
        return [
            idx
            for idx, layer_type in enumerate(self.layers_block_type)
            if layer_type == "linear_attention"
        ]

    def _full_attention_layer_ids(self):
        return [
            idx
            for idx, layer_type in enumerate(self.layers_block_type)
            if layer_type == "attention"
        ]

    def _mamba2_cache_params(self):
        if _is_cpu:
            world_size = get_attention_tp_size()
            adjust_tp_num_heads_if_necessary(self, world_size, False)

        shape = Mamba2StateShape.create(
            tp_world_size=get_attention_tp_size(),
            intermediate_size=self.linear_value_head_dim * self.linear_num_value_heads,
            n_groups=self.linear_num_key_heads,
            num_heads=self.linear_num_value_heads,
            head_dim=self.linear_value_head_dim,
            state_size=self.linear_key_head_dim,
            conv_kernel=self.linear_conv_kernel_dim,
        )

        return Mamba2CacheParams(
            shape=shape, layers=self.linear_layer_ids, dtype=mamba2_state_dtype(self)
        )

    Qwen3_5TextConfig.layers_block_type = property(_layers_block_type)
    Qwen3_5TextConfig.linear_layer_ids = property(_linear_layer_ids)
    Qwen3_5TextConfig.full_attention_layer_ids = property(_full_attention_layer_ids)
    Qwen3_5TextConfig.mamba2_cache_params = property(_mamba2_cache_params)

    original_hybrid_gdn_config = ModelRunner.hybrid_gdn_config.fget

    def _hybrid_gdn_config(self):
        config = self.model_config.hf_config.get_text_config()
        if isinstance(config, (Qwen3_5Config, Qwen3_5TextConfig, qwen3_5_model.Qwen3_5MoeConfig)):
            _ensure_qwen35_attention_layer_ids(config)
            return config
        if isinstance(config, Qwen3_5VisionConfig):
            text_config = getattr(config, "text_config", None)
            if isinstance(text_config, (Qwen3_5Config, Qwen3_5TextConfig, qwen3_5_model.Qwen3_5MoeConfig)):
                _ensure_qwen35_attention_layer_ids(text_config)
                return text_config
        return original_hybrid_gdn_config(self)

    ModelRunner.hybrid_gdn_config = property(_hybrid_gdn_config)


def _get_shadow_model_dir(model_path: str) -> Path:
    source = Path(model_path).resolve()
    source_hash = hashlib.sha256(str(source).encode("utf-8")).hexdigest()[:16]
    cache_root = os.environ.get("SLIME_SGLANG_MODEL_CACHE_DIR")
    if cache_root:
        base_dir = Path(cache_root)
    else:
        base_dir = Path(tempfile.gettempdir()) / "slime-sglang-models"
    return base_dir / f"qwen3_5_text_v6_{source_hash}"


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


def _compute_attention_layer_ids(config) -> tuple[list[int], list[int]]:
    layer_types = getattr(config, "layer_types", None) or getattr(config, "layers_block_type", None) or []
    full_attention_layer_ids = []
    linear_attention_layer_ids = []
    for idx, layer_type in enumerate(layer_types):
        if layer_type in {"full_attention", "attention"}:
            full_attention_layer_ids.append(idx)
        elif layer_type == "linear_attention":
            linear_attention_layer_ids.append(idx)
    return full_attention_layer_ids, linear_attention_layer_ids


def _ensure_qwen35_attention_layer_ids(config) -> None:
    full_attention_layer_ids, linear_attention_layer_ids = _compute_attention_layer_ids(config)
    if not hasattr(config, "full_attention_layer_ids"):
        config.full_attention_layer_ids = full_attention_layer_ids
    if not hasattr(config, "linear_attention_layer_ids"):
        config.linear_attention_layer_ids = linear_attention_layer_ids
