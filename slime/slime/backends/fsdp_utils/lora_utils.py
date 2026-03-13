"""LoRA utilities for FSDP-based training.

Provides helpers to apply HuggingFace PEFT LoRA adapters to a model,
save/load LoRA-only checkpoints, and merge/unmerge adapters for weight sync.
"""

from __future__ import annotations

import logging
from argparse import Namespace
from pathlib import Path

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def apply_lora(model: torch.nn.Module, args: Namespace) -> torch.nn.Module:
    """Wrap *model* with PEFT LoRA adapters according to *args*.

    Returns the PeftModel wrapper.  All base-model parameters are frozen;
    only the LoRA parameters are trainable.
    """
    from peft import LoraConfig, get_peft_model

    target_modules = None
    if args.lora_target_modules:
        target_modules = [m.strip() for m in args.lora_target_modules.split(",")]

    modules_to_save = None
    if args.lora_modules_to_save:
        modules_to_save = [m.strip() for m in args.lora_modules_to_save.split(",")]

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        modules_to_save=modules_to_save,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    if dist.get_rank() == 0:
        model.print_trainable_parameters()

    return model


def propagate_no_split_modules(model: torch.nn.Module) -> torch.nn.Module:
    """Ensure ``_no_split_modules`` is visible through the PEFT wrapper.

    PEFT wraps the original model as ``model.base_model.model``.  FSDP's
    ``apply_fsdp2`` reads ``model._no_split_modules`` to decide which layers
    to shard individually.  This helper copies the attribute up when missing.
    """
    if getattr(model, "_no_split_modules", None):
        return model

    # PeftModel -> LoraModel -> original HF model
    inner = getattr(model, "base_model", None)
    if inner is not None:
        inner = getattr(inner, "model", inner)
    if inner is not None:
        no_split = getattr(inner, "_no_split_modules", None)
        if no_split:
            model._no_split_modules = no_split
            logger.info(f"Propagated _no_split_modules from inner model: {no_split}")

    return model


def save_lora_checkpoint(model: torch.nn.Module, path: Path) -> None:
    """Save only the LoRA adapter weights + config to *path*.

    Only rank 0 writes to disk.  All ranks participate in the state-dict
    gathering (handled by FSDP through ``state_dict()``).
    """
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

    # Gather full state dict (FSDP2 sharded -> full)
    full_state = get_model_state_dict(
        model,
        options=StateDictOptions(full_state_dict=True, cpu_offload=True),
    )

    if dist.get_rank() == 0:
        # Filter to only LoRA keys
        lora_state = {k: v for k, v in full_state.items() if "lora_" in k}
        path.mkdir(parents=True, exist_ok=True)
        torch.save(lora_state, path / "adapter_weights.pt")

        # Save the PEFT config so we can reload
        if hasattr(model, "peft_config"):
            import json

            for adapter_name, cfg in model.peft_config.items():
                cfg_dict = cfg.to_dict()
                # Convert sets to sorted lists for JSON serialization
                for k, v in cfg_dict.items():
                    if isinstance(v, set):
                        cfg_dict[k] = sorted(v)
                with open(path / "adapter_config.json", "w") as f:
                    json.dump(cfg_dict, f, indent=2)
                break  # Only save the first (default) adapter

        logger.info(f"Saved LoRA adapter ({len(lora_state)} tensors) to {path}")

    dist.barrier()


def load_lora_checkpoint(model: torch.nn.Module, path: Path) -> None:
    """Load LoRA adapter weights from *path* into *model*.

    Broadcasts from rank 0 to all other ranks.
    """
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

    adapter_file = path / "adapter_weights.pt"
    if not adapter_file.exists():
        logger.warning(f"No LoRA adapter found at {adapter_file}; skipping load.")
        return

    if dist.get_rank() == 0:
        lora_state = torch.load(adapter_file, map_location="cpu", weights_only=True)
        logger.info(f"Loaded LoRA adapter ({len(lora_state)} tensors) from {path}")
    else:
        lora_state = {}

    # Build a full state dict with LoRA weights overlaid
    full_state = {}
    for name, param in model.named_parameters():
        if "lora_" in name:
            if name in lora_state:
                full_state[name] = lora_state[name]

    if full_state:
        set_model_state_dict(
            model,
            full_state,
            options=StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,
                broadcast_from_rank0=True,
                strict=False,
            ),
        )
        logger.info(f"Loaded {len(full_state)} LoRA parameters from checkpoint.")

    dist.barrier()


def get_merged_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Return a merged (base + LoRA) state dict on CPU.

    Merges the adapter in-place, copies the state dict, then unmerges.
    The returned dict uses base-model keys (no ``lora_`` or ``base_model.`` prefixes).
    """
    model.merge_adapter()
    try:
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

        merged_state = get_model_state_dict(
            model,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )
        # Strip PEFT wrapper prefixes so keys match the original HF model
        cleaned = {}
        for k, v in merged_state.items():
            # Skip lora-specific keys (they've been merged into base)
            if "lora_" in k:
                continue
            # Strip common PEFT prefixes
            clean_key = k
            for prefix in ["base_model.model.", "base_model."]:
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix):]
                    break
            cleaned[clean_key] = v
        return cleaned
    finally:
        model.unmerge_adapter()
