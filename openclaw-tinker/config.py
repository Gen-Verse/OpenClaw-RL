"""Unified configuration for OpenClaw Tinker deployment.

Supports all three training methods: RL, OPD, and Combined (OPD+RL).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TinkerConfig:
    """Configuration for all OpenClaw Tinker training methods.

    Method-specific fields:
      - RL: uses prm_m, prm_temperature, prm_max_tokens for PRM scoring
      - OPD: uses the same PRM fields for hint judging + optional eval_mode
      - Combined: uses w_opd, w_rl to weight OPD vs RL advantage components
    """

    # -- Method --
    method: str = "rl"  # "rl", "opd", or "combine"

    # -- Model --
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    lora_rank: int = 32

    # -- Teacher / Judge model (deployed on Tinker, base model, no LoRA) --
    # Defaults to model_name (same base model used as judge and teacher).
    teacher_model_name: str = ""

    # -- Training --
    learning_rate: float = 1e-4
    batch_size: int = 4
    max_steps: int = 1000
    loss_fn: str = "ppo"
    kl_loss_coef: float = 0.0
    save_weights_timeout: float = 200.0
    save_interval: int = 20
    resume_from_ckpt: str = ""

    # -- Combined method: advantage weights --
    w_opd: float = 1.0
    w_rl: float = 1.0

    # -- OPD: optional eval-mode (compute PRM eval scores alongside OPD) --
    eval_mode: bool = False

    # -- PRM / Hint Judge --
    prm_m: int = 3
    prm_temperature: float = 0.6
    prm_max_tokens: int = 4096

    # -- Proxy Server --
    proxy_host: str = "0.0.0.0"
    proxy_port: int = 30000
    served_model_name: str = "qwen3-4b"
    api_key: str = ""
    max_context_tokens: int = 20000

    # -- Turn classification --
    # Default turn type when neither X-Turn-Type header nor body field is
    # provided by the client.  Standard OpenAI-compatible clients (including
    # OpenClaw) do not send turn-type metadata, so "main" ensures their
    # turns are collected for training by default.  Set to "side" to require
    # explicit opt-in via the X-Turn-Type header.
    default_turn_type: str = "main"

    # -- Logging --
    record_dir: str = "records/"
    wandb_project: str = "openclaw-tinker"

    def resolved_teacher_model(self) -> str:
        """Return the teacher model name, defaulting to policy model if unset."""
        return self.teacher_model_name or self.model_name
