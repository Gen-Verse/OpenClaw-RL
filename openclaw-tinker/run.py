#!/usr/bin/env python3
"""Entry point for OpenClaw Tinker training.

Supports all three methods via --method {rl, opd, combine}.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

from config import TinkerConfig
from trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def parse_args() -> TinkerConfig:
    parser = argparse.ArgumentParser(description="OpenClaw training on Tinker (RL / OPD / Combined)")

    # Method
    parser.add_argument("--method", default=os.getenv("METHOD", "rl"),
                        choices=["rl", "opd", "combine"],
                        help="Training method: rl, opd, or combine (default: rl)")

    # Model
    parser.add_argument("--model-name", default=os.getenv("MODEL_NAME", "Qwen/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--lora-rank", type=int, default=int(os.getenv("LORA_RANK", "32")))
    parser.add_argument("--teacher-model-name", default=os.getenv("TEACHER_MODEL_NAME", ""),
                        help="Teacher/judge model on Tinker (defaults to same as policy model)")

    # Training
    parser.add_argument("--learning-rate", type=float, default=float(os.getenv("LEARNING_RATE", "1e-4")))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("BATCH_SIZE", "4")))
    parser.add_argument("--max-steps", type=int, default=int(os.getenv("MAX_STEPS", "1000")))
    parser.add_argument("--loss-fn", default=os.getenv("LOSS_FN", "ppo"))
    parser.add_argument("--kl-loss-coef", type=float, default=float(os.getenv("KL_LOSS_COEF", "0.02")))
    parser.add_argument("--save-interval", type=int, default=int(os.getenv("SAVE_INTERVAL", "20")))
    parser.add_argument("--resume-from-ckpt", default=os.getenv("RESUME_FROM_CKPT", ""))

    # Combined method weights
    parser.add_argument("--w-opd", type=float, default=float(os.getenv("OPENCLAW_COMBINE_W_OPD", "1.0")),
                        help="OPD advantage weight (combine method only)")
    parser.add_argument("--w-rl", type=float, default=float(os.getenv("OPENCLAW_COMBINE_W_RL", "1.0")),
                        help="RL advantage weight (combine method only)")
    parser.add_argument("--train-epochs", type=int, default=int(os.getenv("TRAIN_EPOCHS", "1")),
                        help="Duplicate samples N times per rollout batch (combine default: 2)")

    # OPD eval mode
    parser.add_argument("--eval-mode", action="store_true",
                        default=os.getenv("EVAL_MODE", "0").strip().lower() in {"1", "true", "yes"},
                        help="Enable PRM eval scoring alongside OPD (opd method only)")

    # PRM / Hint Judge (on Tinker)
    parser.add_argument("--prm-m", type=int, default=int(os.getenv("PRM_M", "3")))
    parser.add_argument("--prm-temperature", type=float, default=float(os.getenv("PRM_TEMPERATURE", "0.6")))
    parser.add_argument("--prm-max-tokens", type=int, default=int(os.getenv("PRM_MAX_TOKENS", "4096")))

    # Proxy
    parser.add_argument("--proxy-host", default=os.getenv("PROXY_HOST", "0.0.0.0"))
    parser.add_argument("--proxy-port", type=int, default=int(os.getenv("PROXY_PORT", "30000")))
    parser.add_argument("--served-model-name", default=os.getenv("SERVED_MODEL_NAME", "qwen3-4b"))
    parser.add_argument("--api-key", default=os.getenv("SGLANG_API_KEY", ""))

    # Context
    parser.add_argument("--max-context-tokens", type=int,
                        default=int(os.getenv("MAX_CONTEXT_TOKENS", "20000")),
                        help="Max tokens per training datum (truncates prompt from left)")

    # Logging
    parser.add_argument("--record-dir", default=os.getenv("RECORD_DIR", "records/"))
    parser.add_argument("--wandb-project", default=os.getenv("WANDB_PROJECT", "openclaw-tinker"))

    args = parser.parse_args()

    return TinkerConfig(
        method=args.method,
        model_name=args.model_name,
        lora_rank=args.lora_rank,
        teacher_model_name=args.teacher_model_name,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        loss_fn=args.loss_fn,
        kl_loss_coef=args.kl_loss_coef,
        save_interval=args.save_interval,
        resume_from_ckpt=args.resume_from_ckpt,
        w_opd=args.w_opd,
        w_rl=args.w_rl,
        train_epochs=args.train_epochs,
        eval_mode=args.eval_mode,
        prm_m=args.prm_m,
        prm_temperature=args.prm_temperature,
        prm_max_tokens=args.prm_max_tokens,
        proxy_host=args.proxy_host,
        proxy_port=args.proxy_port,
        served_model_name=args.served_model_name,
        api_key=args.api_key,
        max_context_tokens=args.max_context_tokens,
        record_dir=args.record_dir,
        wandb_project=args.wandb_project,
    )


def main():
    config = parse_args()
    if not os.getenv("TINKER_API_KEY"):
        print("ERROR: TINKER_API_KEY environment variable is required.", file=sys.stderr)
        sys.exit(1)
    trainer = Trainer(config)
    try:
        asyncio.run(trainer.run())
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("[main] Ctrl+C received, cleaning up ...")
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main()
