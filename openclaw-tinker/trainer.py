"""Unified training loop for all three methods on Tinker.

Supports: RL, OPD, and Combined (OPD + RL).

Cycle:
  1. Resume rollout worker -> collect batch_size samples
  2. Pause rollout worker
  3. Compute advantages and convert to Tinker Datums
  4. forward_backward_async -> optim_step_async
  5. save_weights_and_get_sampling_client_async -> push to rollout worker
  6. Resume rollout worker

The teacher/judge model is deployed on Tinker as a separate base-model
SamplingClient (no LoRA), sharing the same cloud infrastructure as the
policy model.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import os

from transformers import AutoTokenizer

from config import TinkerConfig
from data_formatter import (
    TrainingSample,
    batch_to_datums,
    batch_to_datums_combined,
    compute_grpo_advantages,
)
from rollout import RolloutWorker, drain_output_queue

logger = logging.getLogger(__name__)

_GREEN = "\033[32m"
_RESET = "\033[0m"


class Trainer:
    """End-to-end trainer using Tinker LoRA for RL / OPD / Combined methods."""

    def __init__(self, config: TinkerConfig):
        self.config = config
        self.training_client = None
        self.sampling_client = None
        self.rollout_worker = None
        self._wandb = None
        self._service_clients: list = []

    async def setup(self):
        import tinker

        # Weights & Biases
        if os.environ.get("WANDB_DISABLED", "").strip().lower() not in {"1", "true", "yes"}:
            try:
                wandb = importlib.import_module("wandb")
                self._wandb = wandb.init(
                    project=self.config.wandb_project,
                    name=os.environ.get("WANDB_RUN_NAME", ""),
                )
            except Exception as e:
                logger.warning("[Trainer] wandb init failed: %s", e)

        # Tokenizer
        logger.info("[Trainer] loading tokenizer for %s ...", self.config.model_name)
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, trust_remote_code=True)

        # Tinker service + LoRA training client (policy model)
        logger.info("[Trainer] connecting to Tinker ...")
        service_client = tinker.ServiceClient()
        self._service_clients.append(service_client)
        self.training_client = await service_client.create_lora_training_client_async(
            base_model=self.config.model_name,
            rank=self.config.lora_rank,
        )

        if self.config.resume_from_ckpt:
            logger.info("[Trainer] resuming from checkpoint: %s", self.config.resume_from_ckpt)
            await self.training_client.load_state_async(self.config.resume_from_ckpt)

        self.sampling_client = await self.training_client.save_weights_and_get_sampling_client_async()
        logger.info("[Trainer] initial sampling client ready")

        # Teacher/judge SamplingClient (base model, no LoRA)
        teacher_model = self.config.resolved_teacher_model()
        logger.info("[Trainer] creating teacher SamplingClient for %s ...", teacher_model)
        teacher_service = tinker.ServiceClient()
        self._service_clients.append(teacher_service)
        teacher_client = await teacher_service.create_sampling_client_async(
            base_model=teacher_model,
        )
        logger.info("[Trainer] teacher SamplingClient ready")

        # Create method-specific scorer
        method = self.config.method.lower()
        if method == "rl":
            from scorers import PRMScorer
            scorer = PRMScorer(
                teacher_sampling_client=teacher_client, tokenizer=tokenizer,
                prm_m=self.config.prm_m, temperature=self.config.prm_temperature,
                max_tokens=self.config.prm_max_tokens,
            )
        elif method == "opd":
            from scorers import OPDScorer
            scorer = OPDScorer(
                teacher_sampling_client=teacher_client, tokenizer=tokenizer,
                prm_m=self.config.prm_m, temperature=self.config.prm_temperature,
                max_tokens=self.config.prm_max_tokens,
                eval_mode=self.config.eval_mode,
            )
        elif method == "combine":
            from scorers import CombinedScorer
            scorer = CombinedScorer(
                teacher_sampling_client=teacher_client, tokenizer=tokenizer,
                prm_m=self.config.prm_m, temperature=self.config.prm_temperature,
                max_tokens=self.config.prm_max_tokens,
            )
        else:
            raise ValueError(f"Unknown method: {method!r}")

        # Rollout worker (selects server class based on config.method)
        self.rollout_worker = RolloutWorker(
            config=self.config,
            sampling_client=self.sampling_client,
            scorer=scorer,
        )

    async def _train_on_batch(self, batch: list[TrainingSample], step: int):
        import tinker

        method = self.config.method.lower()

        # Convert batch to Tinker Datums
        max_tok = self.config.max_context_tokens
        if method == "combine":
            datums = batch_to_datums_combined(
                batch,
                w_opd=self.config.w_opd,
                w_rl=self.config.w_rl,
                max_tokens=max_tok,
            )
        else:
            # RL and OPD both use scalar GRPO advantages
            advantages = compute_grpo_advantages(batch)
            datums = batch_to_datums(batch, advantages, max_tokens=max_tok)

        if not datums:
            logger.error("[Trainer] EMPTY batch at step %d — all %d samples failed datum conversion, skipping", step, len(batch))
            return

        if len(datums) < len(batch):
            logger.warning("[Trainer] step %d: only %d/%d samples converted to datums", step, len(datums), len(batch))

        # --- forward_backward + optim_step: run while inference continues ---
        # Inference is NOT paused during this phase.
        logger.info("[Trainer] step %d: forward_backward (%d datums) ...", step, len(datums))
        try:
            fb_future = await self.training_client.forward_backward_async(datums, loss_fn=self.config.loss_fn)
            fb_output = await fb_future.result_async()
            logger.info(
                "[Trainer] step %d: forward_backward done — metrics=%s",
                step, getattr(fb_output, "metrics", None),
            )
        except Exception as e:
            logger.error("[Trainer] step %d: forward_backward FAILED: %s", step, e, exc_info=True)
            raise

        logger.info("[Trainer] step %d: optim_step ...", step)
        try:
            optim_future = await self.training_client.optim_step_async(
                tinker.AdamParams(learning_rate=self.config.learning_rate)
            )
            optim_output = await optim_future.result_async()
            logger.info(
                "[Trainer] step %d: optim_step done — metrics=%s",
                step, getattr(optim_output, "metrics", None),
            )
        except Exception as e:
            logger.error("[Trainer] step %d: optim_step FAILED: %s", step, e, exc_info=True)
            raise

        # --- save weights: briefly pause inference to swap the sampling client ---
        logger.info("[Trainer] step %d: pausing inference for weight swap ...", step)
        self.rollout_worker.pause_submission()

        lora_name = f"openclaw_{method}_lora"
        try:
            self.sampling_client = await asyncio.wait_for(
                self.training_client.save_weights_and_get_sampling_client_async(name=lora_name),
                timeout=self.config.save_weights_timeout,
            )
        except asyncio.TimeoutError:
            logger.error("[Trainer] save_weights timed out at step %d — training may be stuck", step)
            self.rollout_worker.resume_submission()
            raise
        except Exception as e:
            logger.error("[Trainer] step %d: save_weights FAILED: %s", step, e, exc_info=True)
            self.rollout_worker.resume_submission()
            raise

        self.rollout_worker.update_sampling_client(self.sampling_client)
        self.rollout_worker.resume_submission()
        logger.info("[Trainer] step %d: inference resumed with updated weights", step)

        # Save checkpoint at configured interval and on final step
        if step % self.config.save_interval == 0 or step == self.config.max_steps:
            try:
                resolved = await self.training_client.save_state_async(name=f"step_{step:04d}")
                logger.info("[Trainer] checkpoint saved: %s", getattr(resolved, "path", ""))
            except Exception as e:
                logger.error("[Trainer] save_state FAILED at step %d: %s", step, e, exc_info=True)

        # Logging
        self._log_step(batch, step, method)

    def _log_step(self, batch: list[TrainingSample], step: int, method: str):
        rewards = [s.reward for s in batch]
        mean_r = sum(rewards) / len(rewards) if rewards else 0

        log_dict = {
            "train/step": step,
            "train/mean_reward": mean_r,
            "train/batch_size": len(batch),
        }

        if method == "rl":
            success = sum(1 for r in rewards if r > 0) / len(rewards) if rewards else 0
            logger.info(
                "%s[Trainer] step %d done | batch=%d mean_reward=%.3f success=%.2f%s",
                _GREEN, step, len(batch), mean_r, success, _RESET,
            )
            log_dict["train/success_rate"] = success

        elif method == "opd":
            has_teacher = sum(1 for s in batch if s.teacher_logprobs is not None)
            logger.info(
                "%s[Trainer] step %d done | batch=%d mean_reward=%.3f teacher_samples=%d%s",
                _GREEN, step, len(batch), mean_r, has_teacher, _RESET,
            )
            log_dict["train/teacher_samples"] = has_teacher

        elif method == "combine":
            types = {"opd+rl": 0, "opd": 0, "rl": 0}
            for s in batch:
                t = getattr(s, "sample_type", "")
                if t in types:
                    types[t] += 1
            logger.info(
                "%s[Trainer] step %d done | batch=%d mean_reward=%.3f "
                "opd+rl=%d opd=%d rl=%d%s",
                _GREEN, step, len(batch), mean_r,
                types["opd+rl"], types["opd"], types["rl"], _RESET,
            )
            log_dict.update({
                "train/opd_rl_samples": types["opd+rl"],
                "train/opd_only_samples": types["opd"],
                "train/rl_only_samples": types["rl"],
            })

        if self._wandb:
            self._wandb.log(log_dict, step=step)

    async def run(self):
        await self.setup()
        self.rollout_worker.start()
        self.rollout_worker.resume_submission()
        logger.info("[Trainer] proxy starting at %s:%d (method=%s)",
                    self.config.proxy_host, self.config.proxy_port, self.config.method)

        for step in range(1, self.config.max_steps + 1):
            logger.info("[Trainer] step %d/%d - collecting batch (size=%d) ...",
                        step, self.config.max_steps, self.config.batch_size)

            self.rollout_worker.reset_eval_scores()
            # Inference keeps running — we just drain completed samples
            groups = await drain_output_queue(self.config.batch_size, self.rollout_worker)
            batch = [s for group in groups for s in group]

            eval_scores = self.rollout_worker.drain_eval_scores()
            if eval_scores:
                avg = sum(eval_scores) / len(eval_scores)
                logger.info("[Trainer] prm_eval_score=%.4f (n=%d)", avg, len(eval_scores))
                if self._wandb:
                    self._wandb.log({"rollout/prm_eval_score": avg}, step=step)

            # _train_on_batch handles pause/resume internally (only during weight swap)
            await self._train_on_batch(batch, step)

        logger.info("[Trainer] training complete (%d steps)", self.config.max_steps)
        self.cleanup()

    def cleanup(self):
        """Unload model and release Tinker sessions."""
        if self._wandb:
            self._wandb.finish()
        if self.rollout_worker:
            self.rollout_worker.stop()
        for sc in self._service_clients:
            try:
                sc.holder.close()
                logger.info("[Trainer] closed Tinker session")
            except Exception as e:
                logger.warning("[Trainer] failed to close session: %s", e)
        self._service_clients.clear()
