from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.types import Sample

from rollout_agent import create_prm_agent, create_rollout_agent
from custom_types import RunContext, TaskTimeouts
from env_client import create_env_client
from agent_runner import AgentRunner
from sample_builders import build_non_trainable_sample, build_samples_from_outcome
from skill_context import augment_user_message
from trajectory_sink import publish_terminal_trajectory

logger = logging.getLogger(__name__)


def _make_run_context(sample: Sample) -> RunContext:
    metadata = sample.metadata or {}
    uid = (
        metadata["uid"]
        if "uid" in metadata and metadata["uid"]
        else uuid.uuid4().hex[:8]
    )
    group_index = sample.group_index
    sample_index = sample.index
    return RunContext(
        uid=uid,
        group_index=group_index,
        sample_index=sample_index,
        log_dir=Path("build_outputs") / "AgentRunner_Output",
    )


def _make_log_tag(
    task_meta: Dict[str, Any],
    run_ctx: RunContext,
    lease_id: str | None = None,
) -> str:
    parts = [
        f"task={task_meta['task_name']}",
        f"uid={run_ctx.uid}",
        f"group_idx={run_ctx.group_index}",
        f"sample_idx={run_ctx.sample_index}",
    ]
    if lease_id:
        parts.append(f"lease={lease_id}")
    return f"[{' '.join(parts)}]"


def _format_exception(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _log_stage_exception(
    log_tag: str,
    stage: str,
    exc: Exception,
    exc_info: bool = True,
) -> None:
    logger.error(
        "%s %s failed (%s): %s",
        log_tag,
        stage,
        type(exc).__name__,
        exc,
        exc_info=exc_info,
    )


async def generate(
    args,
    sample: Sample,
    sampling_params: Dict[str, Any],
    evaluation: bool = False,
) -> List[Sample]:
    state = GenerateState(args)
    task_meta = sample.metadata
    # task_meta = sample.prompt
    run_ctx = _make_run_context(sample)
    log_tag = _make_log_tag(task_meta, run_ctx)
    timeouts = TaskTimeouts(
        ensure_image=getattr(args, "ensure_image_timeout", 300.0),
        reset_session=getattr(args, "reset_session_timeout", 300.0),
        close_session=getattr(args, "close_session_timeout", 60.0),
        eval=getattr(args, "eval_timeout", 600.0),
    )
    prm_enabled = args.prm_enable and (not evaluation)
    prm_step_coef = args.prm_step_coef
    prm_agent = None
    env_client = None
    lease_id = None
    runner = None
    user_msg = None
    env_ready_for_evaluation = False
    outcome = None
    rollout_error: str | None = None
    retrieved_skills: list[dict[str, Any]] = []

    try:
        task_name = task_meta["task_name"]
        stage = "env_client.create"
        try:
            env_client = create_env_client()

            stage = "env.allocate"
            lease = await env_client.allocate(
                task_key=task_name,
                request_id=run_ctx.run_identity(),
            )
            lease_id = lease["lease_id"]
            log_tag = _make_log_tag(task_meta, run_ctx, lease_id=lease_id)
            logger.info("%s Allocated remote terminal env backend.", log_tag)

            stage = "env.reset"
            reset_payload = await env_client.reset(
                lease_id=lease_id,
                task_meta=task_meta,
                run_ctx=run_ctx.to_payload(),
                task_timeouts=timeouts.to_payload(),
            )
            env_ready_for_evaluation = True

            user_msg = reset_payload["user_msg"]
            user_msg, retrieved_skills = augment_user_message(
                user_msg,
                task_meta["instruction"],
            )
            tool_schemas = reset_payload["tool_schemas"]
            logger.info(
                "%s Remote env reset complete. Starting terminal rollout.", log_tag
            )

            stage = "prm_agent.create"
            if prm_enabled:
                prm_agent = create_prm_agent(
                    args,
                    tokenizer=state.tokenizer,
                    task_instruction=task_meta["instruction"],
                    log_tag=log_tag,
                )

            stage = "runner.create"
            rollout_agent = create_rollout_agent(
                agent_type=getattr(args, "terminal_agent_type", "camel_agent"),
                args=args,
                tokenizer=state.tokenizer,
                sampling_params=sampling_params,
                model_type=getattr(args, "model_type", "Qwen3"),
            )
            runner = AgentRunner(
                rollout_agent=rollout_agent,
                tool_schemas=tool_schemas,
                env_client=env_client,
                lease_id=lease_id,
                prm_agent=prm_agent,
                max_iterations=getattr(args, "max_iteration", 10),
                max_parse_errors=getattr(args, "max_parse_errors", 3),
                log_tag=log_tag,
            )
        except Exception as exc:
            _log_stage_exception(log_tag, stage, exc)
            rollout_error = _format_exception(exc)

        if runner is not None:
            try:
                stage = "runner.run_episode"
                outcome = await runner.run_episode(user_msg)
            except Exception as exc:
                _log_stage_exception(log_tag, stage, exc)
                rollout_error = _format_exception(exc)

        reward = 0.0
        eval_error: str | None = None

        if env_ready_for_evaluation:
            try:
                stage = "env.evaluate"
                reward = await env_client.evaluate(lease_id)
                logger.info("%s Evaluation completed reward=%.4f", log_tag, reward)
            except Exception as exc:
                _log_stage_exception(log_tag, stage, exc)
                eval_error = _format_exception(exc)
                logger.error(
                    "%s Marking sample FAILED after evaluation error.", log_tag
                )

        stage = "sample.build"
        try:
            sample.metadata = sample.metadata or {}
            sample.metadata["retrieved_skills"] = retrieved_skills
            samples = build_samples_from_outcome(
                sample,
                outcome,
                reward=reward,
                rollout_error=rollout_error,
                eval_error=eval_error,
                prm_enabled=prm_enabled,
                prm_step_coef=prm_step_coef,
                log_tag=log_tag,
            )
            if not evaluation:
                await publish_terminal_trajectory(
                    task_meta=task_meta,
                    reward=reward,
                    outcome=outcome,
                    rollout_error=rollout_error,
                    eval_error=eval_error,
                    retrieved_skills=retrieved_skills,
                )
            return samples
        except Exception as exc:
            _log_stage_exception(log_tag, stage, exc)
            return build_non_trainable_sample(
                sample,
                Sample.Status.FAILED,
                reward=reward,
                eval_error=eval_error,
            )
    finally:
        if env_client is not None and lease_id is not None:
            stage = "env.close"
            try:
                await env_client.close(lease_id)
            except Exception as exc:
                _log_stage_exception(log_tag, stage, exc)
