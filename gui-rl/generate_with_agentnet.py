"""
Offline trajectory generation using AgentNet pre-recorded dataset.

Replaces generate_with_gui.py for environments without cloud VMs.
Instead of interacting with a live desktop VM, this module:
  - Reads pre-recorded screenshots from AgentNet JSONL + image directory
  - Feeds each screenshot to the policy model (same as online rollout)
  - Uses the pre-annotated task_completed flag as the reward signal

Usage: set --custom-generate-function-path generate_with_agentnet.generate
           --custom-rm-path generate_with_agentnet.reward_func
"""
from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.misc import load_function
from slime.utils.types import Sample

# Re-use helpers from generate_with_gui to avoid duplication
from generate_with_gui import (
    _build_dynamic_history_samples,
    _build_result_dir,
    _get_gui_trajectory_semaphore,
    _gui_log,
    _save_image,
)

logger = logging.getLogger(__name__)


def _create_agentnet_agent(args: Any, *, max_steps: int, max_image_history_length: int, result_dir: Path):
    agent_cls_path = getattr(args, "gui_agent_class_path", None) or os.getenv("GUI_AGENT_CLASS_PATH")
    if not agent_cls_path:
        raise RuntimeError("GUI_AGENT_CLASS_PATH is required.")
    agent_cls = load_function(agent_cls_path)
    coordinate_type = os.getenv("GUI_COORDINATE_TYPE", "relative")
    return agent_cls(
        max_steps=max_steps,
        max_image_history_length=max_image_history_length,
        coordinate_type=coordinate_type,
        example_result_dir=str(result_dir),
    )


async def generate(args, sample: Sample, sampling_params, evaluation: bool = False) -> Sample | list[Sample]:
    """Offline generate: replay AgentNet screenshots through the policy model."""
    assert not args.partial_rollout, "Partial rollout is not supported for AgentNet offline rollout."

    metadata = sample.metadata or {}
    instruction = metadata.get("instruction", "")
    traj = metadata.get("traj", [])
    image_dir = metadata.get("image_dir", "")
    domain = metadata.get("domain", "agentnet")
    task_id = metadata.get("task_id", "unknown")

    result_dir = _build_result_dir(args, domain, task_id, sample)
    traj_path = result_dir / "traj.jsonl"

    state = GenerateState(args)

    max_steps_cfg = getattr(args, "gui_max_steps", None)
    if max_steps_cfg is None:
        max_steps_cfg = int(os.getenv("GUI_MAX_STEPS", "15"))
    max_steps = int(max_steps_cfg)

    max_image_history_cfg = getattr(args, "gui_max_image_history_length", None)
    if max_image_history_cfg is None:
        max_image_history_cfg = int(os.getenv("GUI_MAX_IMAGE_HISTORY_LENGTH", str(max_steps)))
    max_image_history_length = int(max_image_history_cfg)

    sampling_params = dict(sampling_params)
    if evaluation and getattr(args, "eval_temperature", None) is not None:
        sampling_params["temperature"] = float(args.eval_temperature)
    elif (not evaluation) and getattr(args, "rollout_temperature", None) is not None:
        sampling_params["temperature"] = float(args.rollout_temperature)

    parser = _create_agentnet_agent(
        args,
        max_steps=max_steps,
        max_image_history_length=max_image_history_length,
        result_dir=result_dir,
    )
    parser.reset(logging.getLogger("agentnet.rollout"))

    assistant_responses: list[str] = []
    step_snapshots: list[dict[str, Any]] = []
    fallback_train_messages: list[dict[str, Any]] = [parser.build_train_system_message()]
    fallback_tool_spec: dict[str, Any] | None = None
    last_step_train_messages_for_loss: list[dict[str, Any]] | None = None
    last_step_tool_spec_for_loss: dict[str, Any] | None = None
    final_status = Sample.Status.COMPLETED

    trajectory_semaphore = _get_gui_trajectory_semaphore()
    await trajectory_semaphore.acquire()

    try:
        _gui_log(
            "AgentNet offline rollout start sample=%s group=%s domain=%s task_id=%s steps=%d",
            sample.index,
            sample.group_index,
            domain,
            task_id,
            len(traj),
        )

        steps_to_run = traj[:max_steps]
        if not steps_to_run:
            final_status = Sample.Status.FAILED
        else:
            # Save first screenshot
            first_img_path = Path(image_dir) / steps_to_run[0]["image"]
            if first_img_path.exists():
                _save_image(first_img_path.read_bytes(), result_dir / "step_0.png")

        for step_idx, step in enumerate(steps_to_run):
            img_path = Path(image_dir) / step["image"]
            if not img_path.exists():
                _gui_log("Missing image %s at step %d, stopping", step["image"], step_idx)
                final_status = Sample.Status.FAILED
                break

            obs = {"screenshot": img_path.read_bytes()}

            parse_ctx = parser.build_policy_messages(instruction=instruction, obs=obs)
            policy_messages = parse_ctx["messages"]
            tool_spec = parse_ctx.get("tool_spec")
            fallback_train_messages = policy_messages
            fallback_tool_spec = tool_spec

            response, finish_type = await parser.generate_with_sglang(
                args=args,
                state=state,
                messages=policy_messages,
                sampling_params=sampling_params,
                sampling_seed=((int(sample.index or 0) + 1) * 1000003 + step_idx * 9973),
                tool_spec=tool_spec,
            )

            _gui_log(
                "step sample=%s step=%s finish=%s response=%s",
                sample.index,
                step_idx,
                finish_type,
                repr(response[:]) if response else repr(response),
            )

            if finish_type == "abort":
                final_status = Sample.Status.ABORTED
                break

            # Build training messages for this step (same as generate_with_gui)
            step_train_messages = copy.deepcopy(policy_messages)
            for msg in step_train_messages:
                if msg.get("role") == "assistant":
                    msg["step_loss_mask"] = 0
            step_train_messages.append({"role": "assistant", "content": response, "step_loss_mask": 1})
            last_step_train_messages_for_loss = step_train_messages
            last_step_tool_spec_for_loss = tool_spec
            assistant_responses.append(response)
            step_snapshots.append({
                "step_idx": step_idx,
                "train_messages": step_train_messages,
                "response_text": response,
                "tool_spec": tool_spec,
            })

            original_width = int(parse_ctx["original_width"])
            original_height = int(parse_ctx["original_height"])
            processed_width = int(parse_ctx["processed_width"])
            processed_height = int(parse_ctx["processed_height"])

            natural_action, actions, info_dict = parser.parse_response(
                response=response,
                original_width=original_width,
                original_height=original_height,
                processed_width=processed_width,
                processed_height=processed_height,
            )
            parser.record_policy_turn(
                action_text=natural_action or "Execute action",
                response=response,
                screenshot_bytes=obs["screenshot"],
            )

            # Save screenshot and trajectory record
            step_image_path = result_dir / f"step_{step_idx + 1}.png"
            _save_image(obs["screenshot"], step_image_path)
            with open(traj_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "step_num": step_idx + 1,
                    "action": actions[0] if actions else "",
                    "natural_language_action": info_dict.get("action"),
                    "response": response,
                    "agentnet_code": step["value"].get("code", ""),
                }, ensure_ascii=False) + "\n")

            # Terminate if model outputs DONE/FAIL
            if actions and str(actions[0]).upper() in {"DONE", "FAIL"}:
                final_status = Sample.Status.COMPLETED if str(actions[0]).upper() == "DONE" else Sample.Status.FAILED
                break
        else:
            if steps_to_run:
                final_status = Sample.Status.TRUNCATED

        # Use pre-annotated reward (no live evaluator needed)
        outcome_reward = float(metadata.get("reward", -1.0))
        _gui_log("rollout end sample=%s status=%s reward=%.1f", sample.index, final_status.value, outcome_reward)

        with open(result_dir / "result.txt", "w", encoding="utf-8") as f:
            f.write(f"{outcome_reward}\n")

    except Exception as e:
        final_status = Sample.Status.ABORTED
        tb = traceback.format_exc()
        with open(traj_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"Error": str(e), "Traceback": tb}, ensure_ascii=False) + "\n")
        print(
            f"[AGENTNET_ROLLOUT_ERROR] sample_index={sample.index} domain={domain} task_id={task_id}\n{tb}",
            file=sys.stderr,
            flush=True,
        )
        logger.exception("AgentNet rollout failed for sample %s", sample.index)
        outcome_reward = -1.0
    finally:
        trajectory_semaphore.release()

    # Build final training data (identical to generate_with_gui)
    train_messages_for_loss = last_step_train_messages_for_loss or fallback_train_messages
    tool_spec_for_loss = last_step_tool_spec_for_loss or fallback_tool_spec
    input_ids, loss_mask, mm_train = parser.build_train_data(
        args=args,
        state=state,
        train_messages=train_messages_for_loss,
        tool_spec=tool_spec_for_loss,
    )
    active_positions = [i for i in range(len(loss_mask)) if i < len(input_ids) and int(loss_mask[i]) == 1]
    if active_positions:
        response_start = active_positions[0]
        response_length = len(input_ids) - response_start
        loss_mask = [int(loss_mask[i]) if i < len(loss_mask) else 0 for i in range(response_start, len(input_ids))]
    else:
        response_length = 0
        loss_mask = []

    sample.tokens = input_ids
    sample.loss_mask = loss_mask
    sample.response = "\n".join(assistant_responses)
    sample.response_length = response_length
    sample.multimodal_train_inputs = mm_train
    sample.status = final_status
    sample.metadata = sample.metadata or {}
    sample.metadata["agentnet_reward"] = outcome_reward
    sample.reward = {"score": outcome_reward, "acc": float(outcome_reward > 0)}

    if getattr(args, "dynamic_history", False) and not evaluation:
        dynamic_samples = _build_dynamic_history_samples(
            args=args,
            state=state,
            agent=parser,
            base_sample=sample,
            step_snapshots=step_snapshots,
            outcome_reward=outcome_reward,
            prm_score_by_step=None,
        )
        return dynamic_samples

    return sample


async def reward_func(args, samples: list[Sample], **kwargs) -> list[Sample]:
    """Reward is pre-set in generate(); this is a no-op pass-through."""
    return samples
