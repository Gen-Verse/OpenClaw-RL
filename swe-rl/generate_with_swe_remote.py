"""Custom generate/reward for SWE-Bench RL with REMOTE Docker containers.

Drop-in replacement for generate_with_swe.py. Instead of running
Mini-SWE-Agent with local Docker, this version uses SweEnvClient to
interact with remote Docker containers managed by swe_env_pool_server.

The agent logic (multi-turn bash interaction) is reimplemented here
using the same prompt templates from swebench.yaml, but executes
commands via HTTP instead of `docker exec`.

Usage in training script:
    --custom-generate-function-path generate_with_swe_remote.generate
    --custom-rm-path generate_with_swe_remote.reward_func
"""

import asyncio
import copy
import json
import os
import re
import time
import yaml
from functools import lru_cache
from pathlib import Path

from loguru import logger

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.types import Sample

from swe_env_client import SweEnvClient
from swe_utils import get_docker_image_name
from message_utils import get_response_ids_and_loss_mask_from_messages
from swe_context_manager import get_context_messages


SWEAGENT_CONFIG_PATH = os.getenv(
    "SWE_CONFIG_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "swebench.yaml"),
)


def _extract_assistant_turn_spans(loss_mask: list[int]) -> list[list[int]]:
    """Extract [start, end) spans of consecutive 1s from loss_mask.

    Each span corresponds to one assistant turn's generated tokens.
    """
    spans: list[list[int]] = []
    in_span = False
    start = 0
    for i, m in enumerate(loss_mask):
        if m == 1 and not in_span:
            start = i
            in_span = True
        elif m == 0 and in_span:
            spans.append([start, i])
            in_span = False
    if in_span:
        spans.append([start, len(loss_mask)])
    return spans


@lru_cache(maxsize=1)
def _get_swe_semaphore() -> asyncio.Semaphore:
    return asyncio.Semaphore(int(os.getenv("SWE_MAX_CONCURRENT", "8")))


@lru_cache(maxsize=1)
def _get_sweagent_config() -> dict:
    config_path = os.getenv("SWE_CONFIG_PATH", SWEAGENT_CONFIG_PATH)
    for candidate in [config_path, Path(config_path)]:
        p = Path(candidate)
        if p.exists():
            return yaml.safe_load(p.read_text())
    raise FileNotFoundError(f"SWE config not found: {config_path}")


def _sanitize_filename(value: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in value)


@lru_cache(maxsize=1)
def _get_swe_save_dir() -> Path | None:
    save_dir = os.getenv("SWE_SAVE_TRAJ_DIR", "").strip()
    if not save_dir:
        return None
    path = Path(save_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_rollout_artifacts(*, sample: Sample, iid: str, sampling_params: dict, run_info: dict):
    try:
        save_dir = _get_swe_save_dir()
        if save_dir is None:
            return
        ts_ns = time.time_ns()
        stem = (
            f"{_sanitize_filename(iid)}"
            f"__g{sample.group_index if sample.group_index is not None else 'na'}"
            f"__i{sample.index if sample.index is not None else 'na'}"
            f"__{ts_ns}"
        )
        run_dir = save_dir / stem
        run_dir.mkdir(parents=True, exist_ok=True)
        traj_payload = {
            "messages": run_info.get("messages", []),
            "step_debug": run_info.get("step_debug", []),
            "info": {
                "instance_id": iid,
                "exit_status": run_info.get("exit_status"),
                "error": run_info.get("error"),
                "steps": run_info.get("n_steps"),
                "patch_source": run_info.get("patch_source"),
                "reward": run_info.get("reward"),
                "eval_result": run_info.get("eval_result"),
                "group_index": sample.group_index,
                "index": sample.index,
            },
            "trajectory_format": "slime-mini-swe-remote-1",
        }
        (run_dir / "traj.json").write_text(json.dumps(traj_payload, ensure_ascii=True, indent=2, default=str))
        git_patch = run_info.get("git_patch")
        if isinstance(git_patch, str):
            (run_dir / "patch.diff").write_text(git_patch)
        meta_payload = {
            "instance_id": iid,
            "sampling_params": sampling_params,
            "sample_metadata": sample.metadata,
            "sample_prompt": sample.prompt,
            "group_index": sample.group_index,
            "index": sample.index,
        }
        (run_dir / "meta.json").write_text(json.dumps(meta_payload, ensure_ascii=True, indent=2, default=str))
        logger.info(f"[SWE-R] [{iid}] Saved rollout artifacts to {run_dir}")
    except Exception as e:
        logger.warning(f"[SWE-R] [{iid}] Failed to save rollout artifacts: {e}")


def _parse_bash_action(response_text: str) -> str | None:
    """Extract the bash command from a response containing ```bash ... ```."""
    pattern = r"```bash\s*\n(.*?)```"
    match = re.search(pattern, response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def _extract_patch_from_submission(output: str) -> str:
    """Extract a clean git patch text from submit command output."""
    if not isinstance(output, str):
        return ""
    text = output.lstrip("\n")
    sentinel = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
    if text.startswith(sentinel):
        text = text[len(sentinel):].lstrip("\n")
    return text


def _is_valid_git_patch(patch_text: str) -> bool:
    """Lightweight patch validity check before remote evaluation."""
    if not isinstance(patch_text, str):
        return False
    text = patch_text.strip()
    if not text:
        return False
    if "diff --git " not in text:
        return False
    has_old = ("--- a/" in text) or ("--- /dev/null" in text)
    has_new = "+++ b/" in text
    return has_old and has_new


def _render_observation(config: dict, returncode: int, output: str) -> str:
    """Render the action_observation_template from swebench.yaml."""
    from jinja2 import Template
    template_str = config.get("agent", {}).get("action_observation_template", "")
    if not template_str:
        return f"<returncode>{returncode}</returncode>\n<output>\n{output}\n</output>"
    template = Template(template_str)
    return template.render(output={"returncode": returncode, "output": output})


async def _run_agent_remote(
    env_client: SweEnvClient,
    lease_id: str,
    instance: dict,
    litellm_model_name: str,
    model_config: dict,
    sweagent_config: dict,
    *,
    args: object | None = None,
    prm_agent: object | None = None,
    tokenizer: object | None = None,
    cm_max_input_tokens: int | None = None,
    cm_head_ratio: float = 0.3,
) -> dict:
    """Run the multi-turn agent loop using remote Docker execution."""
    from litellm import acompletion

    iid = instance.get("instance_id", "unknown")
    agent_config = sweagent_config.get("agent", {})
    env_config = sweagent_config.get("environment", {})
    cwd = env_config.get("cwd", "/testbed")
    step_limit = int(agent_config.get("step_limit", 20))
    exec_timeout = int(env_config.get("timeout", 180))

    system_template = agent_config.get("system_template", "You are a helpful assistant.")
    instance_template = agent_config.get("instance_template", "{{task}}")
    from jinja2 import Template
    instance_message = Template(instance_template).render(task=instance["problem_statement"])

    messages = [
        {"role": "system", "content": system_template},
        {"role": "user", "content": instance_message},
    ]

    step_debug = []
    git_patch = None
    patch_source = None
    exit_status = None
    error = None
    n_steps = 0
    prm_pending_tasks: list[tuple[int, asyncio.Task]] = []
    managed_contexts: list[list[dict]] = []
    assistant_texts: list[str] = []

    cm_enabled = tokenizer is not None and cm_max_input_tokens is not None and cm_max_input_tokens > 0

    t0 = time.time()
    for step_idx in range(step_limit):
        n_steps = step_idx + 1
        await env_client.heartbeat(lease_id)

        if cm_enabled:
            ctx_messages = get_context_messages(
                messages, tokenizer,
                max_input_tokens=cm_max_input_tokens,
                head_ratio=cm_head_ratio,
            )
        else:
            ctx_messages = messages

        try:
            resp = await acompletion(
                model=litellm_model_name,
                messages=ctx_messages,
                **model_config.get("model_kwargs", {}),
            )
            assistant_text = resp.choices[0].message.content or ""
        except Exception as e:
            error = f"LLM call failed at step {step_idx}: {e}"
            logger.error(f"[SWE-R] [{iid}] {error}")
            break

        managed_contexts.append(copy.deepcopy(ctx_messages))
        messages.append({"role": "assistant", "content": assistant_text})
        assistant_texts.append(assistant_text)

        bash_cmd = _parse_bash_action(assistant_text)
        if bash_cmd is None:
            observation = _render_observation(sweagent_config, -1, "No valid bash command found in response.")
            messages.append({"role": "user", "content": observation})
            continue

        is_submit = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in bash_cmd

        step_t0 = time.time()
        try:
            exec_result = await env_client.exec(
                lease_id=lease_id, command=bash_cmd, cwd=cwd, timeout=exec_timeout,
            )
            returncode = exec_result.get("returncode", -1)
            output = exec_result.get("output", "")
        except Exception as e:
            returncode = -1
            output = f"Execution error: {e}"
            logger.error(f"[SWE-R] [{iid}] step {step_idx} exec error: {e}")

        step_debug.append({
            "step_idx": step_idx,
            "action": bash_cmd,
            "returncode": returncode,
            "output_len": len(output),
            "output_head": output[:2000],
            "output_tail": output[-2000:] if len(output) > 2000 else output,
            "start_ts": step_t0,
            "end_ts": time.time(),
            "ok": returncode != -1,
        })

        # PRM: dispatch async scoring right after execution result is ready.
        skip_prm = is_submit and getattr(prm_agent, "skip_submit", True)
        if prm_agent is not None and args is not None and not skip_prm:
            prm_pending_tasks.append((
                step_idx,
                prm_agent.submit_step_judge(
                    args,
                    problem_statement=instance.get("problem_statement", ""),
                    step_debug=list(step_debug),
                    policy_response=assistant_text,
                    step_index=step_idx,
                ),
            ))

        if is_submit:
            exit_status = "submitted"
            candidate_patch = _extract_patch_from_submission(output)
            if _is_valid_git_patch(candidate_patch):
                git_patch = candidate_patch
                patch_source = "submission"
            break

        observation = _render_observation(sweagent_config, returncode, output)
        remaining = step_limit - (step_idx + 1)
        if remaining == 1:
            observation += "\nREMINDER: You only have 1 turn left. Please provide the final answer"
        elif remaining > 1:
            observation += f"\nREMINDER: You have {remaining} turns left to arrive at the solution."
        messages.append({"role": "user", "content": observation})

    if git_patch is None:
        try:
            diff_result = await env_client.diff(lease_id=lease_id, cwd=cwd)
            fallback_patch = diff_result if isinstance(diff_result, str) else ""
            if _is_valid_git_patch(fallback_patch):
                git_patch = fallback_patch
                patch_source = "git_diff_fallback"
            if exit_status is None:
                exit_status = "max_steps"
        except Exception as e:
            error = f"diff failed: {e}"

    # PRM: collect all pending results
    prm_step_scores: list[float] = []
    prm_step_details: list[dict] = []
    if prm_agent is not None and prm_pending_tasks:
        prm_step_scores, prm_step_details = await prm_agent.collect_step_results(prm_pending_tasks)

    logger.info(
        f"[SWE-R] [{iid}] Agent done: steps={n_steps}, exit={exit_status}, "
        f"patch={'yes' if git_patch else 'no'}, "
        f"prm_steps={len(prm_step_scores)}, elapsed={time.time()-t0:.1f}s"
    )
    return {
        "messages": messages,
        "step_debug": step_debug,
        "git_patch": git_patch,
        "patch_source": patch_source,
        "exit_status": exit_status,
        "n_steps": n_steps,
        "error": error,
        "prm_step_scores": prm_step_scores,
        "prm_step_details": prm_step_details,
        "managed_contexts": managed_contexts,
        "assistant_texts": assistant_texts,
    }


def _ensure_openai_base_url(args):
    """Set OPENAI_BASE_URL from the framework's auto-detected router address.

    When OPENAI_BASE_URL is 'auto' or unset, derive it from
    args.sglang_router_ip / args.sglang_router_port which are populated
    by _start_router() after the router is actually running.
    """
    current = os.environ.get("OPENAI_BASE_URL", "")
    if current and current != "auto":
        return
    router_ip = getattr(args, "sglang_router_ip", None)
    router_port = getattr(args, "sglang_router_port", None)
    if router_ip and router_port:
        url = f"http://{router_ip}:{router_port}/v1"
        os.environ["OPENAI_BASE_URL"] = url
        logger.info(f"[SWE-R] OPENAI_BASE_URL resolved to {url}")


async def generate(args, sample: Sample, sampling_params: dict) -> Sample | list[Sample]:
    """Called by slime via ``--custom-generate-function-path``.

    Each sample corresponds to a SWE-Bench instance executed inside a
    remote Docker container via swe_env_pool_server + swe_exec_server.
    """
    rollout_timeout = float(os.getenv("SWE_ROLLOUT_TIMEOUT", "1800"))
    iid = (
        sample.metadata.get("instance", {}).get("instance_id", "unknown")
        if isinstance(sample.metadata, dict)
        else "unknown"
    )
    try:
        return await asyncio.wait_for(
            _generate_impl(args, sample, sampling_params),
            timeout=rollout_timeout,
        )
    except (asyncio.TimeoutError, TimeoutError):
        logger.error(
            f"[SWE-R] [{iid}] TOTAL ROLLOUT TIMEOUT ({rollout_timeout}s) exceeded, aborting sample"
        )
        sample.status = Sample.Status.ABORTED
        sample.reward = {"score": 0.0, "acc": 0.0}
        sample.remove_sample = True
        return sample


async def _generate_impl(args, sample: Sample, sampling_params: dict) -> Sample | list[Sample]:
    """Core implementation — called via generate() with a total timeout guard."""
    _ensure_openai_base_url(args)
    state = GenerateState(args)
    instance = sample.metadata.get("instance", {})
    data_source = sample.metadata.get("data_source", "swe-gym")
    iid = instance.get("instance_id", "unknown")

    # PRM agent initialization
    prm_agent = None
    if getattr(args, "prm_enable", False):
        from swe_prm import SweRewardAgent
        prm_agent = SweRewardAgent(
            max_history_steps=int(getattr(args, "swe_prm_max_history_steps", 8)),
            max_problem_len=int(getattr(args, "swe_prm_max_problem_len", 8000)),
            max_output_len=int(getattr(args, "swe_prm_max_output_len", 4000)),
            max_history_output_len=int(getattr(args, "swe_prm_max_history_output_len", 1000)),
            skip_submit=bool(getattr(args, "swe_prm_skip_submit", True)),
            tokenizer=state.tokenizer,
        )

    t_start = time.time()
    swe_env_url = os.getenv("SWE_ENV_SERVER_URL", "?")
    logger.info(
        "[SWE-R] ========== REMOTE ROLLOUT ENTERED ========== instance_id={} | SWE_ENV_SERVER={} | data_source={}",
        iid, swe_env_url, data_source,
    )
    logger.info(f"[SWE-R] [{iid}] Step 1/5: generate() called, data_source={data_source}")

    sweagent_config = _get_sweagent_config()
    image_name = get_docker_image_name(instance, data_source)

    model_config = sweagent_config.get("model", {})
    litellm_model_name = (
        model_config.get("model_name")
        or os.getenv("SWE_LITELLM_MODEL_NAME")
        or "openai/Qwen/Qwen3-8B"
    )
    model_config["model_name"] = litellm_model_name
    model_config.setdefault("model_kwargs", {}).update({
        "temperature": sampling_params.get("temperature", 1.0),
        "max_tokens": sampling_params.get("max_new_tokens", 4096),
    })

    env_client = SweEnvClient()
    swe_semaphore = _get_swe_semaphore()

    logger.info(f"[SWE-R] [{iid}] Step 1/5: Waiting for semaphore...")
    await swe_semaphore.acquire()
    logger.info(f"[SWE-R] [{iid}] Step 1/5: Semaphore acquired ({time.time()-t_start:.1f}s)")

    lease_id = None
    eval_lease_id = None
    run_info = {"messages": [], "step_debug": [], "reward": 0, "error": None,
                "git_patch": None, "patch_source": None, "exit_status": None, "n_steps": 0, "eval_result": None}

    try:
        logger.info(f"[SWE-R] [{iid}] Step 2/5: Allocating container for {image_name}")
        lease = await env_client.allocate(image=image_name, instance_id=iid)
        lease_id = lease["lease_id"]
        logger.info(f"[SWE-R] [{iid}] Step 2/5: Container ready, lease={lease_id}")

        max_context_len = int(getattr(args, "rollout_max_context_len", 0) or 0)
        max_new_tokens = int(model_config.get("model_kwargs", {}).get("max_tokens", 4096))
        cm_max_input_tokens = max(1, max_context_len - max_new_tokens) if max_context_len > 0 else None
        cm_head_ratio = float(getattr(args, "swe_cm_head_ratio", 0.3))

        logger.info(
            f"[SWE-R] [{iid}] Step 3/5: Running agent... "
            f"(cm_max_input_tokens={cm_max_input_tokens}, cm_head_ratio={cm_head_ratio})"
        )
        agent_result = await _run_agent_remote(
            env_client, lease_id, instance, litellm_model_name, model_config, sweagent_config,
            args=args, prm_agent=prm_agent,
            tokenizer=state.tokenizer,
            cm_max_input_tokens=cm_max_input_tokens,
            cm_head_ratio=cm_head_ratio,
        )
        run_info.update(agent_result)

        git_patch = run_info.get("git_patch")
        if git_patch:
            try:
                await env_client.close(lease_id)
                logger.info(f"[SWE-R] [{iid}] Step 3/5: Closed agent container lease={lease_id}")
                lease_id = None
            except Exception:
                logger.exception(f"[SWE-R] [{iid}] Failed to close agent lease before eval")

            logger.info(f"[SWE-R] [{iid}] Step 4/5: Allocating fresh eval container...")
            try:
                eval_lease = await env_client.allocate(image=image_name, instance_id=f"{iid}__eval")
                eval_lease_id = eval_lease["lease_id"]
                logger.info(f"[SWE-R] [{iid}] Step 4/5: Eval container ready, lease={eval_lease_id}")
                eval_result = await env_client.evaluate(
                    lease_id=eval_lease_id,
                    patch=git_patch,
                    eval_script=instance.get("eval_script", ""),
                )
                resolved = eval_result.get("resolved", False)
                run_info["reward"] = int(resolved)
                run_info["eval_result"] = eval_result
                logger.info(f"[SWE-R] [{iid}] Step 4/5: resolved={resolved}")
            except Exception as e:
                run_info["error"] = str(e)
                logger.error(f"[SWE-R] [{iid}] Step 4/5: Eval error: {e}")
        else:
            logger.warning(f"[SWE-R] [{iid}] Step 4/5: No patch, skipping eval")

    except Exception as e:
        run_info["error"] = str(e)
        logger.exception(f"[SWE-R] [{iid}] Error: {e}")
    finally:
        if eval_lease_id is not None:
            try:
                await env_client.close(eval_lease_id)
            except BaseException:
                logger.warning(f"[SWE-R] [{iid}] Failed to close eval lease (may be cancelled)")
        if lease_id is not None:
            try:
                await env_client.close(lease_id)
            except BaseException:
                logger.warning(f"[SWE-R] [{iid}] Failed to close lease (may be cancelled)")
        swe_semaphore.release()
        logger.info(f"[SWE-R] [{iid}] Semaphore released")

    messages = run_info["messages"]
    reward = run_info["reward"]
    error = run_info["error"]
    managed_contexts = run_info.get("managed_contexts", [])
    assistant_texts = run_info.get("assistant_texts", [])

    _save_rollout_artifacts(sample=sample, iid=iid, sampling_params=sampling_params, run_info=run_info)

    if not messages:
        logger.warning(f"[SWE-R] [{iid}] Step 5/5: ABORTED — no messages (error={error})")
        sample.status = Sample.Status.ABORTED
        sample.reward = {"score": 0.0, "acc": 0.0}
        sample.remove_sample = True
        return sample

    use_dynamic_history = getattr(args, "dynamic_history", False) and managed_contexts and assistant_texts

    outcome_reward = 1.0 if reward else -1.0
    prm_step_scores = run_info.get("prm_step_scores", [])
    prm_step_details = run_info.get("prm_step_details", [])

    # ------------------------------------------------------------------
    # Dynamic-history path: one training sample per step, each with
    # the managed context the model actually saw during rollout.
    # ------------------------------------------------------------------
    if use_dynamic_history:
        dynamic_samples: list[Sample] = []
        n_steps = min(len(managed_contexts), len(assistant_texts))

        for step_idx in range(n_steps):
            ctx_msgs = managed_contexts[step_idx]
            resp_text = assistant_texts[step_idx]

            prompt_ids = state.tokenizer.apply_chat_template(
                ctx_msgs, add_generation_prompt=True, tokenize=True,
            )
            resp_msgs = [{"role": "assistant", "content": resp_text}]
            response_ids, loss_mask, _ = get_response_ids_and_loss_mask_from_messages(
                resp_msgs, state.tokenizer, assistant_logprobs=None,
            )

            max_ctx = int(getattr(args, "rollout_max_context_len", 0) or 0)
            if max_ctx > 0:
                max_resp = max(1, max_ctx - len(prompt_ids))
                if len(response_ids) > max_resp:
                    response_ids = response_ids[:max_resp]
                    loss_mask = loss_mask[:max_resp]

            child = copy.deepcopy(sample)
            child.tokens = prompt_ids + response_ids
            child.response = resp_text
            child.response_length = len(response_ids)
            child.loss_mask = loss_mask
            child.rollout_log_probs = None
            child.status = Sample.Status.COMPLETED if response_ids else Sample.Status.ABORTED

            child.metadata = copy.deepcopy(sample.metadata or {})
            child.metadata["dynamic_step_index"] = step_idx
            child.metadata["dynamic_outcome_reward"] = float(outcome_reward)
            child.metadata["num_steps"] = n_steps

            step_reward = float(outcome_reward)
            prm_score = 0.0
            if step_idx < len(prm_step_scores):
                prm_score = float(prm_step_scores[step_idx])

            if getattr(args, "prm_enable", False):
                child.metadata["step_wise"] = {
                    "step_scores": [prm_score],
                    "step_indices": [step_idx],
                    "step_token_spans": [[0, len(response_ids)]],
                    "step_scores_with_outcome": [prm_score + step_reward],
                    "outcome_reward": step_reward,
                }
                child.reward = None
            else:
                child.reward = {"score": step_reward, "acc": float(reward)}

            if child.status == Sample.Status.ABORTED:
                child.reward = child.reward or {"score": 0.0, "acc": 0.0}
                child.remove_sample = True

            dynamic_samples.append(child)

        if not dynamic_samples:
            sample.status = Sample.Status.ABORTED
            sample.reward = {"score": 0.0, "acc": 0.0}
            sample.remove_sample = True
            return [sample]

        if getattr(args, "prm_enable", False) and prm_step_scores:
            for child in dynamic_samples:
                child.metadata["prm"] = {
                    "enabled": True,
                    "step_scores": prm_step_scores,
                    "step_mean_score": (
                        sum(prm_step_scores) / len(prm_step_scores)
                    ),
                    "step_details": prm_step_details,
                }

        elapsed = time.time() - t_start
        logger.info(
            f"[SWE-R] [{iid}] Step 5/5: DONE (dynamic_history) — "
            f"n_samples={len(dynamic_samples)}, outcome_reward={outcome_reward}, "
            f"prm_enabled={getattr(args, 'prm_enable', False)}, "
            f"total_elapsed={elapsed:.1f}s"
        )
        return dynamic_samples

    # ------------------------------------------------------------------
    # Default path: single training sample from full messages.
    # ------------------------------------------------------------------
    prompt_messages = messages[:2]
    response_messages = messages[2:]

    while response_messages and response_messages[-1]["role"] == "user":
        response_messages.pop()

    if not response_messages:
        logger.warning(f"[SWE-R] [{iid}] Step 5/5: ABORTED — no assistant messages")
        sample.status = Sample.Status.ABORTED
        sample.reward = {"score": 0.0, "acc": 0.0}
        sample.remove_sample = True
        return sample

    prompt_ids = state.tokenizer.apply_chat_template(
        prompt_messages, add_generation_prompt=True, tokenize=True
    )
    response_ids, loss_mask, _ = get_response_ids_and_loss_mask_from_messages(
        response_messages, state.tokenizer, assistant_logprobs=None
    )

    max_context_len = getattr(args, "rollout_max_context_len", None)
    if max_context_len is not None:
        max_response_tokens = max(1, int(max_context_len) - len(prompt_ids))
    else:
        max_response_tokens = getattr(args, "rollout_max_response_len", 4096)
    if len(response_ids) > max_response_tokens:
        response_ids = response_ids[:max_response_tokens]
        loss_mask = loss_mask[:max_response_tokens]
        sample.status = Sample.Status.TRUNCATED

    sample.tokens = prompt_ids + response_ids
    sample.response = "\n".join(m["content"] for m in response_messages if m["role"] == "assistant")
    sample.response_length = len(response_ids)
    sample.loss_mask = loss_mask
    sample.rollout_log_probs = None
    if sample.status == Sample.Status.PENDING:
        sample.status = Sample.Status.COMPLETED

    # PRM metadata
    if getattr(args, "prm_enable", False):
        sample.metadata = sample.metadata or {}
        sample.metadata["prm"] = {
            "enabled": True,
            "step_scores": prm_step_scores,
            "step_mean_score": (sum(prm_step_scores) / len(prm_step_scores)) if prm_step_scores else 0.0,
            "step_details": prm_step_details,
        }

        step_token_spans = _extract_assistant_turn_spans(loss_mask)
        n_aligned = min(len(prm_step_scores), len(step_token_spans))
        sample.metadata["step_wise"] = {
            "step_scores": prm_step_scores[:n_aligned],
            "step_indices": list(range(n_aligned)),
            "step_token_spans": step_token_spans[:n_aligned],
            "step_scores_with_outcome": [
                float(s) + outcome_reward for s in prm_step_scores[:n_aligned]
            ],
            "outcome_reward": outcome_reward,
        }
        sample.reward = None
    else:
        sample.reward = {"score": outcome_reward, "acc": float(reward)}

    elapsed = time.time() - t_start
    logger.info(
        f"[SWE-R] [{iid}] Step 5/5: DONE — status={sample.status.name}, "
        f"reward={sample.reward}, response_len={sample.response_length}, "
        f"prm_enabled={getattr(args, 'prm_enable', False)}, "
        f"total_elapsed={elapsed:.1f}s"
    )
    return sample


async def reward_func(args, sample: Sample | list[Sample], **kwargs):
    """Compute reward, integrating PRM step-wise scores when enabled."""

    prm_step_coef = float(getattr(args, "prm_step_coef", 1.0))

    def _get_reward(s: Sample) -> dict:
        if getattr(args, "prm_enable", False) and isinstance(s.metadata, dict):
            prm_meta = s.metadata.get("prm", {})
            step_wise_meta = s.metadata.get("step_wise", {})
            outcome_reward = step_wise_meta.get("outcome_reward", 0.0)
            prm_step_mean = float(prm_meta.get("step_mean_score", 0.0))
            final_score = outcome_reward + prm_step_coef * prm_step_mean
            return {
                "score": final_score,
                "acc": 1.0 if outcome_reward > 0 else 0.0,
                "outcome_reward": outcome_reward,
                "prm_step_mean": prm_step_mean,
                "prm_step_coef": prm_step_coef,
            }

        if isinstance(s.reward, dict):
            return s.reward
        acc = float(s.metadata.get("eval_score", 0.0)) if isinstance(s.metadata, dict) else 0.0
        return {"score": 1.0 if acc == 1.0 else -1.0, "acc": acc}

    if isinstance(sample, list):
        return [_get_reward(s) for s in sample]
    return _get_reward(sample)
