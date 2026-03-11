#!/usr/bin/env python3
"""Standalone eval script: run a stronger LLM on SWE training data.

Uses the same remote Docker eval infrastructure as training, but bypasses
the slime training loop entirely. Useful for:
  - Verifying the eval pipeline is correct before / during RL training
  - Establishing an upper-bound baseline (e.g. Qwen3-32B on train set)
  - Debugging reward signal issues

Usage:
  # Make sure swe_env_pool_server and swe_exec_server(s) are already running.
  PYTHONPATH=. python eval_swe.py \\
    --data     /path/to/train.jsonl \\
    --model    openai/Qwen/Qwen3-32B \\
    --api-base http://<ROUTER_IP>:8000/v1 \\
    --output-dir /path/to/eval_runs/qwen32b_train \\
    --max-concurrent 4 \\
    --max-instances 50 \\
    --step-limit 20 \\
    --max-tokens 4096

Environment variables (override CLI):
  SWE_ENV_SERVER_URL  - pool server URL (default http://localhost:18090)
  OPENAI_API_KEY      - forwarded to litellm
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time

# Parent directory contains swe_env_client, swe_utils, etc.
sys.path.insert(0, str(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
from datetime import datetime
from pathlib import Path

import yaml
from loguru import logger

# ---------------------------------------------------------------------------
# Lazy imports so the script can be syntax-checked without all deps installed.
# ---------------------------------------------------------------------------


def _import_litellm_acompletion():
    from litellm import acompletion  # noqa: PLC0415
    return acompletion


# ---------------------------------------------------------------------------
# Inline copies of helpers from generate_with_swe_remote.py
# (avoids requiring slime to be importable just for the agent loop)
# ---------------------------------------------------------------------------

def _parse_bash_action(response_text: str) -> str | None:
    """Extract the bash command from ```bash ... ```."""
    match = re.search(r"```bash\s*\n(.*?)```", response_text, re.DOTALL)
    return match.group(1).strip() if match else None


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
    template_str = config.get("agent", {}).get("action_observation_template", "")
    if not template_str:
        return f"<returncode>{returncode}</returncode>\n<output>\n{output}\n</output>"
    from jinja2 import Template  # noqa: PLC0415
    return Template(template_str).render(output={"returncode": returncode, "output": output})


async def _run_agent_loop(
    env_client,
    lease_id: str,
    instance: dict,
    litellm_model_name: str,
    model_kwargs: dict,
    sweagent_config: dict,
) -> dict:
    """Multi-turn agent loop.  Same logic as generate_with_swe_remote._run_agent_remote."""
    acompletion = _import_litellm_acompletion()

    iid = instance.get("instance_id", "unknown")
    agent_config = sweagent_config.get("agent", {})
    env_config = sweagent_config.get("environment", {})
    cwd = env_config.get("cwd", "/testbed")
    step_limit = int(agent_config.get("step_limit", 20))
    exec_timeout = int(env_config.get("timeout", 180))

    system_template = agent_config.get("system_template", "You are a helpful assistant.")
    instance_template = agent_config.get("instance_template", "{{task}}")
    from jinja2 import Template  # noqa: PLC0415
    instance_message = Template(instance_template).render(task=instance["problem_statement"])

    messages = [
        {"role": "system", "content": system_template},
        {"role": "user", "content": instance_message},
    ]

    step_debug: list[dict] = []
    git_patch = None
    patch_source = None
    exit_status = None
    error = None
    n_steps = 0

    t0 = time.time()
    for step_idx in range(step_limit):
        n_steps = step_idx + 1
        await env_client.heartbeat(lease_id)

        try:
            resp = await acompletion(
                model=litellm_model_name,
                messages=messages,
                **model_kwargs,
            )
            assistant_text = resp.choices[0].message.content or ""
        except Exception as exc:
            error = f"LLM call failed at step {step_idx}: {exc}"
            logger.error(f"[{iid}] {error}")
            break

        messages.append({"role": "assistant", "content": assistant_text})

        bash_cmd = _parse_bash_action(assistant_text)
        if bash_cmd is None:
            obs = _render_observation(sweagent_config, -1, "No valid bash command found in response.")
            messages.append({"role": "user", "content": obs})
            continue

        is_submit = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in bash_cmd

        step_t0 = time.time()
        try:
            exec_result = await env_client.exec(
                lease_id=lease_id, command=bash_cmd, cwd=cwd, timeout=exec_timeout,
            )
            returncode = exec_result.get("returncode", -1)
            output = exec_result.get("output", "")
        except Exception as exc:
            returncode = -1
            output = f"Execution error: {exc}"

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

        if is_submit:
            exit_status = "submitted"
            candidate_patch = _extract_patch_from_submission(output)
            if _is_valid_git_patch(candidate_patch):
                git_patch = candidate_patch
                patch_source = "submission"
            break

        obs = _render_observation(sweagent_config, returncode, output)
        remaining = step_limit - (step_idx + 1)
        if remaining == 1:
            obs += "\nREMINDER: You only have 1 turn left. Please provide the final answer"
        elif remaining > 1:
            obs += f"\nREMINDER: You have {remaining} turns left to arrive at the solution."
        messages.append({"role": "user", "content": obs})

    if git_patch is None:
        try:
            diff_result = await env_client.diff(lease_id=lease_id, cwd=cwd)
            fallback_patch = diff_result if isinstance(diff_result, str) else ""
            if _is_valid_git_patch(fallback_patch):
                git_patch = fallback_patch
                patch_source = "git_diff_fallback"
            if exit_status is None:
                exit_status = "max_steps"
        except Exception as exc:
            error = f"diff failed: {exc}"

    logger.info(
        f"[{iid}] Agent done: steps={n_steps}, exit={exit_status}, "
        f"patch={'yes' if git_patch else 'no'}, elapsed={time.time()-t0:.1f}s"
    )
    return {
        "messages": messages,
        "step_debug": step_debug,
        "git_patch": git_patch,
        "patch_source": patch_source,
        "exit_status": exit_status,
        "n_steps": n_steps,
        "error": error,
    }


# ---------------------------------------------------------------------------
# Artifact saving (same schema as training traj.json / patch.diff / meta.json)
# ---------------------------------------------------------------------------

def _save_artifacts(run_dir: Path, instance: dict, run_info: dict, args: argparse.Namespace):
    run_dir.mkdir(parents=True, exist_ok=True)
    traj_payload = {
        "messages": run_info.get("messages", []),
        "step_debug": run_info.get("step_debug", []),
        "info": {
            "instance_id": instance.get("instance_id"),
            "exit_status": run_info.get("exit_status"),
            "error": run_info.get("error"),
            "steps": run_info.get("n_steps"),
            "patch_source": run_info.get("patch_source"),
            "reward": run_info.get("reward"),
            "eval_result": run_info.get("eval_result"),
        },
        "trajectory_format": "slime-mini-swe-remote-eval-1",
    }
    (run_dir / "traj.json").write_text(
        json.dumps(traj_payload, ensure_ascii=True, indent=2, default=str)
    )
    git_patch = run_info.get("git_patch")
    if isinstance(git_patch, str) and git_patch.strip():
        (run_dir / "patch.diff").write_text(git_patch)
    meta_payload = {
        "instance_id": instance.get("instance_id"),
        "model": args.model,
        "api_base": args.api_base,
        "step_limit": args.step_limit,
        "max_tokens": args.max_tokens,
    }
    (run_dir / "meta.json").write_text(
        json.dumps(meta_payload, ensure_ascii=True, indent=2, default=str)
    )


# ---------------------------------------------------------------------------
# Per-instance driver
# ---------------------------------------------------------------------------

async def _eval_instance(
    instance: dict,
    data_source: str,
    sweagent_config: dict,
    args: argparse.Namespace,
    semaphore: asyncio.Semaphore,
    output_dir: Path,
    results: list[dict],
):
    from swe_env_client import SweEnvClient  # noqa: PLC0415
    from swe_utils import get_docker_image_name  # noqa: PLC0415

    iid = instance.get("instance_id", "unknown")

    def _sanitize(v: str) -> str:
        return "".join(c if c.isalnum() or c in "._-" else "_" for c in v)

    run_dir = output_dir / f"{_sanitize(iid)}__{time.time_ns()}"

    run_info: dict = {
        "messages": [], "step_debug": [], "reward": 0,
        "error": None, "git_patch": None, "exit_status": None,
        "n_steps": 0, "eval_result": None, "patch_source": None,
    }

    async with semaphore:
        image_name = get_docker_image_name(instance, data_source)
        env_client = SweEnvClient(base_url=args.env_server_url)
        lease_id = None
        eval_lease_id = None
        t0 = time.time()
        try:
            logger.info(f"[{iid}] Allocating container: {image_name}")
            lease = await env_client.allocate(image=image_name, instance_id=iid)
            lease_id = lease["lease_id"]

            model_kwargs = {
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
            }
            # Override step_limit in config for this run
            sweagent_config_copy = json.loads(json.dumps(sweagent_config))
            sweagent_config_copy.setdefault("agent", {})["step_limit"] = args.step_limit

            agent_result = await _run_agent_loop(
                env_client, lease_id, instance,
                args.model, model_kwargs, sweagent_config_copy,
            )
            run_info.update(agent_result)

            git_patch = run_info.get("git_patch")
            if git_patch and git_patch.strip():
                try:
                    await env_client.close(lease_id)
                    logger.info(f"[{iid}] Closed agent container lease={lease_id}")
                    lease_id = None
                except Exception:
                    logger.exception(f"[{iid}] Failed to close agent lease before eval")

                logger.info(f"[{iid}] Allocating fresh eval container...")
                try:
                    eval_lease = await env_client.allocate(image=image_name, instance_id=f"{iid}__eval")
                    eval_lease_id = eval_lease["lease_id"]
                    logger.info(f"[{iid}] Eval container ready, lease={eval_lease_id}")
                    eval_result = await env_client.evaluate(
                        lease_id=eval_lease_id,
                        patch=git_patch,
                        eval_script=instance.get("eval_script", ""),
                    )
                    resolved = eval_result.get("resolved", False)
                    run_info["reward"] = int(resolved)
                    run_info["eval_result"] = eval_result
                    logger.info(f"[{iid}] resolved={resolved}")
                except Exception as exc:
                    run_info["error"] = str(exc)
                    logger.error(f"[{iid}] Eval error: {exc}")
            else:
                logger.warning(f"[{iid}] No patch generated, skipping eval")

        except Exception as exc:
            run_info["error"] = str(exc)
            logger.exception(f"[{iid}] Fatal error: {exc}")
        finally:
            if eval_lease_id is not None:
                try:
                    await env_client.close(eval_lease_id)
                except Exception:
                    pass
            if lease_id is not None:
                try:
                    await env_client.close(lease_id)
                except Exception:
                    pass

    _save_artifacts(run_dir, instance, run_info, args)

    result_row = {
        "instance_id": iid,
        "resolved": bool(run_info["reward"]),
        "exit_status": run_info.get("exit_status"),
        "patch_source": run_info.get("patch_source"),
        "n_steps": run_info.get("n_steps", 0),
        "error": run_info.get("error"),
        "elapsed": round(time.time() - t0, 1),
        "run_dir": str(run_dir),
    }
    results.append(result_row)

    status = "✓ RESOLVED" if result_row["resolved"] else "✗ failed"
    logger.info(f"[{iid}] {status}  steps={result_row['n_steps']}  elapsed={result_row['elapsed']}s")
    return result_row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Eval a stronger LLM on SWE training data via remote Docker."
    )
    parser.add_argument("--data", required=True,
                        help="Path to JSONL training dataset (same format used in training).")
    parser.add_argument("--model", required=True,
                        help="LiteLLM model name, e.g. 'openai/Qwen/Qwen3-32B'.")
    parser.add_argument("--api-base", default=None,
                        help="OpenAI-compatible API base URL. If unset, uses OPENAI_BASE_URL env var.")
    parser.add_argument("--api-key", default=None,
                        help="API key. If unset, uses OPENAI_API_KEY env var.")
    parser.add_argument("--output-dir", default=None,
                        help="Directory to save per-instance results. "
                             "Default: export/eval_runs/<model>_<timestamp>.")
    parser.add_argument("--env-server-url", default=None,
                        help="Pool server URL. Default: SWE_ENV_SERVER_URL env or http://localhost:18090.")
    parser.add_argument("--swe-config", default=None,
                        help="Path to swebench.yaml. Default: SWE_CONFIG_PATH env or swebench.yaml.")
    parser.add_argument("--data-source", default="swe-gym",
                        help="Dataset type: 'swe-gym' or 'swe-bench'. Controls Docker image naming.")
    parser.add_argument("--max-concurrent", type=int, default=4,
                        help="Max concurrent Docker containers. Default: 4.")
    parser.add_argument("--max-instances", type=int, default=None,
                        help="Stop after this many instances. Default: run all.")
    parser.add_argument("--shuffle", action="store_true",
                        help="Shuffle instances before picking --max-instances.")
    parser.add_argument("--step-limit", type=int, default=20,
                        help="Max agent steps per instance. Default: 20.")
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="Max tokens per LLM call. Default: 4096.")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature. Default: 0 (greedy for eval).")
    return parser.parse_args()


async def main():
    args = _parse_args()

    # --- Initialize slime's global HTTP client ---
    # In training this is done by init_http_client(args); we do it directly here.
    import httpx  # noqa: PLC0415
    from slime.utils import http_utils  # noqa: PLC0415
    if http_utils._http_client is None:
        http_utils._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=30.0, read=300.0, write=300.0, pool=300.0),
            limits=httpx.Limits(max_connections=256, max_keepalive_connections=64),
        )
        logger.info("Initialized slime HTTP client")

    # --- env vars ---
    if args.api_base:
        os.environ["OPENAI_BASE_URL"] = args.api_base
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
    if args.env_server_url is None:
        args.env_server_url = os.getenv("SWE_ENV_SERVER_URL", "http://localhost:18090")

    # --- swebench config ---
    _default_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "swebench.yaml")
    config_path = args.swe_config or os.getenv("SWE_CONFIG_PATH", _default_config)
    sweagent_config = yaml.safe_load(Path(config_path).read_text())

    # --- output dir ---
    if args.output_dir is None:
        model_slug = args.model.replace("/", "_").replace(":", "_")
        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        args.output_dir = str(
            Path(__file__).resolve().parent.parent
            / "output" / "eval_runs" / f"{model_slug}_{ts}"
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output dir: {output_dir}")

    # --- load dataset ---
    instances = []
    with open(args.data) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            # Support both raw instance format and slime training format
            if "metadata" in row and "instance" in row["metadata"]:
                instances.append(row["metadata"]["instance"])
            elif "instance_id" in row:
                instances.append(row)
            else:
                logger.warning(f"Skipping unrecognised row (no instance_id): {list(row.keys())}")

    if args.shuffle:
        import random
        random.shuffle(instances)
    if args.max_instances is not None:
        instances = instances[: args.max_instances]

    logger.info(f"Loaded {len(instances)} instances from {args.data}")
    logger.info(f"Model: {args.model}  api_base: {os.environ.get('OPENAI_BASE_URL', '(default)')}")
    logger.info(f"step_limit={args.step_limit}  max_tokens={args.max_tokens}  temperature={args.temperature}")
    logger.info(f"max_concurrent={args.max_concurrent}")

    # --- run ---
    semaphore = asyncio.Semaphore(args.max_concurrent)
    results: list[dict] = []
    tasks = [
        _eval_instance(
            instance=inst,
            data_source=args.data_source,
            sweagent_config=sweagent_config,
            args=args,
            semaphore=semaphore,
            output_dir=output_dir,
            results=results,
        )
        for inst in instances
    ]
    await asyncio.gather(*tasks)

    # --- summary ---
    total = len(results)
    resolved = sum(1 for r in results if r["resolved"])
    errors = sum(1 for r in results if r["error"])
    # Count runs where no valid patch could be extracted from either submission
    # output or fallback git diff.
    no_patch = sum(1 for r in results if r.get("patch_source") is None)
    avg_steps = sum(r["n_steps"] for r in results) / max(total, 1)
    avg_elapsed = sum(r["elapsed"] for r in results) / max(total, 1)

    summary = {
        "model": args.model,
        "data": args.data,
        "total": total,
        "resolved": resolved,
        "resolve_rate": round(resolved / max(total, 1), 4),
        "errors": errors,
        "no_patch": no_patch,
        "avg_steps": round(avg_steps, 1),
        "avg_elapsed_s": round(avg_elapsed, 1),
        "timestamp": datetime.now().isoformat(),
        "args": vars(args),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    # Also save per-instance results table
    (output_dir / "results.jsonl").write_text(
        "\n".join(json.dumps(r) for r in sorted(results, key=lambda x: x["instance_id"]))
    )

    print("\n" + "=" * 60)
    print(f"  Model       : {args.model}")
    print(f"  Data        : {args.data}")
    print(f"  Instances   : {total}")
    print(f"  Resolved    : {resolved} / {total}  ({100*resolved/max(total,1):.1f}%)")
    print(f"  Errors      : {errors}")
    print(f"  No patch    : {no_patch}")
    print(f"  Avg steps   : {avg_steps:.1f}")
    print(f"  Avg elapsed : {avg_elapsed:.1f}s")
    print(f"  Output      : {output_dir}")
    print("=" * 60 + "\n")

    logger.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
