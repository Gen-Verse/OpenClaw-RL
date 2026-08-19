"""Skill retrieval injection and trajectory feedback for SWE rollouts.

Optional, env-gated bridge between swe-rl and the self-evolution skill store:

* ``SWE_SKILL_RETRIEVAL=1`` retrieves recovery skills relevant to the issue
  and appends them to the instance message before the agent loop starts.
* ``SWE_TRAJECTORY_LOG`` / ``EVOLUTION_SERVER_URL`` publish each rollout
  outcome so failed instances can later be distilled into skills.

Both are off by default; held-out evaluation runs stay uncontaminated as long
as the publish variables are unset there.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_TERMINAL_RL_DIR = Path(__file__).resolve().parent.parent / "terminal-rl"
if _TERMINAL_RL_DIR.is_dir() and str(_TERMINAL_RL_DIR) not in sys.path:
    sys.path.append(str(_TERMINAL_RL_DIR))


def retrieve_skills_for_issue(problem_statement: str) -> list[dict[str, Any]]:
    if os.getenv("SWE_SKILL_RETRIEVAL", "0") != "1":
        return []
    skills_dir = os.getenv("SWE_SKILLS_DIR", "")
    if not skills_dir:
        return []
    try:
        from skill_context import retrieve_skills
    except ImportError:
        logger.warning("[SWE-Skill] terminal-rl skill_context unavailable; skipping retrieval")
        return []
    top_k = int(os.getenv("SWE_SKILL_TOP_K", "3"))
    return retrieve_skills(problem_statement, skills_dir, top_k=top_k)


def augment_instance_message(
    instance_message: str,
    problem_statement: str,
) -> tuple[str, list[dict[str, Any]]]:
    skills = retrieve_skills_for_issue(problem_statement)
    if not skills:
        return instance_message, []

    snippets = "\n\n".join(
        f"### Recovery Skill {index}\n{skill['content']}"
        for index, skill in enumerate(skills, start=1)
    )
    block = (
        "\n\n## Retrieved Recovery Skills\n"
        "Use these only when relevant. Verify every command and result in the current environment.\n"
        f"{snippets}"
    )
    return instance_message + block, [{"path": s["path"], "score": s["score"]} for s in skills]


def build_swe_event(
    *,
    instance: dict[str, Any],
    data_source: str,
    resolved: bool,
    run_info: dict[str, Any],
) -> dict[str, Any]:
    eval_result = run_info.get("eval_result") or {}
    policy = run_info.get("policy") or {}
    failure_bits = [
        str(run_info.get("error") or ""),
        str(run_info.get("exit_status") or ""),
        ",".join(policy.get("reasons", [])),
        str(eval_result.get("grading_error") or ""),
    ]
    failure_text = "\n".join(bit for bit in failure_bits if bit)
    return {
        "repo_id": instance.get("repo", "swe"),
        "task_id": instance.get("instance_id", "unknown"),
        "benchmark_id": data_source,
        "commit_base": instance.get("base_commit", ""),
        "action_type": "run_tests",
        "action_payload": {
            "patch_source": run_info.get("patch_source"),
            "n_steps": run_info.get("n_steps", 0),
            "retrieved_skills": run_info.get("retrieved_skills", []),
        },
        "command_results": [
            {
                "command": "swe-patch-eval",
                "exit_code": 0 if resolved else 1,
                "stdout": "",
                "stderr": "" if resolved else failure_text,
            }
        ],
        "test_results": {"passed": resolved, "resolved_by": eval_result.get("resolved_by", "")},
        "final_status": "success" if resolved else "failed",
    }


def _append_jsonl(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def _post_event(url: str, event: dict[str, Any]) -> None:
    request = urllib.request.Request(
        f"{url.rstrip('/')}/v1/trajectories",
        data=json.dumps(event, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=5):
            return
    except (urllib.error.HTTPError, urllib.error.URLError, OSError):
        return


async def publish_swe_trajectory(
    *,
    instance: dict[str, Any],
    data_source: str,
    resolved: bool,
    run_info: dict[str, Any],
) -> None:
    log_path = os.getenv("SWE_TRAJECTORY_LOG", "")
    server_url = os.getenv("EVOLUTION_SERVER_URL", "")
    if not log_path and not server_url:
        return

    event = build_swe_event(
        instance=instance,
        data_source=data_source,
        resolved=resolved,
        run_info=run_info,
    )
    if log_path:
        await asyncio.to_thread(_append_jsonl, Path(log_path), event)
    if server_url:
        await asyncio.to_thread(_post_event, server_url, event)
