"""Publish terminal rollout outcomes to the self-evolution trajectory store."""

from __future__ import annotations

import asyncio
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


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
    except (urllib.error.HTTPError, urllib.error.URLError):
        return


async def publish_terminal_trajectory(
    *,
    task_meta: dict[str, Any],
    reward: float,
    outcome: Any,
    rollout_error: str | None,
    eval_error: str | None,
    retrieved_skills: list[dict[str, Any]],
) -> None:
    if not os.getenv("TERMINAL_TRAJECTORY_LOG") and not os.getenv("EVOLUTION_SERVER_URL"):
        return

    success = reward >= 1.0 and eval_error is None
    diagnostic = eval_error or rollout_error or ""
    event = {
        "repo_id": "terminal-rl",
        "task_id": task_meta.get("task_name", "unknown-task"),
        "benchmark_id": task_meta.get("data_source", "terminal_bench"),
        "action_type": "run_tests",
        "action_payload": {
            "task_instruction": task_meta.get("instruction", ""),
            "retrieved_skills": retrieved_skills,
        },
        "command_results": [
            {
                "command": "terminal-task-evaluation",
                "exit_code": 0 if success else 1,
                "stdout": "",
                "stderr": diagnostic,
            }
        ],
        "test_results": {"passed": success, "score": reward},
        "final_status": "success" if success else "failed",
        "metadata": {
            "model_turn_count": getattr(outcome, "model_turn_count", 0) if outcome is not None else 0,
            "parse_error_count": getattr(outcome, "parse_error_count", 0) if outcome is not None else 0,
        },
    }

    local_path = os.getenv("TERMINAL_TRAJECTORY_LOG", "")
    if local_path:
        await asyncio.to_thread(_append_jsonl, Path(local_path), event)

    server_url = os.getenv("EVOLUTION_SERVER_URL", "")
    if server_url:
        await asyncio.to_thread(_post_event, server_url, event)
