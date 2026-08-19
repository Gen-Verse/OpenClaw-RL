"""Persist task-level held-out terminal evaluation results as JSONL."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _status_value(status: Any) -> str:
    return getattr(status, "value", str(status))


def log_eval_rollout_data(rollout_id, args, data, extra_metrics) -> bool:
    output = os.getenv("TERMINAL_EVAL_RESULTS_PATH", "")
    if not output:
        return False

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for dataset_name, dataset in data.items():
        for sample in dataset.get("samples") or []:
            metadata = sample.metadata or {}
            turn_idx = metadata.get("turn_idx")
            num_turns = metadata.get("num_turns")
            if turn_idx is not None and num_turns is not None and turn_idx != num_turns - 1:
                continue

            reward = sample.reward if isinstance(sample.reward, dict) else {}
            records.append(
                {
                    "rollout_id": rollout_id,
                    "dataset": dataset_name,
                    "task_id": metadata.get("task_name", ""),
                    "attempt_id": str(sample.index),
                    "status": _status_value(sample.status),
                    "success": float(reward.get("accuracy", 0.0)) >= 1.0,
                    "reward": float(reward.get("score", 0.0)),
                    "steps": int(metadata.get("model_turn_count", 0)),
                    "skill_retrieval": metadata.get("retrieved_skills", []),
                }
            )

    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return False
