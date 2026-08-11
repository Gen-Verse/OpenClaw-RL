#!/usr/bin/env python3
"""Summarize Terminal Agent held-out results and compare experiment variants."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    return [
        json.loads(line)
        for line in file_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def summarize(records: list[dict[str, Any]], pass_at_k: int) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("task_id"):
            by_task[record["task_id"]].append(record)

    if not by_task:
        return {
            "tasks": 0,
            "attempts": 0,
            "pass_at_1": 0.0,
            f"pass_at_{pass_at_k}": 0.0,
            "resolve_rate": 0.0,
            "avg_steps": 0.0,
            "skill_retrieval_rate": 0.0,
        }

    def attempt_key(record: dict[str, Any]) -> tuple[int, int | str, int]:
        raw = str(record.get("attempt_id", ""))
        return (0, int(raw), int(record.get("rollout_id", 0))) if raw.isdigit() else (1, raw, int(record.get("rollout_id", 0)))

    ordered = {task_id: sorted(task_records, key=attempt_key) for task_id, task_records in by_task.items()}
    first_passes = sum(records[0].get("success", False) for records in ordered.values())
    pass_k = sum(
        any(record.get("success", False) for record in records[:pass_at_k])
        for records in ordered.values()
    )
    resolves = sum(any(record.get("success", False) for record in records) for records in ordered.values())
    flat = [record for records in ordered.values() for record in records]
    return {
        "tasks": len(ordered),
        "attempts": len(flat),
        "pass_at_1": round(first_passes / len(ordered), 4),
        f"pass_at_{pass_at_k}": round(pass_k / len(ordered), 4),
        "resolve_rate": round(resolves / len(ordered), 4),
        "avg_steps": round(sum(float(record.get("steps", 0)) for record in flat) / len(flat), 2),
        "skill_retrieval_rate": round(
            sum(bool(record.get("skill_retrieval")) for record in flat) / len(flat), 4
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pass-at-k", type=int, default=8)
    args = parser.parse_args()

    summary = summarize(load_jsonl(args.results), args.pass_at_k)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
