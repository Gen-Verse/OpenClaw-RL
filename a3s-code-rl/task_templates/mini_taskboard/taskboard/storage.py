from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_DB = ".taskboard.json"


def load_tasks(db_path: str | Path = DEFAULT_DB) -> list[dict[str, Any]]:
    path = Path(db_path)
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"expected a list of tasks in {path}")
    return [dict(item) for item in raw]


def save_tasks(tasks: list[dict[str, Any]], db_path: str | Path = DEFAULT_DB) -> None:
    path = Path(db_path)
    path.write_text(json.dumps(tasks, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def next_task_id(tasks: list[dict[str, Any]]) -> int:
    return max((int(task.get("id", 0)) for task in tasks), default=0) + 1
