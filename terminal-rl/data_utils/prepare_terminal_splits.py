#!/usr/bin/env python3
"""Build deterministic train and held-out JSONL files from terminal task folders."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

DATA_UTILS_DIR = Path(__file__).resolve().parent
if str(DATA_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_UTILS_DIR))

from load_tasks import load_terminal_bench_tasks


def stable_split(task_names: list[str], holdout_fraction: float, seed: str) -> tuple[list[str], list[str]]:
    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must be between 0 and 1")
    ranked = sorted(
        task_names,
        key=lambda name: hashlib.sha256(f"{seed}:{name}".encode("utf-8")).hexdigest(),
    )
    heldout_count = max(1, min(len(ranked) - 1, round(len(ranked) * holdout_fraction)))
    return ranked[heldout_count:], ranked[:heldout_count]


def write_dataset(tasks: list[Any], dataset_dir: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            metadata = {
                "task_name": task.task_name,
                "task_path": str(task.task_path.relative_to(dataset_dir)),
                "instruction": task.instruction,
                "data_source": "terminal_bench",
            }
            handle.write(
                json.dumps(
                    {
                        "task": [{"role": "user", "content": task.instruction}],
                        "metadata": metadata,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-dir", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--seed", default="terminal-evolution-v1")
    args = parser.parse_args()

    tasks_dir = Path(args.tasks_dir).resolve()
    dataset_dir = Path(args.dataset_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    task_names = sorted(path.name for path in tasks_dir.iterdir() if path.is_dir())
    train_names, heldout_names = stable_split(task_names, args.holdout_fraction, args.seed)
    train_tasks = load_terminal_bench_tasks(tasks_dir, train_names)
    heldout_tasks = load_terminal_bench_tasks(tasks_dir, heldout_names)
    write_dataset(train_tasks, dataset_dir, output_dir / "train.jsonl")
    write_dataset(heldout_tasks, dataset_dir, output_dir / "heldout.jsonl")
    (output_dir / "split_manifest.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "holdout_fraction": args.holdout_fraction,
                "train_tasks": train_names,
                "heldout_tasks": heldout_names,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
