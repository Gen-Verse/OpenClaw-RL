#!/usr/bin/env python3
"""Deterministically split WildClawBench text tasks into train / eval sets.

Writes configs/split.json:
  {"train": [task md paths...], "eval": [...]}

Split is stratified per category and stable for a given seed, so all variants
evaluate on the same held-out set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def split_tasks(tasks: list[Path], train_ratio: float, seed: str) -> tuple[list[str], list[str]]:
    ranked = sorted(
        tasks,
        key=lambda p: hashlib.sha256(f"{seed}:{p}".encode("utf-8")).hexdigest(),
    )
    n_train = max(1, round(len(ranked) * train_ratio)) if len(ranked) > 1 else len(ranked)
    train = [str(p) for p in ranked[:n_train]]
    eval_ = [str(p) for p in ranked[n_train:]]
    return train, eval_


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wcb-root", required=True, help="WildClawBench clone root")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=[
            "01_Productivity_Flow",
            "03_Social_Interaction",
            "04_Search_Retrieval",
            "06_Safety_Alignment",
        ],
    )
    parser.add_argument("--train-ratio", type=float, default=0.65)
    parser.add_argument("--seed", default="wcb-ablation-v1")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    wcb_root = Path(args.wcb_root)
    train_all: list[str] = []
    eval_all: list[str] = []
    per_category = {}

    for cat in args.categories:
        task_dir = wcb_root / "tasks" / cat
        tasks = sorted(task_dir.glob("*.md")) if task_dir.is_dir() else []
        # exclude the annotated template
        tasks = [p for p in tasks if "template" not in p.name]
        train, eval_ = split_tasks(tasks, args.train_ratio, args.seed + cat)
        per_category[cat] = {"train": len(train), "eval": len(eval_)}
        train_all.extend(train)
        eval_all.extend(eval_)

    payload = {
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "per_category": per_category,
        "train": train_all,
        "eval": eval_all,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(per_category, ensure_ascii=False))
    print(f"train={len(train_all)} eval={len(eval_all)} -> {out}")


if __name__ == "__main__":
    main()
