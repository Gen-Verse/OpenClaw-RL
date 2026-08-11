#!/usr/bin/env python3
"""Export the Terminal rollout accuracy curve from local JSONL metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    records = [
        json.loads(line)
        for line in Path(args.metrics).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    curve = [
        {
            "rl_step": record.get("rollout/step"),
            "accuracy": record.get("terminal/accuracy"),
            "reward_mean": record.get("terminal/reward_mean"),
            "failed_ratio": record.get("terminal/failed_ratio"),
        }
        for record in records
        if record.get("terminal/accuracy") is not None
    ]
    summary = {
        "points": curve,
        "latest": curve[-1] if curve else None,
        "best_accuracy": max((point["accuracy"] for point in curve), default=None),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
