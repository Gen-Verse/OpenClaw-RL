#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import yaml

try:
    from .swebench_runner import load_events, summarize
except ImportError:
    from swebench_runner import load_events, summarize


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def evaluate_experiments(experiments, events_dir: Path):
    results = []
    for exp in experiments:
        name = Path(exp["name"]).name
        event_path = events_dir / f"{name}.jsonl"
        events = load_events(str(event_path))
        results.append(
            {
                "name": name,
                "status": "completed" if events else "pending",
                "events": str(event_path),
                "metrics": summarize(events) if events else None,
                "config": exp,
            }
        )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="slime-coding-agent/configs/ablation.yaml")
    parser.add_argument("--events-dir", default="slime-coding-agent/outputs/ablations")
    parser.add_argument("--output", default="slime-coding-agent/outputs/ablation_results.json")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    results = evaluate_experiments(cfg["experiments"], Path(args.events_dir))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ablation] results saved to {out}")


if __name__ == "__main__":
    main()
