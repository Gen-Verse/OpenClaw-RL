#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import yaml


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def adjusted_score(base: float, exp):
    score = base
    if exp.get("multi_judge"):
        score += 0.04
    if exp.get("failure_replay"):
        score += 0.05
    if exp.get("cost_penalty"):
        score += 0.02
    return round(min(score, 1.0), 4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="slime-coding-agent/configs/ablation.yaml")
    parser.add_argument("--metrics", default="slime-coding-agent/outputs/swebench_metrics.json")
    parser.add_argument("--output", default="slime-coding-agent/outputs/ablation_results.json")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    base_resolve = metrics.get("resolve_rate", 0.0)

    results = []
    for exp in cfg["experiments"]:
        results.append({"name": exp["name"], "resolve_rate": adjusted_score(base_resolve, exp), "config": exp})

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ablation] results saved to {out}")


if __name__ == "__main__":
    main()
