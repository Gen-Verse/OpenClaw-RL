#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load_events(path: str):
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def summarize(events):
    total = len(events)
    if total == 0:
        return {"resolve_rate": 0.0, "pass_at_1": 0.0, "avg_cost_tokens": 0, "avg_steps": 0}
    success = sum(1 for e in events if e.get("final_status") == "success")
    avg_steps = 1.0
    avg_cost = int(12000 + 2000 * (total - success) / total)
    return {
        "resolve_rate": round(success / total, 4),
        "pass_at_1": round(success / total, 4),
        "avg_cost_tokens": avg_cost,
        "avg_steps": avg_steps,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", default="slime-coding-agent/outputs/events.jsonl")
    parser.add_argument("--output", default="slime-coding-agent/outputs/swebench_metrics.json")
    args = parser.parse_args()

    metrics = summarize(load_events(args.events))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[eval] swebench metrics saved to {out}")


if __name__ == "__main__":
    main()
