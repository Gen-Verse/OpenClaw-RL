#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load_events(path: str):
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def event_steps(event):
    command_results = event.get("command_results")
    if isinstance(command_results, list):
        return len(command_results)
    commands = event.get("action_payload", {}).get("commands")
    return len(commands) if isinstance(commands, list) else 0


def event_tokens(event):
    total_tokens = event.get("usage", {}).get("total_tokens")
    if isinstance(total_tokens, (int, float)) and not isinstance(total_tokens, bool) and total_tokens >= 0:
        return total_tokens
    return None


def summarize(events):
    grouped = {}
    for event in events:
        key = (event.get("repo_id", ""), event.get("task_id", event.get("event_id", "")))
        grouped.setdefault(key, []).append(event)

    total = len(grouped)
    if total == 0:
        return {
            "instances": 0,
            "attempts": 0,
            "resolve_rate": 0.0,
            "pass_at_1": 0.0,
            "avg_cost_tokens": None,
            "avg_steps": 0.0,
        }

    resolved = sum(
        any(event.get("final_status") == "success" for event in attempts)
        for attempts in grouped.values()
    )
    pass_at_1 = sum(
        attempts[0].get("final_status") == "success"
        for attempts in grouped.values()
    )
    known_tokens = [tokens for event in events if (tokens := event_tokens(event)) is not None]
    avg_steps = sum(event_steps(event) for event in events) / len(events)
    return {
        "instances": total,
        "attempts": len(events),
        "resolve_rate": round(resolved / total, 4),
        "pass_at_1": round(pass_at_1 / total, 4),
        "avg_cost_tokens": round(sum(known_tokens) / len(known_tokens), 2) if known_tokens else None,
        "avg_steps": round(avg_steps, 2),
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
