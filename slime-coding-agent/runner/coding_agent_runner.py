#!/usr/bin/env python3
import argparse
import json
import shlex
import subprocess
import time
import uuid
from pathlib import Path
from typing import Dict, List

import yaml

from agent_core.models import CommandResult, RewardComponents, RolloutEvent
from agent_core.schema import validate_action_type, validate_required_fields, validate_reward_components


SAFE_COMMAND_PREFIXES = ("echo", "python", "pytest", "ls", "pwd", "cat")


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_command(command: str, timeout_sec: int = 30) -> CommandResult:
    parts = shlex.split(command)
    if not parts or parts[0] not in SAFE_COMMAND_PREFIXES:
        return CommandResult(command=command, exit_code=126, stdout="", stderr="command not allowed")
    cp = subprocess.run(parts, capture_output=True, text=True, timeout=timeout_sec)
    return CommandResult(command=command, exit_code=cp.returncode, stdout=cp.stdout[-4000:], stderr=cp.stderr[-4000:])


def calc_reward(command_results: List[Dict], test_passed: bool) -> RewardComponents:
    failed_cmds = sum(1 for x in command_results if x["exit_code"] != 0)
    passed = 1.0 if test_passed else 0.0
    quality = max(0.0, 0.8 - 0.2 * failed_cmds)
    safety = 0.9
    human = 0.6
    cost = min(1.0, 0.1 * len(command_results))
    return RewardComponents(passed=passed, quality=quality, safety=safety, human=human, cost=cost)


def build_event(task: Dict, cmd_results: List[CommandResult], reward: RewardComponents, final_status: str):
    event = RolloutEvent(
        event_id=str(uuid.uuid4()),
        timestamp=int(time.time()),
        repo_id=task["repo_id"],
        task_id=task["task_id"],
        benchmark_id=task.get("benchmark_id", "custom_task"),
        commit_base=task.get("commit_base", "HEAD"),
        action_type="run_tests",
        action_payload={"commands": task["commands"]},
        command_results=[x.to_dict() for x in cmd_results],
        test_results={"passed": final_status == "success", "failed_cases": 0 if final_status == "success" else 1},
        reward_components={
            "pass": reward.passed,
            "quality": reward.quality,
            "safety": reward.safety,
            "human": reward.human,
            "cost": reward.cost,
        },
        total_reward=reward.total,
        final_status=final_status,
    ).to_dict()
    return event


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="slime-coding-agent/configs/base.yaml")
    parser.add_argument("--tasks", default="slime-coding-agent/configs/tasks_demo.json")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    schema = load_yaml("slime-coding-agent/configs/rollout_event_schema.yaml")
    tasks = json.loads(Path(args.tasks).read_text(encoding="utf-8"))

    output_dir = Path(cfg["logging"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    event_path = output_dir / cfg["logging"]["event_log"]

    with open(event_path, "a", encoding="utf-8") as f:
        for task in tasks:
            cmd_results = [run_command(c).to_dict() for c in task["commands"]]
            success = all(r["exit_code"] == 0 for r in cmd_results)
            reward = calc_reward(cmd_results, success)
            event = build_event(task, [CommandResult(**r) for r in cmd_results], reward, "success" if success else "failed")
            validate_required_fields(event, schema["required_fields"])
            validate_action_type(event, schema["action_types"])
            validate_reward_components(event)
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    print(f"[runner] wrote {len(tasks)} event(s) to {event_path}")


if __name__ == "__main__":
    main()
