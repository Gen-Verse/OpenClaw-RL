#!/usr/bin/env bash
set -euo pipefail
python slime-coding-agent/runner/coding_agent_runner.py --config slime-coding-agent/configs/base.yaml --tasks slime-coding-agent/configs/tasks_demo.json
