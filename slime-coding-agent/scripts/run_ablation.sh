#!/usr/bin/env bash
set -euo pipefail
python slime-coding-agent/eval/ablation_runner.py --config slime-coding-agent/configs/ablation.yaml --events-dir slime-coding-agent/outputs/ablations
