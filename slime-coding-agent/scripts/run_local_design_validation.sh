#!/usr/bin/env bash
set -euo pipefail

echo "[Check] rollout event schema exists"
test -f "$(dirname "$0")/../configs/rollout_event_schema.yaml"

echo "[Check] docs exist"
test -f "$(dirname "$0")/../docs/research_report_zh.md"
test -f "$(dirname "$0")/../docs/implementation_plan_zh.md"

echo "Design validation passed."
