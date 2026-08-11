#!/usr/bin/env bash
set -euo pipefail
python -m uvicorn server.agent_server:app --app-dir slime-coding-agent --host 0.0.0.0 --port "${AGENT_SERVER_PORT:-8010}"
