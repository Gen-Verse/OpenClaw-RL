#!/bin/bash
export OPENCLAW_AUTONOMOUS_CONFIG="$(dirname "$0")/autonomous.json"
python3 -m autonomous.run "$@"
