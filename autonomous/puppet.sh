#!/bin/bash
export OPENCLAW_PUPPET_CONFIG="$(dirname "$0")/puppet.json"
python3 -m autonomous.run "$@"
