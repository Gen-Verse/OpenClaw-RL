#!/usr/bin/env bash
set -euo pipefail
python - <<'PY'
import yaml
p = 'slime-coding-agent/deploy/compose_slime_sglang.yaml'
with open(p, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
print('[deploy] services:', ', '.join(cfg.get('services', {}).keys()))
PY
