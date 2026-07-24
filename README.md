# OpenClaw Puppet

Puppet master for OpenClaw. Controls it like a tool.

## Features

- **Memory** — daily notes, long-term storage, search, consolidation
- **Heartbeat** — proactive monitoring, periodic self-checks
- **Cron** — scheduled tasks with flexible intervals
- **Skills** — modular tool system, extensible
- **Workspace** — persistent identity, user context, soul

## Quick Start

```bash
pip install -r autonomous/requirements.txt
python setup.py
# edit ~/.openclaw/puppet.json
openclaw gateway run
python -m autonomous.run run
```

## CLI

```bash
# core
python -m autonomous.run run
python -m autonomous.run status
python -m autonomous.run init
python -m autonomous.run send "message"

# memory
python -m autonomous.run memory today
python -m autonomous.run memory recent
python -m autonomous.run memory long-term
python -m autonomous.run memory consolidate
python -m autonomous.run memory search "query"

# tasks
python -m autonomous.run task add "name" "command" --interval 60
python -m autonomous.run task list
python -m autonomous.run task remove task-0001

# cron
python -m autonomous.run cron add "name" "command" --schedule "every 30m"
python -m autonomous.run cron add "backup" "cmd" --schedule "daily 03:00"
python -m autonomous.run cron list
python -m autonomous.run cron remove cron-0001
python -m autonomous.run cron enable cron-0001
python -m autonomous.run cron disable cron-0001

# heartbeat
python -m autonomous.run heartbeat status

# workspace
python -m autonomous.run workspace list
python -m autonomous.run workspace identity
python -m autonomous.run workspace backup
```

## Architecture

```
Puppet
├── Gateway ──── WebSocket ──── OpenClaw
├── Memory ───── daily + long-term + search
├── Heartbeat ── periodic checks
├── Cron ─────── scheduled jobs
├── Skills ───── modular tools
├── Workspace ── SOUL/IDENTITY/USER
└── Tasks ────── persistent queue
```

## Config

`~/.openclaw/puppet.json`:

```json
{
  "gateway": {"url": "ws://127.0.0.1:18789", "token": "your-token"},
  "poll_interval": 60,
  "heartbeat_interval": 1800
}
```

## License

Apache 2.0
