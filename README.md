# OpenClaw Puppet

Puppet master for OpenClaw. Controls it like a tool, not a chatbot.

## What It Does

- **Memory** — Daily notes, long-term memory, search, consolidation
- **Heartbeat** — Proactive monitoring, periodic self-checks
- **Cron** — Scheduled tasks with flexible intervals
- **Skills** — Modular tool system, extensible
- **Workspace** — Persistent identity, user context, soul

## Quick Start

```bash
pip install -r autonomous/requirements.txt
python setup.py
# Edit ~/.openclaw/puppet.json with your gateway token
openclaw gateway run
python -m autonomous.run run
```

## CLI

```bash
# Core
python -m autonomous.run run          # Start controller
python -m autonomous.run status       # Show status
python -m autonomous.run init         # Init workspace
python -m autonomous.run send "msg"   # Send message

# Memory
python -m autonomous.run memory today
python -m autonomous.run memory recent
python -m autonomous.run memory long-term
python -m autonomous.run memory consolidate
python -m autonomous.run memory search "query"

# Tasks
python -m autonomous.run task add "name" "command" --interval 60
python -m autonomous.run task list
python -m autonomous.run task remove task-0001

# Cron
python -m autonomous.run cron add "name" "command" --schedule "every 30m"
python -m autonomous.run cron add "backup" "backup files" --schedule "daily 03:00"
python -m autonomous.run cron list
python -m autonomous.run cron remove cron-0001
python -m autonomous.run cron enable cron-0001
python -m autonomous.run cron disable cron-0001

# Heartbeat
python -m autonomous.run heartbeat status

# Workspace
python -m autonomous.run workspace list
python -m autonomous.run workspace identity
python -m autonomous.run workspace backup
```

## Architecture

```
┌──────────────────────────────────────────────┐
│              PuppetController                │
│                                              │
│  ┌─────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Memory  │  │ Heartbeat│  │   Cron   │   │
│  │ System  │  │ Monitor  │  │  Engine  │   │
│  └────┬────┘  └────┬─────┘  └────┬─────┘   │
│       │            │             │           │
│  ┌────┴────────────┴─────────────┴────┐     │
│  │         GatewayClient              │     │
│  │      (WebSocket → OpenClaw)        │     │
│  └────────────────────────────────────┘     │
│                                              │
│  ┌──────────┐  ┌──────────┐                 │
│  │ Workspace│  │  Skills  │                 │
│  │ Manager  │  │ Registry │                 │
│  └──────────┘  └──────────┘                 │
└──────────────────────────────────────────────┘
```

## Config

Edit `~/.openclaw/puppet.json`:

```json
{
  "gateway": {
    "url": "ws://127.0.0.1:18789",
    "token": "your-token"
  },
  "poll_interval": 60,
  "heartbeat_interval": 1800
}
```

## Features Inherited from OpenClaw

| Feature | Implementation |
|---------|---------------|
| Memory | `memory.py` — daily notes, MEMORY.md, search, consolidation |
| Heartbeat | `heartbeat.py` — configurable periodic checks |
| Cron | `cron.py` — flexible scheduling (every Ns, daily HH:MM) |
| Skills | `skills.py` — modular tool registration and execution |
| Workspace | `workspace.py` — SOUL.md, IDENTITY.md, USER.md management |
| Sessions | Via Gateway WebSocket client |
| Tasks | `task_manager.py` — persistent task queue |
