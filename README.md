# OpenClaw Puppet

Puppet master for OpenClaw. Pull the strings, run the agent, no human needed.

## What This Does

You feed it a task. It connects to OpenClaw Gateway via WebSocket and executes. Heartbeats, scheduled tasks, direct commands -- all autonomous. The agent does what you tell it. Nothing more.

## Quick Start

```bash
pip install -r autonomous/requirements.txt
python setup.py
# Edit ~/.openclaw/autonomous.json with your gateway token
openclaw gateway run
python -m autonomous.run run
```

## CLI

```bash
python -m autonomous.run run                    # start puppet controller
python -m autonomous.run status                 # check status
python -m autonomous.run send "what's the status?"  # direct command
python -m autonomous.run task add "check email" "check for urgent emails" --interval 300
python -m autonomous.run task list              # list tasks
python -m autonomous.run task remove task-0001  # remove task
```

## Configuration

`~/.openclaw/autonomous.json`:

```json
{
  "gateway": {
    "url": "ws://127.0.0.1:18789",
    "token": "your-gateway-token"
  },
  "agent": {
    "agent_id": "main",
    "model": "github-copilot/claude-opus-4.7"
  },
  "poll_interval": 60,
  "heartbeat_interval": 1800
}
```

## Workspace Overrides

`workspace-overrides/` contains modified workspace files. Setup copies them to `~/.openclaw/workspace/`.

- `SOUL.md` -- Puppet control philosophy
- `AGENTS.md` -- Autonomous behavior rules
- `IDENTITY.md` -- Agent identity
- `USER.md` -- User context
- `HEARTBEAT.md` -- Proactive check tasks

## Architecture

```
┌──────────────────┐     WebSocket      ┌──────────────┐
│  Puppet          │◄──────────────────►│   OpenClaw   │
│  Controller      │                    │   Gateway    │
│                  │     agent.turn     │              │
│  TaskManager     │───────────────────►│   Agent      │
│  HeartbeatLoop   │                    │   (Claude)   │
└──────────────────┘                    └──────────────┘
```

The puppet controller sits between you and the gateway. It manages tasks, sends commands, monitors health. The agent does the work. The controller pulls the strings.

## Workspace Overrides

Copied to `~/.openclaw/workspace/` during setup. These override the default agent personality to make it autonomous and action-first.
