# OpenClaw Autonomous Controller

Autonomous control system for OpenClaw - makes it operate as a self-directed agent.

## What This Does

- Connects to the OpenClaw Gateway via WebSocket
- Sends messages and receives responses programmatically
- Manages scheduled tasks with persistent state
- Runs heartbeat loops for proactive behavior
- Provides CLI for direct interaction

## Quick Start

```bash
# Install dependencies
pip install -r autonomous/requirements.txt

# Run setup (copies workspace overrides, creates config)
python setup.py

# Edit config with your gateway token
# ~/.openclaw/autonomous.json

# Start the gateway
openclaw gateway run

# Start autonomous controller
python -m autonomous.run run
```

## CLI Commands

```bash
# Start autonomous mode
python -m autonomous.run run

# Check status
python -m autonomous.run status

# Send a direct message
python -m autonomous.run send "what's the status?"

# Add a recurring task
python -m autonomous.run task add "check email" "check for urgent emails" --interval 300

# List tasks
python -m autonomous.run task list

# Remove a task
python -m autonomous.run task remove task-0001
```

## Configuration

Edit `~/.openclaw/autonomous.json`:

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

The `workspace-overrides/` directory contains modified workspace files that make OpenClaw more autonomous:

- `SOUL.md` - Direct, action-oriented personality
- `AGENTS.md` - Autonomous behavior rules
- `IDENTITY.md` - Agent identity
- `USER.md` - User context
- `HEARTBEAT.md` - Proactive check tasks

These are copied to `~/.openclaw/workspace/` during setup.

## Architecture

```
┌─────────────────┐     WebSocket      ┌──────────────┐
│  Autonomous     │◄──────────────────►│   OpenClaw   │
│  Controller     │                    │   Gateway    │
│                 │     agent.turn     │              │
│  TaskManager    │───────────────────►│   Agent      │
│  HeartbeatLoop  │                    │   (Claude)   │
└─────────────────┘                    └──────────────┘
```
