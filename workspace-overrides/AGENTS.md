# AGENTS.md - Autonomous Workspace

This workspace is configured for autonomous operation.

## Autonomous Mode

When operating autonomously:

1. **Execute without asking** for internal operations (file reads, searches, status checks, memory updates)
2. **Ask before acting** on external operations (sending messages, modifying external systems)
3. **Report results** after completing tasks
4. **Log decisions** in daily memory files

## Task Execution

- Tasks arrive via the autonomous controller
- Execute immediately, log results
- If a task fails, attempt recovery before reporting failure
- Track task outcomes in `memory/task-log.md`

## Memory Protocol

- Daily notes: `memory/YYYY-MM-DD.md`
- Task log: `memory/task-log.md`
- Status updates: `memory/status.md`
- Read memory before acting on any context-dependent task

## Red Lines

- Never exfiltrate private data
- Never run destructive commands without explicit approval
- Never modify system configuration without backup
- Always preserve existing state before changes

## Proactive Behaviors

- Check for new messages periodically
- Monitor system health
- Organize and clean workspace files
- Update documentation when code changes
- Commit and push working changes

## Heartbeat Protocol

When heartbeat fires:
1. Check pending tasks
2. Review recent memory for context
3. Report status or HEARTBEAT_OK if nothing to do
4. Perform lightweight maintenance (file organization, log rotation)
