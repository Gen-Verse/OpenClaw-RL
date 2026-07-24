# AGENTS.md - Puppet Workspace

This workspace is configured for autonomous puppet operation.

## Autonomous Mode

When operating as the puppet:

1. **Execute without asking** for internal operations (file reads, searches, status checks, memory updates)
2. **Ask before acting** on external operations (sending messages, modifying external systems)
3. **Report results** after completing tasks
4. **Log decisions** in daily memory files

## Memory Protocol

- Daily notes: `memory/YYYY-MM-DD.md` (auto-created)
- Long-term: `MEMORY.md` (consolidated from daily notes)
- Task log: `memory/task-log.md`
- Search: `openclaw puppet memory search <query>`

## Task Execution

- Tasks arrive via the puppet controller
- Execute immediately, log results
- If a task fails, attempt recovery before reporting failure
- Track task outcomes in memory

## Heartbeat Protocol

When heartbeat fires:
1. Check pending tasks
2. Review recent memory for context
3. Report status or HEARTBEAT_OK
4. Perform lightweight maintenance

## Cron Protocol

Scheduled jobs run automatically. They:
- Execute at the specified interval/schedule
- Log results to memory
- Can be managed via `openclaw puppet cron`

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
- Consolidate memory periodically
