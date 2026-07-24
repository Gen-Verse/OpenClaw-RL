# AGENTS.md

Puppet workspace — autonomous operation.

## Rules

1. Execute internal actions without asking.
2. Ask before external actions.
3. Log everything to memory.
4. Report results after tasks.

## Memory

- Daily: `memory/YYYY-MM-DD.md`
- Long-term: `MEMORY.md`
- Search: `python -m autonomous.run memory search "query"`

## Heartbeat

Periodic checks run automatically. Log results. Act on findings.

## Cron

Scheduled jobs execute at their interval. Managed via `python -m autonomous.run cron`.

## Red Lines

- No data exfiltration
- No destructive commands without approval
- Backup before changes
