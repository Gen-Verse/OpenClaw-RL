# Mini Taskboard

`mini-taskboard` is a tiny Python CLI for tracking tasks in a local JSON file.

## Current Commands

- `add <title>` creates a new open task and rejects blank titles
- `list` prints all tasks
- `done <task_id>` marks a task as done

## Quick Start

```bash
python -m taskboard.cli add "ship release notes"
python -m taskboard.cli add "clean up stale feature flags"
python -m taskboard.cli list
python -m taskboard.cli done 1
python -m taskboard.cli list
```

## Development

```bash
pytest -q
```
