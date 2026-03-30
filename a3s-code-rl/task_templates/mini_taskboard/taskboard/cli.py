from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .storage import DEFAULT_DB, load_tasks, next_task_id, save_tasks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="taskboard")
    parser.add_argument("--db", default=DEFAULT_DB, help="Path to the JSON task database.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_parser = subparsers.add_parser("add", help="Create a task.")
    add_parser.add_argument("title", help="Task title.")
    add_parser.set_defaults(func=cmd_add)

    list_parser = subparsers.add_parser("list", help="List all tasks.")
    list_parser.set_defaults(func=cmd_list)

    done_parser = subparsers.add_parser("done", help="Mark a task as done.")
    done_parser.add_argument("task_id", type=int, help="Numeric task id.")
    done_parser.set_defaults(func=cmd_done)

    return parser


def cmd_add(args: argparse.Namespace) -> int:
    tasks = load_tasks(args.db)
    title = args.title.strip()
    if not title:
        print("task title must not be empty", file=sys.stderr)
        return 1
    task = {
        "id": next_task_id(tasks),
        "title": title,
        "status": "open",
    }
    tasks.append(task)
    save_tasks(tasks, args.db)
    print(f"added #{task['id']} {task['title']}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    tasks = load_tasks(args.db)
    if not tasks:
        print("no tasks")
        return 0
    for task in tasks:
        print(f"#{task['id']} [{task['status']}] {task['title']}")
    return 0


def cmd_done(args: argparse.Namespace) -> int:
    tasks = load_tasks(args.db)
    for task in tasks:
        if int(task["id"]) == args.task_id:
            task["status"] = "done"
            save_tasks(tasks, args.db)
            print(f"completed #{task['id']} {task['title']}")
            return 0
    print(f"task {args.task_id} not found", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.db = str(Path(args.db))
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
