#!/usr/bin/env python3
import argparse
import asyncio
import json
import sys

from .config import AutonomousConfig
from .controller import AutonomousController, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenClaw Puppet Controller")
    parser.add_argument("--config", help="Config file path")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("run", help="Start autonomous controller")
    sub.add_parser("status", help="Show controller status")

    msg_p = sub.add_parser("send", help="Send a message to the agent")
    msg_p.add_argument("message", help="Message to send")

    task_p = sub.add_parser("task", help="Manage tasks")
    task_sub = task_p.add_subparsers(dest="task_cmd")
    add_p = task_sub.add_parser("add", help="Add a task")
    add_p.add_argument("name", help="Task name")
    add_p.add_argument("command", help="Command/prompt")
    add_p.add_argument("--interval", type=int, default=60, help="Interval in seconds")
    task_sub.add_parser("list", help="List tasks")
    rm_p = task_sub.add_parser("remove", help="Remove a task")
    rm_p.add_argument("task_id", help="Task ID")

    args = parser.parse_args()
    config = AutonomousConfig.load(args.config)
    setup_logging(config.log_file, args.log_level)

    if args.command == "run":
        controller = AutonomousController(config)
        asyncio.run(controller.run_forever())
    elif args.command == "status":
        async def _status():
            c = AutonomousController(config)
            await c.client.connect()
            status = await c.get_status()
            print(json.dumps(status, indent=2))
            await c.client.disconnect()
        asyncio.run(_status())
    elif args.command == "send":
        async def _send():
            c = AutonomousController(config)
            await c.client.connect()
            reply = await c.send_command(args.message)
            print(reply)
            await c.client.disconnect()
        asyncio.run(_send())
    elif args.command == "task":
        if args.task_cmd == "add":
            async def _add():
                c = AutonomousController(config)
                task = c.task_manager.add_task(args.name, args.command, interval=args.interval)
                print(f"Added: {task.id} - {task.name}")
            asyncio.run(_add())
        elif args.task_cmd == "list":
            async def _list():
                c = AutonomousController(config)
                for t in c.task_manager.list_tasks():
                    status = "ON" if t.enabled else "OFF"
                    print(f"[{status}] {t.id}: {t.name} (runs: {t.run_count})")
            asyncio.run(_list())
        elif args.task_cmd == "remove":
            async def _remove():
                c = AutonomousController(config)
                if c.task_manager.remove_task(args.task_id):
                    print(f"Removed: {args.task_id}")
                else:
                    print(f"Not found: {args.task_id}")
            asyncio.run(_remove())
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
