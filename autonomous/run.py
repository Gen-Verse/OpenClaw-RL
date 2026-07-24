#!/usr/bin/env python3
import argparse
import asyncio
import json

from .config import AutonomousConfig
from .controller import PuppetController, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenClaw Puppet Controller")
    parser.add_argument("--config", help="Config file path")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("run", help="Start puppet controller")
    sub.add_parser("status", help="Show controller status")
    sub.add_parser("init", help="Initialize workspace")

    msg_p = sub.add_parser("send", help="Send a message to the agent")
    msg_p.add_argument("message", help="Message to send")

    mem_p = sub.add_parser("memory", help="Memory operations")
    mem_sub = mem_p.add_subparsers(dest="mem_cmd")
    mem_sub.add_parser("today", help="Show today's memory")
    mem_sub.add_parser("recent", help="Show recent memory")
    mem_sub.add_parser("long-term", help="Show long-term memory")
    mem_sub.add_parser("consolidate", help="Consolidate daily to MEMORY.md")
    s_p = mem_sub.add_parser("search", help="Search memory")
    s_p.add_argument("query", help="Search query")

    task_p = sub.add_parser("task", help="Manage tasks")
    task_sub = task_p.add_subparsers(dest="task_cmd")
    add_p = task_sub.add_parser("add", help="Add a task")
    add_p.add_argument("name", help="Task name")
    add_p.add_argument("command", help="Command/prompt")
    add_p.add_argument("--interval", type=int, default=60, help="Interval in seconds")
    task_sub.add_parser("list", help="List tasks")
    rm_p = task_sub.add_parser("remove", help="Remove a task")
    rm_p.add_argument("task_id", help="Task ID")

    cron_p = sub.add_parser("cron", help="Manage cron jobs")
    cron_sub = cron_p.add_subparsers(dest="cron_cmd")
    cadd = cron_sub.add_parser("add", help="Add a cron job")
    cadd.add_argument("name", help="Job name")
    cadd.add_argument("command", help="Command/prompt")
    cadd.add_argument("--schedule", required=True, help="Schedule (every 30s, every 2h, daily 09:00)")
    cron_sub.add_parser("list", help="List cron jobs")
    crm = cron_sub.add_parser("remove", help="Remove a cron job")
    crm.add_argument("job_id", help="Job ID")
    cron_sub.add_parser("enable", help="Enable a cron job").add_argument("job_id", help="Job ID")
    cron_sub.add_parser("disable", help="Disable a cron job").add_argument("job_id", help="Job ID")

    hb_p = sub.add_parser("heartbeat", help="Heartbeat operations")
    hb_sub = hb_p.add_subparsers(dest="hb_cmd")
    hb_sub.add_parser("status", help="Show heartbeat status")

    ws_p = sub.add_parser("workspace", help="Workspace operations")
    ws_sub = ws_p.add_subparsers(dest="ws_cmd")
    ws_sub.add_parser("list", help="List workspace files")
    ws_sub.add_parser("identity", help="Show identity")
    ws_sub.add_parser("backup", help="Backup workspace")

    args = parser.parse_args()
    config = AutonomousConfig.load(args.config)
    setup_logging(config.log_file, args.log_level)

    if args.command == "run":
        controller = PuppetController(config)
        asyncio.run(controller.run_forever())

    elif args.command == "status":
        async def _status():
            c = PuppetController(config)
            await c.client.connect()
            status = await c.get_status()
            print(json.dumps(status, indent=2))
            await c.client.disconnect()
        asyncio.run(_status())

    elif args.command == "init":
        from .workspace import Workspace
        ws = Workspace(config.workspace_dir)
        created = ws.init()
        print(f"Workspace initialized at {config.workspace_dir}")
        if created:
            print(f"Created: {', '.join(created)}")
        else:
            print("All files already exist.")

    elif args.command == "send":
        async def _send():
            c = PuppetController(config)
            await c.client.connect()
            reply = await c.send_command(args.message)
            print(reply)
            await c.client.disconnect()
        asyncio.run(_send())

    elif args.command == "memory":
        from .memory import Memory
        mem = Memory(config.workspace_dir)
        if args.mem_cmd == "today":
            print(mem.get_today() or "No memory for today.")
        elif args.mem_cmd == "recent":
            print(mem.get_recent())
        elif args.mem_cmd == "long-term":
            print(mem.get_long_term() or "No long-term memory.")
        elif args.mem_cmd == "consolidate":
            print(mem.consolidate())
        elif args.mem_cmd == "search":
            results = mem.search(args.query)
            for r in results:
                print(f"{r['file']}:{r['line']}: {r['text']}")

    elif args.command == "task":
        from .task_manager import TaskManager
        from .gateway_client import GatewayClient
        tm = TaskManager(GatewayClient(config.gateway), config.task_dir)
        if args.task_cmd == "add":
            task = tm.add_task(args.name, args.command, interval=args.interval)
            print(f"Added: {task.id} - {task.name}")
        elif args.task_cmd == "list":
            for t in tm.list_tasks():
                status = "ON" if t.enabled else "OFF"
                print(f"[{status}] {t.id}: {t.name} (runs: {t.run_count})")
        elif args.task_cmd == "remove":
            if tm.remove_task(args.task_id):
                print(f"Removed: {args.task_id}")
            else:
                print(f"Not found: {args.task_id}")

    elif args.command == "cron":
        from .cron import CronEngine
        engine = CronEngine(config.cron_dir)
        if args.cron_cmd == "add":
            job = engine.add(args.name, args.command, args.schedule)
            print(f"Added: {job.id} - {job.name} ({job.schedule})")
        elif args.cron_cmd == "list":
            for j in engine.list_jobs():
                status = "ON" if j.enabled else "OFF"
                print(f"[{status}] {j.id}: {j.name} [{j.schedule}] (runs: {j.run_count})")
        elif args.cron_cmd == "remove":
            if engine.remove(args.job_id):
                print(f"Removed: {args.job_id}")
            else:
                print(f"Not found: {args.job_id}")
        elif args.cron_cmd == "enable":
            engine.enable(args.job_id)
            print(f"Enabled: {args.job_id}")
        elif args.cron_cmd == "disable":
            engine.disable(args.job_id)
            print(f"Disabled: {args.job_id}")

    elif args.command == "heartbeat":
        from .heartbeat import Heartbeat
        hb = Heartbeat(config.workspace_dir)
        if args.hb_cmd == "status":
            status = hb.get_status()
            for name, info in status["checks"].items():
                due = "DUE" if info["due"] else "ok"
                print(f"[{due}] {name}: interval={info['interval']}s")

    elif args.command == "workspace":
        from .workspace import Workspace
        ws = Workspace(config.workspace_dir)
        if args.ws_cmd == "list":
            for f in ws.list_files():
                print(f"{f['name']} ({f['size']} bytes)")
        elif args.ws_cmd == "identity":
            print(json.dumps(ws.get_identity(), indent=2))
        elif args.ws_cmd == "backup":
            path = ws.backup()
            print(f"Backup created: {path}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
