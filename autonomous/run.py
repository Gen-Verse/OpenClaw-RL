#!/usr/bin/env python3
"""OpenClaw Puppet CLI."""
from __future__ import annotations

import argparse
import asyncio
import json

from .config import Config
from .controller import Puppet, setup_logging


def main() -> None:
    p = argparse.ArgumentParser(prog="openclaw-puppet", description="OpenClaw Puppet Controller")
    p.add_argument("--config", help="config file path")
    p.add_argument("--log-level", default="INFO")
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("run", help="start controller")
    sub.add_parser("status", help="show status")
    sub.add_parser("init", help="initialize workspace")

    s = sub.add_parser("send", help="send message to agent")
    s.add_argument("message")

    m = sub.add_parser("memory", help="memory operations")
    ms = m.add_subparsers(dest="mem_cmd")
    ms.add_parser("today")
    ms.add_parser("recent")
    ms.add_parser("long-term")
    ms.add_parser("consolidate")
    sq = ms.add_parser("search")
    sq.add_argument("query")

    t = sub.add_parser("task", help="manage tasks")
    ts = t.add_subparsers(dest="task_cmd")
    ta = ts.add_parser("add")
    ta.add_argument("name")
    ta.add_argument("command")
    ta.add_argument("--interval", type=int, default=60)
    ts.add_parser("list")
    tr = ts.add_parser("remove")
    tr.add_argument("task_id")

    c = sub.add_parser("cron", help="manage cron jobs")
    cs = c.add_subparsers(dest="cron_cmd")
    ca = cs.add_parser("add")
    ca.add_argument("name")
    ca.add_argument("command")
    ca.add_argument("--schedule", required=True)
    cs.add_parser("list")
    cr = cs.add_parser("remove")
    cr.add_argument("job_id")
    ce = cs.add_parser("enable")
    ce.add_argument("job_id")
    cd = cs.add_parser("disable")
    cd.add_argument("job_id")

    hb = sub.add_parser("heartbeat")
    hs = hb.add_subparsers(dest="hb_cmd")
    hs.add_parser("status")

    w = sub.add_parser("workspace")
    ws = w.add_subparsers(dest="ws_cmd")
    ws.add_parser("list")
    ws.add_parser("identity")
    ws.add_parser("backup")

    args = p.parse_args()
    cfg = Config.load(args.config)
    setup_logging(cfg.log_file, args.log_level)

    if args.cmd == "run":
        asyncio.run(Puppet(cfg).run_forever())

    elif args.cmd == "status":
        async def _status():
            c = Puppet(cfg)
            await c.gw.connect()
            print(json.dumps(await c.status(), indent=2))
            await c.gw.close()
        asyncio.run(_status())

    elif args.cmd == "init":
        from .workspace import Workspace
        ws = Workspace(cfg.workspace_dir)
        created = ws.init()
        print(f"workspace: {cfg.workspace_dir}")
        print(f"created: {', '.join(created)}" if created else "all files exist")

    elif args.cmd == "send":
        async def _send():
            c = Puppet(cfg)
            await c.gw.connect()
            print(await c.send(args.message))
            await c.gw.close()
        asyncio.run(_send())

    elif args.cmd == "memory":
        from .memory import Memory
        mem = Memory(cfg.workspace_dir)
        if args.mem_cmd == "today":
            print(mem.today() or "no memory today")
        elif args.mem_cmd == "recent":
            print(mem.recent())
        elif args.mem_cmd == "long-term":
            print(mem.long_term_content() or "no long-term memory")
        elif args.mem_cmd == "consolidate":
            print(mem.consolidate())
        elif args.mem_cmd == "search":
            for r in mem.search(args.query):
                print(f"{r['file']}:{r['line']}: {r['text']}")

    elif args.cmd == "task":
        from .task_manager import TaskManager
        tm = TaskManager(None, cfg.task_dir)  # type: ignore[arg-type]
        if args.task_cmd == "add":
            t = tm.add(args.name, args.command, interval=args.interval)
            print(f"{t.id} - {t.name}")
        elif args.task_cmd == "list":
            for t in tm.list_all():
                print(f"[{'ON' if t.enabled else 'OFF'}] {t.id}: {t.name} ({t.run_count} runs)")
        elif args.task_cmd == "remove":
            print("removed" if tm.remove(args.task_id) else "not found")

    elif args.cmd == "cron":
        from .cron import Cron
        engine = Cron(cfg.cron_dir)
        if args.cron_cmd == "add":
            j = engine.add(args.name, args.command, args.schedule)
            print(f"{j.id} - {j.name} [{j.schedule}]")
        elif args.cron_cmd == "list":
            for j in engine.list_all():
                print(f"[{'ON' if j.enabled else 'OFF'}] {j.id}: {j.name} [{j.schedule}] ({j.run_count} runs)")
        elif args.cron_cmd == "remove":
            print("removed" if engine.remove(args.job_id) else "not found")
        elif args.cron_cmd == "enable":
            engine.enable(args.job_id)
        elif args.cron_cmd == "disable":
            engine.disable(args.job_id)

    elif args.cmd == "heartbeat":
        from .heartbeat import Heartbeat
        hb = Heartbeat(cfg.workspace_dir)
        if args.hb_cmd == "status":
            for n, info in hb.status()["checks"].items():
                tag = "DUE" if info["due"] else "ok"
                print(f"[{tag}] {n}: {info['interval']}s")

    elif args.cmd == "workspace":
        from .workspace import Workspace
        ws = Workspace(cfg.workspace_dir)
        if args.ws_cmd == "list":
            for f in ws.list_files():
                print(f"{f['name']} ({f['size']} bytes)")
        elif args.ws_cmd == "identity":
            print(json.dumps(ws.identity(), indent=2))
        elif args.ws_cmd == "backup":
            print(f"backup: {ws.backup()}")

    else:
        p.print_help()


if __name__ == "__main__":
    main()
