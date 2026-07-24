from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

log = logging.getLogger("openclaw.puppet.tasks")


@dataclass
class Task:
    id: str
    name: str
    command: str
    args: dict = field(default_factory=dict)
    interval: int = 0
    next_run: float = 0
    last_run: float = 0
    enabled: bool = True
    run_count: int = 0
    last_result: str = ""

    def is_due(self) -> bool:
        return self.enabled and time.time() >= self.next_run


class TaskManager:
    def __init__(self, send_fn: Callable, task_dir: str) -> None:
        self._send = send_fn
        self.dir = Path(task_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.tasks: dict[str, Task] = {}
        self._handlers: dict[str, Callable] = {}
        self._load()

    def _load(self) -> None:
        for f in self.dir.glob("*.json"):
            try:
                t = Task(**json.loads(f.read_text()))
                self.tasks[t.id] = t
            except Exception:
                log.exception("skip %s", f.name)

    def _save(self, t: Task) -> None:
        (self.dir / f"{t.id}.json").write_text(json.dumps({
            "id": t.id, "name": t.name, "command": t.command, "args": t.args,
            "interval": t.interval, "next_run": t.next_run, "last_run": t.last_run,
            "enabled": t.enabled, "run_count": t.run_count, "last_result": t.last_result,
        }, indent=2))

    def on(self, command: str, handler: Callable) -> None:
        self._handlers[command] = handler

    def add(self, name: str, command: str, args: dict | None = None, interval: int = 60) -> Task:
        tid = f"task-{len(self.tasks) + 1:04d}"
        t = Task(id=tid, name=name, command=command, args=args or {},
                 interval=interval, next_run=time.time() + interval)
        self.tasks[tid] = t
        self._save(t)
        log.info("added %s (%s)", name, tid)
        return t

    def remove(self, tid: str) -> bool:
        t = self.tasks.pop(tid, None)
        if t:
            p = self.dir / f"{tid}.json"
            p.unlink(missing_ok=True)
            return True
        return False

    async def run(self, t: Task) -> str:
        handler = self._handlers.get(t.command)
        if handler:
            try:
                result = await handler(t)
                t.last_result = str(result)
            except Exception as e:
                t.last_result = f"ERROR: {e}"
        else:
            prompt = t.command
            if t.args:
                prompt += " " + json.dumps(t.args)
            try:
                t.last_result = await self._send(prompt)
            except Exception as e:
                t.last_result = f"ERROR: {e}"
        t.last_run = time.time()
        t.run_count += 1
        if t.interval > 0:
            t.next_run = time.time() + t.interval
        self._save(t)
        return t.last_result

    async def tick(self) -> list[str]:
        out: list[str] = []
        for t in list(self.tasks.values()):
            if t.is_due():
                r = await self.run(t)
                out.append(f"{t.name}: {r[:200]}")
        return out

    def list_all(self) -> list[Task]:
        return sorted(self.tasks.values(), key=lambda t: t.next_run)
