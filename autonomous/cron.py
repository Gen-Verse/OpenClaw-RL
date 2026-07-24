from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

log = logging.getLogger("openclaw.puppet.cron")


def _parse_schedule(spec: str) -> float:
    now = time.time()
    parts = spec.strip().split()
    if len(parts) == 2 and parts[0].isdigit():
        val, unit = int(parts[0]), parts[1]
        multi = {"s": 1, "m": 60, "h": 3600, "d": 86400}
        for suffix, mul in multi.items():
            if unit.startswith(suffix):
                return now + val * mul
    if spec.startswith("daily ") and len(parts) == 2:
        try:
            h, m = map(int, parts[1].split(":"))
            target = datetime.now().replace(hour=h, minute=m, second=0, microsecond=0)
            if target.timestamp() <= now:
                target = target.replace(day=target.day + 1)
            return target.timestamp()
        except Exception:
            pass
    return now + 3600


@dataclass
class Job:
    id: str
    name: str
    command: str
    schedule: str
    args: dict = field(default_factory=dict)
    enabled: bool = True
    last_run: float = 0
    next_run: float = 0
    run_count: int = 0
    last_result: str = ""

    def is_due(self) -> bool:
        return self.enabled and time.time() >= self.next_run


class Cron:
    def __init__(self, jobs_dir: str) -> None:
        self.dir = Path(jobs_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, Job] = {}
        self._handlers: dict[str, Callable] = {}
        self._load()

    def _load(self) -> None:
        for f in self.dir.glob("*.json"):
            try:
                j = Job(**json.loads(f.read_text()))
                if not j.next_run:
                    j.next_run = _parse_schedule(j.schedule)
                self.jobs[j.id] = j
            except Exception:
                log.exception("skip %s", f.name)

    def _save(self, j: Job) -> None:
        (self.dir / f"{j.id}.json").write_text(json.dumps({
            "id": j.id, "name": j.name, "command": j.command, "schedule": j.schedule,
            "args": j.args, "enabled": j.enabled, "last_run": j.last_run,
            "next_run": j.next_run, "run_count": j.run_count, "last_result": j.last_result,
        }, indent=2))

    def on(self, command: str, handler: Callable) -> None:
        self._handlers[command] = handler

    def add(self, name: str, command: str, schedule: str, args: dict | None = None) -> Job:
        jid = f"cron-{len(self.jobs) + 1:04d}"
        j = Job(id=jid, name=name, command=command, schedule=schedule,
                args=args or {}, next_run=_parse_schedule(schedule))
        self.jobs[jid] = j
        self._save(j)
        return j

    def remove(self, jid: str) -> bool:
        j = self.jobs.pop(jid, None)
        if j:
            (self.dir / f"{jid}.json").unlink(missing_ok=True)
            return True
        return False

    def enable(self, jid: str) -> bool:
        j = self.jobs.get(jid)
        if j:
            j.enabled = True
            self._save(j)
            return True
        return False

    def disable(self, jid: str) -> bool:
        j = self.jobs.get(jid)
        if j:
            j.enabled = False
            self._save(j)
            return True
        return False

    async def run(self, j: Job, fallback: Callable | None = None) -> str:
        handler = self._handlers.get(j.command) or fallback
        if handler:
            try:
                r = await handler(j)
                j.last_result = str(r)
            except Exception as e:
                j.last_result = f"ERROR: {e}"
        else:
            j.last_result = f"no handler: {j.command}"
        j.last_run = time.time()
        j.run_count += 1
        j.next_run = _parse_schedule(j.schedule)
        self._save(j)
        return j.last_result

    async def tick(self, fallback: Callable | None = None) -> list[dict]:
        out: list[dict] = []
        for j in list(self.jobs.values()):
            if j.is_due():
                r = await self.run(j, fallback)
                out.append({"job": j.name, "result": r})
        return out

    def list_all(self) -> list[Job]:
        return sorted(self.jobs.values(), key=lambda j: j.next_run)
