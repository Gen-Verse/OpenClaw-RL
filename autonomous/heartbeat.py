from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

log = logging.getLogger("openclaw.puppet.heartbeat")


@dataclass
class Check:
    name: str
    interval: int
    last_run: float = 0
    enabled: bool = True

    def is_due(self) -> bool:
        return self.enabled and time.time() >= self.last_run + self.interval


class Heartbeat:
    def __init__(self, workspace: str) -> None:
        self.root = Path(workspace)
        self.state_file = self.root / "heartbeat-state.json"
        self.checks: dict[str, Check] = {}
        self._handlers: dict[str, Callable] = {}
        self._state: dict = {"lastChecks": {}}
        self._load()

    def _load(self) -> None:
        if self.state_file.exists():
            try:
                self._state = json.loads(self.state_file.read_text())
            except Exception:
                pass
        checks_file = self.root / "heartbeat-checks.json"
        if checks_file.exists():
            try:
                for name, info in json.loads(checks_file.read_text()).items():
                    self.checks[name] = Check(
                        name=name,
                        interval=info.get("interval", 3600),
                        last_run=self._state.get("lastChecks", {}).get(name, 0),
                        enabled=info.get("enabled", True),
                    )
            except Exception:
                log.exception("failed to load checks")

    def _save(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps(self._state, indent=2))

    def register(self, name: str, interval: int, handler: Callable | None = None) -> None:
        self.checks[name] = Check(name=name, interval=interval)
        if handler:
            self._handlers[name] = handler

    def due(self) -> list[str]:
        return [n for n, c in self.checks.items() if c.is_due()]

    def mark(self, name: str) -> None:
        now = time.time()
        if name in self.checks:
            self.checks[name].last_run = now
        self._state.setdefault("lastChecks", {})[name] = now
        self._save()

    async def run_one(self, name: str) -> str:
        c = self.checks.get(name)
        if not c:
            return f"unknown: {name}"
        h = self._handlers.get(name)
        if h:
            try:
                r = await h(c)
                self.mark(name)
                return str(r)
            except Exception as e:
                self.mark(name)
                return f"ERROR: {e}"
        self.mark(name)
        return f"no handler: {name}"

    async def tick(self) -> list[dict]:
        results: list[dict] = []
        for name in self.due():
            log.info("check due: %s", name)
            r = await self.run_one(name)
            results.append({"check": name, "result": r})
        return results

    def status(self) -> dict:
        return {
            "checks": {
                n: {"interval": c.interval, "last_run": c.last_run,
                     "enabled": c.enabled, "due": c.is_due()}
                for n, c in self.checks.items()
            }
        }
