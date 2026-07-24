import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger("openclaw.puppet.heartbeat")


@dataclass
class HeartbeatCheck:
    name: str
    interval: int
    last_run: float = 0
    enabled: bool = True

    def is_due(self) -> bool:
        if not self.enabled:
            return False
        return time.time() >= self.last_run + self.interval


class Heartbeat:
    def __init__(self, workspace_dir: str, state_file: str = ""):
        self.workspace = Path(workspace_dir)
        self.state_file = Path(state_file) if state_file else self.workspace / "heartbeat-state.json"
        self.checks: dict[str, HeartbeatCheck] = {}
        self._callbacks: dict[str, Callable] = {}
        self._state: dict = {"lastChecks": {}}
        self._load_state()
        self._load_checks()

    def _load_state(self) -> None:
        if self.state_file.exists():
            try:
                self._state = json.loads(self.state_file.read_text())
            except Exception:
                self._state = {"lastChecks": {}}

    def _save_state(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps(self._state, indent=2))

    def _load_checks(self) -> None:
        checks_file = self.workspace / "heartbeat-checks.json"
        if checks_file.exists():
            try:
                data = json.loads(checks_file.read_text())
                for name, info in data.items():
                    self.checks[name] = HeartbeatCheck(
                        name=name,
                        interval=info.get("interval", 3600),
                        last_run=self._state.get("lastChecks", {}).get(name, 0),
                        enabled=info.get("enabled", True),
                    )
            except Exception:
                logger.exception("Failed to load heartbeat checks")

    def register(self, name: str, interval: int, callback: Callable | None = None) -> None:
        self.checks[name] = HeartbeatCheck(name=name, interval=interval)
        if callback:
            self._callbacks[name] = callback

    def get_due_checks(self) -> list[str]:
        return [name for name, check in self.checks.items() if check.is_due()]

    def mark_done(self, name: str) -> None:
        if name in self.checks:
            self.checks[name].last_run = time.time()
        self._state.setdefault("lastChecks", {})[name] = time.time()
        self._save_state()

    async def run_check(self, name: str) -> str:
        check = self.checks.get(name)
        if not check:
            return f"Unknown check: {name}"
        callback = self._callbacks.get(name)
        if callback:
            try:
                result = await callback(check)
                self.mark_done(name)
                return result
            except Exception as e:
                self.mark_done(name)
                return f"ERROR: {e}"
        self.mark_done(name)
        return f"No handler for check: {name}"

    async def tick(self) -> list[dict]:
        results = []
        for name in self.get_due_checks():
            logger.info("Heartbeat check due: %s", name)
            result = await self.run_check(name)
            results.append({"check": name, "result": result})
        return results

    def get_status(self) -> dict:
        return {
            "checks": {
                name: {
                    "interval": c.interval,
                    "last_run": c.last_run,
                    "enabled": c.enabled,
                    "due": c.is_due(),
                }
                for name, c in self.checks.items()
            },
            "lastChecks": self._state.get("lastChecks", {}),
        }
