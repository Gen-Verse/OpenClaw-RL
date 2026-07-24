from __future__ import annotations

import asyncio
import logging
import signal
import sys
from typing import Any

from .config import Config
from .gateway_client import Gateway
from .memory import Memory
from .heartbeat import Heartbeat
from .cron import Cron
from .skills import Registry
from .workspace import Workspace
from .task_manager import TaskManager

log = logging.getLogger("openclaw.puppet")


class Puppet:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.gw = Gateway(cfg.gateway)
        self.ws = Workspace(cfg.workspace_dir)
        self.mem = Memory(cfg.workspace_dir)
        self.hb = Heartbeat(cfg.workspace_dir)
        self.cron = Cron(cfg.cron_dir)
        self.skills = Registry(cfg.skills_dir)
        self.tasks = TaskManager(self.gw.send, cfg.task_dir)
        self._running = False
        self._tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        log.info("starting puppet")
        self._running = True
        await self.gw.connect()
        self.ws.init()
        self.mem.log("puppet started", "system")
        self._wire_heartbeats()
        self._tasks.append(asyncio.create_task(self._loop_heartbeat()))
        self._tasks.append(asyncio.create_task(self._loop_poll()))
        self._tasks.append(asyncio.create_task(self._loop_cron()))
        log.info("all systems online")

    async def stop(self) -> None:
        log.info("stopping puppet")
        self._running = False
        for t in self._tasks:
            t.cancel()
        await self.gw.close()
        self.mem.log("puppet stopped", "system")

    def _wire_heartbeats(self) -> None:
        self.hb.register("status", 300)
        self.hb.register("memory_maintenance", 7200)
        self.hb.register("health", 600)

    async def _loop_heartbeat(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(self.cfg.heartbeat_interval)
                if not self._running:
                    break
                for r in await self.hb.tick():
                    self.mem.log(f"heartbeat [{r['check']}]: {r['result'][:200]}", "heartbeat")
            except asyncio.CancelledError:
                break
            except Exception:
                log.exception("heartbeat error")
                await asyncio.sleep(30)

    async def _loop_poll(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(self.cfg.poll_interval)
                if not self._running:
                    break
                for r in await self.tasks.tick():
                    self.mem.log(f"task: {r[:200]}", "task")
            except asyncio.CancelledError:
                break
            except Exception:
                log.exception("poll error")
                await asyncio.sleep(10)

    async def _loop_cron(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(30)
                if not self._running:
                    break
                for r in await self.cron.tick():
                    self.mem.log(f"cron [{r['job']}]: {r['result'][:200]}", "cron")
            except asyncio.CancelledError:
                break
            except Exception:
                log.exception("cron error")
                await asyncio.sleep(10)

    async def send(self, message: str) -> str:
        self.mem.log(f"cmd: {message[:100]}", "command")
        reply = await self.gw.send(message)
        self.mem.log(f"reply: {reply[:200]}", "command")
        return reply

    async def status(self) -> dict:
        try:
            health = await self.gw.health()
        except Exception:
            health = {"status": "disconnected"}
        return {
            "running": self._running,
            "health": health,
            "identity": self.ws.identity(),
            "user": self.ws.user(),
            "memory_files": len(self.mem.list_files()),
            "heartbeat": self.hb.status(),
            "cron_jobs": len(self.cron.list_all()),
            "skills": len(self.skills.list_all()),
            "tasks": [
                {"id": t.id, "name": t.name, "enabled": t.enabled,
                 "next_run": t.next_run, "runs": t.run_count,
                 "last": t.last_result[:100] if t.last_result else ""}
                for t in self.tasks.list_all()
            ],
        }

    async def run_forever(self) -> None:
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, lambda: asyncio.create_task(self.stop()))
            except NotImplementedError:
                pass
        await self.start()
        try:
            while self._running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            await self.stop()


def setup_logging(log_file: str | None = None, level: str = "INFO") -> None:
    fmt = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    handlers: list[Any] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        import os
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format=fmt, handlers=handlers)
