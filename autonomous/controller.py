import asyncio
import logging
import signal
import sys
from typing import Any

from .config import AutonomousConfig
from .gateway_client import GatewayClient
from .memory import Memory
from .heartbeat import Heartbeat
from .cron import CronEngine
from .skills import SkillRegistry
from .workspace import Workspace
from .task_manager import TaskManager

logger = logging.getLogger("openclaw.puppet")


class PuppetController:
    def __init__(self, config: AutonomousConfig):
        self.config = config
        self.client = GatewayClient(config.gateway)
        self.workspace = Workspace(config.workspace_dir)
        self.memory = Memory(config.workspace_dir)
        self.heartbeat = Heartbeat(config.workspace_dir)
        self.cron = CronEngine(config.cron_dir)
        self.skills = SkillRegistry(config.skills_dir)
        self.task_manager = TaskManager(self.client, config.task_dir)
        self._running = False
        self._tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        logger.info("Starting puppet controller")
        self._running = True
        await self.client.connect()
        self.workspace.init()
        self.memory.log("Puppet controller started", "system")
        self._register_default_heartbeats()
        self._register_default_crons()
        self._tasks.append(asyncio.create_task(self._heartbeat_loop()))
        self._tasks.append(asyncio.create_task(self._poll_loop()))
        self._tasks.append(asyncio.create_task(self._cron_loop()))
        logger.info("Puppet controller started — all systems online")

    async def stop(self) -> None:
        logger.info("Stopping puppet controller")
        self._running = False
        for t in self._tasks:
            t.cancel()
        await self.client.disconnect()
        self.memory.log("Puppet controller stopped", "system")
        logger.info("Puppet controller stopped")

    def _register_default_heartbeats(self) -> None:
        self.heartbeat.register("status_check", 300)
        self.heartbeat.register("memory_maintenance", 7200)
        self.heartbeat.register("system_health", 600)

    def _register_default_crons(self) -> None:
        pass

    async def _heartbeat_loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(self.config.heartbeat_interval)
                if not self._running:
                    break
                logger.info("Heartbeat cycle starting")
                results = await self.heartbeat.tick()
                for r in results:
                    self.memory.log(f"Heartbeat [{r['check']}]: {r['result'][:200]}", "heartbeat")
                if not results:
                    self.memory.log("Heartbeat: all checks clean", "heartbeat")
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Heartbeat loop error")
                self.memory.log_error("Heartbeat loop error")
                await asyncio.sleep(30)

    async def _poll_loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(self.config.poll_interval)
                if not self._running:
                    break
                results = await self.task_manager.tick()
                for r in results:
                    self.memory.log(f"Task [{r.split(':')[0]}]: {r.split(':', 1)[1][:200]}", "task")
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Poll loop error")
                await asyncio.sleep(10)

    async def _cron_loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(30)
                if not self._running:
                    break
                results = await self.cron.tick()
                for r in results:
                    self.memory.log(f"Cron [{r['job']}]: {r['result'][:200]}", "cron")
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Cron loop error")
                await asyncio.sleep(10)

    async def send_command(self, message: str) -> str:
        self.memory.log(f"Command sent: {message[:100]}", "command")
        reply = await self.client.send_agent_message(message)
        self.memory.log(f"Reply: {reply[:200]}", "command")
        return reply

    async def get_status(self) -> dict:
        try:
            health = await self.client.health()
        except Exception:
            health = {"status": "disconnected"}
        return {
            "running": self._running,
            "health": health,
            "identity": self.workspace.get_identity(),
            "user": self.workspace.get_user(),
            "memory_files": len(self.memory.list_files()),
            "heartbeat": self.heartbeat.get_status(),
            "cron_jobs": len(self.cron.list_jobs()),
            "skills": len(self.skills.list_skills()),
            "tasks": [
                {
                    "id": t.id,
                    "name": t.name,
                    "enabled": t.enabled,
                    "next_run": t.next_run,
                    "run_count": t.run_count,
                    "last_result": t.last_result[:100] if t.last_result else "",
                }
                for t in self.task_manager.list_tasks()
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
