import asyncio
import logging
import signal
import sys
import time
from typing import Any

from .config import AutonomousConfig
from .gateway_client import GatewayClient
from .task_manager import TaskManager

logger = logging.getLogger("openclaw.autonomous")


class AutonomousController:
    def __init__(self, config: AutonomousConfig):
        self.config = config
        self.client = GatewayClient(config.gateway)
        self.task_manager = TaskManager(self.client, config.task_dir)
        self._running = False
        self._heartbeat_task: asyncio.Task | None = None
        self._poll_task: asyncio.Task | None = None
        self._setup_default_handlers()

    def _setup_default_handlers(self) -> None:
        self.task_manager.register_handler("heartbeat", self._handle_heartbeat)
        self.task_manager.register_handler("poll", self._handle_poll)
        self.task_manager.register_handler("message", self._handle_message)
        self.task_manager.register_handler("status", self._handle_status)

    async def start(self) -> None:
        logger.info("Starting autonomous controller")
        self._running = True
        await self.client.connect()
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self._poll_task = asyncio.create_task(self._poll_loop())
        logger.info("Autonomous controller started")

    async def stop(self) -> None:
        logger.info("Stopping autonomous controller")
        self._running = False
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        if self._poll_task:
            self._poll_task.cancel()
        await self.client.disconnect()
        logger.info("Autonomous controller stopped")

    async def _heartbeat_loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(self.config.heartbeat_interval)
                if not self._running:
                    break
                logger.info("Heartbeat: checking in")
                reply = await self.client.send_agent_message(
                    "HEARTBEAT_CHECK: Autonomous controller periodic check. "
                    "Report status. Check for pending tasks. "
                    "If nothing to report, respond with HEARTBEAT_OK."
                )
                logger.info("Heartbeat response: %s", reply[:200])
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Heartbeat error")
                await asyncio.sleep(30)

    async def _poll_loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(self.config.poll_interval)
                if not self._running:
                    break
                results = await self.task_manager.tick()
                if results:
                    for r in results:
                        logger.info("Task result: %s", r)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Poll error")
                await asyncio.sleep(10)

    async def send_command(self, message: str) -> str:
        return await self.client.send_agent_message(message)

    async def get_status(self) -> dict:
        try:
            health = await self.client.health()
        except Exception:
            health = {"status": "disconnected"}
        return {
            "running": self._running,
            "health": health,
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
