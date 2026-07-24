import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .gateway_client import GatewayClient

logger = logging.getLogger("openclaw.autonomous.tasks")


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
        if not self.enabled:
            return False
        return time.time() >= self.next_run


class TaskManager:
    def __init__(self, client: GatewayClient, task_dir: str):
        self.client = client
        self.task_dir = task_dir
        self.tasks: dict[str, Task] = {}
        self._handlers: dict[str, Callable] = {}
        os.makedirs(task_dir, exist_ok=True)
        self._load_tasks()

    def _load_tasks(self) -> None:
        for f in Path(self.task_dir).glob("*.json"):
            try:
                data = json.loads(f.read_text())
                task = Task(**data)
                self.tasks[task.id] = task
                logger.info("Loaded task: %s (%s)", task.name, task.id)
            except Exception:
                logger.exception("Failed to load task: %s", f.name)

    def _save_task(self, task: Task) -> None:
        path = Path(self.task_dir) / f"{task.id}.json"
        path.write_text(
            json.dumps(
                {
                    "id": task.id,
                    "name": task.name,
                    "command": task.command,
                    "args": task.args,
                    "interval": task.interval,
                    "next_run": task.next_run,
                    "last_run": task.last_run,
                    "enabled": task.enabled,
                    "run_count": task.run_count,
                    "last_result": task.last_result,
                },
                indent=2,
            )
        )

    def register_handler(self, command: str, handler: Callable) -> None:
        self._handlers[command] = handler

    def add_task(
        self,
        name: str,
        command: str,
        args: dict | None = None,
        interval: int = 60,
        enabled: bool = True,
    ) -> Task:
        task_id = f"task-{len(self.tasks) + 1:04d}"
        task = Task(
            id=task_id,
            name=name,
            command=command,
            args=args or {},
            interval=interval,
            next_run=time.time() + interval,
            enabled=enabled,
        )
        self.tasks[task_id] = task
        self._save_task(task)
        logger.info("Added task: %s (%s)", name, task_id)
        return task

    def remove_task(self, task_id: str) -> bool:
        if task_id in self.tasks:
            del self.tasks[task_id]
            path = Path(self.task_dir) / f"{task_id}.json"
            if path.exists():
                path.unlink()
            logger.info("Removed task: %s", task_id)
            return True
        return False

    async def execute_task(self, task: Task) -> str:
        handler = self._handlers.get(task.command)
        if handler:
            try:
                result = await handler(task)
                task.last_result = str(result)
            except Exception as e:
                task.last_result = f"ERROR: {e}"
                logger.exception("Task %s failed", task.id)
        else:
            task.last_result = await self._execute_via_agent(task)
        task.last_run = time.time()
        task.run_count += 1
        if task.interval > 0:
            task.next_run = time.time() + task.interval
        self._save_task(task)
        return task.last_result

    async def _execute_via_agent(self, task: Task) -> str:
        prompt = task.command
        if task.args:
            prompt += " " + json.dumps(task.args)
        try:
            reply = await self.client.send_agent_message(prompt)
            return reply
        except Exception as e:
            return f"AGENT_ERROR: {e}"

    async def tick(self) -> list[str]:
        results = []
        for task in list(self.tasks.values()):
            if task.is_due():
                logger.info("Running task: %s", task.name)
                result = await self.execute_task(task)
                results.append(f"{task.name}: {result}")
        return results

    def list_tasks(self) -> list[Task]:
        return sorted(self.tasks.values(), key=lambda t: t.next_run)

    def get_task(self, task_id: str) -> Task | None:
        return self.tasks.get(task_id)
