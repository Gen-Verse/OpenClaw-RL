import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger("openclaw.puppet.cron")


@dataclass
class CronJob:
    id: str
    name: str
    command: str
    schedule: str  # "every Ns", "every Nh", "every Nd", "cron expr", "daily HH:MM"
    args: dict = field(default_factory=dict)
    enabled: bool = True
    last_run: float = 0
    next_run: float = 0
    run_count: int = 0
    last_result: str = ""

    def compute_next_run(self) -> float:
        now = time.time()
        if self.schedule.startswith("every "):
            parts = self.schedule.split()
            if len(parts) == 2:
                value = int(parts[0])
                unit = parts[1]
                if unit.startswith("s"):
                    return now + value
                elif unit.startswith("m"):
                    return now + value * 60
                elif unit.startswith("h"):
                    return now + value * 3600
                elif unit.startswith("d"):
                    return now + value * 86400
        elif self.schedule.startswith("daily "):
            time_str = self.schedule.split()[1]
            try:
                h, m = map(int, time_str.split(":"))
                now_dt = datetime.now()
                target = now_dt.replace(hour=h, minute=m, second=0, microsecond=0)
                if target.timestamp() <= now:
                    target = target.replace(day=target.day + 1)
                return target.timestamp()
            except Exception:
                pass
        return now + 3600  # fallback: 1 hour

    def is_due(self) -> bool:
        if not self.enabled:
            return False
        return time.time() >= self.next_run


class CronEngine:
    def __init__(self, jobs_dir: str):
        self.jobs_dir = Path(jobs_dir)
        self.jobs_dir.mkdir(exist_ok=True)
        self.jobs: dict[str, CronJob] = {}
        self._handlers: dict[str, Callable] = {}
        self._load_jobs()

    def _load_jobs(self) -> None:
        for f in self.jobs_dir.glob("*.json"):
            try:
                data = json.loads(f.read_text())
                job = CronJob(**data)
                if not job.next_run:
                    job.next_run = job.compute_next_run()
                self.jobs[job.id] = job
            except Exception:
                logger.exception("Failed to load cron job: %s", f.name)

    def _save_job(self, job: CronJob) -> None:
        path = self.jobs_dir / f"{job.id}.json"
        path.write_text(json.dumps({
            "id": job.id,
            "name": job.name,
            "command": job.command,
            "schedule": job.schedule,
            "args": job.args,
            "enabled": job.enabled,
            "last_run": job.last_run,
            "next_run": job.next_run,
            "run_count": job.run_count,
            "last_result": job.last_result,
        }, indent=2))

    def register_handler(self, command: str, handler: Callable) -> None:
        self._handlers[command] = handler

    def add(self, name: str, command: str, schedule: str, args: dict | None = None) -> CronJob:
        job_id = f"cron-{len(self.jobs) + 1:04d}"
        job = CronJob(
            id=job_id,
            name=name,
            command=command,
            schedule=schedule,
            args=args or {},
            next_run=CronJob(name=name, command=command, schedule=schedule).compute_next_run(),
        )
        self.jobs[job_id] = job
        self._save_job(job)
        logger.info("Added cron job: %s (%s) schedule=%s", name, job_id, schedule)
        return job

    def remove(self, job_id: str) -> bool:
        if job_id in self.jobs:
            del self.jobs[job_id]
            path = self.jobs_dir / f"{job_id}.json"
            if path.exists():
                path.unlink()
            return True
        return False

    def enable(self, job_id: str) -> bool:
        if job_id in self.jobs:
            self.jobs[job_id].enabled = True
            self._save_job(self.jobs[job_id])
            return True
        return False

    def disable(self, job_id: str) -> bool:
        if job_id in self.jobs:
            self.jobs[job_id].enabled = False
            self._save_job(self.jobs[job_id])
            return True
        return False

    async def execute(self, job: CronJob, executor: Callable | None = None) -> str:
        handler = self._handlers.get(job.command)
        if handler:
            try:
                result = await handler(job)
                job.last_result = str(result)
            except Exception as e:
                job.last_result = f"ERROR: {e}"
        elif executor:
            try:
                result = await executor(job)
                job.last_result = str(result)
            except Exception as e:
                job.last_result = f"ERROR: {e}"
        else:
            job.last_result = f"No handler for command: {job.command}"
        job.last_run = time.time()
        job.run_count += 1
        job.next_run = job.compute_next_run()
        self._save_job(job)
        return job.last_result

    async def tick(self, executor: Callable | None = None) -> list[dict]:
        results = []
        for job in list(self.jobs.values()):
            if job.is_due():
                logger.info("Cron job due: %s", job.name)
                result = await self.execute(job, executor)
                results.append({"job": job.name, "result": result})
        return results

    def list_jobs(self) -> list[CronJob]:
        return sorted(self.jobs.values(), key=lambda j: j.next_run)

    def get_job(self, job_id: str) -> CronJob | None:
        return self.jobs.get(job_id)
