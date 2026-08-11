#!/usr/bin/env python3
"""Long-running API server that schedules training or skill accumulation."""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any, Dict

import yaml
from fastapi import FastAPI, HTTPException

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_core.evolution import EvolutionCoordinator


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "evolution.yaml"


def create_app(config_path: str | Path = DEFAULT_CONFIG) -> FastAPI:
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    coordinator = EvolutionCoordinator.from_file(config_path)
    stop_event = threading.Event()
    worker: threading.Thread | None = None

    app = FastAPI(title="OpenClaw-RL Self-Evolving Agent Server", version="0.1.0")
    app.state.coordinator = coordinator

    def scheduler_loop() -> None:
        interval = int(config["schedule"].get("poll_seconds", 900))
        while not stop_event.is_set():
            coordinator.run_cycle()
            stop_event.wait(interval)

    @app.on_event("startup")
    def start_scheduler() -> None:
        nonlocal worker
        if config["schedule"].get("automatic", True):
            worker = threading.Thread(target=scheduler_loop, name="evolution-scheduler", daemon=True)
            worker.start()

    @app.on_event("shutdown")
    def stop_scheduler() -> None:
        stop_event.set()
        if worker:
            worker.join(timeout=5)

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {"status": "ok", "mode": coordinator.status().get("mode", "not_started")}

    @app.post("/v1/trajectories")
    def ingest_trajectory(event: Dict[str, Any]) -> Dict[str, Any]:
        try:
            coordinator.ingest(event)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"accepted": True}

    @app.post("/v1/evolution/run")
    def run_evolution(force: bool = False) -> Dict[str, Any]:
        return coordinator.run_cycle(force=force)

    @app.get("/v1/evolution/status")
    def evolution_status() -> Dict[str, Any]:
        return coordinator.status()

    @app.get("/v1/skills")
    def list_skills() -> Dict[str, Any]:
        return {"skills": coordinator.skill_store.list()}

    return app


app = create_app()
