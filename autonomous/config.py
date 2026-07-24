from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

_DEFAULT_BASE = Path.home() / ".openclaw"


@dataclass
class GatewayConfig:
    url: str = "ws://127.0.0.1:18789"
    token: str = ""
    timeout: int = 300


@dataclass
class AgentConfig:
    agent_id: str = "main"
    model: str = ""
    thinking: str = "medium"
    timeout: int = 600


@dataclass
class Config:
    gateway: GatewayConfig = field(default_factory=GatewayConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    poll_interval: int = 60
    heartbeat_interval: int = 1800
    task_dir: str = ""
    cron_dir: str = ""
    skills_dir: str = ""
    workspace_dir: str = ""
    log_file: str = ""
    enabled: bool = True

    @classmethod
    def load(cls, path: str | None = None) -> Config:
        path = path or os.environ.get("OPENCLAW_PUPPET_CONFIG", str(_DEFAULT_BASE / "puppet.json"))
        cfg = cls()
        if os.path.exists(path):
            with open(path) as fh:
                raw = json.load(fh)
            _merge_dataclass(raw, "gateway", cfg.gateway)
            _merge_dataclass(raw, "agent", cfg.agent)
            for key in ("poll_interval", "heartbeat_interval", "task_dir", "cron_dir",
                        "skills_dir", "workspace_dir", "log_file", "enabled"):
                if key in raw and hasattr(cfg, key):
                    setattr(cfg, key, raw[key])
        if not cfg.gateway.token:
            cfg.gateway.token = os.environ.get("OPENCLAW_GATEWAY_TOKEN", "")
        base = str(_DEFAULT_BASE)
        cfg.task_dir = cfg.task_dir or os.path.join(base, "tasks")
        cfg.cron_dir = cfg.cron_dir or os.path.join(base, "cron")
        cfg.skills_dir = cfg.skills_dir or os.path.join(base, "skills")
        cfg.workspace_dir = cfg.workspace_dir or os.path.join(base, "workspace")
        cfg.log_file = cfg.log_file or os.path.join(base, "puppet.log")
        return cfg

    def save(self, path: str | None = None) -> None:
        path = path or os.environ.get("OPENCLAW_PUPPET_CONFIG", str(_DEFAULT_BASE / "puppet.json"))
        data = {
            "gateway": {"url": self.gateway.url, "token": self.gateway.token, "timeout": self.gateway.timeout},
            "agent": {"agent_id": self.agent.agent_id, "model": self.agent.model,
                      "thinking": self.agent.thinking, "timeout": self.agent.timeout},
            "poll_interval": self.poll_interval,
            "heartbeat_interval": self.heartbeat_interval,
            "task_dir": self.task_dir,
            "cron_dir": self.cron_dir,
            "skills_dir": self.skills_dir,
            "workspace_dir": self.workspace_dir,
            "log_file": self.log_file,
            "enabled": self.enabled,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(data, fh, indent=2)


def _merge_dataclass(raw: dict, key: str, target: object) -> None:
    if key not in raw or not isinstance(raw[key], dict):
        return
    for k, v in raw[key].items():
        if hasattr(target, k):
            setattr(target, k, v)
