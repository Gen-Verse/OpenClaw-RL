import os
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GatewayConfig:
    url: str = "ws://127.0.0.1:18789"
    token: str = ""
    timeout: int = 300


@dataclass
class AgentConfig:
    agent_id: str = "main"
    model: str = "github-copilot/claude-opus-4.7"
    thinking: str = "medium"
    timeout: int = 600


@dataclass
class AutonomousConfig:
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
    def load(cls, path: str | None = None) -> "AutonomousConfig":
        if path is None:
            path = os.environ.get(
                "OPENCLAW_PUPPET_CONFIG",
                str(Path.home() / ".openclaw" / "puppet.json"),
            )
        config = cls()
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            if "gateway" in data:
                for k, v in data["gateway"].items():
                    if hasattr(config.gateway, k):
                        setattr(config.gateway, k, v)
            if "agent" in data:
                for k, v in data["agent"].items():
                    if hasattr(config.agent, k):
                        setattr(config.agent, k, v)
            for k, v in data.items():
                if k not in ("gateway", "agent") and hasattr(config, k):
                    setattr(config, k, v)
        if not config.gateway.token:
            config.gateway.token = os.environ.get("OPENCLAW_GATEWAY_TOKEN", "")
        base = str(Path.home() / ".openclaw")
        if not config.task_dir:
            config.task_dir = os.path.join(base, "tasks")
        if not config.cron_dir:
            config.cron_dir = os.path.join(base, "cron")
        if not config.skills_dir:
            config.skills_dir = os.path.join(base, "skills")
        if not config.workspace_dir:
            config.workspace_dir = os.path.join(base, "workspace")
        if not config.log_file:
            config.log_file = os.path.join(base, "puppet.log")
        return config

    def save(self, path: str | None = None) -> None:
        if path is None:
            path = os.environ.get(
                "OPENCLAW_PUPPET_CONFIG",
                str(Path.home() / ".openclaw" / "puppet.json"),
            )
        data = {
            "gateway": {
                "url": self.gateway.url,
                "token": self.gateway.token,
                "timeout": self.gateway.timeout,
            },
            "agent": {
                "agent_id": self.agent.agent_id,
                "model": self.agent.model,
                "thinking": self.agent.thinking,
                "timeout": self.agent.timeout,
            },
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
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
