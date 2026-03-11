"""Async HTTP client for swe_env_pool_server.

Used by generate_with_swe_remote.py (inside the RolloutManager) to interact with
remote Docker containers via the pool server.  Modeled after gui/env_client.py.
"""

from __future__ import annotations

import os
from typing import Any

from slime.utils.http_utils import post


class SweEnvClient:
    def __init__(self, base_url: str | None = None):
        self.base_url = (base_url or os.getenv("SWE_ENV_SERVER_URL", "http://localhost:18090")).rstrip("/")
        self.default_max_retries = int(os.getenv("SWE_ENV_HTTP_MAX_RETRIES", "10"))
        self.evaluate_max_retries = int(os.getenv("SWE_EVALUATE_MAX_RETRIES", "3"))

    async def allocate(self, image: str, instance_id: str = "") -> dict[str, Any]:
        out = await post(
            f"{self.base_url}/allocate",
            {"image": image, "instance_id": instance_id},
            max_retries=self.default_max_retries,
        )
        if not out.get("ok", False):
            raise RuntimeError(f"SWE allocate failed: {out}")
        return out

    async def heartbeat(self, lease_id: str) -> None:
        out = await post(
            f"{self.base_url}/heartbeat",
            {"lease_id": lease_id},
            max_retries=self.default_max_retries,
        )
        if not out.get("ok", False):
            raise RuntimeError(f"SWE heartbeat failed: {out}")

    async def exec(self, lease_id: str, command: str, cwd: str = "/testbed",
                   timeout: int = 180, env: dict | None = None) -> dict[str, Any]:
        """Execute a command in the container. Returns {ok, returncode, output}."""
        out = await post(
            f"{self.base_url}/exec",
            {
                "lease_id": lease_id,
                "command": command,
                "cwd": cwd,
                "timeout": timeout,
                "env": env or {},
            },
            max_retries=self.default_max_retries,
        )
        if not out.get("ok", False):
            raise RuntimeError(f"SWE exec failed: {out}")
        return out

    async def diff(self, lease_id: str, cwd: str = "/testbed") -> str:
        """Get git diff from the container. Returns the patch string."""
        out = await post(
            f"{self.base_url}/diff",
            {"lease_id": lease_id, "cwd": cwd},
            max_retries=self.default_max_retries,
        )
        if not out.get("ok", False):
            raise RuntimeError(f"SWE diff failed: {out}")
        return out.get("patch", "")

    async def evaluate(self, lease_id: str, patch: str, eval_script: str,
                       cwd: str = "/testbed", timeout: int = 3600) -> dict[str, Any]:
        """Apply patch + run eval script. Returns {ok, resolved, ...}."""
        out = await post(
            f"{self.base_url}/evaluate",
            {
                "lease_id": lease_id,
                "patch": patch,
                "eval_script": eval_script,
                "cwd": cwd,
                "timeout": timeout,
            },
            max_retries=self.evaluate_max_retries,
        )
        if not out.get("ok", False):
            raise RuntimeError(f"SWE evaluate failed: {out}")
        return out

    async def close(self, lease_id: str) -> None:
        out = await post(
            f"{self.base_url}/close",
            {"lease_id": lease_id},
            max_retries=self.default_max_retries,
        )
        if not out.get("ok", False):
            raise RuntimeError(f"SWE close failed: {out}")
