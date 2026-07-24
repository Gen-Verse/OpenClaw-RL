from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, Callable

import websockets

from .config import GatewayConfig

log = logging.getLogger("openclaw.puppet.gateway")


class Gateway:
    def __init__(self, cfg: GatewayConfig) -> None:
        self.cfg = cfg
        self._ws: Any = None
        self._seq = 0
        self._pending: dict[str, asyncio.Future] = {}
        self._listeners: list[Callable] = []

    async def connect(self) -> None:
        headers: dict[str, str] = {}
        if self.cfg.token:
            headers["Authorization"] = f"Bearer {self.cfg.token}"
        self._ws = await websockets.connect(
            self.cfg.url,
            additional_headers=headers,
            ping_interval=30,
            ping_timeout=10,
        )
        log.info("connected %s", self.cfg.url)

    async def close(self) -> None:
        if self._ws:
            await self._ws.close()
            self._ws = None
            log.info("disconnected")

    def _next_id(self) -> str:
        self._seq += 1
        return f"p-{self._seq}-{uuid.uuid4().hex[:6]}"

    async def call(self, method: str, params: dict | None = None) -> Any:
        if not self._ws:
            await self.connect()
        mid = self._next_id()
        msg: dict[str, Any] = {"jsonrpc": "2.0", "id": mid, "method": method}
        if params:
            msg["params"] = params
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[mid] = fut
        await self._ws.send(json.dumps(msg))
        try:
            return await asyncio.wait_for(fut, timeout=self.cfg.timeout)
        except asyncio.TimeoutError:
            self._pending.pop(mid, None)
            raise TimeoutError(f"{method} timed out")

    async def send(self, message: str, agent: str = "main", session: str | None = None) -> str:
        params: dict[str, Any] = {"message": message, "agentId": agent}
        if session:
            params["sessionKey"] = session
        result = await self.call("agent.turn", params)
        return result.get("reply", "") if isinstance(result, dict) else str(result)

    async def health(self) -> dict:
        r = await self.call("health")
        return r if isinstance(r, dict) else {"status": "unknown"}

    async def sessions(self) -> list[dict]:
        r = await self.call("sessions.list")
        return r.get("sessions", []) if isinstance(r, dict) else []

    def on_event(self, cb: Callable) -> None:
        self._listeners.append(cb)

    async def _recv_loop(self) -> None:
        try:
            async for raw in self._ws:
                data = json.loads(raw)
                mid = data.get("id")
                if mid and mid in self._pending:
                    fut = self._pending.pop(mid)
                    if "error" in data:
                        fut.set_exception(RuntimeError(data["error"].get("message", "error")))
                    else:
                        fut.set_result(data.get("result"))
                elif "method" in data:
                    for cb in self._listeners:
                        try:
                            cb(data)
                        except Exception:
                            log.exception("listener error")
        except websockets.ConnectionClosed:
            log.warning("connection closed")

    async def listen(self) -> asyncio.Task:
        return asyncio.create_task(self._recv_loop())
