import asyncio
import json
import logging
import uuid
from typing import Any, Callable

import websockets

from .config import GatewayConfig

logger = logging.getLogger("openclaw.autonomous.gateway")


class GatewayClient:
    def __init__(self, config: GatewayConfig):
        self.config = config
        self._ws = None
        self._id_counter = 0
        self._pending: dict[str, asyncio.Future] = {}
        self._subscriptions: list[Callable] = []

    async def connect(self) -> None:
        headers = {}
        if self.config.token:
            headers["Authorization"] = f"Bearer {self.config.token}"
        self._ws = await websockets.connect(
            self.config.url,
            additional_headers=headers,
            ping_interval=30,
            ping_timeout=10,
        )
        logger.info("Connected to gateway at %s", self.config.url)

    async def disconnect(self) -> None:
        if self._ws:
            await self._ws.close()
            self._ws = None
            logger.info("Disconnected from gateway")

    def _next_id(self) -> str:
        self._id_counter += 1
        return f"autonomous-{self._id_counter}-{uuid.uuid4().hex[:8]}"

    async def call(self, method: str, params: dict | None = None) -> Any:
        if not self._ws:
            await self.connect()
        msg_id = self._next_id()
        message = {"jsonrpc": "2.0", "id": msg_id, "method": method}
        if params:
            message["params"] = params
        future = asyncio.get_event_loop().create_future()
        self._pending[msg_id] = future
        await self._ws.send(json.dumps(message))
        try:
            result = await asyncio.wait_for(future, timeout=self.config.timeout)
            return result
        except asyncio.TimeoutError:
            self._pending.pop(msg_id, None)
            raise TimeoutError(f"Gateway call {method} timed out")

    async def send_agent_message(
        self, message: str, agent_id: str = "main", session_key: str | None = None
    ) -> str:
        params = {"message": message, "agentId": agent_id}
        if session_key:
            params["sessionKey"] = session_key
        result = await self.call("agent.turn", params)
        return result.get("reply", "") if isinstance(result, dict) else str(result)

    async def read_messages(self, channel: str = "last", limit: int = 10) -> list[dict]:
        result = await self.call(
            "message.read", {"channel": channel, "limit": limit}
        )
        return result.get("messages", []) if isinstance(result, dict) else []

    async def health(self) -> dict:
        result = await self.call("health")
        return result if isinstance(result, dict) else {"status": "unknown"}

    async def list_sessions(self) -> list[dict]:
        result = await self.call("sessions.list")
        return result.get("sessions", []) if isinstance(result, dict) else []

    async def _listen_loop(self) -> None:
        try:
            async for raw in self._ws:
                data = json.loads(raw)
                if "id" in data and data["id"] in self._pending:
                    future = self._pending.pop(data["id"])
                    if "error" in data:
                        future.set_exception(
                            RuntimeError(data["error"].get("message", "unknown error"))
                        )
                    else:
                        future.set_result(data.get("result"))
                elif "method" in data:
                    for sub in self._subscriptions:
                        try:
                            sub(data)
                        except Exception:
                            logger.exception("Subscription callback error")
        except websockets.ConnectionClosed:
            logger.warning("Gateway connection closed")

    async def run_listener(self) -> asyncio.Task:
        return asyncio.create_task(self._listen_loop())

    def on_event(self, callback: Callable) -> None:
        self._subscriptions.append(callback)
