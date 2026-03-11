"""SWE environment pool server — runs on GPU Head Node.

Manages remote ECS Docker nodes: lease allocation, container lifecycle,
command execution, evaluation.  Modeled after gui/env_pool_server.py.

Each "lease" corresponds to one SWE-Bench instance running inside a Docker
container on one of the remote ECS nodes.  The pool server proxies all
requests to the appropriate swe_exec_server running on the ECS node.

Usage:
    python3 -m swe_env_pool_server \
        --port 18090 \
        --exec-server-urls http://10.0.0.10:5000,http://10.0.0.11:5000
"""

from __future__ import annotations

import argparse
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import requests
from flask import Flask, jsonify, request as flask_request

logger = logging.getLogger("swe.env_pool_server")
app = Flask(__name__)


def _post_exec(url: str, payload: dict, timeout: int = 300) -> dict:
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _get_exec(url: str, timeout: int = 30) -> dict:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


# ── Data structures ───────────────────────────────────────────────────

@dataclass
class ExecNode:
    url: str
    active_containers: int = 0
    max_containers: int = 16
    healthy: bool = True
    last_health_check: float = 0.0


@dataclass
class Lease:
    lease_id: str
    node_url: str
    container_id: str
    image: str
    instance_id: str
    created_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)


class SweEnvPool:
    def __init__(self, exec_server_urls: list[str], max_containers_per_node: int = 16):
        self.nodes = [
            ExecNode(url=url.rstrip("/"), max_containers=max_containers_per_node)
            for url in exec_server_urls
        ]
        self._leases: dict[str, Lease] = {}
        self._lock = threading.RLock()

    def _pick_and_reserve_node(self) -> ExecNode:
        """Atomically pick the least-loaded healthy node and increment its counter.

        The counter is incremented optimistically BEFORE the container is
        actually created, so concurrent callers see the reservation and spread
        across nodes.  The caller MUST call ``_unreserve_node(node)`` if the
        subsequent container creation fails.
        """
        with self._lock:
            candidates = [n for n in self.nodes if n.healthy and n.active_containers < n.max_containers]
            if not candidates:
                raise RuntimeError(
                    f"All exec nodes are full or unhealthy. "
                    f"nodes={[(n.url, n.active_containers, n.healthy) for n in self.nodes]}"
                )
            node = min(candidates, key=lambda n: n.active_containers)
            node.active_containers += 1
            return node

    def _unreserve_node(self, node: ExecNode) -> None:
        with self._lock:
            node.active_containers = max(0, node.active_containers - 1)

    def _find_node(self, url: str) -> ExecNode | None:
        for n in self.nodes:
            if n.url == url:
                return n
        return None

    def allocate(self, image: str, instance_id: str, cwd: str = "/testbed") -> dict[str, Any]:
        node = self._pick_and_reserve_node()
        try:
            result = _post_exec(f"{node.url}/container/create", {
                "image": image, "cwd": cwd,
            }, timeout=120)
        except Exception:
            self._unreserve_node(node)
            raise
        if not result.get("ok"):
            self._unreserve_node(node)
            raise RuntimeError(f"Container create failed on {node.url}: {result}")

        container_id = result["container_id"]
        lease_id = f"swe-lease-{uuid.uuid4().hex[:16]}"
        lease = Lease(
            lease_id=lease_id,
            node_url=node.url,
            container_id=container_id,
            image=image,
            instance_id=instance_id,
        )
        with self._lock:
            self._leases[lease_id] = lease
        logger.info("Allocated lease=%s node=%s cid=%s image=%s", lease_id, node.url, container_id[:12], image)
        return {"lease_id": lease_id, "container_id": container_id, "node_url": node.url}

    def _get_lease(self, lease_id: str) -> Lease:
        with self._lock:
            lease = self._leases.get(lease_id)
        if lease is None:
            raise KeyError(f"Unknown lease_id: {lease_id}")
        return lease

    def heartbeat(self, lease_id: str) -> None:
        lease = self._get_lease(lease_id)
        lease.last_heartbeat = time.time()

    def exec(self, lease_id: str, command: str, cwd: str = "/testbed",
             timeout: int = 180, env: dict | None = None) -> dict[str, Any]:
        lease = self._get_lease(lease_id)
        lease.last_heartbeat = time.time()
        return _post_exec(f"{lease.node_url}/container/exec", {
            "container_id": lease.container_id,
            "command": command,
            "cwd": cwd,
            "timeout": timeout,
            "env": env or {},
        }, timeout=timeout + 30)

    def diff(self, lease_id: str, cwd: str = "/testbed") -> dict[str, Any]:
        lease = self._get_lease(lease_id)
        return _post_exec(f"{lease.node_url}/container/diff", {
            "container_id": lease.container_id,
            "cwd": cwd,
        }, timeout=60)

    def evaluate(self, lease_id: str, patch: str, eval_script: str,
                 cwd: str = "/testbed", timeout: int = 3600) -> dict[str, Any]:
        lease = self._get_lease(lease_id)
        return _post_exec(f"{lease.node_url}/container/evaluate", {
            "container_id": lease.container_id,
            "patch": patch,
            "eval_script": eval_script,
            "cwd": cwd,
            "timeout": timeout,
        }, timeout=timeout + 60)

    def close(self, lease_id: str) -> None:
        with self._lock:
            lease = self._leases.pop(lease_id, None)
        if lease is None:
            return
        try:
            _post_exec(f"{lease.node_url}/container/destroy", {
                "container_id": lease.container_id,
            }, timeout=30)
        except Exception:
            logger.exception("Failed to destroy container for lease %s", lease_id)
        node = self._find_node(lease.node_url)
        if node is not None:
            self._unreserve_node(node)
        logger.info("Closed lease=%s cid=%s", lease_id, lease.container_id[:12])

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "total_leases": len(self._leases),
                "nodes": [
                    {
                        "url": n.url,
                        "active_containers": n.active_containers,
                        "max_containers": n.max_containers,
                        "healthy": n.healthy,
                    }
                    for n in self.nodes
                ],
            }

    def health_check(self) -> None:
        for node in self.nodes:
            try:
                r = _get_exec(f"{node.url}/healthz", timeout=5)
                node.healthy = r.get("ok", False)
            except Exception:
                node.healthy = False
            node.last_health_check = time.time()


# ── Flask routes ──────────────────────────────────────────────────────

POOL: SweEnvPool | None = None


@app.get("/healthz")
def healthz():
    return jsonify({"ok": True})


@app.get("/status")
def status():
    if POOL is None:
        return jsonify({"ok": False, "error": "Pool not initialized"}), 500
    return jsonify({"ok": True, "pool": POOL.status()})


@app.post("/allocate")
def allocate():
    if POOL is None:
        return jsonify({"ok": False, "error": "Pool not initialized"}), 500
    data = flask_request.get_json(force=True) or {}
    image = data.get("image")
    instance_id = data.get("instance_id", "")
    if not image:
        return jsonify({"ok": False, "error": "image is required"}), 400
    logger.info("[SWE-POOL] Allocate request: image=%s instance_id=%s", image, instance_id)
    try:
        result = POOL.allocate(image=image, instance_id=instance_id)
        return jsonify({"ok": True, **result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.post("/heartbeat")
def heartbeat():
    if POOL is None:
        return jsonify({"ok": False, "error": "Pool not initialized"}), 500
    data = flask_request.get_json(force=True) or {}
    lease_id = data.get("lease_id")
    if not lease_id:
        return jsonify({"ok": False, "error": "lease_id required"}), 400
    try:
        POOL.heartbeat(lease_id)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.post("/exec")
def exec_cmd():
    if POOL is None:
        return jsonify({"ok": False, "error": "Pool not initialized"}), 500
    data = flask_request.get_json(force=True) or {}
    lease_id = data.get("lease_id")
    command = data.get("command")
    if not lease_id or not command:
        return jsonify({"ok": False, "error": "lease_id and command required"}), 400
    try:
        result = POOL.exec(
            lease_id=lease_id,
            command=command,
            cwd=data.get("cwd", "/testbed"),
            timeout=int(data.get("timeout", 180)),
            env=data.get("env"),
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.post("/diff")
def diff():
    if POOL is None:
        return jsonify({"ok": False, "error": "Pool not initialized"}), 500
    data = flask_request.get_json(force=True) or {}
    lease_id = data.get("lease_id")
    if not lease_id:
        return jsonify({"ok": False, "error": "lease_id required"}), 400
    try:
        result = POOL.diff(lease_id=lease_id, cwd=data.get("cwd", "/testbed"))
        return jsonify(result)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.post("/evaluate")
def evaluate():
    if POOL is None:
        return jsonify({"ok": False, "error": "Pool not initialized"}), 500
    data = flask_request.get_json(force=True) or {}
    lease_id = data.get("lease_id")
    patch = data.get("patch", "")
    eval_script = data.get("eval_script", "")
    if not lease_id:
        return jsonify({"ok": False, "error": "lease_id required"}), 400
    try:
        result = POOL.evaluate(
            lease_id=lease_id,
            patch=patch,
            eval_script=eval_script,
            cwd=data.get("cwd", "/testbed"),
            timeout=int(data.get("timeout", 3600)),
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.post("/close")
def close():
    if POOL is None:
        return jsonify({"ok": False, "error": "Pool not initialized"}), 500
    data = flask_request.get_json(force=True) or {}
    lease_id = data.get("lease_id")
    if not lease_id:
        return jsonify({"ok": False, "error": "lease_id required"}), 400
    try:
        POOL.close(lease_id)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ── Main ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SWE environment pool server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.getenv("SWE_ENV_SERVER_PORT", "18090")))
    parser.add_argument(
        "--exec-server-urls",
        type=str,
        default=os.getenv("SWE_EXEC_SERVER_URLS", "http://localhost:5000"),
        help="Comma-separated swe_exec_server URLs",
    )
    parser.add_argument(
        "--max-containers-per-node",
        type=int,
        default=int(os.getenv("SWE_MAX_CONTAINERS_PER_NODE", "16")),
    )
    return parser.parse_args()


def _periodic_health_check(pool: "SweEnvPool", interval: int = 30) -> None:
    """Background thread: re-check all exec nodes every `interval` seconds."""
    while True:
        time.sleep(interval)
        try:
            pool.health_check()
            healthy = sum(1 for n in pool.nodes if n.healthy)
            logger.info("[SWE-POOL] Periodic health check: %d/%d nodes healthy", healthy, len(pool.nodes))
        except Exception as exc:
            logger.warning("[SWE-POOL] Periodic health check error: %s", exc)


def main() -> None:
    global POOL
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s %(levelname)s %(name)s] %(message)s")

    urls = [u.strip() for u in args.exec_server_urls.split(",") if u.strip()]
    POOL = SweEnvPool(exec_server_urls=urls, max_containers_per_node=args.max_containers_per_node)
    POOL.health_check()

    healthy = sum(1 for n in POOL.nodes if n.healthy)
    logger.info(
        "SWE env pool: %d/%d nodes healthy, max %d containers/node, listening on %s:%s",
        healthy, len(POOL.nodes), args.max_containers_per_node, args.host, args.port,
    )

    hc_thread = threading.Thread(target=_periodic_health_check, args=(POOL,), daemon=True)
    hc_thread.start()
    logger.info("[SWE-POOL] Periodic health check thread started (interval=30s)")

    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
