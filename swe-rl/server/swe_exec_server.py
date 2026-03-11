"""SWE Docker Exec Server — runs on each Volcengine ECS Docker node.

Wraps local `docker` CLI into HTTP endpoints so that the GPU training cluster
can create/exec/destroy SWE-Bench containers remotely.

Usage (on the ECS node):
    python3 swe_exec_server.py                       # default :5000
    python3 swe_exec_server.py --port 5000 --host 0.0.0.0

Endpoints:
    GET  /healthz           → liveness check
    GET  /images            → list locally available Docker images
    POST /container/create  → docker run -d ... sleep infinity
    POST /container/exec    → docker exec <cid> bash -lc <cmd>
    POST /container/diff    → git add -A && git diff --cached
    POST /container/destroy → docker rm -f <cid>
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import threading
import time
import uuid

from flask import Flask, jsonify, request

app = Flask(__name__)
logger = logging.getLogger("swe_exec_server")

_active_containers: dict[str, dict] = {}
_lock = threading.Lock()


def _docker(*args: str, timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", *args],
        capture_output=True, text=True, timeout=timeout,
    )


def _is_valid_git_patch(patch_text: str) -> bool:
    if not isinstance(patch_text, str):
        return False
    text = patch_text.strip()
    if not text:
        return False
    if "diff --git " not in text:
        return False
    has_old = ("--- a/" in text) or ("--- /dev/null" in text)
    has_new = "+++ b/" in text
    return has_old and has_new


# ── Health ────────────────────────────────────────────────────────────

@app.get("/healthz")
def healthz():
    r = _docker("info", "--format", "{{.ContainersRunning}}", timeout=10)
    running = r.stdout.strip() if r.returncode == 0 else "?"
    return jsonify({"ok": True, "running_containers": running})


@app.get("/images")
def list_images():
    r = _docker("images", "--format", "{{.Repository}}:{{.Tag}}", timeout=30)
    if r.returncode != 0:
        return jsonify({"ok": False, "error": r.stderr}), 500
    images = [l.strip() for l in r.stdout.strip().split("\n") if l.strip()]
    return jsonify({"ok": True, "images": images, "count": len(images)})


@app.get("/status")
def status():
    with _lock:
        active = {cid: info for cid, info in _active_containers.items()}
    return jsonify({"ok": True, "active_containers": len(active), "containers": active})


# ── Container lifecycle ───────────────────────────────────────────────

@app.post("/container/create")
def container_create():
    """Create a detached container from a SWE-Bench image.

    Request JSON:
        image (str):   full image name, e.g. docker.io/xingyaoww/sweb.eval.x86_64.django_s_django-12345:latest
        cwd (str):     working directory inside container, default "/testbed"
        timeout (int): docker run timeout in seconds, default 120
    """
    data = request.get_json(force=True) or {}
    image = data.get("image")
    if not image:
        return jsonify({"ok": False, "error": "image is required"}), 400

    cwd = data.get("cwd", "/testbed")
    timeout = int(data.get("timeout", 120))
    container_name = f"swe-{uuid.uuid4().hex[:12]}"

    r = _docker(
        "run", "-d",
        "--name", container_name,
        "-w", cwd,
        "--pids-limit", "256",
        "--memory", "4g",
        image,
        "sleep", "infinity",
        timeout=timeout,
    )
    if r.returncode != 0:
        return jsonify({"ok": False, "error": r.stderr.strip()}), 500

    container_id = r.stdout.strip()
    with _lock:
        _active_containers[container_id] = {
            "name": container_name,
            "image": image,
            "created_at": time.time(),
        }
    logger.info("Created container %s (%s) from %s", container_id[:12], container_name, image)
    return jsonify({"ok": True, "container_id": container_id, "name": container_name})


@app.post("/container/exec")
def container_exec():
    """Execute a bash command inside a running container.

    Request JSON:
        container_id (str): container ID or name
        command (str):      bash command to execute
        cwd (str):          working directory, default "/testbed"
        timeout (int):      execution timeout in seconds, default 180
    """
    data = request.get_json(force=True) or {}
    container_id = data.get("container_id")
    command = data.get("command")
    if not container_id or not command:
        return jsonify({"ok": False, "error": "container_id and command are required"}), 400

    cwd = data.get("cwd", "/testbed")
    timeout = int(data.get("timeout", 180))

    env_args = []
    for key, value in data.get("env", {}).items():
        env_args.extend(["-e", f"{key}={value}"])

    try:
        r = _docker(
            "exec", "-w", cwd, *env_args, container_id,
            "bash", "-lc", command,
            timeout=timeout,
        )
        output = r.stdout + r.stderr
        return jsonify({
            "ok": True,
            "returncode": r.returncode,
            "output": output,
        })
    except subprocess.TimeoutExpired:
        return jsonify({
            "ok": True,
            "returncode": -1,
            "output": f"Command timed out after {timeout}s",
        })


@app.post("/container/diff")
def container_diff():
    """Get the git patch from the container's working directory.

    Request JSON:
        container_id (str): container ID or name
        cwd (str):          working directory, default "/testbed"
    """
    data = request.get_json(force=True) or {}
    container_id = data.get("container_id")
    if not container_id:
        return jsonify({"ok": False, "error": "container_id is required"}), 400

    cwd = data.get("cwd", "/testbed")
    r = _docker(
        "exec", "-w", cwd, container_id,
        "bash", "-lc", "git add -A && git diff --cached",
        timeout=60,
    )
    return jsonify({
        "ok": True,
        "patch": r.stdout,
        "returncode": r.returncode,
        "error": r.stderr if r.returncode != 0 else "",
    })


@app.post("/container/destroy")
def container_destroy():
    """Stop and remove a container.

    Request JSON:
        container_id (str): container ID or name
    """
    data = request.get_json(force=True) or {}
    container_id = data.get("container_id")
    if not container_id:
        return jsonify({"ok": False, "error": "container_id is required"}), 400

    _docker("rm", "-f", container_id, timeout=30)
    with _lock:
        _active_containers.pop(container_id, None)
    logger.info("Destroyed container %s", container_id[:12])
    return jsonify({"ok": True})


# ── Evaluation (run test script inside a fresh container) ─────────────

@app.post("/container/evaluate")
def container_evaluate():
    """Apply a git patch and run the eval script inside the container.

    This replicates what swe_utils.evaluate_trajectory() does locally,
    but executed on the remote Docker node.

    Request JSON:
        container_id (str): container ID
        patch (str):        git diff patch to apply
        eval_script (str):  bash script to run for evaluation
        cwd (str):          working directory, default "/testbed"
        timeout (int):      eval timeout in seconds, default 3600
    """
    data = request.get_json(force=True) or {}
    container_id = data.get("container_id")
    patch = data.get("patch", "")
    eval_script = data.get("eval_script", "")
    cwd = data.get("cwd", "/testbed")
    timeout = int(data.get("timeout", 3600))

    if not container_id:
        return jsonify({"ok": False, "error": "container_id is required"}), 400
    if not _is_valid_git_patch(patch):
        return jsonify({
            "ok": True,
            "resolved": False,
            "error": "patch validation failed: expected unified git diff with diff --git headers",
        })

    delimiter = f"PATCH_{uuid.uuid4().hex}"
    # Reset the container to HEAD before applying the patch.  The model has
    # already modified files in this same container, so without the reset
    # `git apply` would see the *already-changed* working tree and fail to find
    # the original context lines it needs to apply the diff.
    apply_cmd = (
        f"git reset --hard HEAD && git clean -fd && "
        f"git apply <<'{delimiter}'\n{patch}\n{delimiter}"
    )
    r_apply = _docker(
        "exec", "-w", cwd, container_id,
        "bash", "-lc", apply_cmd,
        timeout=60,
    )
    if r_apply.returncode != 0:
        return jsonify({
            "ok": True,
            "resolved": False,
            "error": f"git apply failed: {r_apply.stderr}",
        })

    eval_cmd = f"bash <<'EOF'\n{eval_script}\nEOF"
    try:
        r_eval = _docker(
            "exec", "-w", cwd, container_id,
            "bash", "-lc", eval_cmd,
            timeout=timeout,
        )
        resolved = r_eval.returncode == 0
        return jsonify({
            "ok": True,
            "resolved": resolved,
            "returncode": r_eval.returncode,
            "output": (r_eval.stdout + r_eval.stderr)[-2000:],
        })
    except subprocess.TimeoutExpired:
        return jsonify({
            "ok": True,
            "resolved": False,
            "error": f"Evaluation timed out after {timeout}s",
        })


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SWE Docker Exec Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
    )
    logger.info("Starting SWE exec server on %s:%s", args.host, args.port)
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
