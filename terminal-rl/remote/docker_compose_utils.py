from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from terminal_bench.handlers.trial_handler import TrialHandler
from terminal_bench.terminal.docker_compose_manager import DockerComposeManager
from terminal_bench.terminal.terminal import Terminal


@dataclass(frozen=True)
class ImagePreparationResult:
    mode: Literal["build", "pull"]
    client_image_name: str | None = None


def _shorten_output(text: str | None, max_chars: int = 4000) -> str:
    if not text:
        return ""
    stripped = text.strip()
    if len(stripped) <= max_chars:
        return stripped
    return f"{stripped[:max_chars]}...(truncated, total={len(stripped)} chars)"


def _build_docker_pull_error_message(
    *,
    image: str,
    cmd: list[str],
    return_code: int | None = None,
    timeout: float | None = None,
    stdout: str | None = None,
    stderr: str | None = None,
) -> str:
    lines = [
        f"Docker image pull failed for image '{image}'.",
        f"Command: {' '.join(cmd)}",
    ]
    if return_code is not None:
        lines.append(f"Exit code: {return_code}")
    if timeout is not None:
        lines.append(f"Timeout: {timeout:.1f}s")

    out = _shorten_output(stdout)
    err = _shorten_output(stderr)
    if out:
        lines.append(f"STDOUT:\n{out}")
    if err:
        lines.append(f"STDERR:\n{err}")

    lines.append(
        "Hints: verify task_name/task image tag, run docker login for the registry, and ensure image exists."
    )
    return "\n".join(lines)


def _build_compose_up_error_message(
    *,
    cmd: list[str],
    return_code: int | None = None,
    timeout: float | None = None,
    stdout: str | None = None,
    stderr: str | None = None,
    note: str | None = None,
) -> str:
    lines = [
        "Docker compose up --no-build failed.",
        f"Command: {' '.join(cmd)}",
    ]
    if return_code is not None:
        lines.append(f"Exit code: {return_code}")
    if timeout is not None:
        lines.append(f"Timeout: {timeout:.1f}s")

    out = _shorten_output(stdout)
    err = _shorten_output(stderr)
    if out:
        lines.append(f"STDOUT:\n{out}")
    if err:
        lines.append(f"STDERR:\n{err}")
    if note:
        lines.append(f"Note: {note}")

    lines.append(
        "Hints: verify `docker compose version`; if unavailable, install Compose plugin or ensure `docker-compose` is on PATH."
    )
    return "\n".join(lines)


def _compose_plugin_maybe_missing(stderr: str | None) -> bool:
    if not stderr:
        return False
    lowered = stderr.lower()
    return (
        "unknown shorthand flag: 'p' in -p" in lowered
        or "docker: 'compose' is not a docker command" in lowered
        or 'unknown command "compose"' in lowered
    )


def build_docker_image(task: dict[str, Any], timeout: float = 1200.0) -> None:
    dataset_dir = str(os.getenv("DATASET_DIR", "")).strip()
    if not dataset_dir:
        raise ValueError("DATASET_DIR is required")

    task_path = Path(dataset_dir) / str(task.get("task_path", ""))
    trial_handler = TrialHandler(
        trial_name="build_run",
        input_path=task_path,
        output_path=Path("build_outputs"),
    )

    compose_manager = DockerComposeManager(
        client_container_name=trial_handler.client_container_name,
        client_image_name=trial_handler.client_image_name,
        docker_image_name_prefix=trial_handler.docker_image_name_prefix,
        docker_compose_path=trial_handler.task_paths.docker_compose_path,
        no_rebuild=True,
        cleanup=False,
        sessions_logs_path=trial_handler.trial_paths.sessions_path,
        agent_logs_path=trial_handler.trial_paths.agent_logging_dir,
    )
    try:
        compose_manager.build(timeout=timeout)
    except TypeError:
        compose_manager.build()


def _resolve_pull_image(task: dict[str, Any]) -> str:
    prefix = str(os.getenv("TBENCH_DOCKER_PULL_PREFIX", "")).strip()
    if not prefix:
        raise ValueError("TBENCH_DOCKER_PULL_PREFIX is required in pull mode")

    task_name = str(task.get("task_name", "")).strip()
    if not task_name:
        raise ValueError("task_name is required to resolve pull image")
    if "<" in task_name and ">" in task_name:
        raise ValueError(
            "task_name appears to still be a placeholder "
            f"('{task_name}'). Please provide a concrete task_name."
        )

    return f"{prefix}{task_name}"


def _docker_image_exists_locally(image: str, timeout: float = 30.0) -> bool:
    cmd = ["docker", "image", "inspect", image]
    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.returncode == 0


def pull_docker_image(image: str, timeout: float = 1200.0) -> None:
    if _docker_image_exists_locally(image):
        return

    cmd = ["docker", "pull", image]
    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            _build_docker_pull_error_message(
                image=image,
                cmd=cmd,
                timeout=timeout,
                stdout=exc.stdout,
                stderr=exc.stderr,
            )
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            _build_docker_pull_error_message(
                image=image,
                cmd=cmd,
                return_code=exc.returncode,
                stdout=exc.stdout,
                stderr=exc.stderr,
            )
        ) from exc


def prepare_task_docker_image(
    task: dict[str, Any],
    timeout: float = 1200.0,
) -> ImagePreparationResult:
    raw_mode = str(os.getenv("TBENCH_DOCKER_IMAGE_SOURCE", "")).strip().lower()
    if not raw_mode:
        raise ValueError("TBENCH_DOCKER_IMAGE_SOURCE is required")

    if raw_mode in {"build", "docker_build"}:
        build_docker_image(task=task, timeout=timeout)
        return ImagePreparationResult(mode="build", client_image_name=None)

    if raw_mode in {"pull", "docker_pull"}:
        image = _resolve_pull_image(task=task)
        pull_docker_image(image=image, timeout=timeout)
        return ImagePreparationResult(mode="pull", client_image_name=image)

    raise ValueError(
        f"Unsupported docker image source '{raw_mode}'. Expected one of: build, pull"
    )


_DEFAULT_CONTAINER_MEMORY_LIMIT = os.getenv("CONTAINER_MEMORY_LIMIT", "16g")


def _apply_container_memory_limit(
    container_name: str,
    memory_limit: str,
    logger: logging.Logger | None = None,
) -> None:
    """Best-effort ``docker update --memory`` on a running container."""
    if not memory_limit:
        return
    cmd = [
        "docker",
        "update",
        f"--memory={memory_limit}",
        f"--memory-swap={memory_limit}",
        container_name,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=30.0)
        if logger is not None:
            logger.info(
                "Applied memory limit %s to container %s", memory_limit, container_name
            )
    except Exception as exc:
        if logger is not None:
            logger.warning(
                "Failed to apply memory limit to container %s: %s", container_name, exc
            )


def compose_up_no_build(
    terminal: Terminal,
    *,
    timeout: float,
    container_name: str,
    logger: logging.Logger | None = None,
) -> None:
    compose_manager = getattr(terminal, "_compose_manager")
    compose_override_path = str(os.getenv("COMPOSE_OVERRIDE_PATH", "")).strip()
    compose_command = ["up", "-d", "--no-build"]
    if compose_override_path:
        compose_command = ["-f", compose_override_path, *compose_command]

    command = compose_manager.get_docker_compose_command(compose_command)
    if logger is not None:
        logger.info("Running docker compose command: %s", " ".join(command))
        if compose_override_path:
            logger.info("Using compose override file: %s", compose_override_path)

    compose_manager.env["http_proxy"] = os.getenv("HTTP_PROXY", "")
    compose_manager.env["https_proxy"] = os.getenv("HTTPS_PROXY", "")

    try:
        subprocess.run(
            command,
            env=compose_manager.env,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        if logger is not None:
            logger.error(
                "Docker compose up --no-build timed out after %.1f sec", timeout
            )
            if exc.stdout:
                logger.error("STDOUT: %s", exc.stdout)
            if exc.stderr:
                logger.error("STDERR: %s", exc.stderr)
        raise RuntimeError(
            _build_compose_up_error_message(
                cmd=command,
                timeout=timeout,
                stdout=exc.stdout,
                stderr=exc.stderr,
            )
        ) from exc
    except subprocess.CalledProcessError as exc:
        if logger is not None:
            logger.error(
                "Docker compose up --no-build failed with code %s", exc.returncode
            )
            if exc.stdout:
                logger.error("STDOUT: %s", exc.stdout)
            if exc.stderr:
                logger.error("STDERR: %s", exc.stderr)
        if _compose_plugin_maybe_missing(exc.stderr):
            fallback_command = ["docker-compose", *command[2:]]
            if shutil.which("docker-compose"):
                if logger is not None:
                    logger.warning(
                        "docker compose plugin may be unavailable; falling back to: %s",
                        " ".join(fallback_command),
                    )
                try:
                    subprocess.run(
                        fallback_command,
                        env=compose_manager.env,
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                    )
                except subprocess.TimeoutExpired as fallback_exc:
                    raise RuntimeError(
                        _build_compose_up_error_message(
                            cmd=fallback_command,
                            timeout=timeout,
                            stdout=fallback_exc.stdout,
                            stderr=fallback_exc.stderr,
                            note=f"Initial command {' '.join(command)} failed with exit code {exc.returncode}.",
                        )
                    ) from fallback_exc
                except subprocess.CalledProcessError as fallback_exc:
                    raise RuntimeError(
                        _build_compose_up_error_message(
                            cmd=fallback_command,
                            return_code=fallback_exc.returncode,
                            stdout=fallback_exc.stdout,
                            stderr=fallback_exc.stderr,
                            note=f"Initial command {' '.join(command)} failed with exit code {exc.returncode}.",
                        )
                    ) from fallback_exc
                container = compose_manager._client.containers.get(container_name)
                terminal.container = container
                compose_manager._client_container = container
                _apply_container_memory_limit(
                    container_name, _DEFAULT_CONTAINER_MEMORY_LIMIT, logger=logger
                )
                return

            raise RuntimeError(
                _build_compose_up_error_message(
                    cmd=command,
                    return_code=exc.returncode,
                    stdout=exc.stdout,
                    stderr=exc.stderr,
                    note="Compose plugin appears unavailable and `docker-compose` binary was not found.",
                )
            ) from exc

        raise RuntimeError(
            _build_compose_up_error_message(
                cmd=command,
                return_code=exc.returncode,
                stdout=exc.stdout,
                stderr=exc.stderr,
            )
        ) from exc

    container = compose_manager._client.containers.get(container_name)
    terminal.container = container
    compose_manager._client_container = container
    _apply_container_memory_limit(
        container_name, _DEFAULT_CONTAINER_MEMORY_LIMIT, logger=logger
    )
