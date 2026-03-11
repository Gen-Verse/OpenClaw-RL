import logging
import os
import shlex
import subprocess
import uuid
from dataclasses import asdict, dataclass, field
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any


@dataclass
class DockerEnvironmentConfig:
    image: str
    cwd: str = "/"
    """Working directory in which to execute commands."""
    env: dict[str, str] = field(default_factory=dict)
    """Environment variables to set in the container."""
    forward_env: list[str] = field(default_factory=list)
    """Environment variables to forward to the container.
    Variables are only forwarded if they are set in the host environment.
    In case of conflict with `env`, the `env` variables take precedence.
    """
    timeout: int = 30
    """Timeout for executing commands in the container."""
    executable: str = os.getenv("MSWEA_DOCKER_EXECUTABLE", "docker")
    """Path to the docker/container executable."""
    run_args: list[str] = field(default_factory=lambda: ["--rm"])
    """Additional arguments to pass to the docker/container executable.
    Default is ["--rm"], which removes the container after it exits.
    """
    container_timeout: str = "2h"
    """Max duration to keep container running. Uses the same format as the sleep command."""
    pull_timeout: int = 120
    """Timeout in seconds for pulling images."""
    exec_mode: str = os.getenv("MSWEA_DOCKER_EXEC_MODE", "subprocess")
    """Command execution backend: 'subprocess' or 'api'."""
    docker_api_base_url: str = os.getenv("DOCKER_HOST", "unix:///var/run/docker.sock")
    """Docker API endpoint used when exec_mode='api'."""


class DockerEnvironment:
    def __init__(self, *, config_class: type = DockerEnvironmentConfig, logger: logging.Logger | None = None, **kwargs):
        """This class executes bash commands in a Docker container using direct docker commands.
        See `DockerEnvironmentConfig` for keyword arguments.
        """
        self.logger = logger or logging.getLogger("minisweagent.environment")
        self.container_id: str | None = None
        self.config = config_class(**kwargs)
        self._docker_api_client = None
        self._start_container()

    def get_template_vars(self) -> dict[str, Any]:
        return asdict(self.config)

    def _start_container(self):
        """Start the Docker container and return the container ID."""
        container_name = f"minisweagent-{uuid.uuid4().hex[:8]}"
        cmd = [
            self.config.executable,
            "run",
            "-d",
            "--name",
            container_name,
            "-w",
            self.config.cwd,
            *self.config.run_args,
            self.config.image,
            "sleep",
            self.config.container_timeout,
        ]
        self.logger.debug(f"Starting container with command: {shlex.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.config.pull_timeout,  # docker pull might take a while
            check=True,
        )
        self.logger.info(f"Started container {container_name} with ID {result.stdout.strip()}")
        self.container_id = result.stdout.strip()

    def execute(self, command: str, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        """Execute a command in the Docker container and return the result as a dict."""
        if self.config.exec_mode == "api":
            return self._execute_via_api(command=command, cwd=cwd, timeout=timeout)
        return self._execute_via_subprocess(command=command, cwd=cwd, timeout=timeout)

    def _execute_via_subprocess(self, command: str, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        """Execute via `docker exec` subprocess."""
        cwd = cwd or self.config.cwd
        assert self.container_id, "Container not started"

        cmd = [self.config.executable, "exec", "-i", "-w", cwd]
        for key in self.config.forward_env:
            if (value := os.getenv(key)) is not None:
                cmd.extend(["-e", f"{key}={value}"])
        for key, value in self.config.env.items():
            cmd.extend(["-e", f"{key}={value}"])
        cmd.extend([self.container_id, "bash", "-lc", command])

        result = subprocess.run(
            cmd,
            text=True,
            timeout=timeout or self.config.timeout,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return {"output": result.stdout, "returncode": result.returncode}

    def _get_api_client(self):
        if self._docker_api_client is not None:
            return self._docker_api_client
        try:
            import docker
        except ImportError as e:
            raise RuntimeError(
                "Docker SDK for Python is required for exec_mode='api'. "
                "Install it with `pip install docker`."
            ) from e
        self._docker_api_client = docker.APIClient(base_url=self.config.docker_api_base_url)
        return self._docker_api_client

    def _build_exec_env(self) -> dict[str, str]:
        env: dict[str, str] = {}
        for key in self.config.forward_env:
            if (value := os.getenv(key)) is not None:
                env[key] = value
        env.update(self.config.env)
        return env

    def _execute_via_api(self, command: str, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        """Execute via Docker Engine API using docker SDK."""
        cwd = cwd or self.config.cwd
        assert self.container_id, "Container not started"
        client = self._get_api_client()
        exec_id = client.exec_create(
            container=self.container_id,
            cmd=["bash", "-lc", command],
            tty=False,
            stdin=True,
            environment=self._build_exec_env(),
            workdir=cwd,
        )["Id"]

        timeout_s = timeout or self.config.timeout
        with ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(client.exec_start, exec_id, tty=False, stream=False, demux=False)
            try:
                out = fut.result(timeout=timeout_s)
            except FuturesTimeoutError as e:
                raise TimeoutError(f"docker API exec timed out after {timeout_s}s") from e

        inspect = client.exec_inspect(exec_id)
        exit_code = inspect.get("ExitCode")
        output = out.decode("utf-8", errors="replace") if isinstance(out, (bytes, bytearray)) else str(out)
        return {"output": output, "returncode": int(exit_code) if exit_code is not None else 1}

    def cleanup(self):
        """Stop and remove the Docker container."""
        if getattr(self, "container_id", None) is not None:  # if init fails early, container_id might not be set
            cmd = f"(timeout 60 {self.config.executable} stop {self.container_id} || {self.config.executable} rm -f {self.container_id}) >/dev/null 2>&1 &"
            subprocess.Popen(cmd, shell=True)

    def __del__(self):
        """Cleanup container when object is destroyed."""
        self.cleanup()
