from __future__ import annotations

import asyncio
import json
import logging
import os
from functools import partial
from pathlib import Path
from typing import Any

from camel.toolkits import FunctionTool, TerminalToolkit

from terminal_bench.handlers.trial_handler import TrialHandler
from terminal_bench.parsers.base_parser import UnitTestStatus
from terminal_bench.parsers.parser_factory import ParserFactory
from terminal_bench.terminal.docker_compose_manager import DockerComposeManager
from terminal_bench.terminal.terminal import Terminal

from ..custom_types import RunContext, TaskTimeouts

from .docker_compose_utils import compose_up_no_build, prepare_task_docker_image

logger = logging.getLogger(__name__)


def _stop_terminal_compat(terminal: Terminal, timeout: float) -> None:
    try:
        terminal.stop(timeout=timeout)
    except TypeError as exc:
        if "unexpected keyword argument 'timeout'" not in str(exc):
            raise
        logger.warning(
            "Terminal.stop(timeout=...) is unsupported; retrying with Terminal.stop()."
        )
        terminal.stop()


def _drain_toolkit_sessions(toolkit: Any) -> None:
    sessions = getattr(toolkit, "shell_sessions", None)
    if not isinstance(sessions, dict):
        return
    lock = getattr(toolkit, "_session_lock", None)
    try:
        if lock is not None:
            lock.acquire()
        for session in sessions.values():
            proc = session.get("process")
            if proc is not None:
                try:
                    if hasattr(proc, "terminate"):
                        proc.terminate()
                    elif hasattr(proc, "close"):
                        proc.close()
                except Exception:
                    pass
            q = session.get("output_stream")
            if q is not None:
                try:
                    while not q.empty():
                        q.get_nowait()
                except Exception:
                    pass
        sessions.clear()
    finally:
        if lock is not None:
            try:
                lock.release()
            except RuntimeError:
                pass


class TerminalEnv:
    def __init__(self) -> None:
        self._closed = False
        self._task_meta: dict[str, Any] | None = None
        self._run_ctx: RunContext | None = None
        self._timeouts: TaskTimeouts | None = None

        self._trial_handler: TrialHandler | None = None
        self._terminal: Terminal | None = None
        self._parser = None
        self._terminal_toolkit: TerminalToolkit | None = None
        self._tools: dict[str, Any] = {}

    async def reset(
        self,
        *,
        task_meta: dict[str, Any],
        run_ctx: RunContext,
        timeouts: TaskTimeouts,
    ) -> tuple[str, list[dict[str, Any]]]:
        await self.close()

        self._closed = False
        self._task_meta = task_meta
        self._run_ctx = run_ctx
        self._timeouts = timeouts

        image_prep = await asyncio.to_thread(
            prepare_task_docker_image,
            task=task_meta,
            timeout=self._timeouts.ensure_image,
        )

        dataset_dir = str(os.getenv("DATASET_DIR", "")).strip()
        if not dataset_dir:
            raise ValueError("DATASET_DIR is required")
        task_path = Path(dataset_dir) / self._task_meta["task_path"]
        output_path = Path(self._run_ctx.log_dir).resolve()
        output_path.mkdir(parents=True, exist_ok=True)

        def _sync_reset() -> tuple[str, list[dict[str, Any]]]:
            self._trial_handler = TrialHandler(
                trial_name=f"{self._task_meta['task_name']}.{self._run_ctx.run_identity()}",
                input_path=task_path,
                output_path=output_path,
            )
            task_config = self._trial_handler.task
            self._parser = ParserFactory.get_parser(task_config.parser_name)
            client_image_name = (
                image_prep.client_image_name or self._trial_handler.client_image_name
            )

            self._terminal = Terminal(
                client_container_name=self._trial_handler.client_container_name,
                client_image_name=client_image_name,
                docker_compose_path=self._trial_handler.task_paths.docker_compose_path,
                docker_image_name_prefix=self._trial_handler.docker_image_name_prefix,
                sessions_logs_path=self._trial_handler.trial_paths.sessions_path,
                agent_logs_path=self._trial_handler.trial_paths.agent_logging_dir,
                no_rebuild=True,
                cleanup=False,
            )
            if image_prep.mode == "pull":
                compose_up_no_build(
                    self._terminal,
                    timeout=self._timeouts.reset_session,
                    container_name=self._trial_handler.client_container_name,
                    logger=logger,
                )
            else:
                try:
                    self._terminal.start(timeout=self._timeouts.reset_session)
                except TypeError:
                    self._terminal.start()
                try:
                    from .docker_compose_utils import (
                        _DEFAULT_CONTAINER_MEMORY_LIMIT,
                        _apply_container_memory_limit,
                    )

                    _apply_container_memory_limit(
                        self._trial_handler.client_container_name,
                        _DEFAULT_CONTAINER_MEMORY_LIMIT,
                        logger=logger,
                    )
                except Exception:
                    pass

            session_logs_dir = (
                self._trial_handler.trial_paths.sessions_path
                / "terminal_toolkit_session_logs"
            )
            self._terminal_toolkit = TerminalToolkit(
                timeout=20.0,
                working_directory=None,
                use_docker_backend=True,
                docker_container_name=self._trial_handler.client_container_name,
                session_logs_dir=session_logs_dir,
                safe_mode=False,
            )
            self._tools = {
                "shell_exec": self._terminal_toolkit.shell_exec,
                "shell_view": self._terminal_toolkit.shell_view,
                "shell_write_to_process": self._terminal_toolkit.shell_write_to_process,
                "shell_write_content_to_file": self._terminal_toolkit.shell_write_content_to_file,
            }

            user_msg = (
                f"Task name:{self._task_meta['task_name']}\n"
                f"Task instruction: {self._task_meta['instruction']}"
            )
            function_tools = [FunctionTool(fn) for fn in self._tools.values()]
            tool_schemas = [
                func_tool.get_openai_tool_schema() for func_tool in function_tools
            ]
            return user_msg, tool_schemas

        return await asyncio.to_thread(_sync_reset)

    async def exec_tool(self, name: str, arguments: dict[str, Any]) -> str:
        if not self._tools:
            raise RuntimeError("env is not initialized; call reset first")

        if name not in self._tools:
            return f"[TOOL_ERROR] unknown tool: {name}"

        fn = self._tools[name]

        try:
            if asyncio.iscoroutinefunction(fn):
                result = await fn(**arguments)
            elif hasattr(fn, "async_call") and callable(fn.async_call):
                result = await fn.async_call(**arguments)
            else:
                result = await asyncio.to_thread(partial(fn, **arguments))
        except Exception as exc:
            return f"[TOOL_ERROR] {name}: {type(exc).__name__}: {exc}"

        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=False)

    async def evaluate(self) -> float:
        if (
            self._trial_handler is None
            or self._terminal is None
            or self._parser is None
            or self._timeouts is None
        ):
            raise RuntimeError("env is not initialized; call reset first")

        def _sync_eval() -> float:
            task_name = (
                self._task_meta["task_name"]
                if self._task_meta is not None
                else "unknown"
            )
            paths: list[Path] = [self._trial_handler.task_paths.run_tests_path]
            if self._trial_handler.task_paths.test_dir.exists():
                paths.append(self._trial_handler.task_paths.test_dir)

            self._terminal.copy_to_container(
                paths=paths,
                container_dir=str(DockerComposeManager.CONTAINER_TEST_DIR),
            )

            test_session = self._terminal.create_session(
                "tests",
                is_active_stream=False,
                as_configured_user=False,
            )
            test_script_path = str(
                DockerComposeManager.CONTAINER_TEST_DIR / "run-tests.sh"
            )
            test_timeout_sec = min(
                self._timeouts.eval,
                4 * self._trial_handler.task.max_test_timeout_sec,
            )
            try:
                test_session.send_keys(
                    [f"bash {test_script_path}", "Enter"],
                    block=True,
                    max_timeout_sec=test_timeout_sec,
                )
            except TimeoutError as exc:
                logger.warning(
                    "Evaluation tests timed out for task=%s after %.1fs.",
                    task_name,
                    test_timeout_sec,
                )
                raise RuntimeError(
                    f"Evaluation tests timed out for task={task_name} after {test_timeout_sec:.1f}s"
                ) from exc

            test_output = test_session.capture_pane(capture_entire=True)
            try:
                parser_results = self._parser.parse(test_output)
            except Exception as exc:
                tail = test_output[-2000:] if test_output else ""
                logger.warning(
                    "Failed to parse test output for task=%s with parser=%s: %s. Output tail:\n%s",
                    task_name,
                    type(self._parser).__name__,
                    exc,
                    tail,
                )
                raise RuntimeError(
                    f"Failed to parse test output for task={task_name} with parser={type(self._parser).__name__}: {exc}"
                ) from exc

            if not parser_results:
                return 0.0
            passed = sum(
                1
                for status in parser_results.values()
                if status == UnitTestStatus.PASSED
            )
            reward = (
                float(passed / len(parser_results)) if len(parser_results) > 0 else 0.0
            )
            return reward

        return await asyncio.wait_for(
            asyncio.to_thread(_sync_eval),
            timeout=self._timeouts.eval + 30.0,
        )

    async def close(self) -> None:
        trial_name = (
            self._trial_handler.trial_name
            if self._trial_handler is not None
            else "unknown"
        )
        if self._closed:
            logger.warning("TerminalEnv %s already closed", trial_name)
            return
        self._closed = True

        terminal = self._terminal
        timeouts = self._timeouts
        toolkit = self._terminal_toolkit

        self._tools = {}
        self._terminal = None
        self._trial_handler = None
        self._parser = None
        self._terminal_toolkit = None
        self._task_meta = None
        self._run_ctx = None
        self._timeouts = None

        if toolkit is not None:
            try:
                await asyncio.to_thread(toolkit.cleanup)
            except Exception:
                logger.exception(
                    "Failed to cleanup terminal toolkit for %s", trial_name
                )
            try:
                await asyncio.to_thread(_drain_toolkit_sessions, toolkit)
            except Exception:
                logger.exception("Failed to drain toolkit sessions for %s", trial_name)

        if terminal is not None and timeouts is not None:
            try:
                await asyncio.to_thread(
                    _stop_terminal_compat, terminal, timeouts.close_session
                )
                logger.info("TerminalEnv %s closed", trial_name)
            except Exception:
                logger.exception("Failed to stop terminal session during close")
