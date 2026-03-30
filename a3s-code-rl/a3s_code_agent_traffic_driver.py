#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import random
import re
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _bootstrap_a3s_code() -> None:
    try:
        import a3s_code  # noqa: F401

        return
    except ImportError:
        pass

    repo_root = Path(
        os.getenv(
            "A3S_CODE_REPO_ROOT",
            str(Path(__file__).resolve().parents[2] / "a3s-lab" / "Code"),
        )
    )
    sdk_python = repo_root / "sdk" / "python"
    import sys

    version_dir = f"python{sys.version_info.major}.{sys.version_info.minor}"
    extra_sites = [
        Path(item).expanduser()
        for item in os.getenv("A3S_CODE_EXTRA_SITE_PACKAGES", "").split(":")
        if item.strip()
    ]
    candidates = [
        Path(sys.prefix) / "lib" / version_dir / "site-packages",
        Path(os.getenv("CONDA_PREFIX", "")) / "lib" / version_dir / "site-packages",
        sdk_python / ".venv" / "lib" / "python3.13" / "site-packages",
        sdk_python / ".venv" / "lib" / "python3.12" / "site-packages",
    ] + extra_sites
    for site in candidates:
        if (site / "a3s_code").exists():
            sys.path.insert(0, str(site))
            return

    raise RuntimeError(
        "a3s_code is not importable. Build it first with:\n"
        "  export PATH=/mnt/shared-storage-user/lixiangtian/.cargo/bin:$PATH\n"
        "  cd /mnt/shared-storage-user/lixiangtian/workspace/a3s-lab/Code/sdk/python\n"
        "  /mnt/shared-storage-user/lixiangtian/miniconda3/envs/openclaw-rl/bin/maturin develop --release"
    )


_bootstrap_a3s_code()
from a3s_code import Agent, SessionOptions  # noqa: E402


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "seed_data"
TEMPLATE_ROOT = SCRIPT_DIR / "task_templates"
WORKSPACE_ROOT = Path(os.getenv("A3S_CODE_WORKSPACE_ROOT", str(SCRIPT_DIR / "generated_workspaces")))
CONFIG_ROOT = Path(os.getenv("A3S_CODE_CONFIG_ROOT", str(SCRIPT_DIR / "generated_configs")))
WORKSPACE_TEMPLATE_CACHE_ROOT = Path(
    os.getenv("A3S_CODE_WORKSPACE_TEMPLATE_CACHE_ROOT", str(SCRIPT_DIR / "workspace_template_cache"))
)
RESULTS_DIR = Path(os.getenv("A3S_CODE_RESULTS_DIR", str(SCRIPT_DIR / "results")))
RECORD_FILE = Path(
    os.getenv("A3S_CODE_TRAFFIC_RECORD_FILE", str(RESULTS_DIR / "a3s_code_agent_traffic.jsonl"))
)
SIMULATED_USER_BACKENDS_FILE = Path(
    os.getenv(
        "A3S_CODE_SIMULATED_USER_BACKENDS_FILE",
        str(SCRIPT_DIR / "simulated_user_backends.json"),
    )
)

RL_BASE_URL = os.getenv("RL_BASE_URL", "http://127.0.0.1:30000").rstrip("/")
A3S_MODEL_NAME = os.getenv("A3S_MODEL_NAME", "qwen3-4b-2507")
SIMULATED_USER_MODEL_URL = os.getenv(
    "SIMULATED_USER_MODEL_URL",
    "",
)
SIMULATED_USER_MODEL_NAME = os.getenv(
    "SIMULATED_USER_MODEL_NAME",
    "kimi-k2.5",
)
SIMULATED_USER_API_KEY = os.getenv(
    "SIMULATED_USER_API_KEY",
    "",
)
A3S_API_KEY = os.getenv("A3S_API_KEY", os.getenv("SGLANG_API_KEY", "apiKey"))

CONCURRENCY = int(os.getenv("A3S_CODE_TRAFFIC_CONCURRENCY", "1"))
SESSION_LIMIT = int(os.getenv("A3S_CODE_TRAFFIC_SESSION_LIMIT", "0"))
MAX_MAIN_TURNS = max(2, int(os.getenv("A3S_CODE_MAX_MAIN_TURNS", "3")))
SESSION_DELAY_SEC = float(os.getenv("A3S_CODE_SESSION_DELAY_SEC", "0.5"))
SIMULATED_USER_TIMEOUT_SEC = float(
    os.getenv("A3S_CODE_SIMULATED_USER_TIMEOUT_SEC", "45")
)
SIMULATED_USER_BACKEND_COOLDOWN_SEC = float(
    os.getenv("A3S_CODE_SIMULATED_USER_BACKEND_COOLDOWN_SEC", "60")
)
SIMULATED_USER_MAX_ATTEMPTS = int(
    os.getenv("A3S_CODE_SIMULATED_USER_MAX_ATTEMPTS", "0")
)
REQUEST_TIMEOUT_SEC = float(os.getenv("A3S_CODE_REQUEST_TIMEOUT_SEC", "600"))
KEEP_WORKSPACES = _env_flag("A3S_CODE_KEEP_WORKSPACES", False)
KEEP_WORKSPACES_ON_ERROR = _env_flag("A3S_CODE_KEEP_WORKSPACES_ON_ERROR", KEEP_WORKSPACES)
KEEP_CONFIGS = _env_flag("A3S_CODE_KEEP_CONFIGS", False)
AGENT_CONFIG_MODE = os.getenv("A3S_CODE_AGENT_CONFIG_MODE", "shared").strip().lower()
if AGENT_CONFIG_MODE not in {"shared", "per_session"}:
    raise RuntimeError(
        "A3S_CODE_AGENT_CONFIG_MODE must be 'shared' or 'per_session', "
        f"got {AGENT_CONFIG_MODE!r}"
    )
SESSION_ID_HEADER_NAME = os.getenv("A3S_CODE_SESSION_ID_HEADER_NAME", "X-Session-Id").strip()
if not SESSION_ID_HEADER_NAME:
    SESSION_ID_HEADER_NAME = "X-Session-Id"
SHARED_CONFIG_NAME = os.getenv("A3S_CODE_SHARED_CONFIG_NAME", "a3s-code-shared.hcl").strip()
if not SHARED_CONFIG_NAME:
    SHARED_CONFIG_NAME = "a3s-code-shared.hcl"
WORKSPACE_COPY_MODE = os.getenv("A3S_CODE_WORKSPACE_COPY_MODE", "reflink_auto").strip().lower()
MAX_TOOL_ROUNDS = int(os.getenv("A3S_CODE_MAX_TOOL_ROUNDS", "8"))
TURN_TIMEOUT_SEC = float(os.getenv("A3S_CODE_TURN_TIMEOUT_SEC", "240"))
TOOL_TIMEOUT_MS = int(os.getenv("A3S_CODE_TOOL_TIMEOUT_MS", "240000"))
MAX_PARSE_RETRIES = int(os.getenv("A3S_CODE_MAX_PARSE_RETRIES", "4"))
CIRCUIT_BREAKER_THRESHOLD = int(os.getenv("A3S_CODE_CIRCUIT_BREAKER_THRESHOLD", "5"))
AUTO_COMPACT = _env_flag("A3S_CODE_AUTO_COMPACT", True)
AUTO_COMPACT_THRESHOLD = float(os.getenv("A3S_CODE_AUTO_COMPACT_THRESHOLD", "0.85"))
THINKING_BUDGET = int(os.getenv("A3S_CODE_THINKING_BUDGET", "12000"))
CONTINUATION_ENABLED = _env_flag("A3S_CODE_CONTINUATION_ENABLED", True)
MAX_CONTINUATION_TURNS = int(os.getenv("A3S_CODE_MAX_CONTINUATION_TURNS", "5"))
MODEL_CONTEXT_TOKENS = int(
    os.getenv(
        "A3S_CODE_MODEL_CONTEXT_TOKENS",
        str(
            max(
                1024,
                int(os.getenv("CODE_RL_MATCHED_CONTEXT_TOKENS", "8192"))
                - int(os.getenv("A3S_CODE_CONTEXT_HEADROOM_TOKENS", "1024")),
            )
        ),
    )
)
MODEL_OUTPUT_TOKENS = int(os.getenv("A3S_CODE_MODEL_OUTPUT_TOKENS", "2048"))
GIT_USER_NAME = os.getenv("A3S_CODE_GIT_USER_NAME", "A3S Code RL")
GIT_USER_EMAIL = os.getenv("A3S_CODE_GIT_USER_EMAIL", "a3s-code-rl@example.com")
EXCLUDED_SEED_TAGS = {
    tag.strip()
    for tag in os.getenv("A3S_CODE_EXCLUDE_SEED_TAGS", "advanced").split(",")
    if tag.strip()
}

RECORD_LOCK = threading.Lock()
COUNTER_LOCK = threading.Lock()
TEMPLATE_CACHE_LOCK = threading.Lock()
CONFIG_BUILD_LOCK = threading.Lock()
SESSION_COUNTER = 0


@dataclass
class SimulatedUserBackend:
    url: str
    model: str
    api_key: str
    label: str
    priority: int = 100
    cooldown_until: float = 0.0
    failures: int = 0


@dataclass(frozen=True)
class SeedTask:
    seed_id: str
    template: str
    seed: str
    acceptance: list[str]
    tags: list[str]


@dataclass(frozen=True)
class TemplateMeta:
    template: str
    repo_summary: str
    key_files: list[str]
    test_command: str
    realism_notes: list[str]


class SimulatedUserBackendPool:
    def __init__(
        self,
        backends: list[SimulatedUserBackend],
        *,
        cooldown_sec: float,
        max_attempts: int,
    ) -> None:
        self._backends = backends
        self._cooldown_sec = max(0.0, cooldown_sec)
        self._max_attempts = max(0, max_attempts)
        self._lock = threading.Lock()
        self._next_index_by_priority: dict[int, int] = {}

    @property
    def backends(self) -> list[SimulatedUserBackend]:
        return self._backends

    def candidate_order(self) -> list[SimulatedUserBackend]:
        with self._lock:
            if not self._backends:
                return []
            now = time.monotonic()
            ready = [backend for backend in self._backends if backend.cooldown_until <= now]
            if not ready:
                ready = list(self._backends)
            grouped: dict[int, list[SimulatedUserBackend]] = {}
            for backend in ready:
                grouped.setdefault(backend.priority, []).append(backend)
            ordered: list[SimulatedUserBackend] = []
            for priority in sorted(grouped):
                group = grouped[priority]
                start = self._next_index_by_priority.get(priority, 0) % len(group)
                self._next_index_by_priority[priority] = (start + 1) % len(group)
                ordered.extend(group[start:] + group[:start])
            if self._max_attempts > 0:
                ordered = ordered[: self._max_attempts]
            return ordered

    def mark_success(self, backend: SimulatedUserBackend) -> None:
        with self._lock:
            backend.failures = 0
            backend.cooldown_until = 0.0

    def mark_failure(self, backend: SimulatedUserBackend) -> None:
        with self._lock:
            backend.failures += 1
            if self._cooldown_sec > 0:
                backend.cooldown_until = time.monotonic() + self._cooldown_sec


def _split_csv_env(raw: str, *, keep_empty: bool = False) -> list[str]:
    if raw == "":
        return []
    items = [item.strip() for item in raw.split(",")]
    if keep_empty:
        return items
    return [item for item in items if item]


def _make_backend_label(index: int, model: str, url: str) -> str:
    return f"{index}:{model}@{url.replace('http://', '').replace('https://', '')}"


def _load_simulated_user_backends_from_config(
    path: Path,
) -> tuple[bool, list[SimulatedUserBackend]]:
    if not path.exists():
        return False, []

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(
            f"[a3s-code-driver] simulated_user config unreadable path={path} "
            f"error={type(exc).__name__}: {exc}",
            flush=True,
        )
        return True, []

    raw_backends = payload.get("backends", []) if isinstance(payload, dict) else []
    if not isinstance(raw_backends, list):
        print(
            f"[a3s-code-driver] simulated_user config malformed path={path} backends_type={type(raw_backends).__name__}",
            flush=True,
        )
        return True, []

    backends: list[SimulatedUserBackend] = []
    for index, item in enumerate(raw_backends, start=1):
        if not isinstance(item, dict):
            continue
        enabled = bool(item.get("enabled", True))
        healthy = bool(item.get("healthy", item.get("probe", {}).get("chat_ok", True)))
        if not enabled or not healthy:
            continue
        url = str(item.get("url", "")).strip()
        model = str(item.get("model", "")).strip()
        if not url or not model:
            continue
        api_key = str(item.get("api_key", "") or "")
        label = str(item.get("label", "")).strip() or _make_backend_label(index, model, url)
        priority = int(item.get("priority", 100) or 100)
        backends.append(
            SimulatedUserBackend(
                url=url,
                model=model,
                api_key=api_key,
                label=label,
                priority=priority,
            )
        )
    return True, backends


def _expand_backend_values(values: list[str], size: int, default: str) -> list[str]:
    if size <= 0:
        return []
    if not values:
        values = [default]
    if len(values) == 1 and size > 1:
        return values * size
    if len(values) != size:
        raise RuntimeError(
            f"simulated-user backend config length mismatch: expected 1 or {size}, got {len(values)}"
        )
    return values


def _build_simulated_user_backends() -> list[SimulatedUserBackend]:
    loaded_from_config, config_backends = _load_simulated_user_backends_from_config(
        SIMULATED_USER_BACKENDS_FILE
    )
    if loaded_from_config:
        return config_backends

    urls = _split_csv_env(os.getenv("SIMULATED_USER_MODEL_URLS", ""))
    if not urls and SIMULATED_USER_MODEL_URL:
        urls = [SIMULATED_USER_MODEL_URL]
    if not urls:
        return []

    names = _split_csv_env(os.getenv("SIMULATED_USER_MODEL_NAMES", ""))
    keys = _split_csv_env(os.getenv("SIMULATED_USER_API_KEYS", ""), keep_empty=True)
    names = _expand_backend_values(names, len(urls), SIMULATED_USER_MODEL_NAME)
    keys = _expand_backend_values(keys, len(urls), SIMULATED_USER_API_KEY)

    backends: list[SimulatedUserBackend] = []
    for index, (url, model, api_key) in enumerate(zip(urls, names, keys), start=1):
        backends.append(
            SimulatedUserBackend(
                url=url,
                model=model,
                api_key=api_key,
                label=_make_backend_label(index, model, url),
                priority=100,
            )
        )
    return backends


def _resolve_simulated_user_backends() -> tuple[list[SimulatedUserBackend], str]:
    loaded_from_config, config_backends = _load_simulated_user_backends_from_config(
        SIMULATED_USER_BACKENDS_FILE
    )
    if loaded_from_config:
        if config_backends:
            return config_backends, f"config:{SIMULATED_USER_BACKENDS_FILE}"
        return [], f"config-empty:{SIMULATED_USER_BACKENDS_FILE}"

    env_backends = _build_simulated_user_backends()
    if env_backends:
        return env_backends, "env"
    return [], "fallback-only"


SIMULATED_USER_BACKENDS, SIMULATED_USER_BACKENDS_SOURCE = _resolve_simulated_user_backends()
SIMULATED_USER_POOL = SimulatedUserBackendPool(
    SIMULATED_USER_BACKENDS,
    cooldown_sec=SIMULATED_USER_BACKEND_COOLDOWN_SEC,
    max_attempts=SIMULATED_USER_MAX_ATTEMPTS,
)


def _ensure_dirs() -> None:
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    WORKSPACE_TEMPLATE_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.parent / (
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp"
    )
    temp_path.write_text(content, encoding="utf-8")
    temp_path.replace(path)


def _render_agent_config(base_url: str, *, session_id_header: str | None = None) -> str:
    session_header_line = ""
    if session_id_header:
        session_header_line = f'  sessionIdHeader = "{session_id_header}"\n'
    return (
        f'default_model = "openai/{A3S_MODEL_NAME}"\n\n'
        "providers {\n"
        '  name = "openai"\n'
        f'  api_key = "{A3S_API_KEY}"\n'
        f'  base_url = "{base_url}"\n'
        f"{session_header_line}\n"
        "  models {\n"
        f'    id = "{A3S_MODEL_NAME}"\n'
        f'    name = "{A3S_MODEL_NAME}"\n'
        "    tool_call = true\n\n"
        "    limit {\n"
        f"      context = {MODEL_CONTEXT_TOKENS}\n"
        f"      output  = {MODEL_OUTPUT_TOKENS}\n"
        "    }\n"
        "  }\n"
        "}\n"
    )


def _load_seed_tasks() -> list[SeedTask]:
    raw = json.loads((DATA_DIR / "code_task_seeds.json").read_text(encoding="utf-8"))
    seeds = [
        SeedTask(
            seed_id=str(item["id"]),
            template=str(item["template"]),
            seed=str(item["seed"]),
            acceptance=[str(x) for x in item.get("acceptance", [])],
            tags=[str(x) for x in item.get("tags", [])],
        )
        for item in raw
    ]
    filtered = [
        seed for seed in seeds if not EXCLUDED_SEED_TAGS.intersection(set(seed.tags))
    ]
    return filtered or seeds


def _load_template_meta(template_name: str) -> TemplateMeta:
    meta_path = TEMPLATE_ROOT / template_name / "template_meta.json"
    item = json.loads(meta_path.read_text(encoding="utf-8"))
    return TemplateMeta(
        template=str(item["template"]),
        repo_summary=str(item["repo_summary"]),
        key_files=[str(x) for x in item.get("key_files", [])],
        test_command=str(item.get("test_command", "pytest -q")),
        realism_notes=[str(x) for x in item.get("realism_notes", [])],
    )


def _append_record(payload: dict[str, Any]) -> None:
    with RECORD_LOCK:
        with RECORD_FILE.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _next_session_index() -> int | None:
    global SESSION_COUNTER
    with COUNTER_LOCK:
        if SESSION_LIMIT and SESSION_COUNTER >= SESSION_LIMIT:
            return None
        SESSION_COUNTER += 1
        return SESSION_COUNTER


def _extract_text(resp_json: dict[str, Any]) -> str:
    return str(resp_json.get("choices", [{}])[0].get("message", {}).get("content", "") or "")


def _extract_json_obj(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def _chat_completion(
    client: httpx.Client,
    url: str,
    *,
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.6,
    max_tokens: int = 1024,
    api_key: str = "",
) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    resp = client.post(url, headers=headers, json=body, timeout=SIMULATED_USER_TIMEOUT_SEC)
    resp.raise_for_status()
    return resp.json()


def _simulated_user_chat_completion(
    client: httpx.Client,
    *,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    if not SIMULATED_USER_POOL.backends:
        raise RuntimeError("no simulated-user backends configured")

    last_exc: Exception | None = None
    for backend in SIMULATED_USER_POOL.candidate_order():
        try:
            resp = _chat_completion(
                client,
                backend.url,
                model=backend.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=backend.api_key,
            )
            SIMULATED_USER_POOL.mark_success(backend)
            return resp
        except Exception as exc:
            SIMULATED_USER_POOL.mark_failure(backend)
            last_exc = exc
            if len(SIMULATED_USER_POOL.backends) > 1:
                print(
                    f"[a3s-code-driver] simulated_user backend_fail "
                    f"backend={backend.label} error={type(exc).__name__}: {exc}",
                    flush=True,
                )

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("simulated-user request failed without a concrete backend error")


def _rewrite_seed_task(client: httpx.Client, seed: SeedTask, template_meta: TemplateMeta) -> str:
    system = (
        "You rewrite terse issue notes into realistic user requests for an autonomous coding agent. "
        "Keep the request natural, concrete, and scoped to the repo summary. "
        "Mention testing or verification when appropriate. "
        "Return strict JSON only with keys user_request and success_checks."
    )
    user = (
        f"Repo summary:\n{template_meta.repo_summary}\n\n"
        f"Key files:\n- " + "\n- ".join(template_meta.key_files) + "\n\n"
        f"Suggested test command: {template_meta.test_command}\n\n"
        f"Realism notes:\n- " + "\n- ".join(template_meta.realism_notes) + "\n\n"
        f"Seed task: {seed.seed}\n"
        f"Acceptance hints: {json.dumps(seed.acceptance, ensure_ascii=False)}\n"
        f"Tags: {json.dumps(seed.tags, ensure_ascii=False)}"
    )
    try:
        resp = _simulated_user_chat_completion(
            client,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.7,
            max_tokens=900,
        )
        parsed = _extract_json_obj(_extract_text(resp)) or {}
        prompt = str(parsed.get("user_request", "")).strip()
        if prompt:
            return prompt
    except Exception as exc:
        print(
            f"[a3s-code-driver] simulated_user rewrite fallback "
            f"seed={seed.seed_id} error={type(exc).__name__}: {exc}",
            flush=True,
        )

    acceptance = "; ".join(seed.acceptance[:3])
    return (
        f"I'm working in a small repo for {template_meta.repo_summary.lower()}. "
        f"Please handle this request: {seed.seed}. "
        f"Keep the change scoped, update tests or docs if they are affected, "
        f"and tell me how you verified it. Acceptance notes: {acceptance}."
    )


def _generate_followup(
    client: httpx.Client,
    *,
    seed: SeedTask,
    template_meta: TemplateMeta,
    original_request: str,
    latest_response: str,
    next_turn_number: int,
    is_final_turn: bool,
) -> tuple[str, bool]:
    system = (
        "You simulate a realistic engineer following up on a coding task. "
        "Look at the original request and the latest assistant message, then write the next user message only. "
        "Usually ask for one concrete verification step, edge case, or small polish item. "
        "If this is the final turn, the reply should close the loop naturally. "
        "Return strict JSON with keys reply and done."
    )
    user = (
        f"Repo summary:\n{template_meta.repo_summary}\n\n"
        f"Original request:\n{original_request}\n\n"
        f"Latest assistant response:\n{latest_response}\n\n"
        f"Next user turn number: {next_turn_number}\n"
        f"Final turn: {json.dumps(is_final_turn)}\n"
        f"Suggested test command: {template_meta.test_command}\n"
        f"Acceptance hints: {json.dumps(seed.acceptance, ensure_ascii=False)}"
    )

    try:
        resp = _simulated_user_chat_completion(
            client,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.5,
            max_tokens=500,
        )
        parsed = _extract_json_obj(_extract_text(resp)) or {}
        reply = str(parsed.get("reply", "")).strip()
        done = bool(parsed.get("done", False))
        if reply:
            if is_final_turn:
                done = True
            return reply, done
    except Exception as exc:
        print(
            f"[a3s-code-driver] simulated_user followup fallback "
            f"turn={next_turn_number} seed={seed.seed_id} error={type(exc).__name__}: {exc}",
            flush=True,
        )

    if is_final_turn:
        return (
            f"Looks good. Please do one final verification pass, mention the exact command you ran "
            f"({template_meta.test_command}), and summarize what changed.",
            True,
        )
    return (
        "Please run the relevant verification, cover one edge case if it is missing, "
        "and update any user-facing docs or help text that changed.",
        False,
    )


def _copy_dir(src: Path, dst: Path, *, mode: str, strip_template_meta: bool = False) -> None:
    if dst.exists():
        shutil.rmtree(dst)

    normalized = mode.strip().lower()
    reflink_mode = None
    if normalized == "reflink_auto":
        reflink_mode = "auto"
    elif normalized == "reflink_always":
        reflink_mode = "always"

    copied = False
    if reflink_mode is not None:
        dst.mkdir(parents=True, exist_ok=True)
        cmd = ["cp", "-a", f"--reflink={reflink_mode}", f"{src}/.", str(dst)]
        result = subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        copied = result.returncode == 0
        if not copied:
            shutil.rmtree(dst, ignore_errors=True)

    if not copied:
        shutil.copytree(src, dst)

    if strip_template_meta:
        meta = dst / "template_meta.json"
        if meta.exists():
            meta.unlink()


def _prepare_workspace_template_cache(template_name: str) -> Path:
    src = TEMPLATE_ROOT / template_name
    if not src.exists():
        raise FileNotFoundError(f"template {template_name!r} not found at {src}")
    cache_dir = WORKSPACE_TEMPLATE_CACHE_ROOT / template_name
    with TEMPLATE_CACHE_LOCK:
        if cache_dir.exists():
            return cache_dir

        tmp_dir = WORKSPACE_TEMPLATE_CACHE_ROOT / f".{template_name}.tmp-{uuid.uuid4().hex[:8]}"
        try:
            _copy_dir(src, tmp_dir, mode=WORKSPACE_COPY_MODE, strip_template_meta=True)
            _init_git_repo(tmp_dir)
            tmp_dir.rename(cache_dir)
        except FileExistsError:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise
        return cache_dir


def _copy_workspace_template(template_name: str, session_id: str) -> Path:
    src = _prepare_workspace_template_cache(template_name)
    dst = WORKSPACE_ROOT / session_id
    _copy_dir(src, dst, mode=WORKSPACE_COPY_MODE)
    return dst


def _init_git_repo(workspace: Path) -> None:
    commands = [
        ["git", "init", "-q"],
        ["git", "add", "."],
        [
            "git",
            "-c",
            f"user.name={GIT_USER_NAME}",
            "-c",
            f"user.email={GIT_USER_EMAIL}",
            "commit",
            "-qm",
            "template baseline",
        ],
    ]
    for cmd in commands:
        subprocess.run(
            cmd,
            cwd=workspace,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def _build_agent_config(session_id: str) -> Path:
    if AGENT_CONFIG_MODE == "shared":
        config_path = CONFIG_ROOT / SHARED_CONFIG_NAME
        config_text = _render_agent_config(RL_BASE_URL, session_id_header=SESSION_ID_HEADER_NAME)
        with CONFIG_BUILD_LOCK:
            if not config_path.exists() or config_path.read_text(encoding="utf-8") != config_text:
                _write_text_atomic(config_path, config_text)
        return config_path

    config_path = CONFIG_ROOT / f"{session_id}.hcl"
    _write_text_atomic(
        config_path,
        _render_agent_config(f"{RL_BASE_URL}/session/{session_id}"),
    )
    return config_path


def _mark_session_done(client: httpx.Client, session_id: str) -> None:
    headers = {"Authorization": f"Bearer {A3S_API_KEY}"} if A3S_API_KEY else {}
    resp = client.post(
        f"{RL_BASE_URL}/session_done",
        headers=headers,
        json={"session_id": session_id},
        timeout=REQUEST_TIMEOUT_SEC,
    )
    resp.raise_for_status()


def _send_with_timeout(session: Any, prompt: str) -> Any:
    done = threading.Event()
    result_box: dict[str, Any] = {}

    def _target() -> None:
        try:
            result_box["result"] = session.send(prompt)
        except Exception as exc:
            result_box["error"] = exc
        finally:
            done.set()

    worker = threading.Thread(target=_target, daemon=True)
    worker.start()

    if done.wait(timeout=TURN_TIMEOUT_SEC):
        if "error" in result_box:
            raise result_box["error"]
        return result_box["result"]

    try:
        session.cancel()
    except Exception:
        pass
    done.wait(timeout=5)
    raise TimeoutError(f"session.send timed out after {TURN_TIMEOUT_SEC:.0f}s")


def _run_one_session(worker_id: int, session_index: int, seeds: list[SeedTask]) -> None:
    seed = random.choice(seeds)
    template_meta = _load_template_meta(seed.template)
    session_id = f"a3s-code-{int(time.time())}-{worker_id}-{session_index}-{uuid.uuid4().hex[:8]}"
    workspace = _copy_workspace_template(seed.template, session_id)
    config_path = _build_agent_config(session_id)
    config_is_ephemeral = AGENT_CONFIG_MODE == "per_session"

    record: dict[str, Any] = {
        "session_id": session_id,
        "worker_id": worker_id,
        "session_index": session_index,
        "seed_id": seed.seed_id,
        "seed": seed.seed,
        "template": seed.template,
        "agent_config_mode": AGENT_CONFIG_MODE,
        "agent_config_path": str(config_path),
        "workspace": str(workspace),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "main_turns": [],
        "status": "started",
    }

    simulated_user_client = httpx.Client(timeout=None, trust_env=False)
    print(
        f"[a3s-code-driver] worker={worker_id} session_index={session_index} "
        f"session_id={session_id} seed={seed.seed_id} template={seed.template}",
        flush=True,
    )
    original_request = _rewrite_seed_task(simulated_user_client, seed, template_meta)
    record["original_request"] = original_request

    try:
        agent = Agent.create(str(config_path))
        opts = SessionOptions()
        opts.session_id = session_id
        opts.builtin_skills = True
        opts.auto_compact = AUTO_COMPACT
        opts.auto_compact_threshold = AUTO_COMPACT_THRESHOLD
        opts.tool_timeout_ms = TOOL_TIMEOUT_MS
        opts.max_parse_retries = MAX_PARSE_RETRIES
        opts.max_tool_rounds = MAX_TOOL_ROUNDS
        opts.circuit_breaker_threshold = CIRCUIT_BREAKER_THRESHOLD
        opts.thinking_budget = THINKING_BUDGET if THINKING_BUDGET > 0 else None
        opts.continuation_enabled = CONTINUATION_ENABLED
        opts.max_continuation_turns = MAX_CONTINUATION_TURNS
        session = agent.session(str(workspace), opts, permissive=True)

        latest_response = ""
        for main_turn_number in range(1, MAX_MAIN_TURNS + 1):
            if main_turn_number == 1:
                user_prompt = original_request
                done_after_response = False
            else:
                user_prompt, done_after_response = _generate_followup(
                    simulated_user_client,
                    seed=seed,
                    template_meta=template_meta,
                    original_request=original_request,
                    latest_response=latest_response,
                    next_turn_number=main_turn_number,
                    is_final_turn=main_turn_number == MAX_MAIN_TURNS,
                )

            print(
                f"[a3s-code-driver] worker={worker_id} session_id={session_id} "
                f"turn={main_turn_number} prompt_chars={len(user_prompt)}",
                flush=True,
            )
            result = _send_with_timeout(session, user_prompt)
            latest_response = result.text
            print(
                f"[a3s-code-driver] worker={worker_id} session_id={session_id} "
                f"turn={main_turn_number} tool_calls={result.tool_calls_count} "
                f"response_chars={len(result.text)}",
                flush=True,
            )
            record["main_turns"].append(
                {
                    "turn": main_turn_number,
                    "user": user_prompt,
                    "assistant": result.text,
                    "tool_calls_count": result.tool_calls_count,
                    "done_after_response": done_after_response,
                }
            )
            if done_after_response:
                break

        _mark_session_done(simulated_user_client, session_id)
        record["status"] = "completed"
        print(
            f"[a3s-code-driver] worker={worker_id} session_id={session_id} status=completed "
            f"main_turns={len(record['main_turns'])}",
            flush=True,
        )
    except Exception as exc:
        try:
            _mark_session_done(simulated_user_client, session_id)
        except Exception:
            pass
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"
        print(
            f"[a3s-code-driver] worker={worker_id} session_id={session_id} status=error "
            f"error={type(exc).__name__}: {exc}",
            flush=True,
        )
    finally:
        simulated_user_client.close()
        _append_record(record)
        should_keep_workspace = KEEP_WORKSPACES or (
            record["status"] != "completed" and KEEP_WORKSPACES_ON_ERROR
        )
        if not should_keep_workspace:
            shutil.rmtree(workspace, ignore_errors=True)
        if config_is_ephemeral and not KEEP_CONFIGS:
            try:
                config_path.unlink(missing_ok=True)
            except TypeError:
                if config_path.exists():
                    config_path.unlink()


def _worker_main(worker_id: int, seeds: list[SeedTask]) -> None:
    while True:
        session_index = _next_session_index()
        if session_index is None:
            return
        try:
            _run_one_session(worker_id, session_index, seeds)
            print(
                f"[a3s-code-driver] worker={worker_id} completed session_index={session_index}",
                flush=True,
            )
        except Exception as exc:
            print(
                f"[a3s-code-driver] worker={worker_id} crashed session_index={session_index} "
                f"error={type(exc).__name__}: {exc}",
                flush=True,
            )
        time.sleep(SESSION_DELAY_SEC)


def main() -> None:
    _ensure_dirs()
    seeds = _load_seed_tasks()
    backend_labels = [backend.label for backend in SIMULATED_USER_POOL.backends]
    print(
        "[a3s-code-driver] "
        f"rl_base={RL_BASE_URL} simulated_user_backends={backend_labels or ['fallback-only']} "
        f"simulated_user_backend_source={SIMULATED_USER_BACKENDS_SOURCE} "
        f"model={A3S_MODEL_NAME} "
        f"agent_config_mode={AGENT_CONFIG_MODE} "
        f"session_id_header={SESSION_ID_HEADER_NAME if AGENT_CONFIG_MODE == 'shared' else 'path'} "
        f"concurrency={CONCURRENCY} session_limit={SESSION_LIMIT or 'inf'} "
        f"max_main_turns={MAX_MAIN_TURNS} max_tool_rounds={MAX_TOOL_ROUNDS} "
        f"tool_timeout_ms={TOOL_TIMEOUT_MS} turn_timeout_sec={TURN_TIMEOUT_SEC:.0f} "
        f"thinking_budget={THINKING_BUDGET} auto_compact={AUTO_COMPACT} "
        f"context={MODEL_CONTEXT_TOKENS} output={MODEL_OUTPUT_TOKENS} "
        f"sim_user_timeout_sec={SIMULATED_USER_TIMEOUT_SEC:.0f} "
        f"sim_user_cooldown_sec={SIMULATED_USER_BACKEND_COOLDOWN_SEC:.0f} "
        f"exclude_seed_tags={sorted(EXCLUDED_SEED_TAGS)} "
        f"record_file={RECORD_FILE}",
        flush=True,
    )

    threads = [
        threading.Thread(target=_worker_main, args=(worker_id, seeds), daemon=False)
        for worker_id in range(CONCURRENCY)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


if __name__ == "__main__":
    main()
