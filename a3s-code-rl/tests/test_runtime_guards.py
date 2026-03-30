from __future__ import annotations

import asyncio
import importlib
import queue
import sys
import threading
import time
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class _FakeTokenizer:
    def __call__(self, text, add_special_tokens: bool = False):
        if isinstance(text, str):
            return {"input_ids": list(range(len(text.split())))}
        return {"input_ids": [0]}

    def apply_chat_template(self, messages, tools=None, tokenize=False, add_generation_prompt=False):
        rendered = "\n".join(
            f"{msg.get('role', 'user')}:{msg.get('content', '')}" for msg in messages
        )
        if tokenize:
            return list(range(len(rendered.split())))
        return rendered

    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        return "decoded"


def _install_slime_stubs(monkeypatch) -> None:
    slime_pkg = types.ModuleType("slime")
    utils_pkg = types.ModuleType("slime.utils")
    rollout_pkg = types.ModuleType("slime.rollout")
    metric_utils = types.ModuleType("slime.utils.metric_utils")
    processing_utils = types.ModuleType("slime.utils.processing_utils")
    types_mod = types.ModuleType("slime.utils.types")
    async_utils = types.ModuleType("slime.utils.async_utils")
    base_types = types.ModuleType("slime.rollout.base_types")
    sglang_rollout = types.ModuleType("slime.rollout.sglang_rollout")

    class FakeSample:
        class Status:
            COMPLETED = "completed"
            ABORTED = "aborted"

        def __init__(self):
            self.reward = {}

    class FakeRolloutFnTrainOutput:
        def __init__(self, samples, metrics=None):
            self.samples = samples
            self.metrics = metrics

    async def fake_eval_rollout(*args, **kwargs):
        return [], None

    metric_utils.has_repetition = lambda text: False
    processing_utils.load_tokenizer = lambda *args, **kwargs: _FakeTokenizer()
    types_mod.Sample = FakeSample
    async_utils.run = lambda awaitable: asyncio.run(awaitable)
    base_types.RolloutFnTrainOutput = FakeRolloutFnTrainOutput
    sglang_rollout.eval_rollout = fake_eval_rollout

    monkeypatch.setitem(sys.modules, "slime", slime_pkg)
    monkeypatch.setitem(sys.modules, "slime.utils", utils_pkg)
    monkeypatch.setitem(sys.modules, "slime.rollout", rollout_pkg)
    monkeypatch.setitem(sys.modules, "slime.utils.metric_utils", metric_utils)
    monkeypatch.setitem(sys.modules, "slime.utils.processing_utils", processing_utils)
    monkeypatch.setitem(sys.modules, "slime.utils.types", types_mod)
    monkeypatch.setitem(sys.modules, "slime.utils.async_utils", async_utils)
    monkeypatch.setitem(sys.modules, "slime.rollout.base_types", base_types)
    monkeypatch.setitem(sys.modules, "slime.rollout.sglang_rollout", sglang_rollout)


def _install_service_stubs(monkeypatch) -> None:
    fastapi = types.ModuleType("fastapi")
    fastapi_responses = types.ModuleType("fastapi.responses")
    uvicorn = types.ModuleType("uvicorn")

    class FakeFastAPI:
        def __init__(self, *args, **kwargs):
            self.state = types.SimpleNamespace()

        def on_event(self, *args, **kwargs):
            def decorator(fn):
                return fn

            return decorator

        def get(self, *args, **kwargs):
            def decorator(fn):
                return fn

            return decorator

        def post(self, *args, **kwargs):
            def decorator(fn):
                return fn

            return decorator

    class FakeHTTPException(Exception):
        def __init__(self, status_code: int, detail=None):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    class FakeResponse:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class FakeConfig:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class FakeServer:
        def __init__(self, config=None):
            self.config = config
            self.should_exit = False

        def run(self):
            return None

    fastapi.FastAPI = FakeFastAPI
    fastapi.Header = lambda default=None: default
    fastapi.HTTPException = FakeHTTPException
    fastapi.Request = object
    fastapi_responses.JSONResponse = FakeResponse
    fastapi_responses.StreamingResponse = FakeResponse
    uvicorn.Config = FakeConfig
    uvicorn.Server = FakeServer

    monkeypatch.setitem(sys.modules, "fastapi", fastapi)
    monkeypatch.setitem(sys.modules, "fastapi.responses", fastapi_responses)
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn)


def _install_a3s_code_stub(monkeypatch) -> None:
    a3s_code = types.ModuleType("a3s_code")
    a3s_code.Agent = type("Agent", (), {})
    a3s_code.SessionOptions = type("SessionOptions", (), {})
    monkeypatch.setitem(sys.modules, "a3s_code", a3s_code)


def _load_module(monkeypatch, module_name: str):
    monkeypatch.syspath_prepend(str(ROOT))
    _install_slime_stubs(monkeypatch)
    _install_service_stubs(monkeypatch)
    if module_name == "a3s_code_agent_traffic_driver":
        _install_a3s_code_stub(monkeypatch)

    sys.modules.pop(module_name, None)
    if module_name == "code_rl_rollout":
        sys.modules.pop("code_rl_api_server", None)
    return importlib.import_module(module_name)


def test_pause_submission_discards_queued_groups(monkeypatch) -> None:
    module = _load_module(monkeypatch, "code_rl_rollout")
    worker = object.__new__(module.AsyncRolloutWorker)
    worker._submission_enabled = threading.Event()
    worker._submission_enabled.set()
    worker.output_queue = queue.Queue()
    worker.output_queue.put((1, ["a", "b"]))
    worker.output_queue.put((2, ["c"]))

    class FakeServer:
        def __init__(self):
            self.purge_calls = 0

        def purge_record_files(self):
            self.purge_calls += 1

    worker._server = FakeServer()

    module.AsyncRolloutWorker.pause_submission(worker)

    assert not worker._submission_enabled.is_set()
    assert worker.output_queue.empty()
    assert worker._server.purge_calls == 1


def test_submit_turn_sample_drops_ready_sample_when_paused(monkeypatch) -> None:
    module = _load_module(monkeypatch, "code_rl_api_server")
    server = object.__new__(module.CodeRLAPIServer)
    server.submission_enabled = threading.Event()
    server.drop_repetitive_samples = False
    server._eval_scores = []
    server._eval_scores_lock = threading.Lock()
    server._session_effective = {}
    server.max_response_tokens = 0
    server.response_trim_margin_tokens = 0
    server.max_train_tokens = 0
    server.tokenizer = _FakeTokenizer()
    server.output_queue = queue.Queue()
    decisions: list[str] = []
    server._append_sample_trace = lambda **kwargs: decisions.append(kwargs["decision"])
    server._maybe_cleanup_session = lambda session_id: None
    server._consume_turn_feedback = lambda session_id, turn_num: []
    server._aggregate_feedback = lambda feedback_events: {
        "sanitized": False,
        "redaction_count": 0,
        "permission_denied": False,
        "injection_blocked": False,
        "event_types": [],
    }
    server._resolve_reward = lambda turn_data, prm_result: {
        "score": 1.0,
        "source": "next_state_prm",
        "details": {},
    }
    server._build_metadata = lambda **kwargs: {"train_metadata": {}}

    turn_data = {
        "turn_num": 1,
        "turn_type": "main",
        "prompt_ids": [1, 2],
        "response_ids": [3, 4],
        "response_logprobs": [-0.1, -0.2],
        "prompt_text": "prompt",
        "response_text": "response",
        "tool_calls": [],
        "channel": "api",
        "session_done": False,
        "has_next_state": True,
        "next_state_role": "user",
        "response_has_repetition": False,
    }

    asyncio.run(server._submit_turn_sample(turn_data=turn_data, session_id="s1", prm_result=None))

    assert server.output_queue.empty()
    assert decisions == ["dropped_paused"]
    assert server._eval_scores == []


def test_cleanup_waits_for_submit_tasks(monkeypatch) -> None:
    module = _load_module(monkeypatch, "code_rl_api_server")
    server = object.__new__(module.CodeRLAPIServer)
    session_id = "s1"

    class FakeTask:
        def __init__(self, done: bool):
            self._done = done

        def done(self) -> bool:
            return self._done

    server._finalizing_sessions = {session_id}
    server._pending_records = {}
    server._pending_turn_data = {}
    server._prm_tasks = {}
    server._submit_tasks = {session_id: {FakeTask(False)}}
    server._overflow_terminated_sessions = {session_id}
    server._session_latest_messages = {session_id: []}
    server._session_last_activity = {session_id: time.time()}
    server._turn_feedback = {session_id: {}}
    server._session_effective = {session_id: 1}
    server._turn_counts = {session_id: 2}

    module.CodeRLAPIServer._maybe_cleanup_session(server, session_id)
    assert session_id in server._finalizing_sessions

    server._submit_tasks = {session_id: {FakeTask(True)}}
    module.CodeRLAPIServer._maybe_cleanup_session(server, session_id)
    assert session_id not in server._finalizing_sessions
    assert session_id not in server._session_latest_messages


def test_prepare_workspace_template_cache_serializes_creation(monkeypatch, tmp_path: Path) -> None:
    module = _load_module(monkeypatch, "a3s_code_agent_traffic_driver")
    template_root = tmp_path / "task_templates"
    cache_root = tmp_path / "workspace_template_cache"
    source = template_root / "demo_template"
    source.mkdir(parents=True)
    (source / "README.md").write_text("demo\n", encoding="utf-8")

    monkeypatch.setattr(module, "TEMPLATE_ROOT", template_root)
    monkeypatch.setattr(module, "WORKSPACE_TEMPLATE_CACHE_ROOT", cache_root)
    monkeypatch.setattr(module, "WORKSPACE_COPY_MODE", "copy")
    monkeypatch.setattr(module, "TEMPLATE_CACHE_LOCK", threading.Lock())

    copy_calls = 0
    init_calls = 0
    counter_lock = threading.Lock()

    def fake_copy_dir(src: Path, dst: Path, *, mode: str, strip_template_meta: bool = False) -> None:
        nonlocal copy_calls
        with counter_lock:
            copy_calls += 1
        time.sleep(0.05)
        dst.mkdir(parents=True, exist_ok=True)
        (dst / "README.md").write_text((src / "README.md").read_text(encoding="utf-8"), encoding="utf-8")

    def fake_init_git_repo(workspace: Path) -> None:
        nonlocal init_calls
        with counter_lock:
            init_calls += 1

    monkeypatch.setattr(module, "_copy_dir", fake_copy_dir)
    monkeypatch.setattr(module, "_init_git_repo", fake_init_git_repo)

    start = threading.Event()
    results: list[Path] = []
    errors: list[Exception] = []

    def target() -> None:
        try:
            start.wait()
            results.append(module._prepare_workspace_template_cache("demo_template"))
        except Exception as exc:  # pragma: no cover - assertion below covers this path
            errors.append(exc)

    threads = [threading.Thread(target=target) for _ in range(2)]
    for thread in threads:
        thread.start()
    start.set()
    for thread in threads:
        thread.join()

    assert not errors
    assert len(results) == 2
    assert results[0] == results[1] == cache_root / "demo_template"
    assert copy_calls == 1
    assert init_calls == 1
