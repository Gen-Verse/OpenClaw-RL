"""Trajectory-mode custom generate for SWE-Bench RL (token-fidelity edition).

Standalone single-file implementation shared by the three algorithm variants:
  * generate_with_swe_kiro
  * generate_with_swe_rllm   (+ compact filter post-processing)
  * generate_with_swe_prorl

Design — ProRL-style per-message token_ids
=========================================

Every message in the rollout carries three fields in addition to role/content:
  msg["token_ids"]        : list[int]   — exact tokens for this message
  msg["token_mask"]       : list[int]   — 0 or 1, same length
  msg["token_logprobs"]   : list[float] — same length; 0.0 for non-assistant segments

For assistant messages, ``token_ids`` is constructed as
  ``gen_prompt_tokens + output_token_ids + trailing_nl_tokens``
so the *content* segment is **verbatim what the LLM generated** (no re-encode),
while the chat-template wrapper tokens are deterministic.

For non-assistant messages, ``token_ids`` comes from
  ``tokenizer.apply_chat_template([msg], tokenize=True, add_generation_prompt=False)``
which, for Qwen3-style templates, is byte-identical to the same-position tokens
in a joint ``apply_chat_template(full_messages)`` render (because the template
wraps each message independently with added-token delimiters).

Rollout-training consistency
----------------------------
During rollout, the LLM's input is
  ``sum(m["token_ids"] for m in messages) + gen_prompt_tokens``
During training, the Sample's tokens are
  ``sum(m["token_ids"] for m in messages)``
(The ``gen_prompt_tokens`` suffix is already embedded as the prefix of the
 *next* assistant message when it's appended.)

So the two tokenizations are byte-perfect identical — no re-tokenize drift,
no monotonicity check needed, no silent corruption risk.

Caveat: rollout_max_context_len must be generous enough to fit the full
trajectory. ``_truncate_input_ids`` (head+sep+tail) is intentionally NOT used
here — it would break rollout/training consistency. Instead, if context grows
too long we abort the trajectory (``exit_status="context_overflow"``) and let
compact filter (in the rllm wrapper) handle it.

Usage
-----
Wrappers import ``generate_trajectory`` and call it from their ``generate``.
The rllm wrapper additionally zeros out ``loss_mask`` when
``exit_status != "submitted"`` to implement DeepSWE's compact filter.
"""

import asyncio
import concurrent.futures
import copy
import json
import os
import re
import tempfile
import threading
import time
from functools import lru_cache
from pathlib import Path

import yaml
from loguru import logger

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post as async_post
from slime.utils.types import Sample

from swe_env_client import SweEnvClient
from swe_utils import get_docker_image_name

from skill_feedback import augment_instance_message, publish_swe_trajectory


# =============================================================================
# Globals & config
# =============================================================================

_grading_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

SWEAGENT_CONFIG_PATH = os.getenv(
    "SWE_CONFIG_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "swebench.yaml"),
)

_eval_script_cache: dict[str, str] = {}
_eval_script_cache_lock = threading.Lock()


@lru_cache(maxsize=1)
def _get_swe_semaphore() -> asyncio.Semaphore:
    return asyncio.Semaphore(int(os.getenv("SWE_MAX_CONCURRENT", "8")))


@lru_cache(maxsize=1)
def _get_sweagent_config() -> dict:
    config_path = os.getenv("SWE_CONFIG_PATH", SWEAGENT_CONFIG_PATH)
    for candidate in (config_path, Path(config_path)):
        p = Path(candidate)
        if p.exists():
            return yaml.safe_load(p.read_text())
    raise FileNotFoundError(f"SWE config not found: {config_path}")


def _sanitize_filename(value: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in value)


@lru_cache(maxsize=1)
def _get_swe_save_dir() -> Path | None:
    save_dir = os.getenv("SWE_SAVE_TRAJ_DIR", "").strip()
    if not save_dir:
        return None
    path = Path(save_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_rollout_artifacts(*, sample: Sample, iid: str, sampling_params: dict, run_info: dict) -> None:
    """Best-effort trajectory dump to ``SWE_SAVE_TRAJ_DIR`` (optional)."""
    try:
        save_dir = _get_swe_save_dir()
        if save_dir is None:
            return
        ts_ns = time.time_ns()
        stem = (
            f"{_sanitize_filename(iid)}"
            f"__g{sample.group_index if sample.group_index is not None else 'na'}"
            f"__i{sample.index if sample.index is not None else 'na'}"
            f"__{ts_ns}"
        )
        run_dir = save_dir / stem
        run_dir.mkdir(parents=True, exist_ok=True)
        # Drop token arrays for readability; keep the chat text.
        msgs_light = [
            {k: v for k, v in m.items() if k in ("role", "content")}
            for m in run_info.get("messages", [])
        ]
        traj_payload = {
            "messages": msgs_light,
            "step_debug": run_info.get("step_debug", []),
            "info": {
                "instance_id": iid,
                "exit_status": run_info.get("exit_status"),
                "error": run_info.get("error"),
                "steps": run_info.get("n_steps"),
                "patch_source": run_info.get("patch_source"),
                "reward": run_info.get("reward"),
                "eval_result": run_info.get("eval_result"),
                "policy": run_info.get("policy"),
                "group_index": sample.group_index,
                "index": sample.index,
            },
            "trajectory_format": "slime-swe-trajectory-1",
        }
        (run_dir / "traj.json").write_text(json.dumps(traj_payload, ensure_ascii=True, indent=2, default=str))
        git_patch = run_info.get("git_patch")
        if isinstance(git_patch, str):
            (run_dir / "patch.diff").write_text(git_patch)
        meta_payload = {
            "instance_id": iid,
            "sampling_params": sampling_params,
            "sample_metadata": sample.metadata,
            "sample_prompt": sample.prompt,
            "group_index": sample.group_index,
            "index": sample.index,
        }
        (run_dir / "meta.json").write_text(json.dumps(meta_payload, ensure_ascii=True, indent=2, default=str))
        logger.info(f"[SWE-T] [{iid}] Saved rollout artifacts to {run_dir}")
    except Exception as e:
        logger.warning(f"[SWE-T] [{iid}] Failed to save rollout artifacts: {e}")


# =============================================================================
# Patch parsing & validation
# =============================================================================

def _parse_bash_action(response_text: str) -> str | None:
    """Extract the bash command from a response containing ```bash ... ```."""
    pattern = r"```bash\s*\n(.*?)```"
    match = re.search(pattern, response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def _extract_patch_from_submission(output: str) -> str:
    if not isinstance(output, str):
        return ""
    text = output.lstrip("\n")
    sentinel = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
    if text.startswith(sentinel):
        text = text[len(sentinel):].lstrip("\n")
    return text


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


def _changed_files_from_patch(patch_text: str) -> list[str]:
    if not isinstance(patch_text, str) or not patch_text:
        return []
    files = []
    for m in re.finditer(r"^diff --git a/(.+?) b/(.+?)$", patch_text, flags=re.M):
        files.append(m.group(2))
    return files


# =============================================================================
# Policy gate (test-file / config-file modification detection)
# =============================================================================

def _is_test_like_path(path: str) -> bool:
    if not isinstance(path, str):
        return False
    return bool(
        re.search(r"(^|/)tests?/", path)
        or re.search(r"(^|/)test_.*", path)
        or re.search(r".*_test\.[^/]+$", path)
        or path.endswith("conftest.py")
    )


def _is_config_like_path(path: str) -> bool:
    if not isinstance(path, str):
        return False
    normalized = path.strip().lower()
    basename = normalized.rsplit("/", 1)[-1]
    config_basenames = {
        "pyproject.toml", "setup.cfg", "setup.py", "tox.ini", "pytest.ini",
        ".coveragerc", ".flake8", "mypy.ini", "ruff.toml",
        "requirements.txt", "requirements-dev.txt",
    }
    if basename in config_basenames:
        return True
    if normalized.startswith(".github/workflows/"):
        return True
    if normalized.startswith(".gitlab/"):
        return True
    if normalized.startswith(".circleci/"):
        return True
    return False


def _extract_eval_test_files(instance: dict) -> list[str]:
    if not isinstance(instance, dict):
        return []
    files: set[str] = set()
    for key in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        value = instance.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            items = [value]
        else:
            try:
                items = list(value)
            except TypeError:
                items = []
        for item in items:
            if not isinstance(item, str):
                continue
            test_name = item.split("::", 1)[0].strip()
            if test_name:
                files.add(test_name)
    return sorted(files)


def _analyze_patch_policy(patch_text: str, instance: dict | None = None) -> dict:
    changed_files = _changed_files_from_patch(patch_text)
    test_files = [f for f in changed_files if _is_test_like_path(f)]
    config_files = [f for f in changed_files if _is_config_like_path(f)]
    strict_no_test = os.getenv("SWE_STRICT_NO_TEST_PATCH", "1").strip() != "0"
    strict_no_config = os.getenv("SWE_STRICT_NO_CONFIG_PATCH", "1").strip() != "0"
    scope = os.getenv("SWE_TEST_PATCH_POLICY_SCOPE", "eval_tests_only").strip().lower()
    if scope not in {"all_tests", "eval_tests_only"}:
        scope = "eval_tests_only"
    eval_test_files = _extract_eval_test_files(instance or {})
    eval_test_file_set = set(eval_test_files)
    matched_eval_test_files = [f for f in test_files if f in eval_test_file_set]

    reasons = []
    if strict_no_test:
        if scope == "all_tests" and test_files:
            reasons.append("test_file_modified")
        elif scope == "eval_tests_only" and matched_eval_test_files:
            reasons.append("eval_test_file_modified")
    if strict_no_config and config_files:
        reasons.append("config_file_modified")

    return {
        "changed_files": changed_files,
        "test_files": test_files,
        "config_files": config_files,
        "eval_test_files": eval_test_files,
        "matched_eval_test_files": matched_eval_test_files,
        "test_policy_scope": scope,
        "strict_no_test": strict_no_test,
        "strict_no_config": strict_no_config,
        "violated": len(reasons) > 0,
        "reasons": reasons,
    }


# =============================================================================
# Eval script resolution & grading (ProRL-style harness import order)
# =============================================================================

def _infer_instance_type(instance: dict) -> str:
    if not isinstance(instance, dict):
        return "swebench"
    data_kind = instance.get("data_kind")
    if isinstance(data_kind, str) and data_kind:
        return data_kind
    if "image_assets" in instance and instance.get("image_assets") is not None:
        return "swebench_multimodal"
    return "swebench"


def _resolve_eval_script(instance: dict) -> str:
    direct = instance.get("eval_script", "")
    if isinstance(direct, str) and direct.strip():
        return direct

    iid = str(instance.get("instance_id", ""))
    if iid:
        with _eval_script_cache_lock:
            cached = _eval_script_cache.get(iid)
        if cached is not None:
            return cached

    inst = copy.deepcopy(instance)
    iid = inst.get("instance_id")
    kind = _infer_instance_type(inst)
    if isinstance(iid, str):
        inst["instance_id"] = iid.lower()
    if "version" not in inst and "base_commit" in inst:
        inst["version"] = inst["base_commit"]

    make_test_spec = None
    if kind == "swebench_multimodal":
        try:
            from swebench.harness.test_spec.test_spec import make_test_spec as _make_test_spec  # type: ignore
            make_test_spec = _make_test_spec
        except ModuleNotFoundError:
            logger.error("[SWE-T] [{}] Cannot import swebench make_test_spec for multimodal eval", iid or "unknown")
            return ""
    else:
        try:
            from swegym.harness.test_spec import make_test_spec as _make_test_spec  # type: ignore
            make_test_spec = _make_test_spec
        except ModuleNotFoundError:
            try:
                from swebench.harness.test_spec.test_spec import make_test_spec as _make_test_spec  # type: ignore
                make_test_spec = _make_test_spec
            except ModuleNotFoundError:
                logger.error("[SWE-T] [{}] Cannot import swegym/swebench make_test_spec; eval_script unavailable", iid or "unknown")
                return ""

    try:
        test_spec = make_test_spec(inst)
        script = getattr(test_spec, "eval_script", "")
        if isinstance(script, str) and script.strip():
            if iid:
                with _eval_script_cache_lock:
                    _eval_script_cache[str(iid)] = script
            return script
        logger.error("[SWE-T] [{}] make_test_spec returned empty eval_script", iid or "unknown")
        return ""
    except Exception as e:
        logger.exception("[SWE-T] [{}] Failed to build eval_script: {}", iid or "unknown", e)
        return ""


def _get_harness_tools(instance: dict):
    kind = _infer_instance_type(instance)
    if kind == "swebench_multimodal":
        from swebench.harness.grading import get_eval_report  # type: ignore
        from swebench.harness.run_evaluation import APPLY_PATCH_PASS  # type: ignore
        from swebench.harness.test_spec.test_spec import make_test_spec  # type: ignore
        return make_test_spec, get_eval_report, APPLY_PATCH_PASS

    try:
        from swegym.harness.grading import get_eval_report  # type: ignore
        from swegym.harness.run_evaluation import APPLY_PATCH_PASS  # type: ignore
        from swegym.harness.test_spec import make_test_spec  # type: ignore
        return make_test_spec, get_eval_report, APPLY_PATCH_PASS
    except ModuleNotFoundError:
        from swebench.harness.grading import get_eval_report  # type: ignore
        from swebench.harness.run_evaluation import APPLY_PATCH_PASS  # type: ignore
        from swebench.harness.test_spec.test_spec import make_test_spec  # type: ignore
        return make_test_spec, get_eval_report, APPLY_PATCH_PASS


def _grade_eval_output(instance: dict, git_patch: str, apply_output: str, eval_output: str) -> dict:
    inst = copy.deepcopy(instance)
    iid = str(inst.get("instance_id", "unknown")).lower()
    inst["instance_id"] = iid
    if "version" not in inst and "base_commit" in inst:
        inst["version"] = inst["base_commit"]

    make_test_spec, get_eval_report, apply_patch_pass = _get_harness_tools(inst)
    test_spec = make_test_spec(inst)
    pass_string = f"[{iid}] {apply_patch_pass}:\n{apply_output}"
    test_output = (
        pass_string + "\n"
        + ">>>>> Start Test Output\n"
        + (eval_output or "") + "\n"
        + ">>>>> End Test Output\n"
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        logs_dir = Path(tmp_dir) / "logs" / iid
        logs_dir.mkdir(parents=True, exist_ok=True)
        test_out_path = logs_dir / "test_output.txt"
        test_out_path.write_text(test_output)
        try:
            grading_report = get_eval_report(
                test_spec=test_spec,
                prediction={"model_patch": git_patch, "instance_id": iid},
                log_path=str(test_out_path),
                include_tests_status=True,
            )
        except Exception as e:
            if "got an unexpected keyword argument" in str(e):
                grading_report = get_eval_report(
                    test_spec=test_spec,
                    prediction={"model_patch": git_patch, "instance_id": iid},
                    test_log_path=str(test_out_path),
                    include_tests_status=True,
                )
            else:
                raise
    report = grading_report[iid]
    return {"resolved": bool(report.get("resolved", False)), "report": report}


# =============================================================================
# Observation rendering
# =============================================================================

def _render_observation(config: dict, returncode: int, output: str) -> str:
    """Render the action_observation_template from swebench.yaml."""
    from jinja2 import Template
    template_str = config.get("agent", {}).get("action_observation_template", "")
    if not template_str:
        return f"<returncode>{returncode}</returncode>\n<output>\n{output}\n</output>"
    template = Template(template_str)
    return template.render(output={"returncode": returncode, "output": output})


# =============================================================================
# Chat-template wrapper token computation (cached per tokenizer)
# =============================================================================

_wrapper_token_cache: dict[int, dict[str, list[int]]] = {}
_wrapper_token_cache_lock = threading.Lock()


def _compute_wrapper_tokens(tokenizer) -> dict[str, list[int]]:
    """Compute the (small, deterministic) wrapper-token sequences the chat
    template inserts. Cached per-tokenizer-id to avoid repeated encoding.

    Returns dict with keys:
      - ``gen_prompt``: tokens for the generation-prompt suffix
                       (e.g., for Qwen3: encode("<|im_start|>assistant\\n"))
      - ``trailing_nl``: tokens for the single "\\n" emitted after
                        ``<|im_end|>`` at the end of each message
      - ``im_end``: the single stop-token id (``<|im_end|>`` for Qwen3);
                   a list of length 1 for uniformity

    For non-Qwen tokenizers the diff-based derivation still works as long as
    the chat template appends a clear generation-prompt suffix.
    """
    # Probe dummy messages once and derive the suffixes.
    probe_user = [{"role": "user", "content": "_"}]
    with_gen = tokenizer.apply_chat_template(
        probe_user, tokenize=True, add_generation_prompt=True,
    )
    without_gen = tokenizer.apply_chat_template(
        probe_user, tokenize=True, add_generation_prompt=False,
    )
    if not isinstance(with_gen, list):
        with_gen = list(with_gen)
    if not isinstance(without_gen, list):
        without_gen = list(without_gen)
    assert with_gen[: len(without_gen)] == without_gen, (
        "Chat template does not append generation prompt as a pure suffix; "
        "trajectory mode relies on this invariant."
    )
    gen_prompt = with_gen[len(without_gen):]

    # "\n" token ids (trailing after <|im_end|>). Encode via tokenizer directly
    # to avoid template edge cases.
    trailing_nl = tokenizer.encode("\n", add_special_tokens=False)

    # <|im_end|> id (or the eos token used as the stop-boundary).
    im_end_id = None
    try:
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    except Exception:
        im_end_id = None
    if im_end_id is None or im_end_id < 0:
        im_end_id = tokenizer.eos_token_id

    return {
        "gen_prompt": list(gen_prompt),
        "trailing_nl": list(trailing_nl),
        "im_end": [im_end_id] if im_end_id is not None else [],
    }


def _get_wrapper_tokens(tokenizer) -> dict[str, list[int]]:
    key = id(tokenizer)
    with _wrapper_token_cache_lock:
        cached = _wrapper_token_cache.get(key)
        if cached is not None:
            return cached
    computed = _compute_wrapper_tokens(tokenizer)
    with _wrapper_token_cache_lock:
        _wrapper_token_cache[key] = computed
    return computed


def _encode_single_message(tokenizer, msg: dict) -> list[int]:
    """Tokenize a single message via the chat template (no generation prompt).

    For Qwen3 the result is byte-identical to the corresponding slice of a
    joint ``apply_chat_template(messages)`` render, because the template wraps
    each message independently with added-token delimiters.
    """
    rendered = tokenizer.apply_chat_template(
        [msg], tokenize=True, add_generation_prompt=False,
    )
    if not isinstance(rendered, list):
        rendered = list(rendered)
    return rendered


# =============================================================================
# Env URL
# =============================================================================

def _ensure_openai_base_url(args) -> None:
    current = os.environ.get("OPENAI_BASE_URL", "")
    if current and current != "auto":
        return
    router_ip = getattr(args, "sglang_router_ip", None)
    router_port = getattr(args, "sglang_router_port", None)
    if router_ip and router_port:
        url = f"http://{router_ip}:{router_port}/v1"
        os.environ["OPENAI_BASE_URL"] = url
        logger.info(f"[SWE-T] OPENAI_BASE_URL resolved to {url}")


# =============================================================================
# Multi-turn rollout (ProRL-style per-message token_ids)
# =============================================================================

async def _run_agent_remote(
    env_client: SweEnvClient,
    lease_id: str,
    instance: dict,
    model_config: dict,
    sweagent_config: dict,
    *,
    args,
    tokenizer,
    sglang_generate_url: str,
) -> dict:
    """Multi-turn agent loop. Produces per-message token_ids that guarantee
    byte-perfect consistency between rollout-time context and training-time tokens.

    Returned ``messages`` is a list where each entry has keys:
      role / content / token_ids / token_mask / token_logprobs
    """
    iid = instance.get("instance_id", "unknown")
    agent_config = sweagent_config.get("agent", {})
    env_config = sweagent_config.get("environment", {})
    cwd = env_config.get("cwd", "/testbed")
    step_limit = int(os.getenv("SWE_STEP_LIMIT", "") or agent_config.get("step_limit", 30))
    exec_timeout = int(env_config.get("timeout", 180))

    system_template = agent_config.get("system_template", "You are a helpful assistant.")
    instance_template = agent_config.get("instance_template", "{{task}}")
    from jinja2 import Template
    instance_message = Template(instance_template).render(task=instance["problem_statement"])
    instance_message, retrieved_skills = augment_instance_message(
        instance_message, instance["problem_statement"]
    )

    wrapper = _get_wrapper_tokens(tokenizer)
    gen_prompt_tokens = wrapper["gen_prompt"]
    trailing_nl_tokens = wrapper["trailing_nl"]
    im_end_tokens = wrapper["im_end"]

    # --- Initial system + user messages, each with its own token_ids ---
    def _build_env_side_msg(role: str, content: str) -> dict:
        msg = {"role": role, "content": content}
        toks = _encode_single_message(tokenizer, msg)
        msg["token_ids"] = toks
        msg["token_mask"] = [0] * len(toks)
        msg["token_logprobs"] = [0.0] * len(toks)
        return msg

    messages: list[dict] = [
        _build_env_side_msg("system", system_template),
        _build_env_side_msg("user", instance_message),
    ]

    step_debug: list[dict] = []
    git_patch: str | None = None
    patch_source: str | None = None
    exit_status: str | None = None
    error: str | None = None
    n_steps = 0

    max_new_tokens = int(getattr(args, "rollout_max_response_len", 0) or 0)
    if max_new_tokens <= 0:
        max_new_tokens = int(model_config.get("model_kwargs", {}).get("max_tokens", 4096))

    sglang_sampling_params = {
        "temperature": model_config.get("model_kwargs", {}).get("temperature", 1.0),
        "max_new_tokens": max_new_tokens,
    }
    stop_token_ids = set()
    if tokenizer.eos_token_id is not None:
        stop_token_ids.add(tokenizer.eos_token_id)
    if im_end_tokens:
        stop_token_ids.add(im_end_tokens[0])
    if stop_token_ids:
        sglang_sampling_params["stop_token_ids"] = list(stop_token_ids)

    max_context_len = int(getattr(args, "rollout_max_context_len", 0) or 0)

    t0 = time.time()
    for step_idx in range(step_limit):
        n_steps = step_idx + 1
        await env_client.heartbeat(lease_id)

        # Build LLM input = flatten per-message token_ids + generation prompt.
        # This is what the LLM *actually sees*, and by construction it equals
        # the eventual training-time token prefix (no re-tokenize drift).
        full_input_ids: list[int] = []
        for m in messages:
            full_input_ids.extend(m["token_ids"])
        full_input_ids.extend(gen_prompt_tokens)

        # Abort if context budget is exceeded. We do NOT do head+sep+tail
        # truncation here — that would split rollout's view from training's.
        if max_context_len > 0 and len(full_input_ids) + max_new_tokens > max_context_len:
            exit_status = "context_overflow"
            error = (
                f"context overflow at step {step_idx}: "
                f"input={len(full_input_ids)}, max_new={max_new_tokens}, "
                f"budget={max_context_len}"
            )
            logger.warning(f"[SWE-T] [{iid}] {error}")
            break

        # --- LLM call via SGLang native /generate (input_ids in, tokens+logprobs out) ---
        payload = {
            "input_ids": full_input_ids,
            "sampling_params": sglang_sampling_params,
            "return_logprob": True,
        }
        try:
            output = await async_post(sglang_generate_url, payload, max_retries=3)
        except Exception as e:
            error = f"SGLang call failed at step {step_idx}: {e}"
            logger.error(f"[SWE-T] [{iid}] {error}")
            exit_status = "error"
            break

        assistant_text = output.get("text", "")
        meta_info = output.get("meta_info", {})
        raw_logprobs = meta_info.get("output_token_logprobs", [])
        if raw_logprobs:
            output_token_ids = [x[1] for x in raw_logprobs]
            output_logprobs = [x[0] for x in raw_logprobs]
        else:
            # No logprobs returned — synthesize zeros and encode the text.
            # This path should be rare (only on SGLang versions without
            # return_logprob support).
            output_token_ids = tokenizer.encode(assistant_text, add_special_tokens=False)
            output_logprobs = [0.0] * len(output_token_ids)

        # Ensure output ends with <|im_end|> so the chat-template wrapper
        # structure is correct. If SGLang stopped *before* emitting the stop
        # token, append it. This keeps the assistant-message token_ids
        # byte-identical to the per-message chat-template render.
        if im_end_tokens:
            if not output_token_ids or output_token_ids[-1] != im_end_tokens[0]:
                output_token_ids = list(output_token_ids) + im_end_tokens
                output_logprobs = list(output_logprobs) + [0.0]

        # --- Build the assistant message with full per-message token_ids ---
        asst_tok_ids = list(gen_prompt_tokens) + list(output_token_ids) + list(trailing_nl_tokens)
        asst_tok_mask = (
            [0] * len(gen_prompt_tokens)
            + [1] * len(output_token_ids)
            + [0] * len(trailing_nl_tokens)
        )
        asst_tok_logprobs = (
            [0.0] * len(gen_prompt_tokens)
            + list(output_logprobs)
            + [0.0] * len(trailing_nl_tokens)
        )
        assistant_msg = {
            "role": "assistant",
            "content": assistant_text,
            "token_ids": asst_tok_ids,
            "token_mask": asst_tok_mask,
            "token_logprobs": asst_tok_logprobs,
        }
        messages.append(assistant_msg)

        # --- Parse bash action ---
        bash_cmd = _parse_bash_action(assistant_text)
        if bash_cmd is None:
            observation = _render_observation(
                sweagent_config, -1, "No valid bash command found in response."
            )
            messages.append(_build_env_side_msg("user", observation))
            continue

        is_submit = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in bash_cmd

        # --- Execute on remote Docker ---
        step_t0 = time.time()
        step_timeout = exec_timeout + 30
        try:
            exec_result = await asyncio.wait_for(
                env_client.exec(
                    lease_id=lease_id, command=bash_cmd, cwd=cwd, timeout=exec_timeout,
                ),
                timeout=step_timeout,
            )
            returncode = exec_result.get("returncode", -1)
            output_text = exec_result.get("output", "")
        except (asyncio.TimeoutError, TimeoutError):
            returncode = -1
            output_text = f"Command timed out after {exec_timeout}s. Please try a different command."
            logger.warning(f"[SWE-T] [{iid}] step {step_idx} timed out after {step_timeout}s")
        except Exception as e:
            returncode = -1
            output_text = f"Execution error: {e}"
            logger.error(f"[SWE-T] [{iid}] step {step_idx} exec error: {e}")

        step_debug.append({
            "step_idx": step_idx,
            "action": bash_cmd,
            "returncode": returncode,
            "output_len": len(output_text),
            "output_head": output_text[:2000],
            "output_tail": output_text[-2000:] if len(output_text) > 2000 else output_text,
            "start_ts": step_t0,
            "end_ts": time.time(),
            "ok": returncode != -1,
        })

        if is_submit:
            exit_status = "submitted"
            candidate_patch = _extract_patch_from_submission(output_text)
            if _is_valid_git_patch(candidate_patch):
                git_patch = candidate_patch
                patch_source = "submission"
            break

        # --- Append observation as a user message with its own token_ids ---
        observation = _render_observation(sweagent_config, returncode, output_text)
        remaining = step_limit - (step_idx + 1)
        if remaining == 1:
            observation += "\nREMINDER: You only have 1 turn left. Please provide the final answer"
        elif remaining > 1:
            observation += f"\nREMINDER: You have {remaining} turns left to arrive at the solution."
        messages.append(_build_env_side_msg("user", observation))

    # --- Fallback patch via git diff when the model never submitted ---
    if git_patch is None and exit_status not in ("error", "context_overflow"):
        try:
            diff_result = await env_client.diff(lease_id=lease_id, cwd=cwd)
            fallback_patch = diff_result if isinstance(diff_result, str) else ""
            if _is_valid_git_patch(fallback_patch):
                git_patch = fallback_patch
                patch_source = "git_diff_fallback"
            if exit_status is None:
                exit_status = "max_steps"
        except Exception as e:
            if error is None:
                error = f"diff failed: {e}"

    logger.info(
        f"[SWE-T] [{iid}] Agent done: steps={n_steps}, exit={exit_status}, "
        f"patch={'yes' if git_patch else 'no'}, elapsed={time.time()-t0:.1f}s"
    )

    return {
        "messages": messages,
        "step_debug": step_debug,
        "git_patch": git_patch,
        "patch_source": patch_source,
        "exit_status": exit_status,
        "n_steps": n_steps,
        "error": error,
        "retrieved_skills": retrieved_skills,
    }


# =============================================================================
# Trajectory sample construction (flatten per-message token_ids)
# =============================================================================

def _build_trajectory_sample(
    sample: Sample,
    messages: list[dict],
    outcome_reward: float,
    resolved: bool,
    run_info: dict,
    iid: str,
) -> Sample:
    """Flatten per-message token_ids into a single Sample.

    Because the rollout produced byte-perfect token_ids for every message,
    this function does not re-tokenize anything; it simply concatenates.
    """
    sample = copy.deepcopy(sample)

    if not messages:
        sample.status = Sample.Status.ABORTED
        sample.reward = {"score": 0.0, "acc": 0.0}
        sample.remove_sample = True
        return sample

    # Prompt = all messages up to (but excluding) the first assistant.
    first_asst_idx: int | None = None
    for i, m in enumerate(messages):
        if m.get("role") == "assistant":
            first_asst_idx = i
            break
    if first_asst_idx is None:
        # LLM never produced an assistant reply (initial error).
        sample.status = Sample.Status.ABORTED
        sample.reward = {"score": 0.0, "acc": 0.0}
        sample.remove_sample = True
        return sample

    prompt_tokens: list[int] = []
    for m in messages[:first_asst_idx]:
        prompt_tokens.extend(m["token_ids"])

    response_tokens: list[int] = []
    loss_mask: list[int] = []
    rollout_log_probs: list[float] = []
    assistant_texts: list[str] = []
    for m in messages[first_asst_idx:]:
        response_tokens.extend(m["token_ids"])
        loss_mask.extend(m["token_mask"])
        rollout_log_probs.extend(m["token_logprobs"])
        if m.get("role") == "assistant":
            assistant_texts.append(m.get("content", ""))

    if not response_tokens:
        sample.status = Sample.Status.ABORTED
        sample.reward = {"score": 0.0, "acc": 0.0}
        sample.remove_sample = True
        return sample

    assert len(response_tokens) == len(loss_mask) == len(rollout_log_probs), (
        f"length mismatch: tokens={len(response_tokens)}, mask={len(loss_mask)}, "
        f"logprobs={len(rollout_log_probs)}"
    )

    sample.tokens = prompt_tokens + response_tokens
    sample.response = "".join(assistant_texts)
    sample.response_length = len(response_tokens)
    sample.loss_mask = loss_mask
    sample.rollout_log_probs = rollout_log_probs
    sample.status = Sample.Status.COMPLETED
    sample.reward = {"score": float(outcome_reward), "acc": float(bool(resolved))}

    sample.metadata = copy.deepcopy(sample.metadata or {})
    sample.metadata["exit_status"] = run_info.get("exit_status", "")
    sample.metadata["n_turns"] = sum(1 for m in messages if m.get("role") == "assistant")
    sample.metadata["n_steps"] = run_info.get("n_steps", 0)
    sample.metadata["patch_source"] = run_info.get("patch_source")
    sample.metadata["retrieved_skills"] = run_info.get("retrieved_skills", [])

    return sample


# =============================================================================
# Orchestration (container alloc → rollout → patch eval → sample)
# =============================================================================

async def _generate_impl_trajectory(args, sample: Sample, sampling_params: dict) -> Sample:
    _ensure_openai_base_url(args)
    eval_timeout = int(os.getenv("SWE_EVAL_TIMEOUT", "300"))
    state = GenerateState(args)

    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    instance = metadata.get("instance", metadata)
    if not isinstance(instance, dict):
        instance = {}
    data_source = metadata.get("data_source", instance.get("data_source", "swe-gym"))
    iid = instance.get("instance_id", "unknown")

    t_start = time.time()
    swe_env_url = os.getenv("SWE_ENV_SERVER_URL", "?")
    logger.info(
        "[SWE-T] ========== TRAJECTORY ROLLOUT ENTERED ========== instance_id={} | "
        "SWE_ENV_SERVER={} | data_source={}",
        iid, swe_env_url, data_source,
    )

    sweagent_config = _get_sweagent_config()
    image_name = get_docker_image_name(instance, data_source)

    model_config = sweagent_config.get("model", {})
    litellm_model_name = (
        model_config.get("model_name")
        or os.getenv("SWE_LITELLM_MODEL_NAME")
        or "openai/Qwen/Qwen3-8B"
    )
    model_config["model_name"] = litellm_model_name
    model_config.setdefault("model_kwargs", {}).update({
        "temperature": sampling_params.get("temperature", 1.0),
        "max_tokens": sampling_params.get("max_new_tokens", 4096),
    })

    env_client = SweEnvClient()
    swe_semaphore = _get_swe_semaphore()

    logger.info(f"[SWE-T] [{iid}] Step 1/5: Waiting for semaphore...")
    await swe_semaphore.acquire()
    logger.info(f"[SWE-T] [{iid}] Step 1/5: Semaphore acquired ({time.time()-t_start:.1f}s)")

    loop = asyncio.get_event_loop()
    eval_script = await loop.run_in_executor(_grading_executor, _resolve_eval_script, instance)

    lease_id: str | None = None
    eval_lease_id: str | None = None
    run_info = {
        "messages": [], "step_debug": [], "reward": 0, "error": None,
        "git_patch": None, "patch_source": None, "exit_status": None,
        "n_steps": 0, "eval_result": None, "policy": None,
    }

    try:
        logger.info(f"[SWE-T] [{iid}] Step 2/5: Allocating container for {image_name}")
        lease = await env_client.allocate(image=image_name, instance_id=iid)
        lease_id = lease["lease_id"]
        logger.info(f"[SWE-T] [{iid}] Step 2/5: Container ready, lease={lease_id}")

        sglang_generate_url: str | None = None
        router_ip = getattr(args, "sglang_router_ip", None)
        router_port = getattr(args, "sglang_router_port", None)
        if router_ip and router_port:
            sglang_generate_url = f"http://{router_ip}:{router_port}/generate"
        if sglang_generate_url is None:
            raise RuntimeError(
                "SGLang native /generate endpoint unavailable; trajectory mode "
                "requires SGLang native generation for token-level fidelity."
            )

        logger.info(f"[SWE-T] [{iid}] Step 3/5: Running agent (trajectory mode)...")
        agent_result = await _run_agent_remote(
            env_client, lease_id, instance, model_config, sweagent_config,
            args=args,
            tokenizer=state.tokenizer,
            sglang_generate_url=sglang_generate_url,
        )
        run_info.update(agent_result)

        git_patch = run_info.get("git_patch")
        skip_eval = os.getenv("SWE_SKIP_EVAL", "0").strip() not in ("0", "", "false", "no")
        if git_patch and skip_eval:
            logger.info(f"[SWE-T] [{iid}] Step 4/5: SWE_SKIP_EVAL=1, skipping eval")
        elif git_patch:
            policy = _analyze_patch_policy(git_patch, instance=instance)
            run_info["policy"] = policy
            if policy.get("violated", False):
                reason = ",".join(policy.get("reasons", []))
                logger.warning(f"[SWE-T] [{iid}] Step 4/5: Policy blocked patch: {reason}")
                run_info["reward"] = 0
                run_info["eval_result"] = {
                    "ok": True,
                    "resolved": False,
                    "resolved_by": "policy_blocked",
                    "policy_blocked": True,
                    "policy": policy,
                }
            elif not eval_script:
                run_info["error"] = "eval_script unavailable for instance"
                logger.error(f"[SWE-T] [{iid}] Step 4/5: eval_script unavailable, skipping eval")
            else:
                try:
                    await env_client.close(lease_id)
                    logger.info(f"[SWE-T] [{iid}] Step 3/5: Closed agent container lease={lease_id}")
                    lease_id = None
                except Exception:
                    logger.exception(f"[SWE-T] [{iid}] Failed to close agent lease before eval")

                logger.info(f"[SWE-T] [{iid}] Step 4/5: Allocating fresh eval container...")
                eval_result = None
                try:
                    eval_lease = await env_client.allocate(image=image_name, instance_id=f"{iid}__eval")
                    eval_lease_id = eval_lease["lease_id"]
                    eval_result = await env_client.evaluate(
                        lease_id=eval_lease_id,
                        patch=git_patch,
                        eval_script=eval_script,
                        timeout=eval_timeout,
                    )
                except Exception as e:
                    run_info["error"] = str(e)
                    logger.error(f"[SWE-T] [{iid}] Step 4/5: Eval error: {e}")
                finally:
                    if eval_lease_id is not None:
                        try:
                            await env_client.close(eval_lease_id)
                        except BaseException:
                            pass
                        eval_lease_id = None

                if eval_result is not None:
                    resolved_by_returncode = bool(eval_result.get("resolved", False))
                    resolved = resolved_by_returncode
                    grading_error = None
                    grading_report = None
                    try:
                        loop = asyncio.get_event_loop()
                        graded = await loop.run_in_executor(
                            _grading_executor,
                            _grade_eval_output,
                            instance,
                            git_patch,
                            str(eval_result.get("apply_output", "")),
                            str(eval_result.get("output", "")),
                        )
                        grading_report = graded.get("report")
                        resolved = bool(graded.get("resolved", False))
                    except Exception as ge:
                        grading_error = str(ge)
                        logger.warning(f"[SWE-T] [{iid}] Harness grading failed, fallback to returncode: {ge}")
                    run_info["reward"] = int(resolved)
                    run_info["eval_result"] = {
                        **eval_result,
                        "resolved_by_returncode": resolved_by_returncode,
                        "resolved": resolved,
                        "resolved_by": "harness_get_eval_report" if grading_report is not None else "returncode_fallback",
                        "grading_report": grading_report,
                        "grading_error": grading_error,
                    }
                    logger.info(
                        f"[SWE-T] [{iid}] Step 4/5: resolved={resolved} "
                        f"(returncode_resolved={resolved_by_returncode})"
                    )
        else:
            logger.warning(f"[SWE-T] [{iid}] Step 4/5: No patch, skipping eval")

    except Exception as e:
        run_info["error"] = str(e)
        logger.exception(f"[SWE-T] [{iid}] Error: {e}")
    finally:
        if lease_id is not None:
            try:
                await env_client.close(lease_id)
            except BaseException:
                logger.warning(f"[SWE-T] [{iid}] Failed to close lease (may be cancelled)")
        swe_semaphore.release()
        logger.info(f"[SWE-T] [{iid}] Semaphore released")

    messages = run_info["messages"]
    reward = run_info["reward"]
    error = run_info["error"]

    _save_rollout_artifacts(sample=sample, iid=iid, sampling_params=sampling_params, run_info=run_info)

    await publish_swe_trajectory(
        instance=instance,
        data_source=data_source,
        resolved=bool(reward),
        run_info=run_info,
    )

    if not messages:
        logger.warning(f"[SWE-T] [{iid}] Step 5/5: ABORTED — no messages (error={error})")
        sample = copy.deepcopy(sample)
        sample.status = Sample.Status.ABORTED
        sample.reward = {"score": 0.0, "acc": 0.0}
        sample.remove_sample = True
        return sample

    # Outcome reward convention: {-1, +1} like terminal-rl / swe-rl prior.
    outcome_reward = 1.0 if reward else -1.0
    resolved = bool(reward)

    trajectory_sample = _build_trajectory_sample(
        sample=sample, messages=messages,
        outcome_reward=outcome_reward, resolved=resolved,
        run_info=run_info, iid=iid,
    )

    elapsed = time.time() - t_start
    logger.info(
        f"[SWE-T] [{iid}] Step 5/5: DONE — "
        f"status={trajectory_sample.status.name}, "
        f"n_turns={sample.metadata.get('n_turns', 0) if isinstance(sample.metadata, dict) else 0}, "
        f"response_length={trajectory_sample.response_length}, "
        f"outcome_reward={outcome_reward}, "
        f"exit_status={run_info.get('exit_status', '')}, "
        f"total_elapsed={elapsed:.1f}s"
    )
    return trajectory_sample


async def generate_trajectory(args, sample: Sample, sampling_params: dict) -> Sample:
    """Entry point with total-rollout timeout watchdog.

    Algorithm wrappers (generate_with_swe_{kiro,rllm,prorl}) delegate here.
    """
    rollout_timeout = float(os.getenv("SWE_ROLLOUT_TIMEOUT", "1800"))
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    instance = metadata.get("instance", metadata)
    if not isinstance(instance, dict):
        instance = {}
    iid = instance.get("instance_id", "unknown")
    try:
        return await asyncio.wait_for(
            _generate_impl_trajectory(args, sample, sampling_params),
            timeout=rollout_timeout,
        )
    except (asyncio.TimeoutError, TimeoutError):
        logger.error(
            f"[SWE-T] [{iid}] TOTAL ROLLOUT TIMEOUT ({rollout_timeout}s) exceeded, aborting sample"
        )
        sample = copy.deepcopy(sample)
        sample.status = Sample.Status.ABORTED
        sample.reward = {"score": 0.0, "acc": 0.0}
        sample.remove_sample = True
        return sample
