"""Resource-aware evolution orchestration for the coding agent server."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml


def utc_timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []

    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


@dataclass(frozen=True)
class GPUInfo:
    index: int
    name: str
    total_mb: int
    free_mb: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "total_mb": self.total_mb,
            "free_mb": self.free_mb,
        }


class ResourceProbe:
    """Uses nvidia-smi when available and returns an empty list otherwise."""

    def probe(self) -> List[GPUInfo]:
        command = [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ]
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=10, check=True)
        except (OSError, subprocess.SubprocessError):
            return []

        gpus = []
        for line in completed.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 4:
                continue
            try:
                gpus.append(
                    GPUInfo(
                        index=int(fields[0]),
                        name=fields[1],
                        total_mb=int(float(fields[2])),
                        free_mb=int(float(fields[3])),
                    )
                )
            except ValueError:
                continue
        return gpus


class SkillDistiller:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @staticmethod
    def _failure_text(event: Dict[str, Any]) -> str:
        results = event.get("command_results", [])
        snippets = []
        for result in results:
            if result.get("exit_code", 0) == 0:
                continue
            command = str(result.get("command", "")).strip()
            error = str(result.get("stderr", "")).strip()
            snippets.append(f"command={command}; error={error}")
        return "\n".join(snippets) or str(event.get("failure_reason", "No diagnostic was recorded."))

    def _heuristic_summary(self, event: Dict[str, Any]) -> str:
        task_id = event.get("task_id", "unknown-task")
        failure_text = self._failure_text(event)[:2000]
        return "\n".join(
            [
                f"# Failure Recovery: {task_id}",
                "",
                "## Trigger",
                f"Task `{task_id}` finished with status `{event.get('final_status', 'failed')}`.",
                "",
                "## Observed Evidence",
                "```text",
                failure_text,
                "```",
                "",
                "## Recovery Procedure",
                "1. Reproduce the failing command in the recorded working directory.",
                "2. Read the first actionable error before modifying code.",
                "3. Make the smallest targeted change and rerun the same command.",
                "4. Preserve the final passing command and result in the next trajectory.",
                "",
                "## Validation",
                "The original command exits with code 0 and its relevant test case passes.",
            ]
        )

    def _remote_summary(self, event: Dict[str, Any]) -> Optional[str]:
        if self.config.get("mode") != "openai-compatible":
            return None

        api_base = os.getenv(self.config.get("api_base_env", "SKILL_LLM_API_BASE"), "").rstrip("/")
        model = os.getenv(self.config.get("model_env", "SKILL_LLM_MODEL"), "")
        if not api_base or not model:
            return None

        prompt = (
            "Convert this failed coding-agent trajectory into a concise reusable Markdown skill. "
            "Include Trigger, Diagnosis, Recovery Procedure, and Validation. "
            "Do not claim an unverified root cause.\n\n"
            + json.dumps(event, ensure_ascii=False)[:12000]
        )
        payload = json.dumps(
            {
                "model": model,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": "You create precise engineering recovery skills."},
                    {"role": "user", "content": prompt},
                ],
            }
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        api_key = os.getenv(self.config.get("api_key_env", "SKILL_LLM_API_KEY"), "")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        try:
            request = urllib.request.Request(
                f"{api_base}/chat/completions",
                data=payload,
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=45) as response:
                content = json.loads(response.read().decode("utf-8"))
            summary = content["choices"][0]["message"]["content"].strip()
            return summary or None
        except (KeyError, urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError):
            return None

    def summarize(self, event: Dict[str, Any]) -> str:
        return self._remote_summary(event) or self._heuristic_summary(event)


class SkillStore:
    def __init__(self, skills_dir: Path, index_path: Path, distiller: SkillDistiller):
        self.skills_dir = skills_dir
        self.index_path = index_path
        self.distiller = distiller
        self._lock = threading.Lock()

    @staticmethod
    def _signature(event: Dict[str, Any]) -> str:
        evidence = []
        for result in event.get("command_results", []):
            if result.get("exit_code", 0) == 0:
                continue
            raw = str(result.get("stderr", "")).strip().lower()
            if not raw:
                continue
            raw = re.sub(r"0x[0-9a-f]+", "<hex>", raw)
            raw = re.sub(r"\b\d+\b", "<num>", raw)
            raw = re.sub(r"([a-z]:)?[/\\][^\s:]+", "<path>", raw)
            evidence.append(raw)
        fallback = str(event.get("failure_reason", "")).strip().lower() or str(event.get("task_id", "")).lower()
        payload = {"failure_evidence": evidence or [fallback]}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

    @staticmethod
    def _slug(event: Dict[str, Any]) -> str:
        raw = f"{event.get('repo_id', 'repo')}-{event.get('task_id', 'task')}"
        return re.sub(r"[^a-zA-Z0-9]+", "-", raw).strip("-").lower()[:64] or "failure-recovery"

    def upsert(self, event: Dict[str, Any]) -> Dict[str, Any]:
        signature = self._signature(event)
        with self._lock:
            index = json.loads(self.index_path.read_text(encoding="utf-8")) if self.index_path.exists() else {}
            if signature in index:
                group = index[signature]
                task_id = event.get("task_id", "")
                task_ids = set(group.get("task_ids", []))
                if task_id:
                    task_ids.add(task_id)
                group["task_ids"] = sorted(task_ids)
                group["occurrences"] = int(group.get("occurrences", 1)) + 1
                group["last_seen_at"] = utc_timestamp()
                self.index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
                return {
                    "created": False,
                    "path": group["path"],
                    "signature": signature,
                    "group_id": group.get("group_id", f"skill-{signature[:10]}"),
                    "occurrences": group["occurrences"],
                }

            self.skills_dir.mkdir(parents=True, exist_ok=True)
            path = self.skills_dir / f"{self._slug(event)}-{signature[:10]}.md"
            path.write_text(self.distiller.summarize(event) + "\n", encoding="utf-8")

            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            index[signature] = {
                "path": str(path),
                "group_id": f"skill-{signature[:10]}",
                "created_at": utc_timestamp(),
                "repo_id": event.get("repo_id", ""),
                "task_ids": [event.get("task_id", "")] if event.get("task_id") else [],
                "occurrences": 1,
            }
            self.index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
            return {
                "created": True,
                "path": str(path),
                "signature": signature,
                "group_id": f"skill-{signature[:10]}",
                "occurrences": 1,
            }

    def list(self) -> List[Dict[str, Any]]:
        if not self.index_path.exists():
            return []
        return list(json.loads(self.index_path.read_text(encoding="utf-8")).values())


class TrainingLauncher:
    def __init__(self, repo_root: Path, config: Dict[str, Any]):
        self.repo_root = repo_root
        self.config = config
        self._process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    def launch(self, failure_batch: Path) -> Dict[str, Any]:
        command = self.config.get("command", [])
        if not isinstance(command, list) or not command:
            return {"state": "not_configured"}

        if not self.config.get("execute", False):
            return {"state": "ready", "command": command, "failure_batch": str(failure_batch)}

        with self._lock:
            if self._process and self._process.poll() is None:
                return {"state": "already_running", "pid": self._process.pid}

            log_path = self.repo_root / self.config.get(
                "log_path", "slime-coding-agent/outputs/training/evolution-train.log"
            )
            log_path.parent.mkdir(parents=True, exist_ok=True)
            environment = os.environ.copy()
            environment.update({str(key): str(value) for key, value in self.config.get("environment", {}).items()})
            environment["OPENCLAW_AGENT_FAILURE_BATCH"] = str(failure_batch)

            try:
                with log_path.open("a", encoding="utf-8") as handle:
                    self._process = subprocess.Popen(
                        command,
                        cwd=self.repo_root,
                        env=environment,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                    )
            except OSError as exc:
                return {"state": "launch_failed", "error": str(exc)}
            return {"state": "started", "pid": self._process.pid, "log_path": str(log_path)}


class EvolutionCoordinator:
    def __init__(self, config: Dict[str, Any], repo_root: Path, resource_probe: Optional[ResourceProbe] = None):
        self.config = config
        self.repo_root = repo_root
        self.resource_probe = resource_probe or ResourceProbe()

        data_config = config["data"]
        self.trajectory_log = self._resolve(data_config["trajectory_log"])
        self.state_path = self._resolve(data_config["state_path"])
        self.training_batch_dir = self._resolve(data_config["training_batch_dir"])

        fallback = config["skill_fallback"]
        self.skill_store = SkillStore(
            self._resolve(fallback["skills_dir"]),
            self._resolve(fallback["index_path"]),
            SkillDistiller(fallback.get("summarizer", {})),
        )
        self.training_launcher = TrainingLauncher(repo_root, config["training"])

    @classmethod
    def from_file(cls, config_path: str | Path, resource_probe: Optional[ResourceProbe] = None) -> "EvolutionCoordinator":
        path = Path(config_path).resolve()
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls(config, path.parents[2], resource_probe=resource_probe)

    def _resolve(self, path: str) -> Path:
        resolved = Path(path)
        return resolved if resolved.is_absolute() else self.repo_root / resolved

    def ingest(self, event: Dict[str, Any]) -> None:
        required = {"repo_id", "task_id", "final_status"}
        missing = sorted(required - set(event))
        if missing:
            raise ValueError(f"trajectory is missing required fields: {missing}")
        append_jsonl(self.trajectory_log, event)

    def _is_idle_window(self, now: datetime) -> bool:
        schedule = self.config["schedule"]
        if not schedule.get("enabled", True):
            return False
        start = int(schedule.get("start_hour", 23))
        end = int(schedule.get("end_hour", 7))
        if start == end:
            return True
        if start < end:
            return start <= now.hour < end
        return now.hour >= start or now.hour < end

    def _eligible_gpus(self, gpus: Iterable[GPUInfo]) -> List[GPUInfo]:
        training = self.config["training"]
        min_free_mb = int(float(training["min_free_vram_gb_per_gpu"]) * 1024)
        return [gpu for gpu in gpus if gpu.free_mb >= min_free_mb]

    def _write_training_batch(self, failures: List[Dict[str, Any]]) -> Path:
        self.training_batch_dir.mkdir(parents=True, exist_ok=True)
        batch_path = self.training_batch_dir / f"failure-batch-{int(time.time())}.jsonl"
        with batch_path.open("w", encoding="utf-8") as handle:
            for event in failures:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        return batch_path

    def run_cycle(self, force: bool = False, now: Optional[datetime] = None) -> Dict[str, Any]:
        now = now or datetime.now()
        gpus = self.resource_probe.probe()
        failures = [event for event in read_jsonl(self.trajectory_log) if event.get("final_status") != "success"]
        max_failures = int(self.config["skill_fallback"].get("max_failures_per_cycle", 50))
        failures = failures[-max_failures:]

        result: Dict[str, Any] = {
            "timestamp": utc_timestamp(),
            "idle_window": self._is_idle_window(now),
            "available_gpus": [gpu.to_dict() for gpu in gpus],
            "failed_trajectories": len(failures),
        }
        if not force and not result["idle_window"]:
            result.update({"mode": "idle_wait", "action": "deferred_until_idle_window"})
        elif not failures:
            result.update({"mode": "waiting_for_failures", "action": "no_failure_trajectory"})
        else:
            eligible = self._eligible_gpus(gpus)
            training = self.config["training"]
            required_gpu_count = int(training["min_gpu_count"])
            if training.get("enabled", True) and len(eligible) >= required_gpu_count:
                batch_path = self._write_training_batch(failures)
                launch = self.training_launcher.launch(batch_path)
                result.update(
                    {
                        "mode": "binary_rl_opd_training",
                        "action": launch["state"],
                        "eligible_gpus": [gpu.to_dict() for gpu in eligible],
                        "failure_batch": str(batch_path),
                        "training": launch,
                    }
                )
            else:
                skills = [self.skill_store.upsert(event) for event in failures]
                result.update(
                    {
                        "mode": "skill_accumulation",
                        "action": "distilled_failure_trajectories",
                        "eligible_gpus": [gpu.to_dict() for gpu in eligible],
                        "skills": skills,
                    }
                )

        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    def status(self) -> Dict[str, Any]:
        if not self.state_path.exists():
            return {"mode": "not_started"}
        return json.loads(self.state_path.read_text(encoding="utf-8"))
