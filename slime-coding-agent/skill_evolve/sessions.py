"""Build structured sessions from WildClawBench run outputs.

Each session carries:
  session_id, task_id, score (official overall_score when present),
  trajectory (compact, lossy-clipped step trace),
  skills_referenced (skill names the agent actually read),
  has_tool_errors.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_CLIP = 400


def _clip(text: Any, limit: int = _CLIP) -> str:
    s = str(text or "").strip().replace("\n", " ")
    return s if len(s) <= limit else s[:limit] + "..."


def _render_message(msg: dict) -> str:
    role = msg.get("role", "?")
    parts = [f"[{role}] {_clip(msg.get('content', ''))}"]
    for call in msg.get("tool_calls") or []:
        fn = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = fn.get("name", call.get("name", "tool"))
        args = _clip(fn.get("arguments", call.get("arguments", "")), 200)
        parts.append(f"  -> {name}({args})")
    if role == "tool":
        parts = [f"[tool_result] {_clip(msg.get('content', ''))}"]
    return "\n".join(parts)


def _detect_skills(text: str, known_skills: list[str]) -> set[str]:
    found = set()
    for name in known_skills:
        if name and name in text:
            found.add(name)
    for match in re.findall(r"skills/([A-Za-z0-9_.-]+)/SKILL\.md", text):
        found.add(match)
    return found


def build_session(run_dir: Path, known_skills: list[str]) -> dict[str, Any] | None:
    chat_path = run_dir / "chat.jsonl"
    if not chat_path.is_file():
        return None

    messages = []
    for line in chat_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(msg, dict):
            messages.append(msg)

    trajectory = "\n".join(_render_message(m) for m in messages[:80])

    score = None
    score_path = run_dir / "score.json"
    if score_path.is_file():
        try:
            score = json.loads(score_path.read_text(encoding="utf-8")).get("overall_score")
        except json.JSONDecodeError:
            score = None

    task_id = run_dir.parent.name
    category = run_dir.parent.parent.name
    full_text = trajectory + "\n" + chat_path.read_text(encoding="utf-8", errors="replace")

    return {
        "session_id": run_dir.name,
        "task_id": f"{category}/{task_id}",
        "score": score,
        "num_turns": len(messages),
        "trajectory": trajectory,
        "skills_referenced": _detect_skills(full_text, known_skills),
        "has_tool_errors": "error" in trajectory.lower() or "✗" in trajectory,
        "run_dir": str(run_dir),
    }


def list_known_skills(skills_root: Path) -> list[str]:
    if not skills_root.is_dir():
        return []
    return sorted(p.name for p in skills_root.iterdir() if (p / "SKILL.md").is_file())


def build_sessions(raw_dir: str | Path, skills_root: str | Path) -> list[dict[str, Any]]:
    raw = Path(raw_dir)
    known = list_known_skills(Path(skills_root))
    sessions = []
    for chat_path in sorted(raw.rglob("chat.jsonl")):
        session = build_session(chat_path.parent, known)
        if session is not None:
            sessions.append(session)
    return sessions
