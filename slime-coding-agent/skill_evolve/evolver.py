"""LLM evolver: refine an existing skill or create a new one from G(empty).

Decision contract (mirrors SkillClaw):
  - successful sessions define invariants that must be preserved
  - failed sessions define targets to fix
  - insufficient evidence -> skip
"""

from __future__ import annotations

import json
import re
from typing import Any

from . import llm

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

_SYSTEM = """You are a skill evolution engineer. You receive:
- the current skill (if any) and its full change history
- sessions that used it (or, for new-skill detection, sessions that used no skill)

Rules:
- Successful sessions define invariants: parts that work and must not change.
- Failed sessions define targets: specific behaviors to correct.
- Choose "refine" only when failures trace to the skill's guidance.
- Choose "create" only when no-skill failures share a teachable, recurring procedure.
- Choose "skip" when evidence is weak or failures come from environment/model limits.

Return EXACTLY one JSON object:
{"action": "refine|create|skip",
 "skill_md": "full new SKILL.md content (empty when skip)",
 "evidence": "what session evidence motivated this (2-6 sentences)",
 "rationale": "brief"}
No markdown fences. No extra text."""


def _parse(text: str | None) -> dict[str, Any] | None:
    if not text:
        return None
    clean = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`")
    match = _JSON_RE.search(clean)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if obj.get("action") not in {"refine", "create", "skip"}:
        return None
    return obj


def _format_evidence(sessions: list[dict[str, Any]], max_sessions: int = 8) -> str:
    blocks = []
    for s in sessions[:max_sessions]:
        score = s.get("score")
        blocks.append(
            f"--- session {s.get('session_id')} task={s.get('task_id')} "
            f"score={score if score is not None else 'unknown'} ---\n"
            f"{s.get('trajectory', '')[:2500]}"
        )
    return "\n\n".join(blocks)


def _heuristic_new_skill(sessions: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Fallback create when no LLM is configured: only with 2+ failed sessions."""
    failed = [s for s in sessions if (s.get("score") is not None and s["score"] < 1.0)]
    if len(failed) < 2:
        return None
    task_ids = ", ".join(s["task_id"] for s in failed[:4])
    skill_md = (
        "# Failure Recovery Notes\n\n"
        "## Trigger\n"
        f"Recurring failures observed in tasks: {task_ids}.\n\n"
        "## Observed Evidence\n"
        "Multiple sessions failed without an applicable skill. Review the "
        "recorded trajectory before acting; check inputs and tool outputs first.\n\n"
        "## Recovery Procedure\n"
        "1. Reproduce the failing step and read the first actionable error.\n"
        "2. Verify required input files and paths exist before tool calls.\n"
        "3. Make the smallest change and rerun the failing check.\n\n"
        "## Validation\n"
        "The previously failing check passes."
    )
    return {
        "action": "create",
        "skill_md": skill_md,
        "evidence": f"{len(failed)} failed sessions without skill coverage: {task_ids}",
        "rationale": "heuristic fallback (no LLM configured)",
    }


def evolve_group(
    name: str,
    sessions: list[dict[str, Any]],
    current_skill: str | None,
    history: list[dict],
) -> dict[str, Any] | None:
    """Return {action, skill_md, evidence, rationale} or None when skipping."""
    is_new = current_skill is None

    if not llm.is_configured():
        if is_new:
            return _heuristic_new_skill(sessions)
        return None  # never refine without LLM judgment

    history_text = "\n\n".join(
        f"### {h['version']}\n{h['content'][:1500]}\nEvidence: {h['evidence'][:500]}"
        for h in history[-4:]
    ) or "(no history)"
    current_text = current_skill[:4000] if current_skill else "(no current skill - decide create vs skip)"
    user = (
        f"Skill under review: {name}\n\n"
        f"## Current skill\n{current_text}\n\n"
        f"## History\n{history_text}\n\n"
        f"## Sessions\n{_format_evidence(sessions)}"
    )
    result = _parse(llm.chat(_SYSTEM, user))
    if result is None or result["action"] == "skip":
        return None
    if not str(result.get("skill_md", "")).strip():
        return None
    return result
