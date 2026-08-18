"""Optional idle-time LLM judge for sessions without a benchmark score.

Enabled only when SKILL_EVOLVE_JUDGE=1 and the LLM client is configured.
Sessions that already carry an official score are never re-judged.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from . import llm

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

_SYSTEM = """You are a session-level evaluator for agent trajectories.
Score the session on a 0.0-1.0 scale for: task_completion (0.55),
response_quality (0.30), efficiency (0.05), tool_usage (0.10).
Return EXACTLY one JSON object:
{"task_completion": f, "response_quality": f, "efficiency": f,
 "tool_usage": f, "overall_score": f, "rationale": "brief"}
No markdown fences. No extra text."""


def _parse(text: str | None) -> float | None:
    if not text:
        return None
    clean = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`")
    match = _JSON_RE.search(clean)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        score = float(obj["overall_score"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    return max(0.0, min(1.0, score))


def maybe_judge(session: dict[str, Any]) -> float | None:
    """Return a judged score, or None when judging is disabled/unneeded."""
    if session.get("score") is not None:
        return session["score"]
    if os.getenv("SKILL_EVOLVE_JUDGE", "0") != "1" or not llm.is_configured():
        return None
    user = (
        f"Task: {session.get('task_id')}\n"
        f"Trajectory:\n{session.get('trajectory', '')[:6000]}"
    )
    return _parse(llm.chat(_SYSTEM, user))


def backfill_scores(sessions: list[dict[str, Any]]) -> int:
    judged = 0
    for session in sessions:
        if session.get("score") is None:
            score = maybe_judge(session)
            if score is not None:
                session["score"] = score
                session["score_source"] = "llm_judge"
                judged += 1
    return judged
