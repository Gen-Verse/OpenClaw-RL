"""Publication gate for candidate skills (LLM gate + heuristic fallback)."""

from __future__ import annotations

import json
import re
from typing import Any

from . import llm

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

_SYSTEM = """You are the final publication gate for evolved agent skills.
Approve only if ALL hold:
- grounded in the provided session evidence
- preserves useful existing specifics (commands, paths, endpoints)
- specific and reusable, not generic advice
Reject if speculative, weakly supported, or mostly generic best practices.
Return EXACTLY one JSON object:
{"decision": "accept|reject", "score": 0..1, "reason": "short"}
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
    if obj.get("decision") not in {"accept", "reject"}:
        return None
    return obj


def _evidence_tokens(evidence: str) -> set[str]:
    return {t for t in re.findall(r"[a-zA-Z][a-zA-Z0-9_./-]{4,}", evidence.lower())}


def verify(candidate: dict[str, Any], sessions: list[dict[str, Any]]) -> dict[str, Any]:
    """Return {accepted: bool, score, reason}."""
    skill_md = str(candidate.get("skill_md", ""))
    evidence_text = "\n".join(s.get("trajectory", "")[:800] for s in sessions[:4])

    if llm.is_configured():
        user = (
            f"## Candidate skill\n{skill_md[:6000]}\n\n"
            f"## Motivating evidence\n{candidate.get('evidence', '')[:1000]}\n\n"
            f"## Session excerpts\n{evidence_text[:3000]}"
        )
        result = _parse(llm.chat(_SYSTEM, user))
        if result is not None:
            return {
                "accepted": result["decision"] == "accept",
                "score": float(result.get("score", 0.0)),
                "reason": str(result.get("reason", "")),
            }

    # Heuristic fallback: candidate must reuse concrete tokens from evidence.
    tokens = _evidence_tokens(evidence_text)
    if not tokens:
        return {"accepted": True, "score": 0.5, "reason": "heuristic: no tokens to check"}
    skill_lower = skill_md.lower()
    overlap = sum(1 for t in tokens if t in skill_lower)
    ratio = overlap / max(1, min(len(tokens), 20))
    accepted = ratio >= 0.1
    return {
        "accepted": accepted,
        "score": round(ratio, 3),
        "reason": f"heuristic evidence-token overlap {overlap}/{min(len(tokens), 20)}",
    }
