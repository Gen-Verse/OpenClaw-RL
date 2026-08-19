"""Group sessions by referenced skill; sessions with no reference go to G(empty)."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

NO_SKILL = "__no_skill__"


def group_sessions(sessions: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for session in sessions:
        skills = session.get("skills_referenced") or set()
        if not skills:
            groups[NO_SKILL].append(session)
        else:
            for name in skills:
                groups[name].append(session)
    return dict(groups)
