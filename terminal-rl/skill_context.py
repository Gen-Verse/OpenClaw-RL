"""Optional lexical retrieval of verified recovery skills for terminal rollouts."""

from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Any


_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]{2,}")
_STOPWORDS = {
    "and",
    "for",
    "from",
    "that",
    "the",
    "this",
    "with",
    "your",
    "task",
    "terminal",
}


def _tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in _TOKEN_RE.findall(text)
        if token.lower() not in _STOPWORDS
    }


def retrieve_skills(
    query: str,
    skills_dir: str | Path,
    top_k: int = 3,
    max_chars_per_skill: int = 1200,
) -> list[dict[str, Any]]:
    directory = Path(skills_dir)
    if not directory.is_dir() or top_k <= 0:
        return []

    query_tokens = _tokens(query)
    if not query_tokens:
        return []

    candidates = []
    for path in sorted(directory.glob("*.md")):
        text = path.read_text(encoding="utf-8", errors="replace")
        skill_tokens = _tokens(text)
        overlap = len(query_tokens & skill_tokens)
        if not overlap:
            continue
        score = overlap / math.sqrt(len(query_tokens) * max(1, len(skill_tokens)))
        candidates.append({"path": str(path), "score": round(score, 6), "content": text[:max_chars_per_skill]})
    return sorted(candidates, key=lambda item: (-item["score"], item["path"]))[:top_k]


def augment_user_message(user_message: str, task_instruction: str) -> tuple[str, list[dict[str, Any]]]:
    if os.getenv("TERMINAL_SKILL_RETRIEVAL", "0") != "1":
        return user_message, []

    skills_dir = os.getenv("TERMINAL_SKILLS_DIR", "")
    if not skills_dir:
        return user_message, []

    top_k = int(os.getenv("TERMINAL_SKILL_TOP_K", "3"))
    skills = retrieve_skills(task_instruction, skills_dir, top_k=top_k)
    if not skills:
        return user_message, []

    snippets = "\n\n".join(
        f"### Recovery Skill {index}\n{skill['content']}"
        for index, skill in enumerate(skills, start=1)
    )
    context = (
        "\n\n## Retrieved Recovery Skills\n"
        "Use these only when relevant. Verify every command and result in the current environment.\n"
        f"{snippets}"
    )
    return user_message + context, [{"path": skill["path"], "score": skill["score"]} for skill in skills]
