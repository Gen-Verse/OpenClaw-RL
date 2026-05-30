"""
ExperienceStore — keyword-based experience persistence and retrieval.

Stores hints extracted by the Judge LLM and retrieves them via simple
keyword matching.  Thread-safe; persists to a single JSON file.

Part of the **openclaw-exploop** project (training-free experience loop).
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from typing import Any

logger = logging.getLogger(__name__)

# Matches word-like tokens (letters, digits, underscores) of length > 3.
_WORD_RE = re.compile(r"[a-zA-Z0-9_]+")

# Common English stop-words that are longer than 3 chars — cheap filter.
_STOP_WORDS = frozenset({
    "that", "this", "with", "from", "your", "have", "been", "were", "will",
    "they", "them", "than", "then", "when", "what", "which", "their", "there",
    "would", "could", "should", "about", "after", "before", "where", "these",
    "those", "some", "also", "just", "more", "very", "into", "over", "only",
    "such", "each", "does", "done", "make", "made", "like", "well", "back",
    "much", "most", "many", "same", "here", "every", "still", "while",
    "because", "through", "between", "being", "other", "under",
})


def _extract_keywords(text: str, max_words: int = 20) -> list[str]:
    """Extract up to *max_words* lowercase keywords (len > 3, deduplicated)."""
    tokens = _WORD_RE.findall(text.lower())
    seen: set[str] = set()
    keywords: list[str] = []
    for tok in tokens:
        if len(tok) <= 3:
            continue
        if tok in _STOP_WORDS:
            continue
        if tok in seen:
            continue
        seen.add(tok)
        keywords.append(tok)
        if len(keywords) >= max_words:
            break
    return keywords


class ExperienceStore:
    """Persists and retrieves experiences from a JSON file.

    Each experience entry looks like::

        {
            "id":                 "<uuid>",
            "timestamp":          "2026-03-05 12:00:00",
            "hint":               "...",
            "keywords":           ["keyword1", "keyword2", ...],
            "response_snippet":   "first 300 chars of assistant response",
            "next_state_snippet": "first 300 chars of next user message",
            "session_id":         "sess-abc"
        }
    """

    SNIPPET_LEN = 300

    def __init__(self, store_path: str = "experiences.json") -> None:
        self._store_path = store_path
        self._lock = threading.Lock()
        self._experiences: list[dict[str, Any]] = []
        self._load()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load existing experiences from disk (if any)."""
        if not os.path.isfile(self._store_path):
            logger.info("[ExperienceStore] no existing file at %s — starting fresh", self._store_path)
            return
        try:
            with open(self._store_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list):
                self._experiences = data
                logger.info(
                    "[ExperienceStore] loaded %d experiences from %s",
                    len(self._experiences),
                    self._store_path,
                )
            else:
                logger.warning("[ExperienceStore] unexpected JSON root type in %s — ignoring", self._store_path)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("[ExperienceStore] failed to load %s: %s", self._store_path, exc)

    def _save(self) -> None:
        """Persist current experiences to disk (caller must hold ``_lock``)."""
        tmp_path = self._store_path + ".tmp"
        try:
            parent = os.path.dirname(self._store_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(self._experiences, fh, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self._store_path)
        except OSError as exc:
            logger.error("[ExperienceStore] failed to save to %s: %s", self._store_path, exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(
        self,
        hint: str,
        response_text: str,
        next_state_text: str,
        session_id: str,
    ) -> dict[str, Any]:
        """Store a new experience entry.  Returns the created entry dict."""
        entry: dict[str, Any] = {
            "id": uuid.uuid4().hex[:12],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "hint": hint.strip(),
            "keywords": _extract_keywords(hint),
            "response_snippet": response_text[:self.SNIPPET_LEN].strip(),
            "next_state_snippet": next_state_text[:self.SNIPPET_LEN].strip(),
            "session_id": session_id,
        }
        with self._lock:
            self._experiences.append(entry)
            self._save()
        logger.info(
            "[ExperienceStore] added experience id=%s keywords=%s session=%s (total=%d)",
            entry["id"],
            entry["keywords"][:5],
            session_id,
            len(self._experiences),
        )
        return entry

    def retrieve(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Keyword-based retrieval.

        Score = number of query keywords that appear in an experience's keywords.
        Returns up to *top_k* experiences sorted by descending score, then recency.
        Only entries with score > 0 are returned.
        """
        query_kws = set(_extract_keywords(query, max_words=30))
        if not query_kws:
            return []

        scored: list[tuple[int, int, dict[str, Any]]] = []
        with self._lock:
            for idx, exp in enumerate(self._experiences):
                exp_kws = set(exp.get("keywords", []))
                overlap = len(query_kws & exp_kws)
                if overlap > 0:
                    scored.append((overlap, idx, exp))

        # Sort by overlap desc, then by index desc (newer first).
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        return [entry for _, _, entry in scored[:top_k]]

    def format_for_injection(self, experiences: list[dict[str, Any]]) -> str:
        """Format retrieved experiences as a prompt prefix string."""
        if not experiences:
            return ""
        lines = ["=== Relevant Past Experiences ==="]
        for i, exp in enumerate(experiences, 1):
            lines.append(f"{i}. {exp['hint']}")
        lines.append("================================")
        lines.append(
            "Use the above experiences as context to improve your response. "
            "They capture lessons learned from previous interactions."
        )
        return "\n".join(lines)

    def stats(self) -> dict[str, Any]:
        """Return summary statistics."""
        with self._lock:
            total = len(self._experiences)
        return {
            "total_experiences": total,
            "store_path": self._store_path,
        }
