"""
PromptInjector — retrieve past experiences and inject them into prompts.

Works with :class:`ExperienceStore` to look up relevant hints and
prepend them to the system message so the upstream LLM can benefit
from accumulated experience without any parameter updates.

Part of the **openclaw-exploop** project (training-free experience loop).
"""

from __future__ import annotations

import copy
import logging
from typing import Any

from experience_store import ExperienceStore

logger = logging.getLogger(__name__)


class PromptInjector:
    """Inject retrieved experience hints into a message list.

    Parameters
    ----------
    store : ExperienceStore
        The backing experience store used for keyword retrieval.
    max_exp_chars : int
        Maximum total characters for the injected experience block.
        Experiences are added in relevance order until the budget is
        exhausted.
    """

    def __init__(
        self,
        store: ExperienceStore,
        max_exp_chars: int = 800,
    ) -> None:
        self._store = store
        self._max_exp_chars = max_exp_chars

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def inject(
        self,
        messages: list[dict[str, Any]],
        query: str,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """Return a *new* messages list with experiences prepended.

        The original list is never mutated.

        Steps:
        1. Retrieve up to *top_k* experiences from the store.
        2. Format them into a prefix string (respecting char budget).
        3. Prepend the prefix to the first system message's content.
           If no system message exists, create one at position 0.

        If *query* is empty no retrieval is attempted and the original
        messages are returned as-is (shallow copy).
        """
        if not query.strip():
            return list(messages)

        experiences = self._store.retrieve(query, top_k=top_k)
        if not experiences:
            return list(messages)  # shallow copy, nothing to inject

        prefix = self._format(experiences)
        if not prefix:
            return list(messages)

        logger.info(
            "[PromptInjector] injecting %d experiences "
            "(%d chars) for query=%r",
            len(experiences), len(prefix), query[:60],
        )

        # Deep-copy to avoid mutating caller's data.
        new_msgs: list[dict[str, Any]] = copy.deepcopy(messages)

        # Find or create the system message.
        sys_idx = self._find_system_index(new_msgs)
        if sys_idx is not None:
            original = new_msgs[sys_idx].get("content", "")
            new_msgs[sys_idx]["content"] = (
                prefix + "\n\n" + original
            )
        else:
            new_msgs.insert(0, {
                "role": "system",
                "content": prefix,
            })

        return new_msgs

    def inject_string(
        self,
        text: str,
        query: str,
        top_k: int = 3,
    ) -> str:
        """Convenience wrapper: inject experiences into a plain string.

        Returns *text* prefixed with the experience block (if any).
        """
        if not query.strip():
            return text

        experiences = self._store.retrieve(query, top_k=top_k)
        if not experiences:
            return text
        prefix = self._format(experiences)
        if not prefix:
            return text
        return prefix + "\n\n" + text

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_system_index(
        messages: list[dict[str, Any]],
    ) -> int | None:
        """Return the index of the first system message, or *None*."""
        for idx, msg in enumerate(messages):
            if msg.get("role") == "system":
                return idx
        return None

    def _format(
        self,
        experiences: list[dict[str, Any]],
    ) -> str:
        """Format experiences into a prompt prefix within char budget."""
        header = "=== Relevant Past Experiences ==="
        footer_line1 = "================================"
        footer_line2 = (
            "Use the above experiences as context to improve "
            "your response."
        )
        # Reserve space for header, footer, and newlines between them.
        reserved = len(header) + len(footer_line1) + len(footer_line2) + 4
        budget = self._max_exp_chars - reserved
        if budget <= 0:
            return ""

        lines: list[str] = [header]
        count = 0

        for exp in experiences:
            hint_text = exp.get("hint", "")
            candidate = f"{count + 1}. {hint_text}"
            if budget - len(candidate) < 0:
                break
            lines.append(candidate)
            budget -= len(candidate)
            count += 1

        if count == 0:
            return ""

        lines.append(footer_line1)
        lines.append(footer_line2)
        return "\n".join(lines)
