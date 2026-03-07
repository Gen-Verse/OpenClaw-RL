"""
ExperienceExtractor — extract improvement hints from conversation pairs.

Uses a Judge LLM (via OpenAI-compatible API) to infer what was lacking in
an assistant response based on the *next* user message.  This is the core
of the **ExpLoop** approach: surface quality signals from conversation
flow without any parameter updates.

Part of the **openclaw-exploop** project (training-free experience loop).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_JUDGE_MODEL = "gpt-4o-mini"

_HINT_EXTRACT_PROMPT = """\
You are an expert conversation analyst.  Given:

1. **Assistant response** (what the model said)
2. **Next user message** (the user's follow-up)

Infer what was insufficient, incorrect, or missing in the assistant \
response that caused the user to follow up.  Produce a concise, \
actionable *improvement hint* that a model could use in future similar \
situations.

Rules:
- The hint MUST be a single paragraph, 1-4 sentences.
- Focus on *what to do differently*, not what went wrong.
- If the next user message is simply a new topic or a "thanks", \
  respond with exactly: NO_HINT
- Do NOT repeat the assistant response verbatim.
"""

_TRIVIAL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^no[_ ]?hint$", re.IGNORECASE),
    re.compile(r"^\s*$"),
]

_MIN_HINT_LENGTH = 20  # chars


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_trivial(hint: str) -> bool:
    """Return True if *hint* carries no useful information."""
    stripped = hint.strip()
    if len(stripped) < _MIN_HINT_LENGTH:
        return True
    for pat in _TRIVIAL_PATTERNS:
        if pat.fullmatch(stripped):
            return True
    return False


def _build_user_content(
    response_text: str,
    next_user_message: str,
) -> str:
    return (
        f"**Assistant response:**\n{response_text[:1500]}\n\n"
        f"**Next user message:**\n{next_user_message[:1500]}"
    )


# ---------------------------------------------------------------------------
# ExperienceExtractor
# ---------------------------------------------------------------------------

class ExperienceExtractor:
    """Extract improvement hints by calling a Judge LLM.

    Parameters
    ----------
    base_url : str | None
        OpenAI-compatible base URL.  Falls back to ``JUDGE_BASE_URL``
        then ``UPSTREAM_BASE_URL`` environment variables.
    api_key : str | None
        API key.  Falls back to ``JUDGE_API_KEY`` then
        ``UPSTREAM_API_KEY`` environment variables.
    model : str | None
        Model name.  Falls back to ``JUDGE_MODEL`` env var.
    timeout : float
        HTTP timeout in seconds for each judge call.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = (
            base_url
            or os.environ.get("JUDGE_BASE_URL")
            or os.environ.get("UPSTREAM_BASE_URL", "")
        ).rstrip("/")
        self.api_key = (
            api_key
            or os.environ.get("JUDGE_API_KEY")
            or os.environ.get("UPSTREAM_API_KEY", "")
        )
        self.model = (
            model
            or os.environ.get("JUDGE_MODEL", _DEFAULT_JUDGE_MODEL)
        )
        self.timeout = timeout

        if not self.base_url:
            logger.warning(
                "[ExperienceExtractor] no base_url configured — "
                "hint extraction will fail until set"
            )

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------

    async def extract_hint(
        self,
        response_text: str,
        next_user_message: str,
        session_id: str = "",
    ) -> str | None:
        """Call the Judge LLM and return an improvement hint, or *None*.

        Returns *None* when:
        - both inputs are empty / whitespace-only (nothing to analyse)
        - the judge says NO_HINT
        - the resulting hint is trivially short or empty
        - an HTTP / parsing error occurs (logged, not raised)
        """
        # BUG FIX: guard against empty inputs — the Judge LLM would
        # produce garbage or hallucinated hints when both fields are
        # blank.
        if not response_text.strip() and not next_user_message.strip():
            logger.debug(
                "[ExperienceExtractor] both inputs empty for "
                "session=%s — skipping",
                session_id,
            )
            return None

        user_content = _build_user_content(
            response_text, next_user_message,
        )
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _HINT_EXTRACT_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.3,
            "max_tokens": 256,
        }
        url = f"{self.base_url}/v1/chat/completions"
        headers: dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    url, json=payload, headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as exc:
            logger.error(
                "[ExperienceExtractor] judge HTTP %s for session=%s: %s",
                exc.response.status_code, session_id, exc,
            )
            return None
        except (httpx.RequestError, Exception) as exc:
            logger.error(
                "[ExperienceExtractor] judge request failed "
                "for session=%s: %s",
                session_id, exc,
            )
            return None

        # Parse response
        try:
            hint = (
                data["choices"][0]["message"]["content"]
                .strip()
            )
        except (KeyError, IndexError, TypeError) as exc:
            logger.warning(
                "[ExperienceExtractor] unexpected judge response "
                "structure for session=%s: %s",
                session_id, exc,
            )
            return None

        if _is_trivial(hint):
            logger.debug(
                "[ExperienceExtractor] trivial hint discarded "
                "for session=%s: %r",
                session_id, hint[:80],
            )
            return None

        logger.info(
            "[ExperienceExtractor] extracted hint for session=%s "
            "(%d chars)",
            session_id, len(hint),
        )
        return hint

    # ------------------------------------------------------------------
    # Batch
    # ------------------------------------------------------------------

    async def batch_extract(
        self,
        pairs: list[dict[str, str]],
    ) -> list[str | None]:
        """Extract hints for multiple (response, next_message) pairs.

        Each element of *pairs* should be a dict with keys:
        ``response_text``, ``next_user_message``, and optionally
        ``session_id``.

        Returns a list of the same length — hint strings or *None*.
        """
        tasks = [
            self.extract_hint(
                response_text=p.get("response_text", ""),
                next_user_message=p.get("next_user_message", ""),
                session_id=p.get("session_id", ""),
            )
            for p in pairs
        ]
        return await asyncio.gather(*tasks)
