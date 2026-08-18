"""Minimal OpenAI-compatible chat client for evolution-time LLM calls.

Configured entirely through env vars. Returns None when unconfigured so
callers can fall back to heuristic behavior.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


def is_configured() -> bool:
    return bool(os.getenv("SKILL_LLM_API_BASE") and os.getenv("SKILL_LLM_MODEL"))


def chat(system: str, user: str, temperature: float = 0.0, timeout: int = 60) -> str | None:
    api_base = os.getenv("SKILL_LLM_API_BASE", "").rstrip("/")
    model = os.getenv("SKILL_LLM_MODEL", "")
    if not api_base or not model:
        return None

    payload = json.dumps(
        {
            "model": model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
    ).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    api_key = os.getenv("SKILL_LLM_API_KEY", "")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = urllib.request.Request(
        f"{api_base}/chat/completions", data=payload, headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]
    except (urllib.error.HTTPError, urllib.error.URLError, OSError, KeyError, json.JSONDecodeError):
        return None
