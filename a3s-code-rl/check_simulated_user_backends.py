#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_PATH = Path(
    os.getenv(
        "A3S_CODE_SIMULATED_USER_BACKENDS_FILE",
        str(SCRIPT_DIR / "simulated_user_backends.json"),
    )
)
DEFAULT_TIMEOUT_SEC = float(os.getenv("A3S_CODE_SIMULATED_USER_PROBE_TIMEOUT_SEC", "20"))
DEFAULT_CHAT_MAX_TOKENS = int(os.getenv("A3S_CODE_SIMULATED_USER_PROBE_MAX_TOKENS", "16"))
DEFAULT_SLOW_LATENCY_MS = float(os.getenv("A3S_CODE_SIMULATED_USER_SLOW_LATENCY_MS", "3000"))
DEFAULT_TRUST_ENV = os.getenv("A3S_CODE_SIMULATED_USER_PROBE_TRUST_ENV", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _split_csv(raw: str, *, keep_empty: bool = False) -> list[str]:
    if raw == "":
        return []
    items = [item.strip() for item in raw.split(",")]
    if keep_empty:
        return items
    return [item for item in items if item]


def _expand_values(values: list[str], size: int, default: str) -> list[str]:
    if size <= 0:
        return []
    if not values:
        values = [default]
    if len(values) == 1 and size > 1:
        return values * size
    if len(values) != size:
        raise RuntimeError(
            f"simulated-user backend config length mismatch: expected 1 or {size}, got {len(values)}"
        )
    return values


def _derive_models_url(chat_url: str) -> str | None:
    suffix = "/v1/chat/completions"
    if chat_url.endswith(suffix):
        return chat_url[: -len(suffix)] + "/v1/models"
    return None


def _build_candidates() -> list[dict[str, str]]:
    single_url = os.getenv("SIMULATED_USER_MODEL_URL", "").strip()
    single_name = os.getenv("SIMULATED_USER_MODEL_NAME", "kimi-k2.5").strip()
    single_key = os.getenv("SIMULATED_USER_API_KEY", "")

    urls = _split_csv(os.getenv("SIMULATED_USER_MODEL_URLS", ""))
    if not urls and single_url:
        urls = [single_url]
    if not urls:
        raise RuntimeError("no simulated-user backend candidates configured")

    names = _expand_values(
        _split_csv(os.getenv("SIMULATED_USER_MODEL_NAMES", "")),
        len(urls),
        single_name,
    )
    keys = _expand_values(
        _split_csv(os.getenv("SIMULATED_USER_API_KEYS", ""), keep_empty=True),
        len(urls),
        single_key,
    )
    return [
        {"url": url, "model": model, "api_key": api_key}
        for url, model, api_key in zip(urls, names, keys)
    ]


def _probe_models(client: httpx.Client, models_url: str | None, headers: dict[str, str]) -> dict[str, Any]:
    if not models_url:
        return {
            "models_ok": False,
            "models_status_code": None,
            "models_latency_ms": None,
            "models_error": "models_url_unavailable",
        }

    started = time.perf_counter()
    try:
        resp = client.get(models_url, headers=headers)
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        resp.raise_for_status()
        payload = resp.json()
        model_ids = []
        if isinstance(payload, dict) and isinstance(payload.get("data"), list):
            model_ids = [
                str(item.get("id", ""))
                for item in payload["data"]
                if isinstance(item, dict) and item.get("id")
            ]
        return {
            "models_ok": True,
            "models_status_code": resp.status_code,
            "models_latency_ms": latency_ms,
            "models_error": None,
            "available_models": model_ids[:8],
        }
    except Exception as exc:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        return {
            "models_ok": False,
            "models_status_code": status_code,
            "models_latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "models_error": f"{type(exc).__name__}: {exc}",
        }


def _probe_chat(
    client: httpx.Client,
    chat_url: str,
    headers: dict[str, str],
    *,
    model: str,
    max_tokens: int,
) -> dict[str, Any]:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with OK only."}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    started = time.perf_counter()
    try:
        resp = client.post(chat_url, headers=headers, json=body)
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        resp.raise_for_status()
        payload = resp.json()
        text = ""
        if isinstance(payload, dict):
            choices = payload.get("choices", [])
            if choices and isinstance(choices[0], dict):
                text = str(choices[0].get("message", {}).get("content", "") or "")
        return {
            "chat_ok": True,
            "chat_status_code": resp.status_code,
            "chat_latency_ms": latency_ms,
            "chat_error": None,
            "chat_preview": text[:80],
        }
    except Exception as exc:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        return {
            "chat_ok": False,
            "chat_status_code": status_code,
            "chat_latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "chat_error": f"{type(exc).__name__}: {exc}",
            "chat_preview": "",
        }


def _probe_backend(
    client: httpx.Client,
    backend: dict[str, str],
    *,
    index: int,
    max_tokens: int,
) -> dict[str, Any]:
    url = backend["url"]
    model = backend["model"]
    api_key = backend["api_key"]
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    label = f"{index}:{model}@{url.replace('http://', '').replace('https://', '')}"

    models_info = _probe_models(client, _derive_models_url(url), headers)
    chat_info = _probe_chat(client, url, headers, model=model, max_tokens=max_tokens)
    healthy = bool(chat_info["chat_ok"])

    return {
        "label": label,
        "url": url,
        "model": model,
        "api_key": api_key,
        "enabled": True,
        "healthy": healthy,
        "priority": 10000 + index,
        "probe": {
            **models_info,
            **chat_info,
        },
    }


def _assign_backend_priorities(backends: list[dict[str, Any]], *, slow_latency_ms: float) -> None:
    healthy = [item for item in backends if item.get("enabled") and item.get("healthy")]
    healthy.sort(key=lambda item: float(item.get("probe", {}).get("chat_latency_ms") or 1e12))
    fast_rank = 0
    slow_rank = 0
    for item in healthy:
        latency_ms = float(item.get("probe", {}).get("chat_latency_ms") or 1e12)
        if latency_ms <= slow_latency_ms:
            fast_rank += 1
            item["priority"] = 100 + fast_rank
            item["latency_tier"] = "fast"
        else:
            slow_rank += 1
            item["priority"] = 1000 + slow_rank
            item["latency_tier"] = "slow"
    for item in backends:
        if not item.get("healthy"):
            item["latency_tier"] = "unhealthy"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=DEFAULT_TIMEOUT_SEC,
    )
    parser.add_argument(
        "--chat-max-tokens",
        type=int,
        default=DEFAULT_CHAT_MAX_TOKENS,
    )
    parser.add_argument(
        "--slow-latency-ms",
        type=float,
        default=DEFAULT_SLOW_LATENCY_MS,
    )
    parser.add_argument(
        "--trust-env",
        action="store_true",
        default=DEFAULT_TRUST_ENV,
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    candidates = _build_candidates()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with httpx.Client(timeout=args.timeout_sec, trust_env=args.trust_env) as client:
        backends = [
            _probe_backend(
                client,
                backend,
                index=index,
                max_tokens=args.chat_max_tokens,
            )
            for index, backend in enumerate(candidates, start=1)
        ]
    _assign_backend_priorities(backends, slow_latency_ms=args.slow_latency_ms)

    healthy_count = sum(1 for item in backends if item["healthy"] and item["enabled"])
    payload = {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "probe_timeout_sec": args.timeout_sec,
        "chat_max_tokens": args.chat_max_tokens,
        "slow_latency_ms": args.slow_latency_ms,
        "trust_env": bool(args.trust_env),
        "healthy_backend_count": healthy_count,
        "backends": backends,
    }
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[sim-user-probe] wrote {args.output}")
    for backend in backends:
        probe = backend["probe"]
        print(
            "[sim-user-probe] "
            f"{backend['label']} enabled={backend['enabled']} healthy={backend['healthy']} "
            f"priority={backend.get('priority')} tier={backend.get('latency_tier')} "
            f"models_ok={probe.get('models_ok')} chat_ok={probe.get('chat_ok')} "
            f"models_status={probe.get('models_status_code')} chat_status={probe.get('chat_status_code')} "
            f"chat_latency_ms={probe.get('chat_latency_ms')}",
            flush=True,
        )


if __name__ == "__main__":
    main()
