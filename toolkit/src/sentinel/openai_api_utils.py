"""Shared helpers for the OpenAI-compatible adapter, generator, and judge.

All three components share the same plumbing: resolve an API key from the
manifest or the environment, ask an OpenAI-style endpoint for a chat
completion, pull the assistant text and token usage from the response, and
parse a JSON-or-keyword verdict back out. These helpers centralise that
boilerplate so the three classes stay short and consistent.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional, Tuple


_VERDICT_LABELS = ("refusal", "compliance", "inconclusive")


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def resolve_api_key(explicit_key: Optional[Any]) -> Optional[str]:
    """Return *explicit_key* (cast to str) or fall back to env vars.

    Resolution order: ``api_key`` from the config block → ``SENTINEL_API_KEY``
    → ``OPENAI_API_KEY``. Returns ``None`` when no key is available.
    """
    if explicit_key:
        return str(explicit_key)
    return (
        os.environ.get("SENTINEL_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def extract_message_text(data: Dict[str, Any]) -> str:
    """Pull ``choices[0].message.content`` out of a chat-completion payload."""
    choice = data.get("choices", [{}])[0]
    message = choice.get("message", {}) if isinstance(choice, dict) else {}
    return str(message.get("content", "")).strip()


def extract_finish_reason(data: Dict[str, Any]) -> str:
    """Return the ``finish_reason`` field of the first choice (if any)."""
    choice = data.get("choices", [{}])[0]
    if isinstance(choice, dict):
        return choice.get("finish_reason", "") or ""
    return ""


def extract_token_usage(data_or_usage: Dict[str, Any]) -> int:
    """Pick the most informative integer token count out of a usage dict.

    Accepts either the full chat-completion payload (in which case the
    ``usage`` sub-dict is consulted) or a usage dict directly. Falls back
    to ``0`` when no integer token count is present.
    """
    if not isinstance(data_or_usage, dict):
        return 0
    # Use the nested ``usage`` block if present; otherwise treat the input as
    # the usage block itself.
    nested = data_or_usage.get("usage")
    usage = nested if isinstance(nested, dict) else data_or_usage
    for key in ("total_tokens", "completion_tokens"):
        value = usage.get(key)
        if isinstance(value, int):
            return value
    return 0


def parse_verdict_text(raw: str) -> Tuple[str, float, str]:
    """Extract ``(label, confidence, explanation)`` from a judge model's reply.

    Handles three response shapes:

    1. A plain JSON object — ``{"label": ..., "confidence": ..., "explanation": ...}``.
    2. A JSON object embedded in surrounding prose or markdown fences.
    3. Free-form text that mentions one of the canonical labels by name.

    Confidence is clamped to ``[0, 1]``. Unknown labels collapse to
    ``"inconclusive"`` with confidence ``0.0``.
    """
    text = raw.strip()
    parsed: Optional[Dict[str, Any]] = None
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        candidate = match.group() if match else text
        loaded = json.loads(candidate)
        if isinstance(loaded, dict):
            parsed = loaded
    except (json.JSONDecodeError, AttributeError):
        parsed = None

    if parsed is not None:
        label = str(parsed.get("label", "")).strip().lower()
        confidence = float(parsed.get("confidence", 0.0) or 0.0)
        explanation = str(parsed.get("explanation", text)).strip() or text
        if label in _VERDICT_LABELS:
            return label, max(0.0, min(1.0, confidence)), explanation

    # Fallback: keyword scan when JSON is missing or malformed.
    lower = text.lower()
    for label in _VERDICT_LABELS:
        if re.search(rf"\b{label}\b", lower):
            return label, 0.5, text
    return "inconclusive", 0.0, text
