"""
Privacy enforcement for outbound LLM payloads.

Two layers of defense:

1. Allowlist — the outbound payload may contain ONLY the fields in
   LLM_PAYLOAD_ALLOWLIST. Everything else is stripped.
2. Denylist — the presence of any field named in
   PATIENT_IDENTIFIER_DENYLIST is a HARD FAILURE, not a silent drop.
   Presence of these fields indicates a mapping bug, so we refuse to
   send anything at all.

This is rule 3.15 in code.
"""
from __future__ import annotations

from typing import Any

from backend.core.config import (
    LLM_PAYLOAD_ALLOWLIST,
    PATIENT_IDENTIFIER_DENYLIST,
)
from backend.core.logging import get_logger

logger = get_logger(__name__)


class PrivacyViolation(Exception):
    """Raised when an outbound payload contains a forbidden identifier."""


def enforce(payload: dict[str, Any]) -> dict[str, Any]:
    """Strip non-allowlisted keys and reject any denylisted key.

    Returns a new dict containing only allowlisted keys.

    Raises PrivacyViolation if any denylisted key is present.
    """
    # Hard failure on denylisted keys, at any nesting level.
    hits = _find_denylisted(payload, path="")
    if hits:
        msg = "denylisted patient identifiers present: " + ", ".join(hits)
        logger.error("privacy violation: %s", msg)
        raise PrivacyViolation(msg)

    # Strip to allowlist.
    cleaned = {k: v for k, v in payload.items() if k in LLM_PAYLOAD_ALLOWLIST}

    dropped = [k for k in payload.keys() if k not in LLM_PAYLOAD_ALLOWLIST]
    if dropped:
        logger.warning(
            "privacy: dropped non-allowlisted payload fields: %s",
            ", ".join(sorted(dropped)),
        )

    return cleaned


def _find_denylisted(obj: Any, path: str) -> list[str]:
    """Recursively find any dict key matching the denylist."""
    hits: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            new_path = f"{path}.{key}" if path else str(key)
            if key in PATIENT_IDENTIFIER_DENYLIST:
                hits.append(new_path)
            hits.extend(_find_denylisted(value, new_path))
    elif isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            hits.extend(_find_denylisted(item, f"{path}[{i}]"))
    return hits


def assert_safe(payload: dict[str, Any]) -> None:
    """Raise PrivacyViolation if the payload contains any denylisted key.

    Does not modify the payload.
    """
    hits = _find_denylisted(payload, path="")
    if hits:
        raise PrivacyViolation(
            "denylisted patient identifiers present: " + ", ".join(hits)
        )