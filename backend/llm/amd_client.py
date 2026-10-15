"""
OpenAI-compatible client for the AMD Token Factory endpoint.

Refuses to run with a vision-capable model (rule 3.5). Sends only
capability-flagged parameters.
"""
from __future__ import annotations

from backend.core.config import (
    AMD_CAPABILITIES,
    AMD_TEXT_ONLY_MODELS,
    AMD_VISION_MODELS,
    LLM_BASE_URL,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_REASONING_EFFORT,
    LLM_REQUEST_TIMEOUT_S,
    RADEON_API_KEY,
)
from backend.core.logging import get_logger

logger = get_logger(__name__)


class LlmNotConfigured(RuntimeError):
    """Raised when the AMD endpoint is not configured."""


class LlmCallFailed(RuntimeError):
    """Raised when the call to the LLM fails."""


def _validate_model() -> None:
    if LLM_MODEL in AMD_VISION_MODELS:
        raise LlmNotConfigured(
            f"Configured LLM_MODEL '{LLM_MODEL}' is vision-capable. "
            "Rule 3.5 forbids using it as the LLM."
        )
    if LLM_MODEL not in AMD_TEXT_ONLY_MODELS:
        raise LlmNotConfigured(
            f"Configured LLM_MODEL '{LLM_MODEL}' is not in the text-only allowlist."
        )


def is_configured() -> bool:
    return bool(RADEON_API_KEY) and bool(LLM_BASE_URL)


def call(system_prompt: str, user_message: str) -> str:
    """Make one text-only chat completion call. Returns the message text.

    Raises LlmNotConfigured if the endpoint is not configured.
    Raises LlmCallFailed if the request or response is bad.
    """
    if not is_configured():
        raise LlmNotConfigured("RADEON_API_KEY or LLM_BASE_URL not set")

    _validate_model()

    # Import here so tests can mock the client without importing openai.
    from openai import OpenAI

    client = OpenAI(
        base_url=LLM_BASE_URL,
        api_key=RADEON_API_KEY,
        timeout=LLM_REQUEST_TIMEOUT_S,
    )

    kwargs: dict = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "max_tokens": LLM_MAX_TOKENS,
    }
    if LLM_REASONING_EFFORT and AMD_CAPABILITIES.get(
        f"supports_reasoning_effort_{LLM_REASONING_EFFORT}", False
    ):
        kwargs["reasoning_effort"] = LLM_REASONING_EFFORT

    try:
        response = client.chat.completions.create(**kwargs)
    except Exception as exc:
        raise LlmCallFailed(f"LLM request failed: {exc}") from exc

    if not response.choices:
        raise LlmCallFailed("LLM returned no choices")

    message = response.choices[0].message
    content = getattr(message, "content", None)
    if not content or not content.strip():
        raise LlmCallFailed(
            "LLM returned empty content "
            "(possibly thinking consumed all tokens)"
        )
    return content.strip()


def list_models() -> list[str]:
    """Call GET /v1/models and return the model IDs. Used by the precheck."""
    if not is_configured():
        raise LlmNotConfigured("RADEON_API_KEY or LLM_BASE_URL not set")
    from openai import OpenAI

    client = OpenAI(
        base_url=LLM_BASE_URL,
        api_key=RADEON_API_KEY,
        timeout=LLM_REQUEST_TIMEOUT_S,
    )
    page = client.models.list()
    return [m.id for m in page.data]