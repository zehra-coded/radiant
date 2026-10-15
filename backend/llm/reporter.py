"""
LLM reporter.

Orchestrates: build payload -> safety gate (pre) -> call AMD ->
safety gate (post) -> fallback to template on any failure.

The generator field on the output is one of:
- "amd_token_factory" if the LLM call succeeded and passed post-check
- "template" if the template was used (with a reason)
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from backend.core.logging import get_logger
from backend.fusion.schema import FusionResult
from backend.llm import amd_client, payload as payload_builder, prompts
from backend.llm.template_reporter import generate as template_generate
from backend.nlp.schema import ReportExtraction
from backend.rag.schema import Chunk
from backend.safety.gate import post_generation_check, pre_generation_check
from backend.safety.schema import SafetyResult
from backend.triage.schema import TriageResult
from backend.vision.schema import VisionResult

logger = get_logger(__name__)


@dataclass
class ReporterOutput:
    text: str
    generator: str          # "amd_token_factory" or "template"
    fallback_reason: str | None
    pre_safety: SafetyResult
    post_safety: SafetyResult


def generate_explanation(
    user_query: str,
    vision: VisionResult | None,
    report: ReportExtraction | None,
    fusion: FusionResult | None,
    triage: TriageResult | None,
    evidence: list[Chunk],
    *,
    vision_failed: bool = False,
    report_failed: bool = False,
    retrieval_failed: bool = False,
) -> ReporterOutput:
    """Build a structured payload, run both safety gates, and produce text.

    Never raises to the caller on LLM failure — falls back to template.
    """
    pre = pre_generation_check(
        user_query, triage, fusion, evidence,
        vision_failed=vision_failed,
        report_failed=report_failed,
        retrieval_failed=retrieval_failed,
    )

    payload = payload_builder.build(vision, report, fusion, triage, evidence)

    fallback_reason: str | None = None
    text: str
    generator: str

    if not pre.allowed:
        text = template_generate(payload)
        generator = "template"
        fallback_reason = (
            "pre_generation_check blocked the LLM: "
            + ", ".join(pre.reason_codes)
        )
    elif not amd_client.is_configured():
        text = template_generate(payload)
        generator = "template"
        fallback_reason = "AMD endpoint not configured (RADEON_API_KEY or LLM_BASE_URL unset)"
    else:
        user_message = prompts.USER_TEMPLATE.format(
            payload_json=json.dumps(payload, indent=2, ensure_ascii=False)
        )
        try:
            text = amd_client.call(prompts.SYSTEM_PROMPT, user_message)
            generator = "amd_token_factory"
        except amd_client.LlmNotConfigured as exc:
            text = template_generate(payload)
            generator = "template"
            fallback_reason = f"LLM not configured: {exc}"
        except amd_client.LlmCallFailed as exc:
            text = template_generate(payload)
            generator = "template"
            fallback_reason = f"LLM call failed: {exc}"

    post = post_generation_check(text, triage, fusion, evidence, generator=generator)

    if not post.allowed and generator == "amd_token_factory":
        # LLM output failed validation; replace with template.
        logger.warning("LLM output failed post-check; falling back to template")
        text = template_generate(payload)
        generator = "template"
        fallback_reason = (
            "post_generation_check rejected LLM output: "
            + ", ".join(post.reason_codes)
        )
        post = post_generation_check(text, triage, fusion, evidence, generator=generator)

    return ReporterOutput(
        text=text,
        generator=generator,
        fallback_reason=fallback_reason,
        pre_safety=pre,
        post_safety=post,
    )