"""
Outbound LLM payload builder.

This module owns the ONLY code path that constructs the JSON sent to
the LLM. It takes structured inputs (VisionResult, ReportExtraction,
FusionResult, TriageResult, retrieved chunks, safety info) and produces
a flat, text-only dict. No image bytes, no paths, no patient
identifiers. The result is passed through privacy.enforce() before
being sent.
"""
from __future__ import annotations

from typing import Any

from backend.fusion.schema import FusionResult
from backend.nlp.schema import ReportExtraction
from backend.rag.schema import Chunk
from backend.safety.privacy import enforce
from backend.safety.schema import DISCLAIMER
from backend.triage.schema import TriageResult
from backend.vision.schema import VisionResult


def build(
    vision: VisionResult | None,
    report: ReportExtraction | None,
    fusion: FusionResult | None,
    triage: TriageResult | None,
    evidence: list[Chunk],
) -> dict[str, Any]:
    """Build the outbound payload. Then strip to allowlist and enforce
    the patient-identifier denylist."""
    payload: dict[str, Any] = {
        "vision_scores": _vision_scores(vision),
        "report_findings": _report_findings(report),
        "fusion": _fusion(fusion),
        "uncertainty_signals": list(triage.source_signals) if triage else [],
        "review_priority": triage.priority if triage else None,
        "review_priority_reasons": list(triage.reasons) if triage else [],
        "evidence_chunks": [_evidence(c) for c in evidence],
        "safety_flags": {
            "requires_human_review": bool(
                triage and triage.priority in ("URGENT_REVIEW", "EMERGENCY_REVIEW")
            ),
        },
        "disclaimer": DISCLAIMER,
    }
    return enforce(payload)


def _vision_scores(vision: VisionResult | None) -> list[dict[str, Any]]:
    if vision is None:
        return []
    return [
        {"name": f.name, "model_score": f.model_score, "status": f.status}
        for f in vision.findings
    ]


def _report_findings(report: ReportExtraction | None) -> list[dict[str, Any]]:
    if report is None:
        return []
    return [
        {
            "name": f.name,
            "status": f.status,
            "anatomy": f.anatomy,
            "evidence_span": f.evidence_span,
        }
        for f in report.findings
    ]


def _fusion(fusion: FusionResult | None) -> dict[str, Any]:
    if fusion is None:
        return {}
    return {
        "agreed": list(fusion.agreed),
        "vision_only": list(fusion.vision_only),
        "report_only": list(fusion.report_only),
        "conflicts": [
            {
                "name": c.name,
                "vision_status": c.vision_status,
                "report_status": c.report_status,
                "reason": c.reason,
            }
            for c in fusion.conflicts
        ],
        "overall_consistency": fusion.overall_consistency,
        "agreement_ratio": fusion.agreement_ratio,
    }


def _evidence(chunk: Chunk) -> dict[str, Any]:
    return {
        "document_id": chunk.document_id,
        "chunk_id": chunk.chunk_id,
        "title": chunk.title,
        "text": chunk.text,
        "provenance": chunk.provenance,
        "license": chunk.license,
    }