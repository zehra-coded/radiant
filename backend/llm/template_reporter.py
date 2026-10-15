"""
Deterministic template reporter.

Used when:
- LLM_BASE_URL or RADEON_API_KEY is not set
- The AMD API call fails (network, 429, 5xx, invalid response)
- The safety gate blocks LLM generation

Sets generator="template". Never presents template output as LLM output.
"""
from __future__ import annotations

from typing import Any

from backend.safety.schema import DISCLAIMER


def generate(payload: dict[str, Any]) -> str:
    """Produce a deterministic explanation from the structured payload."""
    lines: list[str] = []
    lines.append("Summary (deterministic template — no LLM used).")
    lines.append("")

    priority = payload.get("review_priority")
    if priority:
        lines.append(f"Review priority: {priority}")

    fusion = payload.get("fusion") or {}
    consistency = fusion.get("overall_consistency")
    if consistency:
        lines.append(f"Image/report consistency: {consistency}")

    conflicts = fusion.get("conflicts") or []
    if conflicts:
        lines.append("")
        lines.append("Conflicts:")
        for c in conflicts:
            lines.append(
                f"- {c.get('name')}: vision says '{c.get('vision_status')}', "
                f"report says '{c.get('report_status')}'"
            )

    vision_scores = payload.get("vision_scores") or []
    present = [v for v in vision_scores if v.get("status") == "present"]
    if present:
        lines.append("")
        lines.append("Vision findings marked present (model scores, not probabilities):")
        for v in present:
            lines.append(f"- {v['name']}: {v['model_score']:.3f}")

    report_findings = payload.get("report_findings") or []
    if report_findings:
        lines.append("")
        lines.append("Report findings:")
        for r in report_findings:
            lines.append(f"- {r['name']} ({r['status']}, {r['anatomy']})")

    evidence = payload.get("evidence_chunks") or []
    if evidence:
        lines.append("")
        lines.append("Retrieved evidence:")
        for e in evidence:
            lines.append(
                f"- [source:{e['document_id']}:{e['chunk_id']}] "
                f"{e['title']} ({e['provenance']})"
            )
    else:
        lines.append("")
        lines.append("Insufficient evidence.")

    lines.append("")
    lines.append("Limitations:")
    lines.append(
        "This explanation is generated deterministically from the structured "
        "payload. No language model was used. Model scores are not calibrated "
        "probabilities. Thresholds are unvalidated prototype heuristics."
    )

    lines.append("")
    lines.append(DISCLAIMER)
    return "\n".join(lines)