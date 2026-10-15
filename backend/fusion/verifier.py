"""
Cross-modal verifier.

Takes a VisionResult and a ReportExtraction, normalizes both to the 14
NIH labels, and returns a FusionResult.

Rules (all from project spec):

- vision present + report negated  -> conflict (never resolved)
- vision present + report absent   -> vision_only
- vision absent  + report present  -> report_only
- vision present + report present  -> agreed
- vision absent  + report negated  -> agreed
- vision uncertain                 -> never a conflict, never an agreement
- report uncertain                 -> never a conflict, never an agreement
- any conflict on a critical label -> forces overall_consistency = "conflicting"
- both modalities empty            -> insufficient (never "high")
- one modality missing / failed    -> insufficient with a review signal

All thresholds come from config.py. Heuristics are UNVALIDATED PROTOTYPE
HEURISTICS.
"""
from __future__ import annotations

from backend.core.config import (
    FUSION_CRITICAL_LABELS,
    FUSION_HIGH_RATIO,
    FUSION_PARTIAL_RATIO,
)
from backend.core.logging import get_logger
from backend.fusion.schema import FusionConflict, FusionResult
from backend.nlp.schema import ReportExtraction
from backend.vision.schema import NIH_LABELS, VisionResult

logger = get_logger(__name__)


def _vision_status_by_label(vision: VisionResult | None) -> dict[str, str]:
    """Return {label: status} for the 14 NIH labels."""
    if vision is None:
        return {}
    return {f.name: f.status for f in vision.findings}


def _report_status_by_label(report: ReportExtraction | None) -> dict[str, str]:
    """Return {label: status} for the 14 NIH labels.

    A label with multiple report findings (e.g. left present and right
    uncertain) is collapsed as follows:
      - if any finding is present -> "present"
      - elif any is uncertain -> "uncertain"
      - elif any is negated -> "negated"
      - else not present in the dict (not_mentioned)
    """
    if report is None:
        return {}
    result: dict[str, str] = {}
    for finding in report.findings:
        current = result.get(finding.name)
        if current is None:
            result[finding.name] = finding.status
            continue
        # Priority: present > uncertain > negated
        priority = {"present": 3, "uncertain": 2, "negated": 1}
        if priority[finding.status] > priority[current]:
            result[finding.name] = finding.status
    return result


def _classify_pair(vision_status: str | None, report_status: str | None) -> str:
    """Return one of: agreed, vision_only, report_only, conflict, skip.

    vision_status is None if the vision module did not run or the label
    is absent from the vision result.
    report_status is None if the report did not mention the label.
    """
    # Uncertain in either modality: skip. Never agreement, never conflict.
    if vision_status == "uncertain" or report_status == "uncertain":
        return "skip"

    # Vision present cases.
    if vision_status == "present":
        if report_status == "present":
            return "agreed"
        if report_status == "negated":
            return "conflict"
        if report_status is None:
            return "vision_only"

    # Vision absent cases.
    if vision_status == "absent":
        if report_status == "present":
            return "report_only"
        if report_status == "negated":
            return "agreed"
        if report_status is None:
            return "skip"  # absent in image, not mentioned in report: nothing to say

    return "skip"


def _compute_overall(
    agreed: list[str],
    vision_only: list[str],
    report_only: list[str],
    conflicts: list[FusionConflict],
    vision_missing: bool,
    report_missing: bool,
) -> tuple[str, float | None, list[str]]:
    """Return (overall_consistency, ratio_or_None, notes)."""
    notes: list[str] = []

    if vision_missing and report_missing:
        return "insufficient", None, ["both modalities missing"]

    if vision_missing:
        return "insufficient", None, ["vision modality missing or failed"]

    if report_missing:
        return "insufficient", None, ["report modality missing or empty"]

    total = len(agreed) + len(vision_only) + len(report_only) + len(conflicts)
    if total == 0:
        return "insufficient", None, [
            "no comparable labels (both modalities returned no findings)"
        ]

    ratio = len(agreed) / total

    # Critical-label override.
    conflict_labels = {c.name for c in conflicts}
    critical_conflicts = FUSION_CRITICAL_LABELS & conflict_labels
    if critical_conflicts:
        notes.append(
            "critical-label conflict override: "
            + ", ".join(sorted(critical_conflicts))
        )
        return "conflicting", ratio, notes

    if ratio >= FUSION_HIGH_RATIO:
        return "high", ratio, notes
    if ratio >= FUSION_PARTIAL_RATIO:
        return "partial", ratio, notes
    return "conflicting", ratio, notes


def verify(
    vision: VisionResult | None,
    report: ReportExtraction | None,
) -> FusionResult:
    """Compare vision and report findings, returning a FusionResult."""
    vision_missing = vision is None
    report_missing = report is None or not report.findings

    v = _vision_status_by_label(vision)
    r = _report_status_by_label(report)

    agreed: list[str] = []
    vision_only: list[str] = []
    report_only: list[str] = []
    conflicts: list[FusionConflict] = []
    skipped: list[str] = []

    for label in NIH_LABELS:
        v_status = v.get(label)
        r_status = r.get(label)

        # If vision did not report this label at all (e.g. module failed),
        # treat it as unavailable rather than "absent".
        if vision_missing:
            v_status = None

        if v_status is None and r_status is None:
            continue

        classification = _classify_pair(v_status, r_status)

        if classification == "agreed":
            agreed.append(label)
        elif classification == "vision_only":
            vision_only.append(label)
        elif classification == "report_only":
            report_only.append(label)
        elif classification == "conflict":
            reason = (
                f"vision says '{v_status}' but report says '{r_status}'"
            )
            conflicts.append(
                FusionConflict(
                    name=label,
                    vision_status=v_status or "unknown",
                    report_status=r_status or "unknown",
                    reason=reason,
                )
            )
        else:
            skipped.append(label)

    overall, ratio, notes = _compute_overall(
        agreed, vision_only, report_only, conflicts,
        vision_missing=vision_missing,
        report_missing=report_missing,
    )
    if skipped:
        notes.append(f"skipped (uncertain or nothing to compare): {', '.join(skipped)}")

    return FusionResult(
        agreed=agreed,
        vision_only=vision_only,
        report_only=report_only,
        conflicts=conflicts,
        overall_consistency=overall,  # type: ignore[arg-type]
        agreement_ratio=ratio,
        notes=notes,
    )