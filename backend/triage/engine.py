"""
Triage engine.

Takes the uncertainty signals, the fusion result, and the vision/report
outputs and returns a TriageResult.

Precedence (higher wins):

    EMERGENCY_REVIEW  >  URGENT_REVIEW  >  MONITOR  >  ROUTINE

Hard rules:

- model_failure          ->  never ROUTINE
- insufficient           ->  never ROUTINE
- vision/report conflict ->  at least URGENT_REVIEW
- critical label present ->  at least URGENT_REVIEW

All thresholds come from config.py and are UNVALIDATED PROTOTYPE
HEURISTICS.
"""
from __future__ import annotations

from backend.core.config import TRIAGE_CRITICAL_PRESENT_LABELS
from backend.fusion.schema import FusionResult
from backend.nlp.schema import ReportExtraction
from backend.triage.schema import ReviewPriority, TriageResult
from backend.uncertainty import signals as sig
from backend.vision.schema import VisionResult

_PRIORITY_INDEX: dict[str, int] = {
    "ROUTINE": 0,
    "MONITOR": 1,
    "URGENT_REVIEW": 2,
    "EMERGENCY_REVIEW": 3,
}


def _max_priority(a: str, b: str) -> str:
    return a if _PRIORITY_INDEX[a] >= _PRIORITY_INDEX[b] else b


def _critical_present_labels(
    vision: VisionResult | None,
    report: ReportExtraction | None,
) -> list[str]:
    """Return NIH labels that are (a) critical and (b) present in at least
    one modality."""
    present: set[str] = set()
    if vision is not None:
        for f in vision.findings:
            if f.status == "present":
                present.add(f.name)
    if report is not None:
        for f in report.findings:
            if f.status == "present":
                present.add(f.name)
    return sorted(present & TRIAGE_CRITICAL_PRESENT_LABELS)


def triage(
    vision: VisionResult | None,
    report: ReportExtraction | None,
    fusion: FusionResult | None,
    uncertainty: sig.UncertaintySignals | None,
    *,
    vision_failed: bool = False,
    report_failed: bool = False,
    retrieval_failed: bool = False,
) -> TriageResult:
    """Compute a review priority from upstream outputs."""
    if uncertainty is None:
        uncertainty = sig.compute(
            vision, report, fusion,
            vision_failed=vision_failed,
            report_failed=report_failed,
            retrieval_failed=retrieval_failed,
        )

    reasons: list[str] = []
    trace: list[str] = []
    notes: list[str] = []
    priority: str = "ROUTINE"

    # Rule 1: any model failure -> at least URGENT_REVIEW, never ROUTINE.
    if uncertainty.has(sig.SIGNAL_MODEL_FAILURE):
        priority = _max_priority(priority, "URGENT_REVIEW")
        reasons.append("One or more pipeline modules failed; human review required.")
        trace.append("rule_model_failure: FIRED -> URGENT_REVIEW")

    # Rule 2: retrieval failure -> at least MONITOR (evidence missing).
    if uncertainty.has(sig.SIGNAL_RETRIEVAL_FAILURE):
        priority = _max_priority(priority, "MONITOR")
        reasons.append("Evidence retrieval failed; explanation may be unsupported.")
        trace.append("rule_retrieval_failure: FIRED -> MONITOR")

    # Rule 3: insufficient report -> at least MONITOR.
    if uncertainty.has(sig.SIGNAL_INSUFFICIENT_REPORT):
        priority = _max_priority(priority, "MONITOR")
        reasons.append("Report contained no findings; cross-modal comparison incomplete.")
        trace.append("rule_insufficient_report: FIRED -> MONITOR")

    # Rule 4: vision/report conflict -> at least URGENT_REVIEW.
    if uncertainty.has(sig.SIGNAL_VISION_REPORT_CONFLICT):
        priority = _max_priority(priority, "URGENT_REVIEW")
        conflict_names = (
            [c.name for c in fusion.conflicts] if fusion is not None else []
        )
        reasons.append(
            "Image and report disagree on: "
            + ", ".join(sorted(set(conflict_names))) if conflict_names
            else "Image and report disagree on at least one finding."
        )
        trace.append("rule_vision_report_conflict: FIRED -> URGENT_REVIEW")

    # Rule 5: critical label present -> at least URGENT_REVIEW.
    critical = _critical_present_labels(vision, report)
    if critical:
        priority = _max_priority(priority, "URGENT_REVIEW")
        reasons.append(
            "Critical finding present: " + ", ".join(critical)
        )
        trace.append("rule_critical_present: FIRED -> URGENT_REVIEW")

    # Rule 6: ambiguous predictions -> at least MONITOR.
    if uncertainty.has(sig.SIGNAL_AMBIGUOUS_PREDICTIONS):
        priority = _max_priority(priority, "MONITOR")
        reasons.append("Vision model produced many ambiguous predictions.")
        trace.append("rule_ambiguous_predictions: FIRED -> MONITOR")

    # Rule 7: low confidence -> at least MONITOR.
    if uncertainty.has(sig.SIGNAL_LOW_CONFIDENCE):
        priority = _max_priority(priority, "MONITOR")
        reasons.append("Some vision findings have low confidence scores.")
        trace.append("rule_low_confidence: FIRED -> MONITOR")

    # Rule 8: overall fusion consistency high + no other flags -> ROUTINE.
    if fusion is not None and fusion.overall_consistency == "high" and priority == "ROUTINE":
        reasons.append("Image and report are consistent; no uncertainty signals fired.")
        trace.append("rule_consistent: FIRED -> ROUTINE")

    # Rule 9: fusion insufficient and no failure signal -> MONITOR, never ROUTINE.
    if fusion is not None and fusion.overall_consistency == "insufficient":
        if priority == "ROUTINE":
            priority = "MONITOR"
            reasons.append("Cross-modal comparison insufficient; defaulting to MONITOR.")
            trace.append("rule_insufficient_fusion: FIRED -> MONITOR (never ROUTINE)")
        else:
            trace.append("rule_insufficient_fusion: already above ROUTINE")

    # Safety assertions from spec: failure and insufficient must never be ROUTINE.
    if uncertainty.has(sig.SIGNAL_MODEL_FAILURE) and priority == "ROUTINE":
        raise RuntimeError(
            "Triage invariant violated: model_failure produced ROUTINE"
        )
    if fusion is not None and fusion.overall_consistency == "insufficient" and priority == "ROUTINE":
        raise RuntimeError(
            "Triage invariant violated: insufficient fusion produced ROUTINE"
        )

    return TriageResult(
        priority=priority,  # type: ignore[arg-type]
        reasons=reasons,
        source_signals=uncertainty.signals(),
        decision_trace=trace,
        notes=notes,
    )