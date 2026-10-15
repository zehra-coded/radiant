"""Fusion evaluation on labeled synthetic cases."""
from __future__ import annotations

from backend.fusion.verifier import verify
from backend.nlp.schema import ReportExtraction, ReportFinding
from backend.vision.schema import NIH_LABELS, VisionFinding, VisionResult


def _vision(spec: dict[str, str]) -> VisionResult:
    return VisionResult(
        findings=[
            VisionFinding(
                name=n,
                model_score={"present": 0.8, "uncertain": 0.4, "absent": 0.1}[
                    spec.get(n, "absent")
                ],
                status=spec.get(n, "absent"),  # type: ignore[arg-type]
            )
            for n in NIH_LABELS
        ],
        checkpoint="eval",
    )


def _report(spec: dict[str, str]) -> ReportExtraction:
    return ReportExtraction(
        findings=[
            ReportFinding(
                name=name,
                anatomy="unspecified",
                status=status,  # type: ignore[arg-type]
                uncertainty=0.0 if status != "uncertain" else 1.0,
                evidence_span="eval",
                spatial_info=None,
            )
            for name, status in spec.items()
        ],
        raw_text="eval",
    )


CASES = [
    {
        "name": "full_agreement",
        "vision": {"Cardiomegaly": "present", "Effusion": "present"},
        "report": {"Cardiomegaly": "present", "Effusion": "present"},
        "expected_consistency": "high",
        "expected_conflicts": 0,
    },
    {
        "name": "both_empty",
        "vision": {},
        "report": {},
        "expected_consistency": "insufficient",
        "expected_conflicts": 0,
    },
    {
        "name": "critical_conflict",
        "vision": {"Pneumothorax": "present"},
        "report": {"Pneumothorax": "negated"},
        "expected_consistency": "conflicting",
        "expected_conflicts": 1,
    },
    {
        "name": "noncritical_double_conflict",
        "vision": {"Nodule": "present", "Mass": "present"},
        "report": {"Nodule": "negated", "Mass": "negated"},
        "expected_consistency": "conflicting",
        "expected_conflicts": 2,
    },
    {
        "name": "uncertain_vision_skipped",
        "vision": {"Effusion": "uncertain"},
        "report": {"Effusion": "negated"},
        "expected_consistency": "insufficient",
        "expected_conflicts": 0,
    },
    {
        "name": "partial_with_conflict",
        "vision": {"Cardiomegaly": "present", "Effusion": "present"},
        "report": {"Cardiomegaly": "present", "Effusion": "negated"},
        "expected_consistency": "conflicting",
        "expected_conflicts": 1,
    },
    {
        "name": "report_only_not_conflict",
        "vision": {"Effusion": "absent"},
        "report": {"Effusion": "present"},
        "expected_consistency": "conflicting",
        "expected_conflicts": 0,
    },
]


def evaluate() -> dict:
    correct_consistency = correct_conflicts = 0
    rows: list[dict] = []
    for case in CASES:
        result = verify(_vision(case["vision"]), _report(case["report"]))
        cons_ok = result.overall_consistency == case["expected_consistency"]
        conf_ok = len(result.conflicts) == case["expected_conflicts"]
        correct_consistency += int(cons_ok)
        correct_conflicts += int(conf_ok)
        rows.append({
            "name": case["name"],
            "expected_consistency": case["expected_consistency"],
            "actual_consistency": result.overall_consistency,
            "consistency_match": cons_ok,
            "expected_conflicts": case["expected_conflicts"],
            "actual_conflicts": len(result.conflicts),
            "conflicts_match": conf_ok,
        })

    n = len(CASES)
    return {
        "cases": n,
        "consistency_accuracy": correct_consistency / n,
        "conflict_count_accuracy": correct_conflicts / n,
        "rows": rows,
        "notes": [
            "7 labeled synthetic cases. Not a clinical evaluation — "
            "tests the verifier logic against its own specification.",
        ],
    }