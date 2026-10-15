"""Integration test: run the extractor on the three synthetic reports.

These reports are synthetic (see data/synthetic/README.md). They are
NOT real patient documents. The test's purpose is to verify that the
extractor handles the three designed scenarios.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from backend.nlp.extractor import extract

REPORTS_DIR = Path(__file__).resolve().parents[2] / "data" / "synthetic"


def _load(name: str) -> str:
    return (REPORTS_DIR / name).read_text(encoding="utf-8")


@pytest.mark.integration
def test_report_agreement() -> None:
    text = _load("report_agreement.txt")
    r = extract(text)
    names = {f.name for f in r.findings}
    print("\n=== report_agreement.txt ===")
    for f in r.findings:
        print(f"  {f.name:<20s} {f.status:<10s} {f.anatomy:<12s} unc={f.uncertainty}")
    assert "Cardiomegaly" in names
    assert "Effusion" in names
    assert "Pneumothorax" in names
    # Pneumothorax should be negated in this report.
    pt = [f for f in r.findings if f.name == "Pneumothorax"]
    assert pt and pt[0].status == "negated"


@pytest.mark.integration
def test_report_conflict() -> None:
    text = _load("report_conflict.txt")
    r = extract(text)
    print("\n=== report_conflict.txt ===")
    for f in r.findings:
        print(f"  {f.name:<20s} {f.status:<10s} {f.anatomy:<12s} unc={f.uncertainty}")
    # This report negates everything. Every extracted finding should be negated.
    assert r.findings, "Expected at least one finding to be extracted"
    assert all(f.status == "negated" for f in r.findings)


@pytest.mark.integration
def test_report_negation_uncertainty() -> None:
    text = _load("report_negation_uncertainty.txt")
    r = extract(text)
    print("\n=== report_negation_uncertainty.txt ===")
    for f in r.findings:
        print(f"  {f.name:<20s} {f.status:<10s} {f.anatomy:<12s} unc={f.uncertainty}")
    statuses = {f.status for f in r.findings}
    # Should have at least one negated and one uncertain.
    assert "negated" in statuses
    assert "uncertain" in statuses