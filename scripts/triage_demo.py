"""Manual demo: run vision + NLP + fusion + triage end to end."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.fusion.verifier import verify
from backend.nlp.extractor import extract
from backend.triage.engine import triage
from backend.vision.model import VisionModel

IMG = Path(r"D:\NIH-ChestXray14\images\00000001_000.png")
REPORT = Path(__file__).resolve().parents[1] / "data" / "synthetic" / "report_conflict.txt"


def main() -> None:
    print(f"Image:  {IMG}")
    print(f"Report: {REPORT.name}")
    print()

    v = VisionModel().predict(IMG)
    r = extract(REPORT.read_text(encoding="utf-8"))
    fusion = verify(v, r)
    result = triage(v, r, fusion, None)

    print(f"Fusion consistency:  {fusion.overall_consistency}")
    print(f"Conflicts:           {[c.name for c in fusion.conflicts]}")
    print(f"Priority:            {result.priority}")
    print()
    print("Reasons:")
    for reason in result.reasons:
        print(f"  - {reason}")
    print()
    print("Source signals:")
    for s in result.source_signals:
        print(f"  - {s}")
    print()
    print("Decision trace:")
    for t in result.decision_trace:
        print(f"  - {t}")


if __name__ == "__main__":
    main()