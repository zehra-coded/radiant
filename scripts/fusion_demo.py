"""Manual demo: run vision + NLP + fusion on one NIH image and one synthetic report.

This is a developer-facing sanity check, not a test.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pathlib import Path

from backend.fusion.verifier import verify
from backend.nlp.extractor import extract
from backend.vision.model import VisionModel

IMG = Path(r"D:\NIH-ChestXray14\images\00000001_000.png")
REPORT = Path(__file__).resolve().parents[1] / "data" / "synthetic" / "report_conflict.txt"


def main() -> None:
    print(f"Image:  {IMG}")
    print(f"Report: {REPORT}")
    print()

    print("Running vision model...")
    v = VisionModel().predict(IMG)
    print("Vision present labels:",
          [f.name for f in v.findings if f.status == "present"])

    print()
    print("Running NLP extractor...")
    text = REPORT.read_text(encoding="utf-8")
    r = extract(text)
    print("Report present labels:",
          [f.name for f in r.findings if f.status == "present"])
    print("Report negated labels:",
          [f.name for f in r.findings if f.status == "negated"])

    print()
    print("Running fusion...")
    fusion = verify(v, r)
    print(f"Overall consistency: {fusion.overall_consistency}")
    print(f"Agreement ratio:     {fusion.agreement_ratio}")
    print(f"Agreed:              {fusion.agreed}")
    print(f"Vision only:         {fusion.vision_only}")
    print(f"Report only:         {fusion.report_only}")
    print(f"Conflicts:")
    for c in fusion.conflicts:
        print(f"  - {c.name}: {c.reason}")
    print(f"Notes:")
    for n in fusion.notes:
        print(f"  - {n}")


if __name__ == "__main__":
    main()