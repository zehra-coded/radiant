"""Re-run only the Rad-SpRL evaluation with the fixed parser."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.evaluation import nlp_eval, radsprl_parser

ROOT = Path(__file__).resolve().parents[1]
XML = (
    ROOT / "data" / "radsprl"
    / "A dataset of chest X-ray reports annotated with Spatial Role Labeling annotations"
    / "Rad-SpRL.xml"
)


def main() -> int:
    print(f"Parsing {XML.name}...")
    docs = radsprl_parser.parse(XML)
    print(f"Documents: {len(docs)}")

    counts = radsprl_parser.summary(docs)
    print("Tag counts:")
    print(json.dumps(counts, indent=2))

    print("\nEvaluating on 200 documents...")
    result = nlp_eval.evaluate(docs, sample=200)

    print(f"\nDiagnosis spans total:    {result['diagnosis_spans_total']}")
    print(f"Diagnosis spans matched:  {result['diagnosis_spans_matched']}")
    print(f"Diagnosis span recall:    {result['diagnosis_span_recall']}")
    print(f"Hedge spans total:        {result['hedge_spans_total']}")
    print(f"Hedge spans matched:      {result['hedge_spans_matched']}")
    print(f"Hedge agreement:          {result['hedge_agreement']}")
    print(f"Landmark spans total:     {result['landmark_spans_total']}")
    print(f"Landmark spans matched:   {result['landmark_spans_matched']}")
    print(f"Anatomy agreement:        {result['anatomy_agreement']}")

    # Also do the full 2000-doc pass since it's fast
    print("\nEvaluating on all 2000 documents...")
    full = nlp_eval.evaluate(docs, sample=len(docs))
    print(f"Diagnosis span recall (2000 docs): {full['diagnosis_span_recall']}")
    print(f"Hedge agreement (2000 docs):       {full['hedge_agreement']}")
    print(f"Anatomy agreement (2000 docs):     {full['anatomy_agreement']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())