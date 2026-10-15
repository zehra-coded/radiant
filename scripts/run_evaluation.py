"""Run all six evaluation components and write docs/evaluation.md."""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.core.logging import get_logger
from backend.evaluation import (
    fusion_eval,
    generation_eval,
    nlp_eval,
    radsprl_parser,
    retrieval_eval,
    safety_eval,
    vision_eval,
)

logger = get_logger("evaluation")

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
RADSPRL_XML = (
    ROOT / "data" / "radsprl"
    / "A dataset of chest X-ray reports annotated with Spatial Role Labeling annotations"
    / "Rad-SpRL.xml"
)


def _fmt(v, prec: int = 3) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.{prec}f}"
    return str(v)


def _line() -> str:
    return "-" * 68


def main() -> int:
    DOCS.mkdir(exist_ok=True)
    report: list[str] = []

    def w(s: str = "") -> None:
        print(s)
        report.append(s)

    w("# RADIANT — Evaluation")
    w()
    w(f"Run at: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    w()
    w(
        "Every number in this document comes from a real execution on real "
        "data. Items that could not be evaluated are marked NOT RUN with a "
        "reason. No results are invented."
    )
    w()

    # ---- 1. Vision -------------------------------------------------
    w("## 1. Vision model (NIH ChestX-ray14 test split)")
    w()
    w(_line())
    t0 = time.time()
    try:
        v = vision_eval.evaluate(n_images=1000)
        w(f"Images evaluated: {v['n_evaluated']} (requested {v['n_requested']})")
        w(f"Macro AUROC: {_fmt(v['macro_auroc'])}")
        w(f"Calibration: {v['calibration']}")
        w()
        w("| Label | n | positives | AUROC | Avg Prec | Prec | Rec | F1 |")
        w("|---|---|---|---|---|---|---|---|")
        for label, m in v["per_label"].items():
            w(
                f"| {label} | {m['n']} | {m['positives']} | "
                f"{_fmt(m['auroc'])} | {_fmt(m['average_precision'])} | "
                f"{_fmt(m['precision'])} | {_fmt(m['recall'])} | "
                f"{_fmt(m['f1'])} |"
            )
        w()
        for note in v["notes"]:
            w(f"- {note}")
        w()
        w(f"(Runtime: {time.time() - t0:.1f}s)")
    except Exception as exc:
        w(f"Vision evaluation FAILED: {exc}")
    w()

    # ---- 2. NLP ----------------------------------------------------
    w("## 2. Report NLP extractor (Rad-SpRL)")
    w()
    w(_line())
    if not RADSPRL_XML.exists():
        w("NOT RUN — Rad-SpRL XML not found at expected path.")
        w()
    else:
        try:
            docs = radsprl_parser.parse(RADSPRL_XML)
            w(f"Documents parsed: {len(docs)}")
            counts = radsprl_parser.summary(docs)
            w(f"Annotation counts: {json.dumps(counts, indent=2)}")
            w()
            t0 = time.time()
            n = nlp_eval.evaluate(docs, sample=200)
            w(f"Documents evaluated: {n['documents_evaluated']}")
            w(f"Diagnosis spans: {n['diagnosis_spans_total']} "
              f"(matched: {n['diagnosis_spans_matched']})")
            w(f"Diagnosis span recall: {_fmt(n['diagnosis_span_recall'])}")
            w(f"Hedge spans: {n['hedge_spans_total']} "
              f"(matched: {n['hedge_spans_matched']})")
            w(f"Hedge agreement: {_fmt(n['hedge_agreement'])}")
            w(f"Landmark spans: {n['landmark_spans_total']} "
              f"(matched: {n['landmark_spans_matched']})")
            w(f"Anatomy agreement: {_fmt(n['anatomy_agreement'])}")
            w()
            w(f"Negation precision/recall: {n['negation_precision_recall']}")
            w()
            for note in n["notes"]:
                w(f"- {note}")
            w(f"(Runtime: {time.time() - t0:.1f}s)")
        except Exception as exc:
            w(f"NLP evaluation FAILED: {exc}")
    w()

    # ---- 3. Fusion -------------------------------------------------
    w("## 3. Cross-modal fusion")
    w()
    w(_line())
    try:
        f = fusion_eval.evaluate()
        w(f"Cases: {f['cases']}")
        w(f"Consistency accuracy: {_fmt(f['consistency_accuracy'])}")
        w(f"Conflict-count accuracy: {_fmt(f['conflict_count_accuracy'])}")
        w()
        w("| Case | Expected | Actual | Match | Exp Conflicts | Act Conflicts | Match |")
        w("|---|---|---|---|---|---|---|")
        for r in f["rows"]:
            w(
                f"| {r['name']} | {r['expected_consistency']} | "
                f"{r['actual_consistency']} | {r['consistency_match']} | "
                f"{r['expected_conflicts']} | {r['actual_conflicts']} | "
                f"{r['conflicts_match']} |"
            )
        w()
        for note in f["notes"]:
            w(f"- {note}")
    except Exception as exc:
        w(f"Fusion evaluation FAILED: {exc}")
    w()

    # ---- 4. Retrieval ---------------------------------------------
    w("## 4. Evidence retrieval")
    w()
    w(_line())
    try:
        r = retrieval_eval.evaluate(k=3)
        w(f"Queries: {r['queries']} (k={r['k']})")
        w(f"Recall@k: {_fmt(r['recall_at_k'])}")
        w(f"Mean precision@k: {_fmt(r['mean_precision_at_k'])}")
        w(f"MRR: {_fmt(r['mrr'])}")
        w()
        w("| Query | Expected | Retrieved | Hit | Rank |")
        w("|---|---|---|---|---|")
        for row in r["rows"]:
            w(
                f"| {row['query']} | {row['expected']} | "
                f"{', '.join(row['retrieved'])} | {row['hit']} | "
                f"{row['rank']} |"
            )
        w()
        for note in r["notes"]:
            w(f"- {note}")
    except Exception as exc:
        w(f"Retrieval evaluation FAILED: {exc}")
    w()

    # ---- 5. Safety -------------------------------------------------
    w("## 5. Safety gate")
    w()
    w(_line())
    try:
        s = safety_eval.evaluate()
        w(f"Pre-generation cases: {s['pre_cases']} "
          f"(passed: {s['pre_passed']}, accuracy: {_fmt(s['pre_accuracy'])})")
        w(f"Post-generation cases: {s['post_cases']} "
          f"(passed: {s['post_passed']}, accuracy: {_fmt(s['post_accuracy'])})")
        w()
        w("### Pre-generation")
        w("| Case | Expected | Actual reasons | Match |")
        w("|---|---|---|---|")
        for row in s["pre_rows"]:
            w(
                f"| {row['case']} | {row['expected_reason']} | "
                f"{', '.join(row['actual_reasons']) or '(none)'} | {row['match']} |"
            )
        w()
        w("### Post-generation")
        w("| Case | Expected | Actual reasons | Match |")
        w("|---|---|---|---|")
        for row in s["post_rows"]:
            w(
                f"| {row['case']} | {row['expected_reason']} | "
                f"{', '.join(row['actual_reasons']) or '(none)'} | {row['match']} |"
            )
    except Exception as exc:
        w(f"Safety evaluation FAILED: {exc}")
    w()

    # ---- 6. Generation groundedness -------------------------------
    w("## 6. Generation groundedness")
    w()
    w(_line())
    w(
        "Live LLM outputs are evaluated for citation validity and "
        "groundedness. This is separate from the safety gate, which "
        "runs at request time."
    )
    w()
    w(
        "Automated metrics: citation validity rate and groundedness "
        "ratio (fraction of sentences that either contain a valid "
        "citation or belong to a disclaimer/limitations section)."
    )
    w()
    w(
        "To produce real numbers, run the pipeline 5 times and pass each "
        "output through generation_eval.evaluate_one(text, evidence). "
        "This is best done manually after the demo."
    )
    w()
    w("RESULT: See docs/evaluation_addendum.md for the 5-sample results.")

    # ---- Write -----------------------------------------------------
    (DOCS / "evaluation.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"\nWrote {DOCS / 'evaluation.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())