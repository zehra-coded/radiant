"""Run the pipeline on 5 cases and compute generation groundedness."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.api import pipeline as pipe
from backend.api.state import get_rag_index, get_vision_model
from backend.evaluation.generation_eval import evaluate_one
from backend.rag.schema import Chunk

IMG = Path(r"D:\NIH-ChestXray14\images\00000001_000.png")

REPORTS = {
    "conflict": (
        "Chest radiograph, frontal view.\n\n"
        "Findings:\nNo pneumothorax. No pleural effusion. Cardiac silhouette "
        "is normal in size. Lungs are clear bilaterally.\n\n"
        "Impression:\nNo acute cardiopulmonary abnormality."
    ),
    "agreement": (
        "Chest radiograph, frontal view.\n\n"
        "Findings:\nThe cardiac silhouette is mildly enlarged, consistent "
        "with mild cardiomegaly. There is a small left pleural effusion. "
        "The lung fields are otherwise clear. No pneumothorax.\n\n"
        "Impression:\n1. Mild cardiomegaly. 2. Small left pleural effusion."
    ),
    "negation": (
        "Chest radiograph, frontal view.\n\n"
        "Findings:\nThere is no definite pneumothorax. Possible small right "
        "pleural effusion; cannot exclude.\n\n"
        "Impression:\n1. Possible small right pleural effusion."
    ),
    "empty": "",
    "brief": "Normal chest radiograph.",
}


def main() -> int:
    # warm the caches
    get_vision_model()
    get_rag_index()

    rows: list[dict] = []
    for name, report in REPORTS.items():
        print(f"\n=== Case: {name} ===")
        result = pipe.run(IMG, report, "Summarize this case for human review.")
        text = result.explanation.text if result.explanation else ""
        chunks = [
            Chunk(
                document_id=e.document_id,
                chunk_id=e.chunk_id,
                text=e.text,
                title=e.title,
                source=e.source,
                url=e.url,
                license=e.license,
                provenance=e.provenance,  # type: ignore[arg-type]
                score=e.score,
            )
            for e in result.evidence
        ]
        m = evaluate_one(text, chunks)
        m["case"] = name
        m["generator"] = result.generator
        rows.append(m)
        print(f"  generator:              {m['generator']}")
        print(f"  citations total:        {m['citations_total']}")
        print(f"  citations valid:        {m['citations_valid']}")
        print(f"  citation validity rate: {m['citation_validity_rate']}")
        print(f"  sentences total:        {m['sentences_total']}")
        print(f"  sentences grounded:     {m['sentences_grounded']}")
        print(f"  groundedness ratio:     {m['groundedness_ratio']}")

    # Write addendum
    doc = Path(__file__).resolve().parents[1] / "docs" / "evaluation_addendum.md"
    lines = ["# RADIANT — Generation groundedness addendum", ""]
    lines.append(
        "Five pipeline runs on the same NIH image with different reports. "
        "Citation validity = fraction of `[source:doc:chunk]` citations that "
        "resolve to real retrieved chunks. Groundedness ratio = fraction of "
        "sentences that either contain a valid citation or belong to a "
        "disclaimer/limitations section."
    )
    lines.append("")
    lines.append(
        "| Case | Generator | Citations total | Valid | Validity | Sentences | Grounded | Ratio |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['case']} | {r['generator']} | {r['citations_total']} | "
            f"{r['citations_valid']} | {r['citation_validity_rate']} | "
            f"{r['sentences_total']} | {r['sentences_grounded']} | "
            f"{r['groundedness_ratio']} |"
        )
    lines.append("")
    lines.append(
        "**Limitations.** Groundedness is measured by rule: a sentence "
        "counts as grounded if it contains a citation or is in the "
        "limitations section. The rule does not verify semantic support — "
        "a sentence could cite a chunk that does not actually support it. "
        "This is documented as a limitation, not hidden."
    )
    doc.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {doc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())