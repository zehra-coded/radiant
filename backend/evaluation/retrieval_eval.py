"""Retrieval evaluation on hand-labeled query-chunk pairs."""
from __future__ import annotations

from backend.core.config import DATA_ROOT
from backend.rag.indexer import build_index
from backend.rag.retriever import retrieve

LABELED_QUERIES = [
    {"query": "pneumothorax", "expected": "fixture_pneumothorax"},
    {"query": "collapsed lung", "expected": "fixture_pneumothorax"},
    {"query": "pleural effusion", "expected": "fixture_pleural_effusion"},
    {"query": "fluid in the chest", "expected": "fixture_pleural_effusion"},
    {"query": "enlarged heart", "expected": "fixture_cardiomegaly"},
    {"query": "cardiomegaly", "expected": "fixture_cardiomegaly"},
    {"query": "consolidation", "expected": "fixture_consolidation"},
    {"query": "lung opacity", "expected": "fixture_consolidation"},
    {"query": "atelectasis", "expected": "fixture_atelectasis"},
    {"query": "lung collapse", "expected": "fixture_atelectasis"},
]


def evaluate(k: int = 3) -> dict:
    index = build_index(DATA_ROOT / "knowledge_base")

    hits = 0
    precision_sum = 0.0
    rr_sum = 0.0
    rows: list[dict] = []

    for item in LABELED_QUERIES:
        rr = retrieve(index, item["query"], k=k)
        retrieved = [c.document_id for c in rr.chunks]
        hit = item["expected"] in retrieved
        rank = retrieved.index(item["expected"]) + 1 if hit else None
        p = (
            sum(1 for d in retrieved if d == item["expected"]) / len(retrieved)
            if retrieved else 0.0
        )
        rr_score = (1.0 / rank) if rank else 0.0

        hits += int(hit)
        precision_sum += p
        rr_sum += rr_score
        rows.append({
            "query": item["query"],
            "expected": item["expected"],
            "retrieved": retrieved,
            "hit": hit,
            "rank": rank,
            "precision_at_k": p,
            "rr": rr_score,
        })

    n = len(LABELED_QUERIES)
    return {
        "queries": n,
        "k": k,
        "recall_at_k": hits / n,
        "mean_precision_at_k": precision_sum / n,
        "mrr": rr_sum / n,
        "rows": rows,
        "notes": [
            f"Evaluated on {n} hand-labeled queries against the small "
            "development-fixture knowledge base (5 documents).",
            "Numbers are indicative, not a general retrieval quality measure.",
        ],
    }