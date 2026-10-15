"""
Vision model evaluation on the NIH ChestX-ray14 test split.

Runs DenseNet-121 on N images from test_list.txt and computes per-label
AUROC, precision, recall, F1 against Data_Entry_2017.csv labels.

Calibration is NOT assessed. Model scores are raw sigmoid outputs.
AUROC is a ranking metric, not a measure of clinical validity.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from backend.core.config import (
    NIH_IMAGES_DIR,
    NIH_METADATA_CSV,
    NIH_TEST_LIST,
    VISION_PRESENT_THRESHOLD,
)
from backend.core.logging import get_logger
from backend.vision.model import VisionModel
from backend.vision.schema import NIH_LABELS

logger = get_logger(__name__)


def load_labels(csv_path: Path) -> dict[str, set[str]]:
    labels: dict[str, set[str]] = {}
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image = row["Image Index"].strip()
            raw = row["Finding Labels"].strip()
            labels[image] = set() if raw == "No Finding" else set(raw.split("|"))
    return labels


def load_test_images(list_path: Path, n: int) -> list[str]:
    names = [
        line.strip()
        for line in list_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return names[:n]


def evaluate(n_images: int = 1000) -> dict:
    labels = load_labels(NIH_METADATA_CSV)
    image_names = load_test_images(NIH_TEST_LIST, n_images)

    model = VisionModel()
    scores: dict[str, list[float]] = {label: [] for label in NIH_LABELS}
    y_true: dict[str, list[int]] = {label: [] for label in NIH_LABELS}

    used: list[str] = []
    for name in image_names:
        path = NIH_IMAGES_DIR / name
        if not path.exists():
            continue
        try:
            result = model.predict(path)
        except Exception as exc:
            logger.warning("skipped %s: %s", name, exc)
            continue
        used.append(name)
        truth = labels.get(name, set())
        for finding in result.findings:
            scores[finding.name].append(float(finding.model_score))
            y_true[finding.name].append(1 if finding.name in truth else 0)

    per_label: dict[str, dict] = {}
    for label in NIH_LABELS:
        y = np.asarray(y_true[label], dtype=int)
        s = np.asarray(scores[label], dtype=float)
        if y.sum() == 0:
            per_label[label] = {
                "n": int(len(y)),
                "positives": 0,
                "auroc": None,
                "average_precision": None,
                "precision": None,
                "recall": None,
                "f1": None,
                "note": "no positive cases in this sample",
            }
            continue
        try:
            auroc = float(roc_auc_score(y, s))
        except Exception:
            auroc = None
        try:
            ap = float(average_precision_score(y, s))
        except Exception:
            ap = None
        preds = (s >= VISION_PRESENT_THRESHOLD).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            y, preds, average="binary", zero_division=0
        )
        per_label[label] = {
            "n": int(len(y)),
            "positives": int(y.sum()),
            "auroc": auroc,
            "average_precision": ap,
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
            "threshold": VISION_PRESENT_THRESHOLD,
            "note": "threshold is UNVALIDATED PROTOTYPE HEURISTIC",
        }

    valid_aurocs = [v["auroc"] for v in per_label.values() if v["auroc"] is not None]
    macro_auroc = float(np.mean(valid_aurocs)) if valid_aurocs else None

    return {
        "n_requested": n_images,
        "n_evaluated": len(used),
        "per_label": per_label,
        "macro_auroc": macro_auroc,
        "calibration": "NOT ASSESSED — raw sigmoid outputs",
        "notes": [
            "Model scores are raw sigmoid outputs, not calibrated probabilities.",
            f"Classification threshold = {VISION_PRESENT_THRESHOLD} "
            "(UNVALIDATED PROTOTYPE HEURISTIC).",
            "AUROC is a ranking metric; it does not imply clinical validity.",
        ],
    }