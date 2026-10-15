"""
Deterministic lexicons for report NLP extraction.

These are V1 heuristic vocabularies. They are designed from general
radiology-report conventions and are informed by — but not identical to —
the Rad-SpRL annotation rules (which are not yet available to inspect).
"""
from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# Findings: canonical NIH label -> list of surface-form variants.
# Order matters only for logging; matching uses word boundaries.
# ---------------------------------------------------------------------------

FINDING_SYNONYMS: Final[dict[str, tuple[str, ...]]] = {
    "Atelectasis": (
        "atelectasis",
        "atelectatic",
        "collapse",
        "collapsed lung",
        "lung collapse",
    ),
    "Cardiomegaly": (
        "cardiomegaly",
        "enlarged heart",
        "enlarged cardiac silhouette",
        "cardiac enlargement",
        "heart is enlarged",
        "heart size is enlarged",
        "mild cardiomegaly",
        "moderate cardiomegaly",
    ),
    "Effusion": (
        "effusion",
        "pleural effusion",
        "pleural fluid",
        "fluid in the pleural space",
    ),
    "Infiltration": (
        "infiltrate",
        "infiltration",
        "infiltrates",
        "infiltrations",
    ),
    "Mass": (
        "mass",
        "masses",
        "pulmonary mass",
    ),
    "Nodule": (
        "nodule",
        "nodules",
        "pulmonary nodule",
        "pulmonary nodules",
    ),
    "Pneumonia": (
        "pneumonia",
        "pneumonic",
        "pneumonitis",
    ),
    "Pneumothorax": (
        "pneumothorax",
        "ptx",
        "collapsed lung due to air",
    ),
    "Consolidation": (
        "consolidation",
        "consolidative",
        "consolidations",
    ),
    "Edema": (
        "edema",
        "pulmonary edema",
        "pulmonary oedema",
        "interstitial edema",
    ),
    "Emphysema": (
        "emphysema",
        "emphysematous",
    ),
    "Fibrosis": (
        "fibrosis",
        "fibrotic",
        "fibrotic changes",
        "scarring",
    ),
    "Pleural_Thickening": (
        "pleural thickening",
        "pleural thickening is present",
        "thickened pleura",
    ),
    "Hernia": (
        "hernia",
        "hiatal hernia",
        "diaphragmatic hernia",
    ),
}

# ---------------------------------------------------------------------------
# Negation cues: if one appears immediately before a finding, the finding
# is marked as negated.
# ---------------------------------------------------------------------------

NEGATION_CUES: Final[tuple[str, ...]] = (
    "no",
    "no evidence of",
    "no definite",
    "without",
    "without evidence of",
    "absence of",
    "absent",
    "is absent",
    "are absent",
    "negative for",
    "free of",
    "no significant",
    "no acute",
    "not seen",
    "not identified",
    "not present",
    "ruled out",
    "no focal",
)

# ---------------------------------------------------------------------------
# Uncertainty cues: if one appears near a finding, the finding is marked
# as uncertain, and the heuristic score is set accordingly.
# ---------------------------------------------------------------------------

UNCERTAINTY_CUES: Final[dict[str, float]] = {
    "possible": 1.0,
    "possibly": 1.0,
    "cannot exclude": 1.0,
    "cannot be excluded": 1.0,
    "cannot be ruled out": 1.0,
    "may represent": 1.0,
    "may be": 1.0,
    "might be": 1.0,
    "suspicious for": 1.0,
    "suspected": 1.0,
    "concerning for": 1.0,
    "questionable": 0.8,
    "equivocal": 0.8,
    "indeterminate": 0.8,
    "likely": 0.6,
    "likely represents": 0.6,
    "probably": 0.6,
    "appears": 0.5,
    "apparent": 0.5,
    "at the upper limit of normal": 0.5,
    "borderline": 0.5,
}

# ---------------------------------------------------------------------------
# Anatomy terms: mapped to canonical Anatomy values.
# ---------------------------------------------------------------------------

ANATOMY_TERMS: Final[dict[str, str]] = {
    "left": "left",
    "right": "right",
    "bilateral": "bilateral",
    "bilaterally": "bilateral",
    "both": "bilateral",
    "upper": "upper",
    "upper lobe": "upper",
    "lower": "lower",
    "lower lobe": "lower",
    "basilar": "basilar",
    "basal": "basilar",
    "apical": "apical",
    "apex": "apical",
    "diffuse": "diffuse",
    "focal": "focal",
    "localized": "focal",
    "localised": "focal",
}

# ---------------------------------------------------------------------------
# Spatial terms: multi-word phrases that carry positional meaning.
# These are captured verbatim into spatial_info and are more specific
# than the single anatomy tokens above.
# ---------------------------------------------------------------------------

SPATIAL_TERMS: Final[tuple[str, ...]] = (
    "left upper",
    "left lower",
    "right upper",
    "right lower",
    "left basilar",
    "right basilar",
    "bilateral lower",
    "bilateral upper",
    "left apical",
    "right apical",
)