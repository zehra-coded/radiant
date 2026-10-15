"""
RADIANT configuration.

Every threshold, path, model name, allowlist, denylist and environment
variable used anywhere in the codebase lives here. No magic numbers
elsewhere. Prototype heuristics are explicitly labeled as unvalidated.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
BACKEND_ROOT: Final[Path] = PROJECT_ROOT / "backend"
DATA_ROOT: Final[Path] = PROJECT_ROOT / "data"
MODELS_ROOT: Final[Path] = PROJECT_ROOT / "models"
DOCS_ROOT: Final[Path] = PROJECT_ROOT / "docs"
LOGS_ROOT: Final[Path] = PROJECT_ROOT / "logs"
# NIH ChestX-ray14 dataset (external, not committed to the repo).
# Defaults point at the local installation; override via .env if needed.
NIH_IMAGES_DIR: Final[Path] = Path(
    os.getenv("NIH_IMAGES_DIR", r"D:\NIH-ChestXray14\images")
)
NIH_METADATA_CSV: Final[Path] = Path(
    os.getenv("NIH_METADATA_CSV", r"D:\Data_Entry_2017.csv")
)
NIH_TRAIN_VAL_LIST: Final[Path] = Path(
    os.getenv("NIH_TRAIN_VAL_LIST", r"D:\train_val_list.txt")
)
NIH_TEST_LIST: Final[Path] = Path(
    os.getenv("NIH_TEST_LIST", r"D:\test_list.txt")
)
# ---------------------------------------------------------------------------
# Environment loading
# ---------------------------------------------------------------------------

load_dotenv(PROJECT_ROOT / ".env", override=False)

# ---------------------------------------------------------------------------
# API / LLM (values may be overridden by .env)
# ---------------------------------------------------------------------------

LLM_BASE_URL: Final[str] = os.getenv(
    "LLM_BASE_URL", "https://developer.amd.com.cn/radeon/api/v1"
)
LLM_MODEL: Final[str] = os.getenv("LLM_MODEL", "DeepSeek-V4-Flash")
LLM_MAX_TOKENS: Final[int] = int(os.getenv("LLM_MAX_TOKENS", "700"))
LLM_REASONING_EFFORT: Final[str] = os.getenv("LLM_REASONING_EFFORT", "none")
LLM_REQUEST_TIMEOUT_S: Final[int] = int(os.getenv("LLM_REQUEST_TIMEOUT_S", "60"))
RADEON_API_KEY: Final[str | None] = os.getenv("RADEON_API_KEY")

# Text-only models permitted as the LLM. Vision-capable models are
# explicitly forbidden by rule 3.5.
AMD_TEXT_ONLY_MODELS: Final[frozenset[str]] = frozenset(
    {
        "DeepSeek-V4-Flash",
        "DeepSeek-V4.1-Flash",   # permitted only when no image parts are sent
        "GLM-5.3-Flash",
        "MiniCPM5-2B",
    }
)

# Vision-capable models that must never be used as the LLM.
AMD_VISION_MODELS: Final[frozenset[str]] = frozenset(
    {
        "DeepSeek-V4-Flash-Vision-Exp",
        "MiMo-V2.6-Flash",
        "Qwen3.8-Flash-Next",
        "Qwen3.8-27B",
        "Gemma-4-31B-IT",
    }
)

# Populated by the Stage 10 read-only precheck; defaults are the safe minimum.
AMD_CAPABILITIES: Final[dict[str, bool]] = {
    "supports_reasoning_effort_none": True,
    "supports_reasoning_effort_low": False,
    "returns_reasoning_field": False,
    "usage_endpoint_available": False,
}

# ---------------------------------------------------------------------------
# Outbound payload allowlist / denylist (rule 3.15)
# ---------------------------------------------------------------------------

LLM_PAYLOAD_ALLOWLIST: Final[frozenset[str]] = frozenset(
    {
        "vision_scores",
        "report_findings",
        "fusion",
        "uncertainty_signals",
        "review_priority",
        "review_priority_reasons",
        "evidence_chunks",
        "safety_flags",
        "disclaimer",
    }
)

# Hard failure if any of these appear on an outbound LLM payload.
PATIENT_IDENTIFIER_DENYLIST: Final[frozenset[str]] = frozenset(
    {
        "patient_name",
        "mrn",
        "dob",
        "date_of_birth",
        "ssn",
        "accession",
        "study_uid",
        "patient_id",
        "nhs_number",
        "insurance_id",
    }
)

# ---------------------------------------------------------------------------
# Vision (Stage 4) — all thresholds are UNVALIDATED PROTOTYPE HEURISTICS
# ---------------------------------------------------------------------------

VISION_CHECKPOINT: Final[str] = "densenet121-res224-nih"
VISION_INPUT_SIZE: Final[int] = 224
VISION_PRESENT_THRESHOLD: Final[float] = 0.5   # UNVALIDATED PROTOTYPE HEURISTIC
VISION_UNCERTAIN_LOW: Final[float] = 0.3       # UNVALIDATED PROTOTYPE HEURISTIC
VISION_UNCERTAIN_HIGH: Final[float] = 0.5      # UNVALIDATED PROTOTYPE HEURISTIC

# ---------------------------------------------------------------------------
# Fusion (Stage 6) — UNVALIDATED PROTOTYPE HEURISTICS
# ---------------------------------------------------------------------------

FUSION_HIGH_RATIO: Final[float] = 0.70
FUSION_PARTIAL_RATIO: Final[float] = 0.40
FUSION_CRITICAL_LABELS: Final[frozenset[str]] = frozenset(
    {"Pneumothorax", "Effusion", "Consolidation", "Edema"}
)
# ---------------------------------------------------------------------------
# Triage / Uncertainty (Stage 7) — UNVALIDATED PROTOTYPE HEURISTICS
# ---------------------------------------------------------------------------

# A vision finding is "low confidence" if its model score is in the
# uncertain band AND it appears as a vision_only finding in fusion.
TRIAGE_LOW_CONFIDENCE_MAX_SCORE: Final[float] = 0.60

# A vision result is "ambiguous" if more than this fraction of its
# findings are in the uncertain band.
TRIAGE_AMBIGUOUS_FRACTION: Final[float] = 0.30

# Critical NIH labels that elevate priority on their own, when marked
# present by either modality. UNVALIDATED PROTOTYPE HEURISTIC.
TRIAGE_CRITICAL_PRESENT_LABELS: Final[frozenset[str]] = frozenset(
    {"Pneumothorax", "Effusion", "Consolidation", "Edema"}
)

# Priority ordering (higher wins).
TRIAGE_PRIORITY_ORDER: Final[tuple[str, ...]] = (
    "ROUTINE",
    "MONITOR",
    "URGENT_REVIEW",
    "EMERGENCY_REVIEW",
)
# ---------------------------------------------------------------------------
# RAG (Stage 8) — UNVALIDATED PROTOTYPE HEURISTIC
# ---------------------------------------------------------------------------

RAG_EMBEDDING_MODEL: Final[str] = "sentence-transformers/all-MiniLM-L6-v2"
RAG_CHUNK_TOKENS: Final[int] = 200
RAG_CHUNK_OVERLAP_TOKENS: Final[int] = 40
RAG_MIN_SIMILARITY: Final[float] = 0.20
RAG_TOP_K_DEFAULT: Final[int] = 5


# ---------------------------------------------------------------------------
# API (Stage 12)
# ---------------------------------------------------------------------------

API_TITLE: Final[str] = "RADIANT"
API_VERSION: Final[str] = "0.12.0-stage12"
API_CORS_ORIGINS: Final[tuple[str, ...]] = (
    "http://localhost:8501",
    "http://127.0.0.1:8501",
)

# Maximum upload size for the image, in bytes (10 MB).
API_MAX_UPLOAD_BYTES: Final[int] = 10 * 1024 * 1024

# Default user query sent with each analyze request if the client omits one.
API_DEFAULT_QUERY: Final[str] = "Summarize this case for human review."
# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_LEVEL: Final[str] = os.getenv("LOG_LEVEL", "INFO")

# ---------------------------------------------------------------------------
# Sanity checks (fail fast)
# ---------------------------------------------------------------------------

def _validate() -> None:
    if LLM_MODEL in AMD_VISION_MODELS:
        raise RuntimeError(
            f"Configured LLM_MODEL '{LLM_MODEL}' is a vision-capable model. "
            "Rule 3.5 forbids using it as the LLM."
        )
    if LLM_MODEL not in AMD_TEXT_ONLY_MODELS:
        raise RuntimeError(
            f"Configured LLM_MODEL '{LLM_MODEL}' is not in the text-only allowlist."
        )


_validate()
# Base URL that the Streamlit frontend uses to reach the API.
API_BASE_URL: Final[str] = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")