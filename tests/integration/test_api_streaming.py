"""Integration test for /analyze/stream — verifies real SSE progress."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.api import main as api_main
from backend.core.config import NIH_IMAGES_DIR, NIH_TRAIN_VAL_LIST

pytestmark = pytest.mark.integration

_EXPECTED_ORDER = [
    "vision_done",
    "nlp_done",
    "fusion_done",
    "priority_done",
    "retrieval_done",
    "safety_done",
    "generation_done",
    "done",
]


def _first_nih_image() -> Path | None:
    if not NIH_TRAIN_VAL_LIST.exists() or not NIH_IMAGES_DIR.exists():
        return None
    for line in NIH_TRAIN_VAL_LIST.read_text(encoding="utf-8").splitlines():
        name = line.strip()
        if not name:
            continue
        p = NIH_IMAGES_DIR / name
        if p.exists():
            return p
    return None


@pytest.mark.skipif(_first_nih_image() is None, reason="NIH image not available")
def test_stream_yields_events_in_order() -> None:
    img = _first_nih_image()
    assert img is not None
    report = (
        Path(__file__).resolve().parents[2] / "data" / "synthetic" / "report_conflict.txt"
    ).read_text(encoding="utf-8")

    client = TestClient(api_main.app)
    with img.open("rb") as f:
        with client.stream(
            "POST",
            "/analyze/stream",
            files={"image": ("xray.png", f, "image/png")},
            data={"report": report},
        ) as r:
            assert r.status_code == 200
            steps: list[str] = []
            final_payload: dict | None = None
            for line in r.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                payload = json.loads(line[len("data: "):])
                steps.append(payload["step"])
                if payload["step"] == "done":
                    final_payload = payload["payload"]

    assert steps == _EXPECTED_ORDER, f"got {steps}"
    assert final_payload is not None
    assert final_payload["request_id"].startswith("req-")
    assert final_payload["generator"] in ("amd_token_factory", "template")