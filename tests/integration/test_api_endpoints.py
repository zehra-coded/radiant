"""Integration tests for the FastAPI endpoints.

These load the real vision model and RAG index (via the app state),
so they are marked integration and skip if the NIH dataset is not
available.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.api import main as api_main
from backend.api import state
from backend.core.config import NIH_IMAGES_DIR, NIH_TRAIN_VAL_LIST

pytestmark = pytest.mark.integration


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


@pytest.fixture(scope="module")
def client() -> TestClient:
    if _first_nih_image() is None:
        pytest.skip("NIH image not available")
    return TestClient(api_main.app)


@pytest.fixture(scope="module")
def report_text() -> str:
    p = Path(__file__).resolve().parents[2] / "data" / "synthetic" / "report_conflict.txt"
    return p.read_text(encoding="utf-8")


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_analyze_returns_radiant_result(client: TestClient, report_text: str) -> None:
    img = _first_nih_image()
    assert img is not None
    with img.open("rb") as f:
        r = client.post(
            "/analyze",
            files={"image": ("xray.png", f, "image/png")},
            data={"report": report_text, "query": "Summarize"},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["request_id"].startswith("req-")
    assert body["vision"] is not None
    assert len(body["vision"]["findings"]) == 14
    assert body["report_analysis"] is not None
    assert body["cross_modal_analysis"] is not None
    assert body["generator"] in ("amd_token_factory", "template")
    assert "disclaimer" in body


def test_analyze_empty_report_is_valid(client: TestClient) -> None:
    img = _first_nih_image()
    assert img is not None
    with img.open("rb") as f:
        r = client.post(
            "/analyze",
            files={"image": ("xray.png", f, "image/png")},
            data={"report": ""},
        )
    assert r.status_code == 200
    assert r.json()["request_id"].startswith("req-")


def test_analyze_corrupt_image(client: TestClient, report_text: str) -> None:
    r = client.post(
        "/analyze",
        files={"image": ("bad.png", b"not an image", "image/png")},
        data={"report": report_text},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["request_id"].startswith("req-")
    # Vision should be None or missing; the pipeline should not raise.
    assert body["human_review"]["required"] is True


def test_retrieve_endpoint(client: TestClient) -> None:
    r = client.post("/retrieve", json={"query": "pneumothorax", "k": 2})
    assert r.status_code == 200
    items = r.json()
    assert len(items) >= 1
    assert all("document_id" in it for it in items)