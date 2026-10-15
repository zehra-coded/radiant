"""Send one real /analyze request and print key fields from the response."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

IMG = Path(r"D:\NIH-ChestXray14\images\00000001_000.png")
REPORT = Path(__file__).resolve().parents[1] / "data" / "synthetic" / "report_conflict.txt"


def main() -> int:
    with IMG.open("rb") as f:
        files = {"image": ("xray.png", f, "image/png")}
        data = {
            "report": REPORT.read_text(encoding="utf-8"),
            "query": "Summarize this case for human review.",
        }
        r = httpx.post("http://127.0.0.1:8000/analyze", files=files, data=data, timeout=120.0)

    if r.status_code != 200:
        print(f"FAILED: HTTP {r.status_code}")
        print(r.text[:2000])
        return 1

    body = r.json()
    print(f"request_id:    {body['request_id']}")
    print(f"generator:     {body['generator']}")
    print(f"priority:      {body['triage']['priority']}")
    print(f"consistency:   {body['cross_modal_analysis']['overall_consistency']}")
    print(f"conflicts:     {len(body['cross_modal_analysis']['conflicts'])}")
    print(f"evidence:      {len(body['evidence'])} chunks")
    print(f"human_review:  {body['human_review']['required']}")
    print(f"fallback_reason: {body['explanation'].get('fallback_reason')}")
    print(f"pre_safety: {body.get('safety', {}).get('reason_codes', 'n/a')}")
    print()
    text = body['explanation']['text']
    print("Explanation (first 400 chars):")
    print(text[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())