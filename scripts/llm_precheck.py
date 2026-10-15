"""Read-only precheck against the live AMD Token Factory API.

Verifies:
- The configured model appears in GET /v1/models
- A minimal chat call succeeds and returns non-empty content
- Whether reasoning_effort="none" is accepted or rejected
- Whether /v1/usage is available

Writes results to docs/amd_setup.md.
Does not modify any project file other than that doc.

Run only when a real API key is configured.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.core.config import (  # noqa: E402
    LLM_BASE_URL,
    LLM_MODEL,
    RADEON_API_KEY,
)
from backend.llm import amd_client, quota  # noqa: E402


def main() -> int:
    out: list[str] = []
    out.append("# AMD Token Factory precheck")
    out.append("")
    out.append(f"- Run at: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    out.append(f"- Base URL: {LLM_BASE_URL}")
    out.append(f"- Configured model: {LLM_MODEL}")
    out.append(f"- API key present: {bool(RADEON_API_KEY)}")
    out.append("")

    if not RADEON_API_KEY:
        out.append("STATUS: RADEON_API_KEY not set. No live calls performed.")
        _write(out)
        print("RADEON_API_KEY not set; precheck skipped.")
        return 0

    # 1. Models list
    try:
        models = amd_client.list_models()
        out.append(f"## GET /v1/models")
        out.append(f"- Returned {len(models)} models")
        out.append(f"- Configured model present: {LLM_MODEL in models}")
        out.append(f"- Sample: {', '.join(models[:5])}")
    except Exception as exc:
        out.append(f"## GET /v1/models — FAILED")
        out.append(f"- Error: {exc}")

    # 2. Minimal chat call
    out.append("")
    out.append("## Minimal chat call")
    try:
        text = amd_client.call(
            "You are a test. Reply with the single word: OK",
            "Reply with OK.",
        )
        out.append(f"- Response: {text[:100]!r}")
        out.append(f"- Non-empty: {bool(text)}")
    except Exception as exc:
        out.append(f"- FAILED: {exc}")

    # 3. reasoning_effort="none"
    out.append("")
    out.append("## reasoning_effort acceptance")
    try:
        from openai import OpenAI
        client = OpenAI(base_url=LLM_BASE_URL, api_key=RADEON_API_KEY, timeout=20.0)
        try:
            client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=8,
                reasoning_effort="none",
            )
            out.append('- reasoning_effort="none": ACCEPTED')
        except Exception as exc:
            out.append(f'- reasoning_effort="none": rejected ({exc})')
    except Exception as exc:
        out.append(f"- Could not test reasoning_effort: {exc}")

    # 4. /v1/usage
    out.append("")
    out.append("## /v1/usage")
    snap = quota.get_quota(force=True)
    if snap is None:
        out.append("- endpoint unavailable or unauthorized")
    else:
        out.append(
            f"- remaining_pts={snap.remaining_pts}, used_pts={snap.used_pts}"
        )

    _write(out)
    print("Precheck written to docs/amd_setup.md")
    return 0


def _write(lines: list[str]) -> None:
    path = Path(__file__).resolve().parents[1] / "docs" / "amd_setup.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())