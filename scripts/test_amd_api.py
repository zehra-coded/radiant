"""One-shot AMD API test. Reads key from .env, makes one chat call,
prints the raw response. Uses Python's JSON handling — no shell escaping."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.core.config import LLM_BASE_URL, LLM_MODEL, RADEON_API_KEY


def main() -> int:
    if not RADEON_API_KEY:
        print("RADEON_API_KEY not set in .env")
        return 1

    print(f"Base URL: {LLM_BASE_URL}")
    print(f"Model:    {LLM_MODEL}")
    print(f"Key len:  {len(RADEON_API_KEY)} chars")
    print(f"Key prefix: {RADEON_API_KEY[:6]}...")
    print(f"Key suffix: ...{RADEON_API_KEY[-4:]}")
    print()

    from openai import OpenAI

    client = OpenAI(
        base_url=LLM_BASE_URL,
        api_key=RADEON_API_KEY,
        timeout=30.0,
    )

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=8,
        )
        print("SUCCESS")
        print("Response:", response.choices[0].message.content)
        print("Usage:", response.usage)
    except Exception as exc:
        print("FAILED")
        print("Exception type:", type(exc).__name__)
        print("Exception:", exc)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())