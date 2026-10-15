"""Measure API-level latency of AMD Token Factory calls.

Makes N real chat completions, records end-to-end client-observed
latency, and writes results to docs/amd_benchmarks.md.

Does NOT measure GPU-side performance. There is no GPU.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.core.config import LLM_BASE_URL, LLM_MODEL, RADEON_API_KEY


def main(n: int = 10) -> int:
    if not RADEON_API_KEY:
        print("RADEON_API_KEY not set. Cannot run benchmark.")
        return 1

    from openai import OpenAI

    client = OpenAI(base_url=LLM_BASE_URL, api_key=RADEON_API_KEY, timeout=60.0)

    # A typical RADIANT payload is structured text; approximate with a short prompt.
    prompt = "Reply with the single word: OK"

    latencies: list[float] = []
    errors: list[str] = []

    print(f"Running {n} calls to {LLM_MODEL} at {LLM_BASE_URL}...")
    for i in range(n):
        t0 = time.perf_counter()
        try:
            r = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=8,
            )
            dt = time.perf_counter() - t0
            latencies.append(dt)
            print(f"  [{i+1}/{n}] {dt*1000:.0f} ms — {r.choices[0].message.content!r}")
        except Exception as exc:
            dt = time.perf_counter() - t0
            errors.append(f"{type(exc).__name__}: {exc}")
            print(f"  [{i+1}/{n}] FAILED after {dt*1000:.0f} ms — {exc}")

    if not latencies:
        print("All calls failed. No benchmark written.")
        return 1

    latencies.sort()
    n_ok = len(latencies)
    p50 = statistics.median(latencies)
    p95 = latencies[min(int(0.95 * n_ok), n_ok - 1)]
    p99 = latencies[min(int(0.99 * n_ok), n_ok - 1)]

    lines = [
        "# AMD Token Factory — API-level benchmark",
        "",
        f"Run at: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "## Scope",
        "",
        "These measurements are **client-observed API-level latencies** for the",
        "AMD Token Factory endpoint. They are NOT GPU benchmarks. RADIANT has",
        "no GPU. No ROCm, no VRAM, no throughput in tokens-per-second-per-GPU",
        "are measured or claimed. What is measured is: how long does it take,",
        "from this client, for the hosted model to return a response?",
        "",
        "## Configuration",
        "",
        f"- Base URL: `{LLM_BASE_URL}`",
        f"- Model: `{LLM_MODEL}`",
        f"- Prompt: `{prompt!r}`",
        f"- Max tokens per response: 8",
        f"- Client: `openai` Python SDK",
        f"- Hardware: Windows 11 x86_64, Python 3.11.9, no GPU",
        "",
        "## Results",
        "",
        f"- Calls attempted: {n}",
        f"- Calls succeeded: {n_ok}",
        f"- Calls failed: {len(errors)}",
        "",
        "| Percentile | Latency (ms) |",
        "|---|---|",
        f"| min | {latencies[0]*1000:.0f} |",
        f"| p50 | {p50*1000:.0f} |",
        f"| p95 | {p95*1000:.0f} |",
        f"| p99 | {p99*1000:.0f} |",
        f"| max | {latencies[-1]*1000:.0f} |",
        "",
    ]

    if errors:
        lines.append("## Errors observed")
        lines.append("")
        for e in errors:
            lines.append(f"- `{e}`")
        lines.append("")

    lines.extend([
        "## What is NOT measured",
        "",
        "- GPU utilization",
        "- ROCm version",
        "- VRAM consumption",
        "- Throughput in tokens per second per GPU",
        "- Concurrent request behavior beyond a single sequential stream",
        "",
        "These require GPU access, which this project does not have.",
        "See `docs/amd_setup.md` for the exact integration boundary.",
        "",
        "## Interpretation",
        "",
        "Latency is dominated by network round-trip time between this client",
        "and the AMD endpoint, plus model inference time on AMD's side. The",
        "client cannot distinguish the two. A larger sample would give more",
        "stable percentiles. This is a snapshot, not a benchmark suite.",
    ])

    out = Path(__file__).resolve().parents[1] / "docs" / "amd_benchmarks.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())