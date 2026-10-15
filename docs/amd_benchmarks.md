# AMD Token Factory — API-level benchmark

Run at: 2026-10-15T21:50:35+00:00

## Scope

These measurements are **client-observed API-level latencies** for the
AMD Token Factory endpoint. They are NOT GPU benchmarks. RADIANT has
no GPU. No ROCm, no VRAM, no throughput in tokens-per-second-per-GPU
are measured or claimed. What is measured is: how long does it take,
from this client, for the hosted model to return a response?

## Configuration

- Base URL: `https://developer.amd.com.cn/radeon/api/v1`
- Model: `DeepSeek-V4-Flash`
- Prompt: `'Reply with the single word: OK'`
- Max tokens per response: 8
- Client: `openai` Python SDK
- Hardware: Windows 11 x86_64, Python 3.11.9, no GPU

## Results

- Calls attempted: 10
- Calls succeeded: 10
- Calls failed: 0

| Percentile | Latency (ms) |
|---|---|
| min | 823 |
| p50 | 932 |
| p95 | 12052 |
| p99 | 12052 |
| max | 12052 |

## What is NOT measured

- GPU utilization
- ROCm version
- VRAM consumption
- Throughput in tokens per second per GPU
- Concurrent request behavior beyond a single sequential stream

These require GPU access, which this project does not have.
See `docs/amd_setup.md` for the exact integration boundary.

## Interpretation

Latency is dominated by network round-trip time between this client
and the AMD endpoint, plus model inference time on AMD's side. The
client cannot distinguish the two. A larger sample would give more
stable percentiles. This is a snapshot, not a benchmark suite.
