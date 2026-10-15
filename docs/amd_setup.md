# AMD Token Factory precheck

- Run at: 2026-10-14T20:40:16+00:00
- Base URL: https://developer.amd.com.cn/radeon/api/v1
- Configured model: DeepSeek-V4-Flash
- API key present: True

## GET /v1/models
- Returned 12 models
- Configured model present: True
- Sample: DeepSeek-V4.1-Flash, DeepSeek-V4-Flash, DeepSeek-V4-Flash-Vision-Exp, Gemma-4-31B-IT, GLM-5.3-Flash

## Minimal chat call
- Response: 'OK'
- Non-empty: True

## reasoning_effort acceptance
- reasoning_effort="none": ACCEPTED

## /v1/usage
- remaining_pts=None, used_pts=None


## Scope of this integration

RADIANT consumes AMD-hosted LLM endpoints via the AMD Token Factory
public API. The integration consists of:

- An OpenAI-compatible client (`backend/llm/amd_client.py`) pointed at
  `https://developer.amd.com.cn/radeon/api/v1`.
- A model allowlist that restricts the configured LLM to text-only
  models. Vision-capable models from the same catalog are rejected.
- A precheck script (`scripts/llm_precheck.py`) that documents the
  live state of the endpoint on each run.
- A usage reader (`backend/llm/quota.py`) that queries `/v1/usage`.
- Automatic fallback to a deterministic template reporter whenever the
  endpoint is unavailable, times out, or returns invalid output.

**RADIANT does not run any workload on AMD GPUs.** There is no ROCm
installation, no GPU-backed vLLM, and no kernel-level measurement in
this project. The only AMD resource RADIANT uses is the hosted model
API.

## What is NOT claimed

- No GPU throughput numbers.
- No ROCm version.
- No VRAM measurements.
- No "runs on AMD" claim. The correct statement is: "RADIANT consumes
  AMD-hosted LLM endpoints via the AMD Token Factory public API."

## API-level benchmark

See `docs/amd_benchmarks.md` for client-observed latency measurements.
Those measurements describe how long a hosted model takes to respond
from a specific client; they are not GPU benchmarks.