# Third-party licenses

This document records the external software, models and datasets that
RADIANT depends on, with each one's license and permitted use.

RADIANT's own code is licensed under MIT (see `LICENSE`). Third-party
components keep their original licenses.

## Python libraries

| Library | Version | License | Purpose |
|---|---|---|---|
| `torch` | 2.14.1+cpu | BSD-3-Clause | Neural network runtime |
| `torchvision` | 0.29.1+cpu | BSD-3-Clause | Image transforms |
| `torchxrayvision` | 1.5.5 | Apache-2.0 | Vision model + weights |
| `sentence-transformers` | 6.1.0 | Apache-2.0 | Embedding model |
| `transformers` | 5.19.0 | Apache-2.0 | Tokenizer backend (transitive) |
| `faiss-cpu` | 1.15.1 | MIT | Vector index |
| `numpy` | 2.4.6 | BSD-3-Clause | Arrays |
| `pillow` | 12.3.0 | HPND | Image I/O |
| `fastapi` | 0.143.0 | MIT | API framework |
| `uvicorn` | 0.54.0 | BSD-3-Clause | ASGI server |
| `python-multipart` | 0.0.32 | Apache-2.0 | Multipart form parsing |
| `pydantic` | 2.14.1 | MIT | Data validation |
| `python-dotenv` | 1.2.4 | BSD-3-Clause | .env loader |
| `openai` | 3.29.0 | Apache-2.0 | LLM API client |
| `streamlit` | 1.65.0 | Apache-2.0 | Frontend |
| `httpx` | 0.28.1 | BSD-3-Clause | HTTP client |
| `pytest` | 9.1.1 | MIT | Test framework |
| `scikit-learn` | 1.9.1 | BSD-3-Clause | Metrics (transitive) |
| `scipy` | 1.17.1 | BSD-3-Clause | Numerics (transitive) |

Attribution: each library retains its original copyright notice and
license text, available in its distribution.

## Models

### `densenet121-res224-nih`

- **Source:** torchxrayvision project
  (https://github.com/mlmed/torchxrayvision)
- **Weights:** downloaded at first use via torchxrayvision
- **Training data:** NIH ChestX-ray14
- **Library license:** Apache-2.0
- **Weights license:** `LICENSE STATUS: UNVERIFIED — the weights are
  distributed with the torchxrayvision library, which is Apache-2.0.
  The checkpoint's own license note has not been independently
  re-verified. Used here for research and evaluation only. Not
  redistributed by RADIANT.`

### `sentence-transformers/all-MiniLM-L6-v2`

- **Source:** Hugging Face
  (https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **License:** Apache-2.0
- **Use:** embedding for evidence retrieval
- **Redistribution:** not redistributed by RADIANT; downloaded at runtime.

## Datasets

### NIH ChestX-ray14

- **Source:** https://nihcc.app.box.com/v/ChestXray-NIHCC
- **License:** CC0 1.0 Universal (Public Domain Dedication)
- **Use in RADIANT:** model evaluation, real-image smoke test
- **Redistribution:** permitted under CC0, but RADIANT does not
  redistribute the dataset — it is git-ignored and stays on the local
  disk. The evaluation runs locally and produces derived metrics only.

### Rad-SpRL (Spatial Role Labeling for Radiology Reports)

- **Source:** https://data.mendeley.com/datasets/yhb26hfz8n/1
- **License:** `LICENSE STATUS: UNVERIFIED — the Mendeley Data page
  terms must be confirmed before any redistribution.`
- **Use in RADIANT:** NLP extractor evaluation (span recall, hedge
  agreement, anatomy agreement)
- **Redistribution:** not redistributed. `data/radsprl/` is git-ignored.

### Synthetic data authored for RADIANT

- **Source:** RADIANT project
- **License:** MIT (same as the project)
- **Files:** `data/synthetic/`, `data/sample_xrays/`
- **Use:** unit tests, integration tests, demo scenarios
- **Marking:** every synthetic file is clearly labeled `SYNTHETIC`.

## APIs

### AMD Token Factory

- **Source:** https://developer.amd.com.cn/radeon/api/v1
- **Terms of use:** "Trial and development only. No SLA. Do not run
  production workloads. Do not send sensitive personal, health or
  financial data."
- **Model used:** `DeepSeek-V4-Flash` (text-only)
- **Use in RADIANT:** optional explanation generation. Never receives
  images, patient identifiers, or raw report text. Only structured
  JSON from an allowlist.
- **Redistribution:** none; the API is consumed at runtime only.
- **Underlying model license:** `LICENSE STATUS: UNVERIFIED — the AMD
  Token Factory does not publish per-model licenses for the shared
  endpoints. RADIANT calls the API; it does not redistribute weights.`

## What RADIANT does NOT bundle

- No model weights are committed to the repository.
- No datasets are committed to the repository.
- No API keys are committed.
- No patient data of any kind.

See `docs/datasets.md` for the full dataset license record.