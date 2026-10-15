# RADIANT — Architecture

This document is a one-page technical description of how RADIANT is
built and why. It is intended for engineers and reviewers, not for
clinical readers. See `README.md` for the overview and `docs/safety.md`
for the safety model.

## Data flow

```
        Chest X-ray                       Report text
             │                                 │
      DenseNet-121                            NLP
   (torchxrayvision)                    (regex + lexicons)
             │                                 │
    14 model_scores                     ReportExtraction
    (raw sigmoid)                      (findings, status, anatomy)
             │                                 │
             └──────────────┬──────────────────┘
                            │
                 CROSS-MODAL VERIFIER
                            │
                FusionResult(agreed, conflicts, ...)
                            │
             UNCERTAINTY SIGNALS + TRIAGE ENGINE
                            │
              TriageResult(priority, reasons, trace)
                            │
                    EVIDENCE RAG (FAISS)
                            │
                list[Chunk] with provenance
                            │
                SAFETY GATE (pre-generation)
                            │
              LLM explanation  ─or─  template reporter
                            │
                SAFETY GATE (post-generation)
                            │
                    RadiantResult
                            │
                     HUMAN REVIEW
```

The **cross-modal verifier** is the differentiator. Everything else is
in service of it.

## Module responsibilities

Every module owns its own Pydantic schema. No model is defined twice;
`backend/schemas/result.py` re-exports and composes. A test asserts
object identity for every schema.

| Module | Responsibility | Schema owner |
|---|---|---|
| `backend/vision` | DenseNet-121 wrapper, preprocessing, thresholds | `VisionResult`, `VisionFinding` |
| `backend/nlp` | Deterministic extractor, lexicons | `ReportExtraction`, `ReportFinding` |
| `backend/fusion` | Cross-modal comparison | `FusionResult`, `FusionConflict` |
| `backend/uncertainty` | Signal computation | (no schema; dict) |
| `backend/triage` | Priority engine | `TriageResult`, `ReviewPriority` |
| `backend/rag` | FAISS index, retriever | `Chunk`, `RetrievalResult` |
| `backend/safety` | Pre/post gate, privacy | `SafetyResult`, reason codes |
| `backend/llm` | AMD client, template fallback, payload | `ReporterOutput` (dataclass) |
| `backend/schemas` | Composition | `RadiantResult`, `EvidenceItem`, `FinalFinding`, `UncertaintySignalsModel`, `ExplanationBlock`, `HumanReviewBlock` |
| `backend/evaluation` | Six evaluation modules | (no schema) |
| `backend/api` | FastAPI app, pipeline, state | (uses schemas) |

## Pipeline orchestration

`backend/api/pipeline.py` is the single source of truth for the
pipeline. It defines a `_PipelineRun` class with one method per step.
Both drivers call those methods:

- `run()` — synchronous, returns a `RadiantResult`
- `run_streaming()` — generator, yields `(step_name, None)` after each
  step really completes, then `("done", result)`

The two drivers share the same step methods. There is no duplicated
logic between the sync and streaming paths.

Every step is wrapped in `try/except`. On failure:

- the step is skipped
- `vision_failed` / `report_failed` / `retrieval_failed` is set
- human review is required in the final result
- the pipeline continues
- **the caller never sees a stack trace**

## Why these choices

**Why torchxrayvision DenseNet-121 and not a newer model?**
It is trained on NIH ChestX-ray14 and its 14-label output aligns with
the dataset RADIANT evaluates against. It is small enough to run on CPU
in under a second per image. The weights are distributed with the
library (Apache-2.0). Any alternative would need to justify the same
three properties.

**Why a regex NLP extractor and not a transformer?**
The extractor's output must be traceable to specific text spans and
must be deterministic for the same input. Regex plus a small lexicon
gives exactly that. It is also auditable: a clinician can read the
lexicon and understand what the extractor will and won't catch. A
transformer would hide the rules behind weights and would require
training data we do not have. `transformers` is installed only as a
dependency of `sentence-transformers`.

**Why FAISS on CPU and not a hosted vector store?**
The knowledge base is 5 files. A 300 MB local index would be
overkill and would require network access. FAISS-CPU is fast enough
(20 ms per query), small, and runs offline. Scale to a hosted store
only if the corpus grows.

**Why the LLM is optional and falls back to a template?**
The template path was demonstrated to work during a network outage in
the evaluation stage. It produces the same citations, the same
structure, and the same disclaimer as the LLM. The LLM adds reading
quality, not correctness. The pipeline does not depend on it.

**Why one safety gate before generation and one after?**
The pre-check decides whether generation may proceed at all. The
post-check validates the LLM's output — specifically, that every
citation resolves to a real retrieved chunk and that the text contains
no diagnosis or treatment language. Either gate can block. Neither
gate trusts the LLM.

## Tracing

`RadiantResult` contains every layer's structured output. A test
(`tests/unit/test_schemas.py::test_explanation_text_is_isolated`)
proves that mutating the explanation text cannot change any other
field. The explanation is a leaf, not an input.

## Deployment

- **Local development:** `uvicorn` on port 8000, `streamlit` on 8501
- **Frontend:** Streamlit consumes `/analyze/stream` for real per-step
  progress
- **No Docker:** a Dockerfile was not produced in this build; the
  environment is documented in `requirements.txt` and the README
- **No GPU:** all inference is CPU
- **No authentication:** the API has no auth in this prototype. It is
  not exposed to the public internet.

## What is not in this architecture

- No model training — pre-trained checkpoints only
- No fine-tuning — evaluation uses the model as shipped
- No persistent store — every request is stateless
- No multi-tenancy — single-user prototype
- No authentication, rate limiting, or audit log

These are deliberate omissions for a research prototype and are noted
in `README.md` and `docs/safety.md`.