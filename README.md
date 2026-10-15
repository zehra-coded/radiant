# RADIANT

**Multimodal Chest X-Ray Triage & Evidence Assistant**

> This system is a research prototype for clinical decision support only.
> It is not a diagnostic tool. All outputs must be reviewed by a licensed
> physician.

RADIANT combines chest X-ray analysis with structured radiology-report
understanding to surface potentially urgent findings, identify
image–report inconsistencies, retrieve supporting medical evidence, and
route uncertain or high-risk cases to human review.

**[Repository](https://github.com/zehra-coded/radiant)** ·
**[Evaluation](docs/evaluation.md)** ·
**[Architecture](docs/architecture.md)** ·
**[Safety](docs/safety.md)** ·
**[Demo guide](docs/demo_guide.md)**

---

## The problem

Chest X-ray machine learning models are good at producing ranked finding
scores. In a clinical workflow, those scores are read alongside a
radiologist's structured report — and the two signals sometimes
disagree. When they do, the disagreement is easy to miss.

RADIANT is built around that gap. It runs a vision model and a report
parser in parallel, compares their outputs finding by finding, and
surfaces conflicts for human review. It never picks a winner.

## What RADIANT is

- A clinical review-prioritization prototype
- A multimodal pipeline (image model + report NLP)
- A cross-modal image/report consistency verifier
- An uncertainty-aware review-priority system
- An evidence-grounded RAG system
- A human-review routing system

## What RADIANT is NOT

- A diagnostic system
- A medical chatbot
- A treatment recommender
- A radiologist replacement
- A patient-facing product
- A clinically validated system
- A workload that runs on AMD GPUs

RADIANT produces structured review-priority information, evidence
citations, and a generated explanation. It does not diagnose, does not
recommend treatment, and does not make clinical decisions.

## The pipeline

```
           Chest X-ray                     Report text
                │                               │
          DenseNet-121                       NLP (regex)
         (torchxrayvision)                  + lexicons
                │                               │
         14 finding scores              14 findings + status
                │                               │
                └───────────────┬───────────────┘
                                │
                     CROSS-MODAL VERIFIER
                                │
                     UNCERTAINTY + PRIORITY
                                │
                        EVIDENCE RAG
                                │
                    SAFETY GATE (pre-generation)
                                │
                 LLM explanation or template fallback
                                │
                    SAFETY GATE (post-generation)
                                │
                        RadiantResult
                                │
                         HUMAN REVIEW
```

The center of the diagram is the project's contribution: **a
cross-modal verifier** that compares the image model's findings against
the report's findings and surfaces disagreements without resolving them.

## Quick facts

| | |
|---|---|
| **Vision model** | DenseNet-121 (torchxrayvision, trained on NIH ChestX-ray14) |
| **Vision AUROC** | macro 0.668 on 1000 real NIH test images |
| **NLP extractor** | Deterministic regex + lexicons (no LLM) |
| **NLP diagnosis recall** | 0.60 on 2000 real Rad-SpRL reports |
| **Cross-modal verifier** | 1.000 consistency accuracy on 7 labeled cases |
| **Retrieval** | Recall@3 = 1.000, MRR = 0.900 |
| **Safety gate** | 11/11 reason-code accuracy |
| **Generation** | Citation validity rate 1.000 (both LLM and template paths) |
| **Tests** | 105 passing |
| **LLM** | Optional, via AMD Token Factory (DeepSeek-V4-Flash) |
| **Fallback** | Deterministic template always available |

Every number above comes from a real execution. See
[`docs/evaluation.md`](docs/evaluation.md) and
[`docs/evaluation_addendum.md`](docs/evaluation_addendum.md).

## Architecture

RADIANT is organized into layered modules. Each owns its own Pydantic
schema. The pipeline is orchestrated by `backend/api/pipeline.py`.

```
backend/
├── core/          configuration and logging
├── vision/        DenseNet-121 wrapper, preprocessing, thresholds
├── nlp/           deterministic extractor, lexicons, schema
├── fusion/        cross-modal verifier
├── uncertainty/   signal computation
├── triage/        review-priority engine
├── rag/           FAISS index + retriever
├── safety/        pre/post generation gate, privacy enforcement
├── llm/           AMD client, template reporter, prompts, payload
├── schemas/       RadiantResult (composed output type)
├── evaluation/    six evaluation modules
└── api/           FastAPI app, pipeline, cached state
frontend/
└── streamlit_app.py
```

See [`docs/architecture.md`](docs/architecture.md) for details.

## Installing and running

**Prerequisites:**

- Windows 10/11, macOS, or Linux
- Python 3.10 or 3.11
- ~5 GB free disk space for the virtual environment

**Setup:**

```bash
git clone https://github.com/zehra-coded/radiant.git
cd radiant
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

**Configure the LLM (optional):**

Copy `.env.example` to `.env` and fill in `RADEON_API_KEY`. Without a
key, RADIANT still runs — the LLM reporter falls back to the
deterministic template.

**Run the API:**

```bash
uvicorn backend.api.main:app --port 8000
```

**Run the dashboard:**

```bash
streamlit run frontend/streamlit_app.py
```

Open `http://localhost:8501`.

**Run the tests:**

```bash
pytest tests/ -m "not integration"       # fast unit tests
pytest tests/ -m integration              # real models, real data
```

**Run the evaluation:**

```bash
python scripts/run_evaluation.py
```

Writes `docs/evaluation.md`. Takes ~5 minutes on CPU.

## Safety and human review

RADIANT requires human review when any of the following is true:

- review priority is `URGENT_REVIEW` or `EMERGENCY_REVIEW`
- the cross-modal verifier found any conflict
- any pipeline step failed
- the safety gate blocked or flagged the output
- retrieved evidence was insufficient

Every reason is recorded in the `human_review.reasons` list in the
structured output.

The safety gate runs **before** and **after** LLM generation. It fails
closed: if evidence is missing, the LLM is instructed to write
"Insufficient evidence."; if the LLM produces a fabricated citation or
out-of-scope language, its output is rejected and the template reporter
is used instead. The required disclaimer is always present in the
user-facing message.

See [`docs/safety.md`](docs/safety.md).

## Evidence retrieval

RADIANT ships with five development fixtures covering pneumothorax,
pleural effusion, cardiomegaly, consolidation, and atelectasis. Each
fixture carries the banner
`DEVELOPMENT FIXTURE — NOT AUTHORITATIVE CLINICAL GUIDANCE`, plus
`source:` and `license:` lines.

Retrieved chunks carry one of three provenance values:
`DEVELOPMENT_FIXTURE`, `LICENSED_SOURCE`, or `UNVERIFIED_SOURCE`.
The frontend displays a visible warning banner for
`DEVELOPMENT_FIXTURE` chunks and blocks `UNVERIFIED_SOURCE` chunks from
generation.

## Datasets

- **NIH ChestX-ray14** — CC0 1.0, real chest X-rays, used for vision
  evaluation and real-image smoke tests. Not redistributed.
- **Rad-SpRL** — license unverified, real radiology reports with
  spatial role labeling, used for NLP extractor evaluation. Not
  redistributed.
- **Synthetic** — RADIANT-authored reports and images, MIT-licensed,
  used for unit tests and demos.

See [`docs/datasets.md`](docs/datasets.md) and
[`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md).

## AMD relationship

RADIANT consumes **AMD-hosted LLM endpoints** via the AMD Token Factory
public API. The LLM is used for explanation generation only; it never
sees the image, never receives patient identifiers, and cannot introduce
new findings.

RADIANT does not run its own workloads on AMD GPUs. There is no ROCm
installation, no GPU-backed vLLM, and no GPU benchmark in this project.
The integration is API-only.

See:

- [`docs/amd_setup.md`](docs/amd_setup.md) — the full integration boundary
- [`docs/amd_benchmarks.md`](docs/amd_benchmarks.md) — client-observed API
  latency measurements

## Limitations

RADIANT is a prototype. Its limitations are documented, not hidden:

- **The vision threshold is unvalidated.** `VISION_PRESENT_THRESHOLD =
  0.5` is a placeholder. F1 at this threshold is low. It is labeled
  `UNVALIDATED PROTOTYPE HEURISTIC` in the code and in every report.
- **Model scores are not calibrated probabilities.** They are raw
  sigmoid outputs. The field name is `model_score`, not `probability`.
- **The NLP extractor's hedge and anatomy vocabularies are narrow.**
  Hedge agreement with Rad-SpRL is 0.11; anatomy agreement is 0.23.
- **Negation precision/recall is not measured.** The Rad-SpRL corpus
  does not annotate present/negated status. This is documented as
  `NOT RUN`.
- **Review priorities are prototype routing suggestions.** They are
  not validated clinical triage categories.
- **A conflict is not a medical error.** It is a disagreement between
  two structured inputs. Neither is treated as ground truth.
- **The LLM is optional and always safety-gated.** Its text cannot
  alter any structured field.

## Project status

All stages 0 through 18 have been implemented and tested. The full
pipeline runs on real NIH images and real Rad-SpRL reports. Every
number in `docs/evaluation.md` comes from a real execution.

## License

MIT — see [`LICENSE`](LICENSE). Third-party components keep their own
licenses — see [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md).

## Disclaimer

This system is a research prototype for clinical decision support only.
It is not a diagnostic tool. All outputs must be reviewed by a licensed
physician.