# RADIANT — Pitch

## One sentence

RADIANT reads a chest X-ray and a radiology report, finds where the
two disagree, cites evidence for the disagreement, and routes it to a
human reviewer.

## The problem

Chest X-ray AI models produce ranked finding scores. In clinical use,
those scores are read alongside a radiologist's report — and the two
signals sometimes disagree. A model that says "effusion present" on an
image where the report says "no pleural effusion" is not necessarily
wrong. But the disagreement is exactly what a reviewer needs to see,
and today there is no tool that surfaces it.

## The solution

RADIANT adds the missing layer: a **cross-modal verifier** that
compares the image model's 14 finding scores against the report's 14
extracted findings, detects conflicts, assigns a review priority, and
routes the case to a human with full evidence and traceability.

RADIANT does not diagnose. It does not treat. It does not pick a
winner in a disagreement. It escalates.

## Architecture

```
      Chest X-ray                        Report text
           │                                  │
    DenseNet-121 (CPU)                    NLP (regex)
    14 model_scores                    14 findings + status
           │                                  │
           └──────────────┬───────────────────┘
                          │
                CROSS-MODAL VERIFIER
                          │
               UNCERTAINTY + TRIAGE
                          │
                  EVIDENCE RAG
                          │
              SAFETY GATE (pre-generation)
                          │
         LLM explanation  ─or─  template fallback
                          │
              SAFETY GATE (post-generation)
                          │
                    RadiantResult
                          │
                    HUMAN REVIEW
```

The center box is the pitch. Everything else is engineering rigor.

## Differentiation

**Most chest X-ray AI projects stop at "my model outputs a label."**
RADIANT does the harder and more useful thing: it compares two
independent signals, quantifies the disagreement, and surfaces it with
evidence and traceability.

**Most hackathon projects hide their weaknesses.** RADIANT publishes
them. Every number in the evaluation comes from a real execution.
Every limitation is documented. Every NOT RUN item has a reason.

**The safety architecture is real, not decorative.** During
evaluation, our network dropped. The LLM endpoint became unreachable.
The pipeline fell back to the deterministic template reporter and
produced 5 valid citations per case. Zero fabricated citations reached
the output. This is what fail-closed means in practice.

## Real numbers

All measured, all traceable to `docs/evaluation.md`.

| Metric | Value | Source |
|---|---|---|
| Vision macro AUROC | 0.668 | 1000 real NIH test images |
| Cardiomegaly AUROC | 0.812 | same run |
| Edema AUROC | 0.765 | same run |
| NLP diagnosis span recall | 0.601 | 2000 real Rad-SpRL reports |
| Fusion consistency accuracy | 1.000 | 7 labeled cases |
| Fusion conflict-count accuracy | 1.000 | same 7 cases |
| Retrieval Recall@3 | 1.000 | 10 hand-labeled queries |
| Retrieval MRR | 0.900 | same 10 queries |
| Safety gate accuracy | 11/11 | 8 pre + 3 post cases |
| Citation validity (both paths) | 1.000 | 5 live runs |
| Tests passing | 105 | `pytest tests/` |

**Items marked NOT RUN:** negation precision/recall (Rad-SpRL does not
annotate present/negated status), GPU benchmarks (no GPU).

## Honest weaknesses

- Hedge agreement with Rad-SpRL is **0.11**. Our uncertainty lexicon
  doesn't match Rad-SpRL's hedging style.
- Anatomy agreement is **0.23**. Our anatomy lexicon is too narrow.
- The vision threshold `0.5` is **unvalidated**. F1 at this threshold
  is low. We do not tune it without data.
- Model scores are **raw sigmoid outputs**, not calibrated
  probabilities. The field is named `model_score`, not `probability`.
- The knowledge base is **5 development fixtures**, not a clinical
  corpus. They are labeled as such in the UI.

These are listed because they are true, not because we were forced to.

## AMD

**What AMD gave us:** an API key to hosted models through the AMD
Token Factory.

**What AMD did NOT give us:** GPUs, credits, or inference on our own
hardware.

**RADIANT's honest relationship:**
- LLM inference: real AMD-hosted `DeepSeek-V4-Flash` via the AMD Token
  Factory API.
- Model allowlist prevents accidentally using a vision-capable model.
- Quota reader shows remaining budget when the endpoint responds.
- API-level latency measured (p50 ~925 ms from a 10-call run). No GPU
  numbers, because there is no GPU.

**The one-line claim:** *"RADIANT consumes AMD-hosted LLM endpoints
via the AMD Token Factory public API. It does not run its own
workloads on AMD GPUs."*

## What RADIANT is NOT

Saying what you are not is more credible than claiming what you are.

- Not a diagnostic system
- Not a medical chatbot
- Not a treatment recommender
- Not a radiologist replacement
- Not a patient-facing product
- Not a clinically validated system

## Business value

- **Reduces reviewer time** by prioritizing which cases need attention
  first, with traceable reasons.
- **Reduces missed disagreements** by systematically comparing two
  signals that today are compared only by a human reading both at once.
- **Runs on CPU** — no GPU budget required for the vision model.
- **LLM is optional** — the pipeline works without any external API.
- **Fits into existing workflows** — it doesn't replace the report or
  the radiologist; it adds a review-priority signal on top.

## Team

Solo submission. [zehra-coded](https://github.com/zehra-coded)

## Links

- **Repository:** https://github.com/zehra-coded/radiant
- **Evaluation:** `docs/evaluation.md`
- **Architecture:** `docs/architecture.md`
- **Safety:** `docs/safety.md`
- **Demo guide:** `docs/demo_guide.md`

## Required disclaimer

This system is a research prototype for clinical decision support only.
It is not a diagnostic tool. All outputs must be reviewed by a licensed
physician.