# RADIANT — Pitch deck

Six slides for a 3-minute pitch. Each slide has a title, content, and
suggested narration.

---

## Slide 1 — Title

**RADIANT**
*Multimodal Chest X-Ray Triage & Evidence Assistant*

**Narration (20s):**
> "RADIANT reads a chest X-ray and a radiology report, finds where
> the two disagree, cites evidence for the disagreement, and routes
> it to a human reviewer. It is a research prototype. It does not
> diagnose. The disclaimer is in every output."

---

## Slide 2 — The problem

**Title:** *Chest X-ray AI has a hidden gap*

**Content:**
- Vision models produce ranked finding scores
- Radiologists produce structured reports
- The two signals sometimes disagree
- Today, no tool surfaces that disagreement systematically

**Narration (30s):**
> "Chest X-ray AI models are good at producing ranked finding
> scores. In clinical use, those scores are read alongside the
> radiologist's report. The two signals sometimes disagree — the
> model says effusion present, the report says no pleural effusion.
> That disagreement is not a defect; it's exactly what a reviewer
> needs to see. But today there is no tool that surfaces it
> systematically. RADIANT does."

---

## Slide 3 — The solution

**Title:** *A cross-modal verifier, not another classifier*

**Content:**
- Vision model: DenseNet-121 on NIH ChestX-ray14
- Report parser: deterministic regex (no LLM, auditable)
- **Cross-modal verifier: the differentiator**
- Uncertainty + triage
- Evidence RAG
- Two safety gates (pre + post generation)
- LLM explanation — optional, always safety-gated

**Narration (30s):**
> "Most chest X-ray AI projects stop at 'my model outputs a label.'
> RADIANT does the harder and more useful thing: it compares two
> independent signals, quantifies the disagreement, and surfaces it
> with evidence and traceability. The cross-modal verifier is the
> project. Everything else is engineering rigor around it."

---

## Slide 4 — Real numbers

**Title:** *Measured, not estimated*

| Metric | Value | Source |
|---|---|---|
| Vision macro AUROC | 0.668 | 1000 real NIH test images |
| Cardiomegaly AUROC | 0.812 | same run |
| NLP diagnosis recall | 0.601 | 2000 real Rad-SpRL reports |
| Fusion accuracy | 1.000 | 7 labeled conflict cases |
| Retrieval Recall@3 | 1.000 | 10 hand-labeled queries |
| Safety gate accuracy | 11/11 | 8 pre + 3 post cases |
| Citation validity | 1.000 | 5 live runs, both paths |
| Tests passing | 105 | `pytest tests/` |

**Note:** *Items we could not measure are marked `NOT RUN` with a
reason. Nothing is estimated.*

**Narration (30s):**
> "Every number here comes from a real execution on real data. A
> thousand real NIH images for vision. Two thousand real Rad-SpRL
> reports for NLP. Six evaluation components. And the numbers are
> honest — our NLP hedge agreement is 0.11 and our anatomy
> agreement is 0.23. Those are weaknesses, and they're documented.
> Anything we couldn't measure is marked 'not run' with a reason."

---

## Slide 5 — Safety architecture

**Title:** *Fail-closed by design*

**Content:**
- Pre-generation gate decides whether generation may run
- Post-generation gate validates citations, blocks diagnosis language
- Fail-closed: missing evidence → "Insufficient evidence."
- Patient-identifier denylist → hard failure, call refused
- LLM never sees the image, never receives patient identifiers
- Template fallback works during network outages
- **During evaluation: 3 of 5 LLM outputs rejected by post-check and replaced.**

**Narration (30s):**
> "The safety gate runs twice. Before generation, it decides whether
> the LLM may run at all. After generation, it validates every
> citation against the retrieved set and blocks diagnosis or
> treatment language. During our own evaluation, our network
> dropped mid-run. The pipeline fell back to the deterministic
> template reporter and produced the same citations, the same
> structure, the same disclaimer. Zero fabricated citations reached
> the output. This is what fail-closed means in practice."

---

## Slide 6 — AMD and closing

**Title:** *AMD relationship — honest and API-only*

**Content:**
- **Consumes:** AMD-hosted LLM endpoints via the AMD Token Factory
- **Model:** `DeepSeek-V4-Flash` (text-only)
- **Precheck:** live endpoint state documented in `docs/amd_setup.md`
- **Benchmark:** client-observed API latency, p50 ~925 ms
- **Does NOT claim:** GPU throughput, ROCm, VRAM — there is no GPU
- One-line claim: *"RADIANT consumes AMD-hosted LLM endpoints. It
  does not run its own workloads on AMD GPUs."*

**Links:**
- Repository: https://github.com/zehra-coded/radiant
- Evaluation: `docs/evaluation.md`
- Architecture: `docs/architecture.md`
- Safety: `docs/safety.md`

**Narration (30s):**
> "RADIANT consumes AMD-hosted LLM endpoints via the AMD Token
> Factory API. It does not run its own workloads on AMD GPUs, and
> we don't claim otherwise. That's the honest relationship.
> Everything is public at github.com/zehra-coded/radiant. MIT
> licensed. Every commit is traceable. RADIANT is a research
> prototype, not a medical device. Thank you."

---

## Converting to PDF

Use Pandoc, Marp, or the VS Code "Markdown PDF" extension.

**VS Code option (easiest):**
1. Install extension "Markdown PDF" by yzane
2. Open `pitch_deck.md`
3. Right-click in the editor → **Markdown PDF: Export (pdf)**
4. A `pitch_deck.pdf` appears next to the file

**Or paste the content manually into Google Slides** — six slides,
about 15 minutes.

## Runtime

**3 minutes at a normal speaking pace.** If the hackathon caps at
2 minutes, merge slides 3 and 4.