# RADIANT — Demo guide

Step-by-step instructions for running and demonstrating RADIANT. This
guide assumes Windows with PowerShell, but the commands translate
directly to macOS and Linux.

## Prerequisites

- Python 3.10 or 3.11 installed
- The virtual environment created (`F:\radiant-venv` in the original
  setup; substitute your own path)
- The NIH ChestX-ray14 dataset at `D:\NIH-ChestXray14\images\` for
  real-image inference, or use the synthetic images in
  `data\sample_xrays\` for UI testing
- The AMD Token Factory API key in `.env` for LLM explanations (optional;
  the template reporter works without it)

## Starting the demo

You need **two** PowerShell windows.

### Window 1 — API backend

```powershell
cd C:\Users\Anyone\radiant-workspace\radiant
F:\radiant-venv\Scripts\Activate.ps1
F:\radiant-venv\Scripts\python.exe -m uvicorn backend.api.main:app --port 8000
```

Wait for:

```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

**Leave this window alone.** It runs the API.

### Window 2 — Streamlit dashboard

```powershell
cd C:\Users\Anyone\radiant-workspace\radiant
F:\radiant-venv\Scripts\Activate.ps1
streamlit run frontend\streamlit_app.py
```

Streamlit prints a URL: `http://localhost:8501`.

Open it in a browser.

## Running the demo

### Step 1 — Show the header and disclaimer

The dashboard opens with:

- The title "RADIANT — Multimodal Chest X-Ray Review"
- The required disclaimer in a red banner at the top

**Say:** *"The disclaimer is always visible. RADIANT never hides what
it is."*

### Step 2 — Upload an image

Use one of these:

- **Real NIH image:** `D:\NIH-ChestXray14\images\00000001_000.png`
- **Synthetic placeholder:** any file in `data\sample_xrays\`

**Say:** *"This is a real NIH ChestX-ray14 image, ground-truth labeled
Cardiomegaly."*

### Step 3 — Paste a report

Use the default report text, or paste the contents of
`data\synthetic\report_conflict.txt`.

**Say:** *"This synthetic report explicitly negates both pneumothorax
and pleural effusion. Watch what happens when the image model sees
those findings."*

### Step 4 — Click Analyze

Watch the progress steps tick off one by one:

```
⏳ Starting…
✅ Vision model analyzed the image
✅ Report findings extracted
✅ Image and report compared
✅ Review priority computed
✅ Evidence retrieved
✅ Safety gate (pre) passed
✅ Explanation generated
```

**Say:** *"Each step appears only after the backend confirms it
completed. There is no fake progress. The frontend consumes a
Server-Sent Events stream from the API."*

### Step 5 — Walk through the tabs

**Findings** — vision model scores per label with progress bars.
**Say:** *"Model scores are raw sigmoid outputs, not calibrated
probabilities. The label says so."*

**Report Analysis** — extracted findings with status and evidence span.
**Say:** *"The extractor correctly parsed 'No pneumothorax' as negated.
No false positives."*

**Consistency** — the fusion result.
**Say:** *"Here's the differentiator. The image model says Effusion and
Pneumothorax are present. The report says they are negated. RADIANT
flags both as conflicts. It does not pick a winner."*

**Uncertainty** — low-confidence findings and conflicts, with reasons.
**Say:** *"Every reason the priority is URGENT_REVIEW is listed here,
traceable to a specific signal."*

**Evidence** — retrieved chunks with provenance labels.
**Say:** *"Each chunk is labeled 'Development fixture — not
authoritative clinical guidance.' This is not the evidence — this is
the prototype for how evidence would be surfaced once we have licensed
sources."*

**Explanation** — the generated explanation.
**Say:** *"The LLM cites real chunks. If it fails — network, quota,
fabricated citation — the template reporter produces the same
structure without the LLM. We demonstrated that path during our
evaluation when our network dropped."*

**Safety** — reason codes, disclaimer, human-review state.
**Say:** *"Allowed, requires human review. The reason codes match the
exact safety rules in the code."*

### Step 6 — Point at the LLM backend panel

At the bottom of the page:

- Configured model: `DeepSeek-V4-Flash`
- Key present: yes
- Generator used: `amd_token_factory` or `template`

**Say:** *"LLM inference is provided by AMD-hosted models via the AMD
Token Factory API. RADIANT runs on CPU. It does not run its own
workloads on AMD GPUs, and we don't claim otherwise."*

## Running the evaluation (optional)

If a judge asks "do you have numbers?":

```powershell
python scripts\run_evaluation.py
```

Takes ~5 minutes. Writes `docs\evaluation.md` with:

- 1000 real NIH images, vision AUROC per label
- 2000 real Rad-SpRL reports, NLP span recall
- Cross-modal fusion accuracy on 7 labeled cases
- Retrieval Recall@K and MRR
- Safety gate accuracy on 11 cases
- Generation groundedness on 5 live runs

**Say:** *"Every number comes from a real execution. Items we could
not measure are marked NOT RUN with a reason."*

## Stopping the demo

In Window 1 (running uvicorn): press **Ctrl+C**.

Nothing is corrupted. You can restart at any time.

## What to say if something breaks during the demo

1. **API not responding:** check Window 1 for a stack trace.
2. **Streamlit shows "Could not reach the API":** the backend isn't
   running, or is on a different port. Check `API_BASE_URL`.
3. **LLM calls fail:** the template fallback engages automatically.
   The demo continues. Point at the fallback as a feature.
4. **Model file missing:** the first DenseNet call downloads ~30 MB.
   Needs network once.

## Timing

Full walkthrough of all seven tabs: **4–5 minutes.**

For a 5-minute submission video: skip the evaluation run, focus on
Steps 1–6.