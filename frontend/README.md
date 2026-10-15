# RADIANT frontend

A single Streamlit application that renders the RADIANT analysis result
in a clinical-review layout.

## Running

The dashboard calls the FastAPI backend at `API_BASE_URL` (default
`http://127.0.0.1:8000`). **Start the backend first**, in a terminal:

```powershell
cd C:\Users\Anyone\radiant-workspace\radiant
F:\radiant-venv\Scripts\Activate.ps1
F:\radiant-venv\Scripts\python.exe -m uvicorn backend.api.main:app --port 8000
```

Then, in a **second** terminal, run the dashboard:

```powershell
cd C:\Users\Anyone\radiant-workspace\radiant
F:\radiant-venv\Scripts\Activate.ps1
streamlit run frontend\streamlit_app.py
```

Streamlit will open `http://localhost:8501` in your browser.

## What the dashboard shows

- Header with the project name and the required disclaimer.
- Two-column input: image upload on the left, report text + reviewer query on the right.
- Real SSE-driven progress: each step appears only after the API confirms it completed.
- A priority badge with four color states.
- A human-review banner when required.
- Seven tabs: Findings, Report Analysis, Consistency, Uncertainty, Evidence, Explanation, Safety.
- Every evidence chunk shows its provenance. Development fixtures are visibly labelled.
- An "LLM backend" panel showing the configured model and the actual generator used.

## Design notes

- Not a chatbot. No free-form conversation.
- No image metrics from a GPU panel. This project has no GPU.
- The disclaimer is always visible.
- Conflicts are never resolved in the UI; they are shown to the reviewer.