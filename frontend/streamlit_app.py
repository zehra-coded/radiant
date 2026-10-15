"""
RADIANT Streamlit clinical-review dashboard.

Consumes the FastAPI backend's /analyze/stream endpoint so that each
progress step is shown only after its real completion event arrives.

Not a chatbot. Structured clinical-review layout.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Make 'import backend...' work regardless of where streamlit is run from.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
import streamlit as st

from backend.core.config import API_BASE_URL, LLM_MODEL, RADEON_API_KEY
from backend.safety.schema import DISCLAIMER

PRIORITY_COLORS = {
    "ROUTINE": "#2e7d32",           # green
    "MONITOR": "#f9a825",           # yellow
    "URGENT_REVIEW": "#ef6c00",     # orange
    "EMERGENCY_REVIEW": "#c62828",  # red
}

SSE_STEPS = [
    ("vision_done", "Vision model analyzed the image"),
    ("nlp_done", "Report findings extracted"),
    ("fusion_done", "Image and report compared"),
    ("priority_done", "Review priority computed"),
    ("retrieval_done", "Evidence retrieved"),
    ("safety_done", "Safety gate (pre) passed"),
    ("generation_done", "Explanation generated"),
]

st.set_page_config(page_title="RADIANT — Chest X-Ray Review", layout="wide")

# ---- Header --------------------------------------------------------
st.markdown(
    "<h1 style='margin-bottom:0'>RADIANT — Multimodal Chest X-Ray Review</h1>"
    "<p style='color:#666; margin-top:4px'>Research prototype for clinical "
    "decision support. Not for diagnostic use.</p>",
    unsafe_allow_html=True,
)

# Disclaimer, always visible, at the top.
st.error(f"**Disclaimer.** {DISCLAIMER}")

# ---- Two-column input row -----------------------------------------
col_image, col_report = st.columns([1, 1])

with col_image:
    st.subheader("Chest X-ray")
    uploaded = st.file_uploader(
        "Upload a PNG or JPG chest X-ray",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=False,
    )
    if uploaded is not None:
        st.image(uploaded, caption=uploaded.name, width="stretch")
with col_report:
    st.subheader("Radiology report")
    default_report = (
        "Chest radiograph, frontal view.\n\n"
        "Findings:\nNo pneumothorax. No pleural effusion. Cardiac silhouette "
        "is normal in size.\n\nImpression:\nNo acute cardiopulmonary abnormality."
    )
    report_text = st.text_area(
        "Paste the report text here",
        value=default_report,
        height=260,
    )
    user_query = st.text_input(
        "Reviewer query",
        value="Summarize this case for human review.",
    )

analyze_clicked = st.button("Analyze", type="primary", width="stretch")

# ---- Analysis ------------------------------------------------------
if analyze_clicked and uploaded is not None:
    progress_placeholder = st.empty()
    status_lines: list[str] = []
    with progress_placeholder.container():
        st.markdown("### Analyzing…")
        status_box = st.empty()

    def _render_status() -> None:
        with status_box.container():
            for line in status_lines:
                st.write(line)

    status_lines.append("⏳ Starting…")
    _render_status()

    files = {"image": (uploaded.name, uploaded.getvalue(), uploaded.type)}
    data = {"report": report_text, "query": user_query}

    final_payload: dict | None = None
    try:
        with httpx.stream(
            "POST",
            f"{API_BASE_URL}/analyze/stream",
            files=files,
            data=data,
            timeout=180.0,
        ) as r:
            if r.status_code != 200:
                st.error(f"API error: HTTP {r.status_code}")
                st.stop()
            for line in r.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                evt = json.loads(line[len("data: "):])
                step = evt.get("step")
                label = dict(SSE_STEPS).get(step)
                if label:
                    status_lines[-1] = status_lines[-1].replace("⏳", "✅")
                    status_lines.append(f"⏳ {label}")
                    _render_status()
                if step == "done":
                    final_payload = evt.get("payload")
                    status_lines[-1] = status_lines[-1].replace("⏳", "✅")
                    _render_status()
    except Exception as exc:
        st.error(f"Could not reach the API at {API_BASE_URL}: {exc}")
        st.stop()

    st.session_state["radiant_result"] = final_payload
    progress_placeholder.empty()

# ---- Result rendering ---------------------------------------------
result = st.session_state.get("radiant_result")
if result is None:
    st.info("Upload an image and click **Analyze** to run the pipeline.")
    st.stop()

st.divider()

# ---- Priority badge + human review banner -------------------------
triage = result.get("triage") or {}
priority = triage.get("priority", "UNKNOWN")
color = PRIORITY_COLORS.get(priority, "#666")

st.markdown(
    f"<div style='display:inline-block; padding:10px 18px; "
    f"background:{color}; color:white; border-radius:6px; "
    f"font-weight:600; font-size:20px'>Review priority: {priority}</div>",
    unsafe_allow_html=True,
)

if result.get("human_review", {}).get("required"):
    reasons = result["human_review"].get("reasons", [])
    st.warning("**Human review required.** " + "; ".join(reasons))

st.write("")

# ---- Tabs ----------------------------------------------------------
tabs = st.tabs(
    ["Findings", "Report Analysis", "Consistency", "Uncertainty",
     "Evidence", "Explanation", "Safety"]
)

# Tab 1 — Findings
with tabs[0]:
    st.subheader("Vision findings")
    st.caption(
        "Model scores are raw sigmoid outputs, NOT calibrated probabilities. "
        "Thresholds are unvalidated prototype heuristics."
    )
    vision = result.get("vision")
    if vision is None:
        st.warning("Vision module did not produce a result.")
    else:
        for f in vision["findings"]:
            score = f["model_score"]
            st.write(f"**{f['name']}** — `{score:.3f}` — status: `{f['status']}`")
            st.progress(min(max(score, 0.0), 1.0))

# Tab 2 — Report Analysis
with tabs[1]:
    st.subheader("Structured report findings")
    report_analysis = result.get("report_analysis")
    if report_analysis is None or not report_analysis.get("findings"):
        st.info("No findings extracted from the report.")
    else:
        for f in report_analysis["findings"]:
            st.write(
                f"**{f['name']}** — status: `{f['status']}` — "
                f"anatomy: `{f['anatomy']}` — uncertainty: `{f['uncertainty']}`"
            )
            st.caption(f"Evidence span: _{f['evidence_span']}_")

# Tab 3 — Consistency
with tabs[2]:
    st.subheader("Image / report consistency")
    fusion = result.get("cross_modal_analysis")
    if fusion is None:
        st.info("No comparison available.")
    else:
        st.write(f"Overall consistency: **{fusion['overall_consistency']}**")
        if fusion.get("agreement_ratio") is not None:
            st.write(f"Agreement ratio: `{fusion['agreement_ratio']:.3f}`")
        if fusion.get("conflicts"):
            st.error("**Conflicts detected:**")
            for c in fusion["conflicts"]:
                st.write(
                    f"- **{c['name']}**: vision says `{c['vision_status']}`, "
                    f"report says `{c['report_status']}` — {c['reason']}"
                )
        if fusion.get("vision_only"):
            st.write("Vision-only findings: " + ", ".join(fusion["vision_only"]))
        if fusion.get("report_only"):
            st.write("Report-only findings: " + ", ".join(fusion["report_only"]))

# Tab 4 — Uncertainty
with tabs[3]:
    st.subheader("Uncertainty signals")
    uncertainty = result.get("uncertainty", {}).get("signals", [])
    if not uncertainty:
        st.write("No uncertainty signals.")
    else:
        for s in uncertainty:
            st.write(f"- **{s['signal']}**: {s['reason']}")
    triage_reasons = triage.get("reasons", [])
    if triage_reasons:
        st.subheader("Why this priority?")
        for r in triage_reasons:
            st.write(f"- {r}")

# Tab 5 — Evidence
with tabs[4]:
    st.subheader("Retrieved evidence")
    evidence = result.get("evidence", [])
    if not evidence:
        st.info("No evidence chunks retrieved.")
    for e in evidence:
        banner = ""
        if e.get("provenance") == "DEVELOPMENT_FIXTURE":
            banner = "Development fixture — not authoritative clinical guidance."
        elif e.get("provenance") == "UNVERIFIED_SOURCE":
            banner = "⚠ UNVERIFIED source — must not be presented as clinical guidance."
        title = f"[{e['document_id']}:{e['chunk_id']}] {e['title']}"
        with st.expander(title):
            if banner:
                st.warning(banner)
            st.write(f"Score: `{e.get('score')}`")
            st.write(f"Source: `{e.get('source')}`")
            st.write(f"License: `{e.get('license')}`")
            st.write(e.get("text", ""))

# Tab 6 — Explanation
with tabs[5]:
    st.subheader("Generated explanation")
    exp = result.get("explanation")
    if exp is None:
        st.info("No explanation was generated.")
    else:
        st.caption(f"Generator: `{exp['generator']}`")
        if exp.get("fallback_reason"):
            st.warning(f"Fallback reason: {exp['fallback_reason']}")
        st.markdown(exp.get("text", ""))

# Tab 7 — Safety
with tabs[6]:
    st.subheader("Safety gate results")
    safety = result.get("safety")
    if safety is None:
        st.info("No safety result available.")
    else:
        st.write(f"Allowed: `{safety['allowed']}`")
        st.write(f"Requires human review: `{safety['requires_human_review']}`")
        st.write(f"Blocked or redirected: `{safety['blocked_or_redirected']}`")
        if safety.get("reason_codes"):
            st.write("Reason codes:")
            for rc in safety["reason_codes"]:
                st.write(f"- `{rc}`")
        st.info(safety.get("user_safe_message", ""))

# ---- LLM Backend panel --------------------------------------------
st.divider()
st.subheader("LLM backend")
col_a, col_b, col_c = st.columns(3)
col_a.metric("Configured model", LLM_MODEL)
col_b.metric("Key present", "yes" if RADEON_API_KEY else "no")
gen = result.get("explanation", {}).get("generator", "unknown")
col_c.metric("Generator used", gen)
st.caption(
    "LLM inference is provided by AMD-hosted models via the AMD Token "
    "Factory API. RADIANT itself runs on CPU; it does not execute "
    "workloads on AMD GPUs."
)
if result.get("explanation", {}).get("fallback_reason"):
    st.info(f"Fallback: {result['explanation']['fallback_reason']}")

# ---- Footer --------------------------------------------------------
st.divider()
st.caption(DISCLAIMER)