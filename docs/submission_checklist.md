# RADIANT — Submission checklist

Everything needed to submit RADIANT.

## Required artifacts

| Item | Status | Notes |
|---|---|---|
| Public GitHub repository | ✅ | https://github.com/zehra-coded/radiant |
| MIT license file | ✅ | `LICENSE` |
| README with setup + run | ✅ | `README.md` |
| Short description | ✅ | Below |
| Long description | ✅ | Below |
| Demo video (5 min MP4) | ⏳ | Script in `docs/demo_script.md` |
| Slide presentation (PDF) | ⏳ | Deck in `docs/pitch_deck.md` |
| Cover image (16:9) | ⏳ | Dashboard screenshot |
| Application URL | ⏳ | Local Streamlit or deploy |
| Team info | ✅ | Solo — zehra-coded |

## Short description (paste into form — 100 chars max)

> RADIANT compares chest X-ray model findings against report
> findings and surfaces disagreements for human review.

## Long description (paste into form)

> RADIANT is a multimodal chest X-ray review prototype. It runs a
> vision model (DenseNet-121 trained on NIH ChestX-ray14) and a
> deterministic report parser in parallel, compares their outputs
> through a cross-modal verifier, and surfaces disagreements for
> human review. It never picks a winner in a conflict. It assigns a
> review priority, retrieves supporting evidence, generates an
> explanation through AMD-hosted LLM endpoints (with a
> deterministic fallback), and passes the output through two safety
> gates. Every number in the project's evaluation comes from a real
> execution; items that could not be measured are marked NOT RUN
> with a reason. RADIANT is a research prototype, not a diagnostic
> tool. All outputs require review by a licensed physician.

## Project links

| Field | Value |
|---|---|
| GitHub | https://github.com/zehra-coded/radiant |
| Demo video | _(paste URL after recording)_ |
| Slide deck PDF | _(paste Drive link or attach)_ |
| Live app URL | _(see options below)_ |
| Cover image | `docs/screenshots/cover.png` |

## Application URL — two options

### Option A — Local (simplest, recommended)

Do not deploy. State in the form:

> "The application runs locally. Setup instructions are in the
> README. The screenshot and demo video show the live UI."

Judges usually accept this for research prototypes. The demo video
is the proof.

### Option B — Deploy to Streamlit Community Cloud

1. Go to https://share.streamlit.io
2. Sign in with GitHub
3. Click **New app**
4. Repository: `zehra-coded/radiant`
5. Branch: `main`
6. Main file: `frontend/streamlit_app.py`
7. Add secret `RADEON_API_KEY` (optional)
8. Deploy

**Caveat:** the API backend cannot be reached from Streamlit Cloud
unless you host it separately. Without it, the UI runs in template
mode. **Do not claim the cloud version uses the LLM unless the
backend is actually deployed there.**

## Cover image

Screenshot of the dashboard after a successful Analyze click. Crop to
16:9. Save as `docs/screenshots/cover.png`. Suggest the Consistency
tab showing the Effusion conflict.

## Video upload

- **YouTube unlisted** works. Paste the URL in the form.
- **Google Drive** with "Anyone with the link" works.
- Do **not** use a platform that requires a login.

## Final check before Submit

- [ ] Repository is public and renders in a browser
- [ ] README shows the disclaimer at the top
- [ ] No `.env` file in the repository
- [ ] No API key visible anywhere in the file tree
- [ ] Demo video uploaded and accessible without login
- [ ] Slide PDF exported and accessible
- [ ] Short description, long description, all links pasted
- [ ] Team info correct (solo — zehra-coded)
- [ ] Cover image uploaded

## After submission

- Keep the repository public until results are announced
- Do not push commits that break the build during judging
- Do not rotate or revoke the AMD API key during judging
- Monitor the Discord for questions

## Common judge follow-ups

| Question | Where the answer lives |
|---|---|
| "Is this clinically validated?" | `README.md` limitations + `docs/safety.md` |
| "What's the accuracy?" | AUROC 0.668 macro in `docs/evaluation.md` |
| "How does the LLM see the image?" | It doesn't. `docs/safety.md`. |
| "What does AMD provide?" | Hosted LLM endpoints. `docs/amd_setup.md`. |
| "Does this run on AMD GPUs?" | No. `README.md` AMD section. |
| "What's NOT done?" | Listed as `NOT RUN` in `docs/evaluation.md`. |