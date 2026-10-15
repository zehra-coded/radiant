# Synthetic reports

**These are synthetic radiology reports authored for RADIANT. They are
not real patient reports.**

Every file in this folder is fictional. Each is designed to exercise a
specific behaviour of the pipeline:

- `report_agreement.txt` — image-model findings and report findings
  should agree.
- `report_conflict.txt` — image-model findings and report findings
  should conflict (a finding is positive in one modality and negated
  in the other).
- `report_negation_uncertainty.txt` — exercises negation cues
  ("no X", "without X") and uncertainty cues ("possible X",
  "cannot exclude X").

These reports are used for:
- Unit tests of the NLP extractor (Stage 5)
- Unit tests of the cross-modal verifier (Stage 6)
- Integration tests of the full pipeline (Stage 12)
- Demo scenarios (Stage 18)

They are **not** clinical documents and must never be presented as
such. Any UI showing a synthetic report must label it clearly as
`SYNTHETIC`.