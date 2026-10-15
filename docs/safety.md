# RADIANT — Safety model

This document describes how RADIANT protects against unsafe or
unsupported output. It is written for engineers and reviewers.

RADIANT is a research prototype. It is not a medical device. It does
not diagnose, does not recommend treatment, and does not make clinical
decisions. Every output requires human review.

## Fail-closed principle

Every safety-relevant decision in RADIANT fails closed. When the
information needed to make a decision is missing, the system blocks or
requires human review rather than assuming safety.

Examples:

- Missing evidence → the LLM is instructed to write "Insufficient
  evidence." Template reporter writes "Insufficient evidence."
- Retrieval failure → retrieval_failure signal fires; priority is at
  least MONITOR, never ROUTINE.
- Model failure → model_failure signal fires; priority is at least
  URGENT_REVIEW, never ROUTINE.
- Invalid citation in LLM output → post-generation check blocks,
  template reporter is used instead.
- Patient identifier in the outbound payload → the LLM call is refused
  entirely (not stripped, not sanitized — refused).

## The two safety gates

### Pre-generation check

Runs before any LLM or template generation.

Inputs: user query, triage result, fusion result, evidence chunks,
failure flags.

Reason codes it can emit:

| Code | Trigger |
|---|---|
| `malformed_input` | empty user query |
| `model_failure` | vision or NLP module failed |
| `retrieval_failure` | evidence retrieval failed |
| `vision_report_conflict` | fusion result contains conflicts |
| `urgent_or_emergency` | triage priority is URGENT_REVIEW or EMERGENCY_REVIEW |
| `low_confidence_finding` | uncertainty signals include low_confidence |
| `diagnosis_request_out_of_scope` | query matches diagnosis patterns |
| `treatment_request_out_of_scope` | query matches treatment patterns |
| `insufficient_evidence` | no evidence chunks retrieved |

Hard blocks (generation refused): `diagnosis_request_out_of_scope`,
`treatment_request_out_of_scope`, `malformed_input`,
`privacy_violation_hard_fail`.

Everything else: generation proceeds, `requires_human_review` is set.

### Post-generation check

Runs after generation.

Validates:

1. Every `[source:doc_id:chunk_id]` citation resolves to a real
   retrieved chunk.
2. The generated text contains no diagnosis or treatment language.
3. If evidence was empty, the text must not contain finding claims.

Reason codes it can emit:

| Code | Trigger |
|---|---|
| `citation_validation_failure` | one or more citations do not resolve |
| `grounding_validation_failure` | diagnosis or treatment language present |
| `insufficient_evidence` | finding claims present but no evidence |

Hard blocks: `citation_validation_failure`,
`grounding_validation_failure`.

When post-generation blocks the LLM output, `backend/llm/reporter.py`
replaces it with the deterministic template reporter and records the
fallback reason.

## Privacy enforcement

Rule: the LLM never sees the image, never sees patient identifiers, and
receives only structured JSON from an explicit allowlist.

Implementation: `backend/safety/privacy.py`.

- **Allowlist:** the outbound payload may contain only fields in
  `LLM_PAYLOAD_ALLOWLIST` (vision_scores, report_findings, fusion,
  uncertainty_signals, review_priority, review_priority_reasons,
  evidence_chunks, safety_flags, disclaimer). Any other field is
  silently dropped.
- **Denylist (hard failure):** the payload must not contain any field
  named in `PATIENT_IDENTIFIER_DENYLIST` (patient_name, mrn, dob,
  date_of_birth, ssn, accession, study_uid, patient_id, nhs_number,
  insurance_id). Presence of any of these causes `PrivacyViolation`
  and blocks the LLM call entirely.

The denylist is a hard failure, not a silent drop. A denylisted key
indicates a mapping bug, so the call is refused rather than sanitized.

## Human review triggers

Human review is required whenever any of the following is true:

- review priority is `URGENT_REVIEW` or `EMERGENCY_REVIEW`
- the fusion result contains any conflicts
- any pipeline step failed (`vision_failed`, `report_failed`,
  `retrieval_failed`)
- the safety gate sets `requires_human_review`
- the LLM output failed the post-generation check

`human_review.reasons` records every reason. Nothing is inferred; the
reasons list is populated from the pipeline state directly.

## Cross-modal conflicts are not medical errors

A conflict is a disagreement between two structured inputs. It is not
a claim that the model is right or that the report is wrong. RADIANT
never picks a winner. The UI says "image–report disagreement," never
"error."

## Review priorities are prototype routing suggestions

`ROUTINE`, `MONITOR`, `URGENT_REVIEW`, `EMERGENCY_REVIEW` are routing
suggestions within a research prototype. They are not validated
clinical triage categories. They were chosen so that the demo produces
sensible output on real data and synthetic conflict cases. They have
not been validated against any real triage standard.

## Prototype heuristics (all explicitly labeled)

Every threshold in the project is labeled as an
`UNVALIDATED PROTOTYPE HEURISTIC` in code and in the docs. None are
clinical thresholds. The complete list:

- `VISION_PRESENT_THRESHOLD = 0.5`
- `VISION_UNCERTAIN_LOW = 0.3`, `VISION_UNCERTAIN_HIGH = 0.5`
- `FUSION_HIGH_RATIO = 0.70`, `FUSION_PARTIAL_RATIO = 0.40`
- `FUSION_CRITICAL_LABELS = {Pneumothorax, Effusion, Consolidation, Edema}`
- `TRIAGE_LOW_CONFIDENCE_MAX_SCORE = 0.60`
- `TRIAGE_AMBIGUOUS_FRACTION = 0.30`
- `TRIAGE_CRITICAL_PRESENT_LABELS = {Pneumothorax, Effusion, Consolidation, Edema}`
- `RAG_MIN_SIMILARITY = 0.20`
- `RAG_CHUNK_TOKENS = 200`, `RAG_CHUNK_OVERLAP_TOKENS = 40`

Evaluation (`docs/evaluation.md`) reports the observed behavior of
these thresholds. Tuning them without data would be guessing.

## What the safety gates do NOT do

- They do not verify that a finding is clinically correct.
- They do not verify that evidence is authoritative.
- They do not detect adversarial input.
- They do not authenticate the caller.
- They do not validate that the report or image are well-formed
  beyond the checks in each module.
- They do not prevent a human from misreading the output.

The safety model is designed around the assumption that a human
reviewer reads the output and makes the final decision. That assumption
is load-bearing. It is stated in the README, in the UI, and in every
generated explanation.

## Test coverage for safety

`tests/unit/test_safety.py` covers 17 cases, including:

- all 8 pre-generation reason codes
- all 3 post-generation reason codes plus a valid case
- privacy allowlist stripping and denylist hard-fail
- disclaimer always present
- fail-closed behavior on model failure, retrieval failure

`backend/evaluation/safety_eval.py` runs the same categories on a
labeled matrix and records accuracy. Both are real and both pass.

## Required disclaimer

The following text is always present in the user-facing message
produced by either safety gate:

> This system is a research prototype for clinical decision support
> only. It is not a diagnostic tool. All outputs must be reviewed by
> a licensed physician.