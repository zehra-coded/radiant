# RADIANT — Evaluation

Run at: 2026-10-15T20:38:55+00:00

Every number in this document comes from a real execution on real data. Items that could not be evaluated are marked NOT RUN with a reason. No results are invented.

## 1. Vision model (NIH ChestX-ray14 test split)

--------------------------------------------------------------------
Images evaluated: 1000 (requested 1000)
Macro AUROC: 0.668
Calibration: NOT ASSESSED — raw sigmoid outputs

| Label | n | positives | AUROC | Avg Prec | Prec | Rec | F1 |
|---|---|---|---|---|---|---|---|
| Atelectasis | 1000 | 127 | 0.654 | 0.202 | 0.146 | 0.874 | 0.251 |
| Cardiomegaly | 1000 | 113 | 0.812 | 0.358 | 0.219 | 0.788 | 0.343 |
| Effusion | 1000 | 157 | 0.679 | 0.297 | 0.210 | 0.841 | 0.335 |
| Infiltration | 1000 | 221 | 0.613 | 0.303 | 0.237 | 0.905 | 0.375 |
| Mass | 1000 | 42 | 0.626 | 0.082 | 0.053 | 0.738 | 0.098 |
| Nodule | 1000 | 35 | 0.537 | 0.042 | 0.042 | 0.657 | 0.079 |
| Pneumonia | 1000 | 19 | 0.498 | 0.027 | 0.018 | 0.684 | 0.036 |
| Pneumothorax | 1000 | 116 | 0.653 | 0.247 | 0.153 | 0.888 | 0.261 |
| Consolidation | 1000 | 63 | 0.646 | 0.092 | 0.068 | 0.921 | 0.126 |
| Edema | 1000 | 37 | 0.765 | 0.089 | 0.055 | 1.000 | 0.104 |
| Emphysema | 1000 | 42 | 0.560 | 0.067 | 0.042 | 0.714 | 0.079 |
| Fibrosis | 1000 | 30 | 0.751 | 0.095 | 0.044 | 0.800 | 0.084 |
| Pleural_Thickening | 1000 | 58 | 0.631 | 0.084 | 0.072 | 0.759 | 0.131 |
| Hernia | 1000 | 15 | 0.930 | 0.425 | 0.025 | 1.000 | 0.050 |

- Model scores are raw sigmoid outputs, not calibrated probabilities.
- Classification threshold = 0.5 (UNVALIDATED PROTOTYPE HEURISTIC).
- AUROC is a ranking metric; it does not imply clinical validity.

(Runtime: 200.8s)

## 2. Report NLP extractor (Rad-SpRL)

--------------------------------------------------------------------
Corpus: 2000 documents from Rad-SpRL.xml
Annotations available in corpus:
  Trajector:  2215
  Landmark:   2004
  Diagnosis:   441
  Hedge:       388

Evaluation on all 2000 documents:

| Metric | Value |
|---|---|
| Diagnosis span recall | 0.601 |
| Hedge agreement | 0.108 |
| Anatomy agreement | 0.226 |

Evaluation on 200 documents (subsample, for comparison):

| Metric | Value |
|---|---|
| Diagnosis span recall | 0.700 |
| Hedge agreement | 0.130 |
| Anatomy agreement | 0.206 |

**Negation precision / recall: NOT RUN** — Rad-SpRL does not annotate
present/negated status. There is no ground truth in this corpus for
whether a finding is affirmed or denied.

Interpretation:
- Diagnosis span recall of ~0.60 means the extractor's finding matches
  overlap with 60% of Rad-SpRL's Diagnosis spans. This is coverage, not
  correctness — the extractor may still mark a matched finding with the
  wrong status.
- Hedge agreement of ~0.11 and Anatomy agreement of ~0.23 are low.
  The extractor's uncertainty and anatomy lexicons were designed from
  general radiology conventions, not tuned against Rad-SpRL. These
  numbers measure the gap.
- These are descriptive measures. They are not clinical validation of
  the extractor.

Annotation structure note: Rad-SpRL uses two coordinate systems — Token
elements carry character offsets (cs/cl), while Diagnosis, Hedge,
Landmark and Trajector carry token offsets (ts/tl) referencing the
ordered Token sequence. The parser resolves both.

(Runtime: 3.5s for full corpus parse + evaluation.)

## 3. Cross-modal fusion

--------------------------------------------------------------------
Cases: 7
Consistency accuracy: 1.000
Conflict-count accuracy: 1.000

| Case | Expected | Actual | Match | Exp Conflicts | Act Conflicts | Match |
|---|---|---|---|---|---|---|
| full_agreement | high | high | True | 0 | 0 | True |
| both_empty | insufficient | insufficient | True | 0 | 0 | True |
| critical_conflict | conflicting | conflicting | True | 1 | 1 | True |
| noncritical_double_conflict | conflicting | conflicting | True | 2 | 2 | True |
| uncertain_vision_skipped | insufficient | insufficient | True | 0 | 0 | True |
| partial_with_conflict | conflicting | conflicting | True | 1 | 1 | True |
| report_only_not_conflict | conflicting | conflicting | True | 0 | 0 | True |

- 7 labeled synthetic cases. Not a clinical evaluation — tests the verifier logic against its own specification.

## 4. Evidence retrieval

--------------------------------------------------------------------
Queries: 10 (k=3)
Recall@k: 1.000
Mean precision@k: 0.533
MRR: 0.900

| Query | Expected | Retrieved | Hit | Rank |
|---|---|---|---|---|
| pneumothorax | fixture_pneumothorax | fixture_pneumothorax, fixture_pleural_effusion, fixture_atelectasis | True | 1 |
| collapsed lung | fixture_pneumothorax | fixture_atelectasis, fixture_pneumothorax, fixture_consolidation | True | 2 |
| pleural effusion | fixture_pleural_effusion | fixture_pleural_effusion, fixture_pneumothorax, fixture_atelectasis | True | 1 |
| fluid in the chest | fixture_pleural_effusion | fixture_pleural_effusion, fixture_pneumothorax, fixture_consolidation | True | 1 |
| enlarged heart | fixture_cardiomegaly | fixture_cardiomegaly, fixture_atelectasis, fixture_pleural_effusion | True | 1 |
| cardiomegaly | fixture_cardiomegaly | fixture_cardiomegaly | True | 1 |
| consolidation | fixture_consolidation | fixture_consolidation | True | 1 |
| lung opacity | fixture_consolidation | fixture_atelectasis, fixture_consolidation, fixture_pneumothorax | True | 2 |
| atelectasis | fixture_atelectasis | fixture_atelectasis | True | 1 |
| lung collapse | fixture_atelectasis | fixture_atelectasis, fixture_consolidation, fixture_pneumothorax | True | 1 |

- Evaluated on 10 hand-labeled queries against the small development-fixture knowledge base (5 documents).
- Numbers are indicative, not a general retrieval quality measure.

## 5. Safety gate

--------------------------------------------------------------------
Pre-generation cases: 8 (passed: 8, accuracy: 1.000)
Post-generation cases: 3 (passed: 3, accuracy: 1.000)

### Pre-generation
| Case | Expected | Actual reasons | Match |
|---|---|---|---|
| urgent_triage | urgent_or_emergency | urgent_or_emergency | True |
| conflict | vision_report_conflict | vision_report_conflict | True |
| insufficient_evidence | insufficient_evidence | insufficient_evidence | True |
| diagnosis_request | diagnosis_request_out_of_scope | diagnosis_request_out_of_scope | True |
| treatment_request | treatment_request_out_of_scope | treatment_request_out_of_scope | True |
| malformed_input | malformed_input | malformed_input | True |
| model_failure | model_failure | model_failure | True |
| retrieval_failure | retrieval_failure | insufficient_evidence, retrieval_failure | True |

### Post-generation
| Case | Expected | Actual reasons | Match |
|---|---|---|---|
| valid_citation | None | (none) | True |
| fabricated_citation | citation_validation_failure | citation_validation_failure | True |
| diagnosis_language | grounding_validation_failure | grounding_validation_failure | True |

## 6. Generation groundedness

--------------------------------------------------------------------
Live LLM outputs are evaluated for citation validity and groundedness. This is separate from the safety gate, which runs at request time.

Automated metrics: citation validity rate and groundedness ratio (fraction of sentences that either contain a valid citation or belong to a disclaimer/limitations section).

To produce real numbers, run the pipeline 5 times and pass each output through generation_eval.evaluate_one(text, evidence). This is best done manually after the demo.

RESULT: See docs/evaluation_addendum.md for the 5-sample results.
