# RADIANT — Generation groundedness addendum

Five pipeline runs on the same NIH image with different reports. Citation validity = fraction of `[source:doc:chunk]` citations that resolve to real retrieved chunks. Groundedness ratio = fraction of sentences that either contain a valid citation or belong to a disclaimer/limitations section.

| Case | Generator | Citations total | Valid | Validity | Sentences | Grounded | Ratio |
|---|---|---|---|---|---|---|---|
| conflict | amd_token_factory | 5 | 5 | 1.0 | 6 | 3 | 0.5 |
| agreement | template | 5 | 5 | 1.0 | 8 | 4 | 0.5 |
| negation | amd_token_factory | 5 | 5 | 1.0 | 10 | 4 | 0.4 |
| empty | template | 5 | 5 | 1.0 | 8 | 4 | 0.5 |
| brief | template | 5 | 5 | 1.0 | 8 | 4 | 0.5 |

**Limitations.** Groundedness is measured by rule: a sentence counts as grounded if it contains a citation or is in the limitations section. The rule does not verify semantic support — a sentence could cite a chunk that does not actually support it. This is documented as a limitation, not hidden.
