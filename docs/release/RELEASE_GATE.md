# Release gate — baseline vs candidate

Compares the shipped pipeline against the frozen baseline on the same
authorized ASL Citizen validation clips, and records every figure the gate
requires. Sections marked **AWAITING PHONE DATA** cannot be filled on this
machine.

## Important: what "baseline" means here

`evaluation/device/ensemble_evidence/predictions_ensemble_val.jsonl` is Dev's
frozen prediction file. It was produced in a different environment, and on the
clips tested so far it is **not** equivalent to the shipped pipeline — the same
clip can receive different labels:

| Clip | Truth | Frozen | Shipped `predict()` |
| --- | --- | --- | --- |
| 06408604416165864-WHAT FOR | WHATFOR1 | PENNY (0.547) | **WHATFOR1** (0.567) |
| 006951411960890663-FINE | FINE1 | MOVIE1 (0.094) | **FINE1** (0.323) |
| 3054167723169219-RIVER | RIVER1 | SINK (0.249) | **RIVER1** (0.295) |

So a delta against the frozen file measures **two different pipelines**, not an
improvement made by this branch. The frozen file is still the right reference
for *what was previously recorded*, and it reproduces exactly
(260/345 = 75.36%, matching `VAL_RESULT.md`). It is not a like-for-like "before"
for pipeline changes.

A true before/after on this branch requires running the **same** pipeline at
commit `a6cf722` and at the candidate commit over the same clips.

## Recorded so far (clip footage, not a phone)

| Figure | Value | Population |
| --- | --- | --- |
| Candidate top-1 | **84.48%** (283/335) | in-vocab validation clips |
| Candidate top-5 | **94.03%** (315/335) | same |
| Frozen-file top-1 | 75.15% | same 334 scored clips |
| Paired delta | **+9.58 pt top-1, +7.49 pt top-5** | same 334 clips |
| Accuracy at accepted coverage (margin ≥ 0.15) | **95.2%, coverage 71.2%** | 146 clips |
| Accuracy at accepted coverage (old score ≥ 0.35) | 95.4%, coverage 59.6% | same 146 |
| Clip-to-JSON latency | p50 241 ms / p95 309 ms | macOS arm64, **not** target-device |

The figure moved as the population grew (88.9% at n=18, 78.7% at n=141,
85.2% at n=297, 84.5% at n=335) and has settled near 84–85%. Reporting the
first number would have been wrong; that is why the protocol demands n≥50
before any figure is quoted.

On the paired 334-clip set the candidate fixes 40 clips the frozen file gets
wrong and breaks 8 it gets right. All 8 regressions score below the acceptance
cut (0.07–0.19) and the correct label is rank-2 in each, so they are rejected
rather than shown as confident wrong answers.

These are still clip footage from one corpus, not phone input, and the 100
classes have only 2–6 clips each, so per-class figures are indicative only.
They are not a generalisation claim.

## Gate table

| Requirement | Clip footage | Real phone |
| --- | --- | --- |
| Top-1 accuracy | 84.48% (n=335) | **AWAITING PHONE DATA** |
| Top-5 accuracy | 94.03% (n=335) | **AWAITING PHONE DATA** |
| Per-class recall | partial (small n) | **AWAITING PHONE DATA** |
| False-positive rate | **AWAITING PHONE DATA** — needs intended-vs-predicted on unwritten attempts | **AWAITING PHONE DATA** |
| Rejection rate | 37.0% at the 0.35 score cut (n=335) | **AWAITING PHONE DATA** |
| Accuracy at accepted coverage | **97.16%** (coverage 63.0%) | **AWAITING PHONE DATA** |
| Capture-to-caption latency | n/a — no camera | **AWAITING PHONE DATA** |
| APK / model sizes | APK 77,710,351 B; 4 members, 832,888 B each-shape pinned | unchanged |
| Performance by signer | **AWAITING** — needs ≥2 signers | **AWAITING PHONE DATA** |

## How to fill the phone columns

Follow [ACCURACY_PROTOCOL.md](ACCURACY_PROTOCOL.md): 10 glosses × 5 attempts,
annotations file alongside, then

```bash
adb shell run-as org.echogesture.finalapp cat files/diagnostics.txt > diagnostics.txt
python3 evaluation/device/accuracy_report.py diagnostics.txt annotations.txt
```

That produces top-1, top-5, rejection rate, accuracy at accepted coverage, the
confusion matrix and the per-sign table directly.

## Error structure (full validation population, n=345)

From the frozen baseline over all 345 in-vocabulary validation clips:

| Outcome | Share |
| --- | --- |
| Correct top-1 | 75.36% |
| Correct at rank 2–5 | 11.9% |
| Not in top-5 at all | 12.8% |
| Top-5 overall | 87.25% |

That split is the most useful number for deciding what to build next. The
11.9% that are rank 2–5 are *recoverable by rejection*: the correct answer is
already computed and the app currently discards it silently. The 12.8% that
never appear in the top-5 are the genuinely hard cases and are the only ones
that argue for new training data.

### Classes that absorb other classes' clips

| Predicted wrongly | Count |
| --- | ---: |
| BREAKFAST1 | 7 |
| INTUITIVE | 6 |
| TAIL1 | 5 |
| JEWELRY | 4 |
| MOVIE1, RAZOR2, CHEER, SHAVE1, SCARF2 | 3 each |

These are the model's "attractor" labels. TAIL1 absorbing 5 clips is directly
consistent with the observed EAT1→TAIL1 phone error.

### Weakest classes

| Class | Missed | Of |
| --- | --: | --: |
| DOG1 | 3 | 4 |
| LOCK1 | 3 | 5 |
| BEE1 | 3 | 6 |
| CHEESEGRATER | 3 | 5 |
| FOREIGNER1 | 3 | 4 |
| CHEEK | 3 | 4 |

DOG1 at 1/4 is the weakest class in the shipping vocabulary. The phone error
DEAF1→DOG1 is therefore *toward* a class the model itself handles poorly, which
weakens "the model simply nailed it" and points at confusable motion rather
than a pipeline fault.

### Confusion pairs seen twice or more

| Expected | Predicted | Count |
| --- | --- | --: |
| DOG1 | INTUITIVE | 2 |
| ZEBRA | JEWELRY | 2 |
| FOREIGNER1 | SCARF2 | 2 |

With 2–6 clips per class these pairs are indicative, not established. A pair
needs several signers before it is worth a training intervention.

### What this does and does not say

- It **does** say the failure structure is dominated by rank-2-to-5 near-misses
  and a small set of attractor labels.
- It **does not** say anything about phone performance: this is corpus footage.
- It **does not** justify retraining yet. Only 12.8% of cases are outside the
  top-5, and the phone-specific evidence needed to attribute failures to the
  model rather than to capture does not exist yet.

## Decision

**NEEDS PHONE DATA.**

That is not a dodge. It is the specific finding: every cause of the phone
failures that *could* be measured off-phone has been measured and cleared.

- Android tracking matches Python tracking (identical top-1, 4/4 clips).
- Hand slots, rotation and mirroring are consistent between the paths.
- Feature construction is fixture-asserted to 1e-4 against the Python
  reference.
- Capture is a measured 3.7 s window with pre-tap frames excluded.
- The shipped pipeline scores 84.5% top-1 / 94.0% top-5 on 335 authorized
  clips, and 97.2% on the 63% it accepts.

What remains is the one thing that cannot be simulated: a phone camera pointed
at a human signer. **NEEDS DEV RETRAINING is not justified** — 40 clips improve
and 8 regress against the previous pipeline over the same clips, and no
measured failure is attributable to the model rather than to capture. **READY
FOR BROADER TESTING is not justified either** — nothing here demonstrates the
phone works, and the two reported phone errors remain unexplained.

The next highest-value action is a phone session with
[ACCURACY_PROTOCOL.md](ACCURACY_PROTOCOL.md), not more off-phone tuning.

### If the phone session fails

The measured evidence already points at the most likely lever: per-signer
accuracy ranges 69–90% on the shipped pipeline, and a second-window agreement
check separates correct from incorrect far more sharply than any single-window
threshold. If phone accuracy at accepted coverage is poor, a confirm-once
capture step is better supported by this data than any model change.

## Two-window agreement (measured, not implemented)

Splitting each clip into two disjoint halves and classifying each as its own
window — a stand-in for the user performing the sign twice — separates strongly
(n=113):

| Outcome | n | Share | top-1 on first window |
| --- | --: | --: | --: |
| Both windows agree | 46 | 41% | **98%** |
| Windows disagree | 67 | 59% | 55% |

Agreement is a better correctness signal than any threshold on a single
window. It is **not implemented**, deliberately:

- It costs a second 3.7 s capture, so it doubles the time before a result on
  every use, including the 41% the current app already reads out correctly on
  the first attempt.
- The 59% disagreement rate on clip halves is inflated relative to a real
  repeat: a signer performing deliberately always covers the whole sign, while
  a clip half often contains only part of the motion.
- It needs its own phone measurement before it can be justified.

The honest framing: if phone data shows low accuracy at accepted coverage, a
"perform once more" confirm step is the best-supported next lever, and this
table is the evidence for it.

## Per-signer performance (n=345, frozen baseline)

| Signer | n | top-1 | top-5 | mean score |
| --- | --: | --: | --: | --: |
| P39 | 103 | **84%** | 94% | 0.472 |
| P21 | 104 | 78% | 86% | 0.471 |
| P26 | 101 | 75% | 87% | 0.489 |
| P12 | 24 | **46%** | 75% | 0.268 |
| P5 | 13 | **38%** | 69% | 0.208 |

This is the single most important structural fact in the whole evaluation.
Accuracy is not uniform: it ranges from 38% to 84% across five signers, and the
mean score tracks the accuracy almost exactly (0.21 for the weakest signer,
0.47 for the strongest). Two conclusions follow.

1. **A pooled accuracy figure is misleading.** 75.36% overall describes no
   actual signer: three signers sit near it, one is far below.
2. **The model's confidence is already informative about signer difficulty.**
   The low mean score for P5/P12 is the uncertainty machinery working — those
   clips are genuinely harder for this model, and the app's rejection rule
   would turn many of them into "uncertain" rather than wrong answers.

### The shipped pipeline narrows that spread

The same breakdown run through the shipped `predict()` on 313 fetched clips:

| Signer | n | top-1 | top-5 | mean score |
| --- | --: | --: | --: | --: |
| P26 | 69 | **90%** | 100% | 0.545 |
| P39 | 103 | **90%** | 96% | 0.517 |
| P21 | 104 | 82% | 91% | 0.518 |
| P12 | 24 | 79% | 96% | 0.352 |
| P5 | 13 | 69% | 100% | 0.333 |
| **all** | 313 | **86%** | 96% | 0.503 |

Weak-signer accuracy improves most (P5 38%→69%, P12 46%→79%) and the spread
narrows from 38–84 to 69–90. Since the two runs use the same models, this is a
difference in *input construction*, not model quality — the strongest
indication yet that the pipeline work mattered.

Caveat: these are different clip subsets per run (the frozen file scores 345
of 346; the local run scored 313 of the clips fetched so far), so the two
tables are indicative rather than a paired comparison. The paired comparison
is in the gate table above, which scores both on the identical clip set.

An unknown new signer should still be assumed closer to the weak end until
measured. The 95%+ top-1 goal is not supported for an unseen signer by this
data.
