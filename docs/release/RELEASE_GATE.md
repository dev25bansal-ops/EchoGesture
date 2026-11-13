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
| Candidate top-1 | 88.89% | 18 in-vocab validation clips |
| Candidate top-5 | 94.44% | same |
| Frozen-file top-1 | 50.00% | same 18 |
| Accuracy at accepted coverage (score ≥ 0.35) | 100% (8/8), coverage 44.4% | same |
| Clip-to-JSON latency | p50 225 ms / p95 513 ms | macOS arm64, **not** target-device |

These are interim and on a small, non-random population. They are not a
generalisation claim.

## Gate table

| Requirement | Clip footage | Real phone |
| --- | --- | --- |
| Top-1 accuracy | interim 88.89% (n=18) | **AWAITING PHONE DATA** |
| Top-5 accuracy | interim 94.44% (n=18) | **AWAITING PHONE DATA** |
| Per-class recall | partial (small n) | **AWAITING PHONE DATA** |
| False-positive rate | **AWAITING PHONE DATA** — needs intended-vs-predicted on unwritten attempts | **AWAITING PHONE DATA** |
| Rejection rate | 55.6% at the 0.35 cut (n=18) | **AWAITING PHONE DATA** |
| Accuracy at accepted coverage | 100% (8/8) | **AWAITING PHONE DATA** |
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

## Decision

**NEEDS PHONE DATA** before any of the three verdicts can be given. The clip
evidence supports neither READY FOR BROADER TESTING (the phone failures are
unexplained by anything measurable here) nor NEEDS DEV RETRAINING (the model
classifies in-vocabulary footage correctly, so retraining is not yet
justified).