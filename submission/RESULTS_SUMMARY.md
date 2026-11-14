# Results summary

One page. Every number with its population and its source. Nothing here is
merged across experiments.

---

## Headline

| Metric | Value | Population | Source |
| --- | --: | --- | --- |
| Top-1 | **77.02%** | ASL Citizen test, 1,249 clips, 11 unseen signers, 100 classes | Dev, `results/results.csv` |
| Top-5 | **92.79%** | same | same |
| Top-1 (validation) | 87.25% | 345 clips, 5 unseen signers | same |

**These are corpus figures measured on a laptop.** There is no phone accuracy
measurement in this project. Chance is 1.00%.

### Independent check from the submission ZIP

The packaged evaluator was extracted to an empty directory and run there:

| | |
| --- | --: |
| Clips | 30 (`val_30`) |
| Correct | **23** |
| Top-1 | **76.67%** |
| Errors | 0 |

23/30 from a clean extraction is consistent with 77.02% on the full split and
confirms the ZIP contains a working model, not a stale one.

---

## Model

| | |
| --- | --- |
| Members | 9 (4 × 255-wide x,y; 5 × 381-wide x,y,z) |
| Precision | **float16** — not INT8 |
| Ensemble bytes | 8,508,832 (8.51 MB) |
| Detector bytes | 7,819,105 |
| Shipped total | 16,327,937 (16.33 MB) — under the 20 MB budget either way |
| Ordered ensemble SHA-256 | `995cc8d0b6f6ac25abb9f7ddca01c3ce71edf8bc2d34e34f4813a2b99dd3b4bb` |
| Vocabulary | 100 fixed classes |

The attention mask is applied: `present.max(axis=1) > 0.5` over resampled
presence, identical for every member. Supplying it was worth +4.24 points —
every earlier artifact had omitted a mask the model was trained with.

---

## Latency

| Measurement | Value | What it is |
| --- | --: | --- |
| Warm p50 | 236.4 ms | full clip-to-JSON, laptop CPU, includes MediaPipe |
| Warm p95 | 314.9 ms | same, n=29 |
| Cold first call | 556.2 ms | model init |
| Ensemble inference only | ~6.56 ms | nine members, laptop |

**The ≤50 ms criterion is FAIL**, and it fails on tracking, not classification.
MediaPipe accounts for roughly 90% of end-to-end time. No phone measurement
exists.

---

## Robustness — two harnesses, two answers

| Level | Worst drop | Verdict | What it corrupts |
| --- | --: | --- | --- |
| Landmark | +4.40 pp | **PASS** (≤5) | the 21 points, after detection |
| Pixel | −10.00 pp | **FAIL** (≤5) | the video frames, before detection |

Both are real. They are not the same experiment and must never be averaged or
quoted interchangeably. The deployed app sees pixels, so the pixel number is the
deployment-relevant one. Worst pixel conditions: crop −10.00, occlusion −6.67.

---

## Robustness and accuracy claims that must never be made

| Never say | Actually |
| --- | --- |
| "77% accurate on a phone" | Corpus/laptop only |
| "184 independent signs tested" | 184 evaluations over **72 unique clips** |
| "INT8 quantised" | FP16 ships |
| "Meets the 50 ms target" | 314.9 ms p95 warm |
| "Robust to occlusion" | −6.67 pp on occlusion |
| "Continuous translation" | 100 isolated signs |

---

## Test status

| Suite | Result |
| --- | --- |
| Android JVM | **110 / 110 pass**, 0 failed, 0 skipped |
| Python `evaluation/device` | 424 / 493 pass; 3 failed, 23 errors, 43 skipped |
| Official bundle suite | 15 / 15 pass |
| Clean-extraction `predict()` | 30 / 30 clips, 0 errors |
| Android instrumentation | 11 / 11 at V4.4; not re-run this cycle |
| Physical phone | **NOT TESTED** |

The 26 Python failures are identical on the untouched V4.4 tip and all trace to
three gitignored artifacts (a `summary.json`, `landmark_classifier_fp16.tflite`,
a `.npy` fixture). They are blocked on assets, not code.