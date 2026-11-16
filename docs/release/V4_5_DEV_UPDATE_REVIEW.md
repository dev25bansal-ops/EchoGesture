# V4.5 — review of Dev's work, 14–15 November

Reviewed range: `11bbddf` (last reviewed) → `73bff0d` (current `hackathon/dev-model`).

**Note on the brief's stated HEAD:** the task named `a30e60d` as Dev's new head.
That commit exists, but Dev's branch has since advanced **twelve commits** past
it. Everything below covers the actual current head, `73bff0d`, not `a30e60d`.

---

## 1. What Dev found

Two findings matter for the submission, and one matters for honesty.

### 1.1 The app runs at a geometry the models were never trained on

| landmark source / geometry | test top-1 |
| --- | --: |
| 256 squash — what the models were trained on | 82.47% |
| 512 letterbox | 85.67% |
| 640×480 letterboxed to a square | 86.31% |
| native 640×480, no resize — the app path | **85.51%** |

Paired McNemar on the same 1,249 clips, squash256 vs native: 52 clips only
squash correct, 90 only native, z=3.10, p=0.0019.

The Android app does no resize — `CameraScreen.kt` requests 640×480 and
`HandTracker.kt` builds the MediaPipe image from the bitmap with no crop or
scale.

**The Python evaluator does resize**, to 256×256, which is a *squash* that
distorts a 4:3 frame to a square.

### 1.2 The landmark cache is degraded, and 77.02% rests on it

| landmark source | test top-1 |
| --- | --: |
| `.scratch/landmarks/` cache (built 8 Nov) | **76.94%** |
| fresh extraction, same geometry | **82.47%** |

Presence patterns differ on **364 of 1,255 test clips (29.0%)**, concentrated in
test: train sampled 138 clips with **zero** differences, val 35 clips with 1.
Cause **not established** — Dev says so plainly.

`models/landmark_model_meta.json` records `ensemble_test_top1: 0.7702` from this
cache, so **77.02% understates the model** by roughly five points.

**This does not invalidate our submission ZIP**, which extracts landmarks from
video on every call and never reads the cache. Verified: no cache path exists in
`src/main.py` or `src/landmark_frontend.py`.

### 1.3 The robustness harness used the wrong aggregator

`robustness.py` averaged per-member **softmax probabilities**; the shipped code
averages **raw logits** then applies one softmax. Corrected, landmark robustness
goes from **4.40 PASS to 5.36 FAIL**. See `V4_5_ROBUSTNESS_RECONCILIATION.md`.

Dev also retracted a 15.7 ms/frame claim that had no supporting artifact.

---

## 2. Classification of every changed file

| File | Class | Decision |
| --- | --- | --- |
| `src/landmark_frontend.py` | **A** — adds `size` param, defaults unchanged | import |
| `src/main.py` | **A** — warm-up, `size` param, latency excludes cold start | see §3 |
| `scripts/ml/geometry_pair.py` | **B** — evaluation correctness | import |
| `scripts/ml/latency_per_frame.py` | **B** — replaces a retracted claim | import |
| `scripts/ml/robustness.py` | **B** — logit aggregator fix | import |
| `scripts/ml/robustness_pixel_parallel.py` | **B** — full-split pixel harness | import |
| `scripts/ml/eval_tflite_cache.py` | **B** | import |
| `scripts/ml/confidence_calibration.py` | **B** | import |
| `scripts/ml/extract_parallel.py` | **C** — research | import |
| `scripts/ml/train.py` | **C** — research; its retrain scored 76.26%, **worse** | import, reject the model |
| `docs/hackathon/*.md` (8 files) | **D** — evidence | import |
| `results/*.json` (per-clip, large) | **D/F** — evidence; may contain clip IDs | inspect before import |
| `models/*.tflite` | **F** — restricted | **never import** |
| Any retrained checkpoint | **E** — worse (76.26% vs 82.47%) | **reject** |

**Rejected outright:** the retrained model. Dev measured it at 76.26% test, below
the existing nine-member ensemble at any geometry. Using it would be a
regression dressed as progress.

---

## 3. Production `src/main.py` — surgical, not replaced

Dev's reference file is short. Ours carries hash pinning, instrumentation,
parallel tracking and the full pin table. **Not overwritten.**

Three specific changes were considered:

| Change | Verdict |
| --- | --- |
| MediaPipe warm-up inside `_init_model()` | **Adopt.** Costs ~135 s once, but it moves a one-time graph build out of the first timed `predict()`. |
| `size` parameter, default 256, `"native"` sentinel | **Adopt** — it is what makes the geometry comparison possible through production code. |
| `latency_ms` excludes cold start | **Adopt.** Ours sets `start_time` before `_init_model()`, so our published 556 ms first-call figure includes model init. That is a reporting defect. |

None of these changes model weights, the feature contract, the mask, the
aggregator, or `predict()`'s signature and output shape.

---

## 4. What I verified myself

| Claim | Verified? |
| --- | --- |
| Dev's branch advanced past the stated HEAD | **Yes** — 12 commits |
| Our evaluator never reads the landmark cache | **Yes** — decode path only |
| Source clips are 640×480 (4:3) | **Yes** — 30/30 val_30 and 40/40 sampled val_346 |
| 256-squash distorts aspect by 1.33× | **Yes**, by construction |
| Native beats squash on our own frozen evaluator | **Yes** — 86.67% vs 76.67% on val_30 |
| Native landmark robustness FAILs at 5.76 | **Yes** — from `robustness_native.json` |
| Native pixel robustness passes | **Not established for the app path.** Dev's 4.48 PASS is at 512-letterbox, a different geometry. |

---

## 5. The 15.7 ms retraction reaches us

`docs/release/FINAL_EVIDENCE.md` on the frozen branch still carries:

> Per-frame p95 (streaming) | <50 ms | … MediaPipe 15.7 ms/frame p50 + 0.55 ms
> classifier ≈ 32.80 ms p95 … TEAM-REPORTED

Dev has since measured per-frame median at **~142–151 ms** and retracted the
figure. That row must be corrected to FAIL; it currently reads as a pass on a
criterion worth 20% of the rubric.

The shipped `submission/results.csv` is **clean** — it has no per-frame row. The
defect is documentation only.

---

## 6. Summary

Dev's geometry finding is real, independently reproduced in direction and
magnitude on our own evaluator, and worth acting on. His cache finding means the
headline accuracy we have been quoting is too low, not too high.

His robustness corrections make one previously-passing criterion **fail**, and
one previously-failing pixel criterion pass at a geometry the app does not use.

The retrained model is worse and is rejected.