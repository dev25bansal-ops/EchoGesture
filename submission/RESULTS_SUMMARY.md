# Results summary

One page. Every number with its population and its source. Nothing here is
merged across experiments.

---

## Headline

| Metric | Value | Population | Source |
| --- | --: | --- | --- |
| Top-1 | **85.45%** | ASL Citizen test, **1,251** clips, 11 unseen signers, 100 classes, **native 640×480 cache** | Dev, `results/results.csv` |
| Top-5 | **96.64%** | same | same |
| Top-1 (validation) | **89.31%** | 346 clips, 5 unseen signers, native cache | same |
| Top-1, **256 squash** geometry | **82.47%** | same test split | Dev, `GEOMETRY_FINDING.md` |

**These are corpus figures measured on a laptop.** There is no phone accuracy
measurement in this project. Chance is 1.00%.

### Which number belongs to which artifact — read this before quoting one

> **CORRECTED 2026-11-22.** The table below used to say the Python entry point
> squashes to 256 and therefore scores 82.47%. That stopped being true at
> `8740f27`, which removed both the squash default and the `ECHOGESTURE_FRAME_SIZE`
> override. `predict()` now runs **native 640×480** unconditionally, so the
> Python and Android paths share one geometry and one accuracy figure.

| Artifact | Geometry it actually uses | Its accuracy |
| --- | --- | ---: |
| **Android app** (CameraX native frames → MediaPipe) | native 640×480 | **85.45%** corpus figure (1,251 clips) |
| **`src/main.py` `predict()`** | native 640×480 — `clip_to_landmarks(..., size=None)` | **85.45%** — the same path, same geometry |
| Physical phone | — | **no accuracy measured**; one device run proves the pipeline only |

**85.45% is now the right number for both.** The 82.47% squash figure is
historical (RESULTS.md row 3) and describes code that no longer exists. Any
ZIP built before `8740f27` still runs the squash path and would measure ~82.5%;
no such ZIP is in this tree.

The 256-squash and native geometries differ by about **+3.0 pp** on the same
weights. The geometry, not the model, carries that difference.

### The 77.02% → 85.45% change was a measurement correction, not a model change

The nine weights are identical. 77.02% came from an 8-Nov cache whose presence
mask is wrong on 364 of 1,255 test clips (29%). Re-extracting at native geometry
gives 85.45% on the same models. Say "the evaluation was wrong", not "the model
improved".

**The test split has been inspected during development.** 85.45% is genuinely
held-out from training, but it is **not blind** — geometry, robustness and
calibration work all repeatedly scored it.

### Independent check from the submission ZIP

The packaged evaluator was extracted to an empty directory and run there:

| | |
| --- | --: |
| Clips | 30 (`val_30`) |
| Correct | **23** |
| Top-1 | **76.67%** |
| Errors | 0 |

23/30 from a clean extraction sits between the 82.47% (256 squash) and 85.45%
(native) full-split figures, which is what a 30-clip sample of that population
should do. It confirms the ZIP contains a working model, not a stale one.

**The current ZIP was rebuilt on 2026-11-17** and re-verified the same way: a
clean extraction ran `predict()` on an authorised clip and returned
`WHATFOR1` at 0.9749 confidence with top-5 `[WHATFOR1, OFFEND, DEAF1, KNIGHT1,
PENNY]` — the same answer the Android result card shows for that clip.

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

**There is no single latency number. There are five, and they are not
interchangeable.**

| Measurement | Value | What it is |
| --- | --: | --- |
| Nine-member classifier only | **~4.2 ms p95** (3.1 ms p50) | desktop x86, warm process, all nine `.tflite` interpreters |
| MediaPipe detector, per frame | **contested: 15.34 ms vs ~135 ms** | see `docs/release/LATENCY_RECONCILIATION.md` |
| Full `predict()` clip→JSON | **4,650 ms p50** | desktop x86, warm, 32 frames — Dev's corrected figure |
| Cold start | **159,000 ms** | one-time per process; 134.8 s is MediaPipe graph build |
| **On-device / on-phone** | **NOT MEASURED** | — |

**The ≤50 ms criterion is NOT MET on desktop and is UNMEASURED on device.** The
classifier is 0.15% of end-to-end time; the detector dominates.

**Do not quote a per-frame PASS from the laptop figures.** Dev's branch contains
two mutually incompatible per-frame figures in the same commit (15.34 ms PASS and
135 ms FAIL). For a 32-frame clip at 15.34 ms, detection would be ~0.5 s — but the
measured end-to-end is 4.65 s, which only ~135 ms/frame explains. Those laptop
numbers were never reconciled against each other.

**They were superseded on 2026-11-21 by a direct on-device measurement**, which
does not need reconciling because it comes from the app's own profiler:

| Path, Vivo V2404, 640×480 VIDEO | p50 | p95 |
| --- | --: | --: |
| CPU delegate | 95 ms | 166 ms |
| **GPU delegate, no hand in frame** | — | **46–54 ms** |
| **GPU delegate, hand in frame** | 50–59 ms | **68–70 ms** |

The app had been building its detector with no delegate at all, so it ran on the
CPU path. With `Delegate.GPU` requested and CPU as fallback, the detector is
2.3× faster and moves from "clearly over budget" to "at or inside it" in the
empty-frame regime — and 18–20 ms over once a hand is present.

All of these are DEBUGGABLE builds, so they are upper bounds. The classifier adds
~40 ms warm on top; it is no longer 0.15% of end-to-end, because the detector
stopped dominating.

> "About 40 ms for the classifier; the detector takes 46–70 ms p95 on the GPU
> delegate depending on whether a hand is in frame. Measured on one physical
> phone, debug build. Desktop-CPU figures were 142–151 ms and are historical."

---

## Robustness — two harnesses, two answers

| Level | Worst drop | Verdict | What it corrupts |
| --- | --: | --- | --- |
| Landmark, unconditional (harness: presence **and** coords zeroed) | **+5.76 pp** | **FAIL** (≤5) | the 21 points, after detection |
| Landmark, at app gate (margin ≥ 0.15, **86% clean / 80% corrupted coverage**) | +1.24 pp | PASS *on accepted clips only* | same corruption, fewer clips scored |
| Pixel (512 letterbox) | +4.48 pp | PASS (≤5) | the video frames, before detection. **Not the app's geometry.** |
| Pixel (256 squash, 4-model, n=30) | −10.00 pp | **FAIL** — older model generation | the video frames, before detection |

**The honest unconditional landmark number is +5.76 pp, and it FAILS.**

There are two other figures circulating and both are wrong to quote as-is:

- **+5.36** was an earlier measurement on the old 256-squash cache. Superseded.
- **+5.52** appears in `results/robustness_gated.json` labelled "raw (every
  clip)", but it is computed from the **coords-kept diagnostic** (79.94%, delta
  actually 5.51), not from the harness corruption (79.70%, delta 5.76). Dev's own
  `DROP_HAND_ANALYSIS.md` states both variants explicitly.

**Do not quote the gated +1.24 without the coverage.** It is earned by rejecting
14% of clean and 20% of corrupted clips. Whether the competition permits
abstentions in its robustness metric is **UNKNOWN** — not verified against the
rubric. Until it is, do not claim a robustness PASS.

The gate is **top-1 minus top-2 probability margin ≥ 0.15** plus a 20% score
floor, verified at `CameraScreen.kt:958`. It is **not** a 0.8 confidence
threshold and **not** a 0.5 abstention, and it does **not** withhold the
prediction — the card still shows the top-1, labelled *"Uncertain — this is the
model's best guess, not a confirmed reading."*

The pixel PASS at 4.48 is at **512 letterbox**, which the Android app does not
use — it runs 640×480 native. **Native-geometry pixel robustness has not been
measured** and is not claimed.

See `docs/release/ROBUSTNESS_RECONCILIATION.md` for the full derivation.

---

## Robustness and accuracy claims that must never be made

| Never say | Actually |
| --- | --- |
| "77% accurate on a phone" | Corpus/laptop only; one phone run proves the pipeline, not an accuracy |
| "184 independent signs tested" | 184 evaluations over **72 unique clips** |
| "INT8 quantised" | FP16 ships |
| "Meets the 50 ms target" unconditionally | GPU p95 46–54 ms empty-frame, 68–70 ms with a hand, debug build |
| "Robust to occlusion" | Landmark robustness FAILs at 5.76 pp; native pixel unmeasured |
| "Continuous translation" | 100 isolated signs |
| "Robustness criterion passed" | **FAIL** under the shipped aggregator |
| "The live camera works as well as a clip" | Same sign 0.969 from a file, 0.246 live |

---

## Test status

| Suite | Result |
| --- | --- |
| Android JVM, `submission/final` @ `29c48b6` | **150 / 150 pass**, 0 failed, 0 skipped |
| Python `evaluation/device` | 433 / 502 pass; 3 failed, 23 errors, 43 skipped |
| Official bundle suite | 15 / 15 pass |
| Clean-extraction `predict()` | 30 / 30 clips, 0 errors |
| Android instrumentation | 16 pass, 0 fail, 1 skip (emulator, API 34) |
| Physical phone | **Run on one Vivo V2404, 2026-11-21**; pipeline verified, no accuracy measured |

The 26 Python failures are identical on the untouched frozen baseline and all
trace to three gitignored artifacts (a `summary.json`,
`landmark_classifier_fp16.tflite`, a `.npy` fixture). They are blocked on
assets, not code.