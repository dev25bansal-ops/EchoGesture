# Reconciling Dev's numbers with V4.4's

Two branches report different values for the same-looking metrics. This file says
which is which, so nobody quotes the wrong one. **Nothing was overwritten** —
both measurements stand, because they measure different things.

Source of Dev's figures: `hackathon/dev-model` @ `11bbddf`.
Source of V4.4 figures: `fix/v4-4-final-release` @ `8dace91`.

---

## 1. Robustness — the one that looks like a contradiction

| | Dev (`results.csv`) | V4.4 (`robustness_ensemble/`) |
| --- | --- | --- |
| Worst-case drop | **4.40 pp** | **10.00 pp** |
| Verdict | **PASS** (≤5) | **FAIL** (≤5) |
| Corruption level | **Landmark** — noise injected into the 21 points after detection | **Pixel/video** — corrupted before MediaPipe sees it |
| Conditions | jitter 0.01/0.03, drop_landmarks, drop_hand, half_framerate, camera_further | crop, occlusion, low_light, motion_blur, frame_drop, reencode |
| Model | 9-member masked | 9-member masked |
| Clips | full split | 30 clips, 5 signers, seed 42 |

**Both are true.** They are not the same experiment:

- **Dev's harness corrupts landmarks.** MediaPipe has already run; detection
  cannot fail. It measures how the *classifier* tolerates degraded input.
- **V4.4's harness corrupts the video.** MediaPipe runs on damaged frames, so
  detection failures propagate. It measures the *deployed pipeline*.

A pass at the landmark level does not imply a pass at the pixel level. The
deployed app sees pixels.

**Worst pixel-level conditions in V4.4:** crop −10.00, occlusion −6.67. Both
beyond the 5-point budget.

### Dev's own pixel numbers independently agree with the FAIL

Dev's `results/robustness_pixel.json` — written at `4594bfa`, the 4-member
pre-mask model — reports:

| Condition | Clean | Corrupted | Drop |
| --- | --: | --: | --: |
| blur_k5 | 83.33% | 70.00% | **13.33** |
| low_light_40pct | 83.33% | 73.33% | **10.00** |
| downscale_35pct | 83.33% | 48.33% | **35.00** |
| jpeg_q25 | 83.33% | 70.00% | **13.33** |

**Every pixel condition fails the 5-point budget, worst case 35 points.** This
is an older model, but it is the same conclusion reached with a different
harness on a different code day. The pixel-level FAIL is not an artifact of one
branch's methodology.

**What to quote:** landmark robustness **passes** at 4.40 pp; pixel robustness
**fails** at 10.00 pp on the current model. Never quote 4.40 as the answer to a
camera-robustness question.

---

## 2. Accuracy — not actually a conflict

| Metric | Dev | V4.4 | Agree? |
| --- | --: | --: | --- |
| Test top-1, 9-member masked | 77.02% | not independently re-run | — |
| Val top-1, 9-member masked | 87.25% | not independently re-run | — |

V4.4 did not re-measure accuracy on the full splits; it cited Dev's figure and
used the models. The 79.3% in V4.3's A/B/C report is a **different experiment**
on 72 authorised clips under synthetic idle — not comparable, and never to be
quoted as independent accuracy on 184 videos.

**All accuracy figures are Dev's laptop measurements, not phone measurements.**

---

## 3. Latency — three different things

| Measurement | Value | Whose | What |
| --- | --: | --- | --- |
| Ensemble inference only | 6.56 ms | Dev | nine members, laptop |
| `predict()` end-to-end, warm | 245–322 ms | V4.4 | laptop, MediaPipe dominates |
| Docker container, one clip | 1201 ms | V4.4 | includes decode |

**None is a phone number.** The organizer's ≤50 ms criterion is **FAIL** on
every configuration measured, and capture interaction is seconds by design.

Dev's `results.csv` records a `latency_model_only_p95` of 6.05 ms as **pass**.
That is the model in isolation. It is not the criterion as written.

---

## 4. `results.csv` — why the two files disagree

Dev's version reports 77.02% test with `device = "dev laptop (Windows 11)"` and
robustness PASS. V4.4's reports 72.78% (4-member) with
`device = "macOS arm64 laptop"` and robustness FAIL.

Neither is wrong about its own measurement; each is stale about the other's.
Dev's is **newer as an accuracy record**. V4.4's is **newer as a pixel-robustness
record**.

No merged CSV was produced here, because a single row cannot honestly carry two
different robustness verdicts from two different harnesses without an extra
dimension. The authoritative per-branch records are preserved instead:
`results/robustness.json` (landmark) and
`evaluation/device/robustness_ensemble/RESULTS.md` (pixel).

**Reconciling these into one organizer-facing table is an open task.** The
honest form needs a `corruption_level` column.

---

## 5. Model identity — no conflict at all

| | |
| --- | --- |
| Ordered ensemble SHA-256, both branches | `995cc8d0b6f6ac25abb9f7ddca01c3ce71edf8bc2d34e34f4813a2b99dd3b4bb` |
| `models/landmark_model_meta.json` | byte-identical |
| `models/landmark_labels.json` | byte-identical |
| `src/features.py` | byte-identical |

Verified by hashing Dev's committed members in his own branch order and
comparing to the V4.4 pin. **The two branches ship the same nine models.**