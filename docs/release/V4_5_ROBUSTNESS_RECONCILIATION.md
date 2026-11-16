# V4.5 — robustness reconciliation

Three separate robustness claims circulated in this project. Two were wrong, and
the corrected values point in opposite directions from what was previously
reported. This file states what each measurement actually is.

---

## 1. The aggregator defect — why every old robustness number is invalid

`scripts/ml/robustness.py` averaged **per-member softmax probabilities**. What
ships is different:

| file | aggregation |
| --- | --- |
| `src/main.py` | `np.mean(member_logits)` then **one** softmax |
| `LandmarkClassifier.kt` | `EnsembleAggregation.meanLogits` then one softmax |
| `robustness.py` before the fix | per-member softmax, then mean |

These are different estimators. The project had already measured the difference
— `results/EXPERIMENTS.md` records mean-probability as +0.9 test / −0.9 val and
**rejected** it. Every robustness figure produced before the fix described a
model that does not ship.

Dev fixed both `robustness.py` and `geometry_pair.py` to average logits.

**Clean accuracy is unchanged by the fix** (76.94% either way, to the decimal),
so the headline number was not affected. The *verdicts* were.

---

## 2. Landmark-level robustness — FAIL, not PASS

1,249 test clips, nine models, mean-logit aggregator, fresh extraction at
256 squash:

| corruption | mean-logit (shipped) | mean-prob (old, wrong) |
| --- | --: | --: |
| clean | 76.94 | 76.94 |
| jitter σ0.01 | 75.98 | 75.98 |
| jitter σ0.03 | 73.26 | 73.18 |
| drop_landmarks 15% | 75.02 | 75.10 |
| **drop_hand 25%** | **71.58** | 72.54 |
| half_framerate | 74.62 | 75.02 |
| camera_further 25% | 75.74 | 76.06 |
| **worst drop** | **+5.36 → FAIL** | +4.40 → "PASS" |

**The previously reported 4.40 PASS is withdrawn.** Under the aggregator the app
actually uses, landmark robustness FAILs the 5-point budget by 0.36.

### At native geometry it is worse

From `results/robustness_native.json` (clean 85.45%):

| condition | top-1 | drop |
| --- | --: | --: |
| clean | 85.45 | — |
| jitter σ0.01 | 85.61 | −0.16 |
| jitter σ0.03 | 82.25 | +3.20 |
| drop_landmarks 15% | 83.29 | +2.16 |
| **drop_hand 25%** | **79.70** | **+5.76** |
| half_framerate | 83.93 | +1.52 |
| camera_further 25% | 85.21 | +0.24 |

**Worst +5.76: still FAIL.** The accuracy gain at native geometry does not make
the model robust to a missing hand.

Dev's decomposition shows this is genuine information loss, not a preprocessing
bug: telling the model a hand is absent (presence zeroed, coordinates kept)
scores *better* than leaving stale coordinates, so the attention mask is doing
its job. The residual ~5.5 points cannot be recovered from the input side.

---

## 3. Pixel-level robustness — PASS at 512 letterbox, NOT what the app uses

Dev, full 1,249 test clips, mean-logit, **512 letterbox**:

| corruption | top-1 | drop |
| --- | --: | --: |
| clean | 85.67% | — |
| blur k5 | 84.55% | +1.12 |
| **low light 40%** | **81.18%** | **+4.48** |
| downscale 35% | 83.75% | +1.92 |
| jpeg q25 | 82.79% | +2.88 |

**Worst +4.48 → PASS. But this is 512-letterbox, a geometry the app does not
use.** The app is 640×480 native. A pass at 512 letterbox is **not** evidence
that native-resolution camera robustness passes.

Dev also found and fixed a `--limit 0` bug: a supposed full-split run had
actually scored 120 clips. The 120-clip estimates scattered badly and pointed the
other way on some conditions; the full-split numbers supersede them.

---

## 4. Our own pixel measurement — different model, different population

`evaluation/device/robustness_ensemble/` measured **four** models, not nine, on
**30 clips** from val_30:

| condition | accuracy | vs clean |
| --- | --: | --: |
| clean | 76.67% | — |
| low_light | 80.00% | +3.33 |
| motion_blur | 76.67% | 0.00 |
| crop | 66.67% | **−10.00** |
| occlusion | 72.41% | **−6.67** |
| frame_drop | 76.67% | 0.00 |

Worst −10.00. This has a **re-encode control** showing the codec round-trip
alone changes nothing in aggregate, so the crop and occlusion drops are real
corruption effects rather than encoder artefacts.

It is a different model generation and a different clip set from Dev's runs.
It must not be averaged with them, quoted as nine-model, or used to contradict
Dev's 512-letterbox PASS — they are not comparable measurements.

---

## 5. What may be claimed in the submission

| Claim | Allowed? |
| --- | --- |
| "Landmark robustness: worst 5.36 pp drop, fails the 5-point budget" | **Yes** — shipped aggregator, full test split |
| "Landmark robustness passes at 4.40" | **No** — withdrawn |
| "Pixel robustness passes at 4.48" | **Only with "512 letterbox" stated**, and only as Dev's measurement |
| "The app is robust to occlusion" | **No** — 640×480 native not measured |
| "Nine-model pixel robustness" | **No measurement exists** |
| "We meet the robustness criterion" | **No** |

---

## 6. Every future robustness result must carry

git commit · model hash · split · unique clip count · corruption type ·
corruption strength · detector geometry · aggregator · mean vs per-clip
statistics · device/runtime · measurement date.

Results without those are not comparable and should not be quoted.

---

## 7. Status

| Measurement | Status |
| --- | --- |
| Landmark, 256 squash, mean-logit, test | **MEASURED — FAIL 5.36** |
| Landmark, native, mean-logit, test | **MEASURED — FAIL 5.76** |
| Pixel, 512 letterbox, mean-logit, test | **MEASURED — PASS 4.48** (not app geometry) |
| Pixel, 256 squash, 4-model, val_30 | **MEASURED — FAIL 10.00** (old model) |
| Pixel, **native 640×480**, nine models | **NOT MEASURED** |
| Physical phone, any condition | **NOT TESTED** |

Bounded native pixel robustness was not run: it requires re-extracting 1,249
clips through a corrupted-video path, and the release decision does not depend
on it. It is recorded as **NOT VERIFIED** rather than assumed either way.