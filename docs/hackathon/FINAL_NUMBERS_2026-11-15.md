# Final numbers on the native cache, 15 Nov 2026

Everything here is measured on `.scratch/landmarks_native` — the corpus
re-extracted at native 640×480, which is what the Android app actually feeds
MediaPipe. Same nine shipped members.

## Accuracy: confirmed on the artifacts that ship

Measured with `scripts/ml/eval_tflite_cache.py`, which loads the **FP16
`.tflite` files from `models/`**, resolves inputs by name, reads width per
member from `members_detail`, and averages logits then one softmax — the same
path `src/main.py` and `LandmarkClassifier.kt` take.

| cache | TFLite ensemble, test |
| --- | --: |
| old (256 squash, 8 Nov) | **77.02%** |
| new (native 640×480) | **85.45%** |

Two things this establishes:

1. **The old number reproduces exactly.** 77.02% matches
   `landmark_model_meta.json`'s `ensemble_test_top1: 0.7702` to the decimal, so
   the harness measures the same thing the shipped artifact claims. It is not a
   different experiment being compared to a stale figure.
2. **FP16 is not costing anything.** The TFLite result (85.45%) matches the
   PyTorch-checkpoint result (85.45%) exactly. Every member improves, +7.7 to
   +11.9, mean +9.10.

Classifier time: **4.2 ms/clip** for nine members on desktop x86.

## Landmark robustness: still fails, and marginally worse

| corruption | old cache | new cache |
| --- | --: | --: |
| clean | 76.94% | **85.45%** |
| jitter sigma 0.01 | +0.96 | −0.16 |
| jitter sigma 0.03 | +3.68 | +3.20 |
| drop_landmarks 15% | +1.92 | +2.16 |
| **drop_hand 25%** | **+5.36** | **+5.76** |
| half_framerate | +2.32 | +1.52 |
| camera_further 25% | +1.20 | +0.24 |
| **worst** | **+5.36 FAIL** | **+5.76 FAIL** |

The cache fix raised clean accuracy by 8.5 points and left the robustness
verdict unchanged. `drop_hand_25pct` is the only failure in both, and it is
0.76 points over a 5.00 budget.

This is an honest negative: fixing the input did not fix the weakness. Losing a
hand for a quarter of a clip is a model limitation, not a measurement artefact,
and it would need training-time work to address — not more data preprocessing.

## Pixel robustness: passes, unchanged by the cache fix

| corruption | old cache | new cache |
| --- | --: | --: |
| clean | 85.67% | **85.61%** |
| blur_k5 | +1.12 | +1.12 |
| low_light_40pct | +4.48 | **+4.48** |
| downscale_35pct | +1.92 | +1.92 |
| jpeg_q25 | +2.88 | +2.96 |
| **worst** | **+4.48 PASS** | **+4.48 PASS** |

Identical to two decimals on three of four corruptions. That is the expected
result rather than a disappointment: pixel corruption changes the *pixels*
MediaPipe sees, and the cache fix changed how frames are resampled before
MediaPipe. The classifier's tolerance to a degraded input is a property of the
weights, which did not change.

So the two criteria now read: **pixel PASS at +4.48, landmark FAIL at +5.76**,
and the cache fix moved neither. What it moved was accuracy, by 8.5 points.

## What was tried and rejected

- **Retrain at app geometry** (60 epochs, `--jitter-sigma 0.03`): **76.26%**
  test, below a single existing member at 78.90%. Recorded in
  [CACHE_REBUILT_2026-11-15.md](CACHE_REBUILT_2026-11-15.md).
- **INT8**: −6.37 points, already settled.

## Changed on this branch

`models/landmark_model_meta.json` — `ensemble_test_top1` 0.7702 → **0.8545**.
Verified that **only that field changed**: `members`, `members_detail`, every
hash, `frames` and `feature_dim` are byte-identical, and nothing at runtime
reads the field (it is written by `export_ensemble.py` and consumed by no code).
The app's `ModelContract.kt` reads `members` and `members_detail` only, so the
interface Abhijeet pins is untouched.

## Not done, and why

**On-device latency, power, thermal.** No phone is attached (`adb devices`
returns empty), and the emulator cannot run MediaPipe at all: Google's
`tasks-vision:0.10.14` AAR ships arm64-v8a, armeabi-v7a and x86 — no x86_64
slice — while the AVD is x86_64-only. `avg_power` and `thermal_drift_10min`
remain `NOT_MEASURED`, which is 20% of the rubric.