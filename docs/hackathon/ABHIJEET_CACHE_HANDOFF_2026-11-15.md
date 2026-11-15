# Your app is right. Our number was wrong.

Written 15 Nov 2026 by Dev, for Abhijeet. Read this before you change anything —
the conclusion may be the opposite of what you'd expect from the repo.

## Headline

The ensemble scores **85.45%** on the signer-disjoint test split, not 77.02%.
**Your app did not change. Our evaluation cache did.**

## What was wrong

The landmark cache at `.scratch/landmarks` was built on 8 Nov at 256×256
squashed frames. Audited against a fresh extraction, its hand-presence mask is
wrong on **29% of test clips** (364 of 1,255): it holds a detection open across
the back half of a clip where the detector actually loses the hand.

The models' attention masking reads exactly that signal, so every accuracy
number in the repo was measured against a known-wrong input.

## What I rebuilt

All 3,390 clips re-extracted at **native 640×480, no resize** — which is what
`CameraScreen.kt` (`setTargetResolution(640,480)`) and `HandTracker.kt` (zero
crop/scale/letterbox) already do. Your pipeline was correct; our Python
reference was the thing out of step, reading at 256 squash.

## The numbers, from the artifacts that actually ship

Measured on the **FP16 `.tflite` files in `models/`**, not the `.pt`
checkpoints, with mean-logit aggregation as `LandmarkClassifier.kt` does:

| cache | TFLite ensemble, test |
| --- | --: |
| old (256 squash) | **77.02%** |
| new (native 640×480) | **85.45%** |

The 77.02% reproduces the shipped `landmark_model_meta.json` to the decimal, so
the harness is measuring the same thing you validated. The 85.45% is the same
nine models on input that matches the app.

Every member improves, from +7.7 to +11.9 points (mean +9.10). Not one lucky
model.

Also verified: the new cache's predictions are **per-clip identical to a live
re-extraction on all 1,249 test clips** — so this is a deterministic record of
the app's real input path, not a fluke.

## What this means for your side

**Nothing in the app needs to change to get 85%.** It already feeds MediaPipe
native frames. If the phone is capturing 640×480 and not resizing, you are
already at this operating point.

**What does need changing is any document that quotes 77.02%.**

- `models/landmark_model_meta.json` → `ensemble_test_top1: 0.7702` should be
  **0.8545**. Nothing at runtime reads this field (checked: it is written by
  `export_ensemble.py` and consumed by no code), so updating it is safe. The
  `members` list, `members_detail` and all hashes are untouched.
- `results/results.csv` — I have already updated this on my branch.
- Any deck, README or demo script quoting 77.02 or "four-model ensemble".

## Robustness, re-measured on the new cache

| criterion | old cache | new cache | verdict |
| --- | --: | --: | --- |
| landmark worst | +5.36 | **+5.76** | FAIL both |
| pixel worst | +4.48 | running | — |

Landmark robustness did **not** improve with the cache fix, and got marginally
worse. Both failures are `drop_hand_25pct` — dropping a hand for 25% of a clip.
That is the one weakness the cache fix does not touch: it is a model limitation,
not a measurement artefact.

## What I tried and rejected, so you do not repeat it

- **Retraining at the app's geometry**: 60 epochs, `--jitter-sigma 0.03` →
  **76.26% test**, below a single existing member (78.90%). Do not replace the
  ensemble on that evidence.
- **INT8**: −6.37 points, two runtimes disagreed on the file. FP16 loses 0.00
  against FP32. Already settled.

## What is still open, and only you can do

**On-device latency, power and thermal.** `avg_power` and `thermal_drift_10min`
are still `NOT_MEASURED`. I cannot measure them: no phone is attached to this
machine, and the emulator cannot run MediaPipe at all (Google's `tasks-vision`
AAR ships arm64-v8a, armeabi-v7a and x86 — **no x86_64 slice**). That is 20% of
the rubric and it needs your hardware.

The classifier itself is fast: nine members measure **4.2 ms/clip** on desktop
x86, and everything scales linearly over four. But that is not a device number.

## If you want to verify any of this

```bash
python scripts/ml/eval_tflite_cache.py --cache .scratch/landmarks_native --split test
python scripts/ml/robustness.py --level landmark --cache .scratch/landmarks_native
python scripts/ml/geometry_pair.py --limit 0
```

Full detail in `docs/hackathon/`, particularly `CACHE_REBUILT_2026-11-15.md`
and `AUDIT_2026-11-15.md` — the latter records twelve claims in my own earlier
drafts that an independent audit caught as wrong, so this branch's docs are the
corrected versions.