# The cache is rebuilt — and it is worth 8.5 points

Written 15 Nov 2026 by Dev. This is the actionable outcome of the cache
investigation: the re-extraction is done, verified, and on disk.

## Result

| landmark source | test top-1 |
| --- | --: |
| old cache (256 squash, built 8 Nov) | **76.94%** |
| **new cache (native 640×480, built today)** | **85.45%** |

**+8.51 points, same nine checkpoints, same features, same aggregator.**

## Why this is trustworthy

Two independent checks:

1. **Reproduces the live measurement.** Scoring landmarks re-extracted on the
   fly at native resolution gave 85.51% (`results/app_path_eval.json`). The new
   stored cache gives 85.45%. Independent code paths, 0.06 points apart.
2. **Per-clip identical.** Comparing predictions from the new cache against the
   live app-path run: **1,249 of 1,249 clips agree, 100.00%.** Not a
   same-accuracy-different-clips coincidence — the same answer on every clip.

The old cache, by contrast, disagreed with a fresh extraction on 364 of 1,255
clips (29%), concentrated in test.

## What was built

`scripts/ml/extract_parallel.py` gained `--size` and `--out`:

```bash
python scripts/ml/extract_parallel.py --workers 6 --size native \
    --out .scratch/landmarks_native --splits val,test,train
```

- `--size native` passes frames at their **native resolution with no resize**,
  which is exactly what the Android app does (`CameraScreen.kt`
  `setTargetResolution(640,480)`, no crop/scale/letterbox in `HandTracker.kt`).
- `--size 256` (the default) preserves the original geometry byte-for-byte —
  verified against the old literal `cv2.resize(f,(256,256))` path, 8/8 frames
  identical. Nothing that already exists is disturbed.
- `--out` writes to a **new directory**, so the old cache is untouched.

Completed: 3,390 clips, train 1,789 / val 346 / test 1,255 — matching the old
cache's counts exactly.

Two of six workers died part-way through with no error output (the known
MediaPipe silent-abort mode). The script is resumable and the survivors finished
the queue, so all clips landed. Worth knowing that a 6-worker run may need a
resume pass.

## What this means

**Every accuracy figure in this repo was about 8.5 points low.** The shipped
77.02% was measured on a cache whose presence mask is wrong on 29% of test
clips. On a cache that matches the app's real input, the same models score
85.45%.

**This is not a free win for the submission**, because the shipped checkpoints
were *trained* on 256-squash landmarks. Feeding them native-resolution
landmarks at inference is the "retrain at app geometry" improvement showing up
partly without the retrain — the models generalise well enough to gain from it.
A model trained natively should do better still.

## What is ready to run next

A retrain on `.scratch/landmarks_native` is unblocked. `train.py` gained
`--landmarks` so it can point at the new cache without moving directories.

**I ran it, and it lost.** 60 epochs, `--jitter-sigma 0.03`, seed 0:

| configuration (test split, 1,249-1,251 clips) | top-1 |
| --- | --: |
| **existing nine-member ensemble, new cache** | **85.45%** |
| one existing member (tcn21), new cache | 78.90% |
| **retrained model (60 ep, σ=0.03), new cache** | **76.26%** |

The retrain is **below a single existing member**, let alone the ensemble. Its
val was 84.4%, so the val/test gap is the usual signer-split gap, not a bug —
the model simply is not as good as what the original augmentation sweep
produced. Training at app geometry does not, by itself, beat re-feeding the
existing models at app geometry.

**So the recommendation changes.** Do not replace the shipped ensemble with a
fresh retrain on this evidence. The 85.45% is already available today by
building the app to use native-resolution landmarks with the *existing* nine
members — no retraining required.

A retrain could still win with the original recipe's full augmentation set
rather than a single jitter change, but that is a larger experiment than the
one run here, and it now needs to beat 85.45% rather than 76.94%. Nothing about
the current numbers supports spending the remaining time on it.

## The actual recommendation

**Ship the existing nine models fed native-resolution landmarks.** That is
worth +8.51 points over what the repo currently reports, it needs no retraining,
and it is verified per-clip against a live app-path measurement.

## Caveats

- **Robustness has not been re-measured on the new cache.** The pixel
  robustness PASS (+4.48) and landmark FAIL (+5.36) were both computed on the
  old cache. Those comparisons are internally consistent, but they are not
  numbers about the new cache.
- The old cache is still present at `.scratch/landmarks` and is what
  `results.csv` currently reflects. Switching the repo to the new cache is a
  deliberate decision, not something to do by accident.