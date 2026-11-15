# Geometry: the app runs at a better operating point than its training data

Written 15 Nov 2026 by Dev. Full 1,249-clip signer-disjoint test split, nine
shipped checkpoints, mean-logit aggregator (the one the app uses).

## The numbers

| landmark source / geometry | test top-1 |
| --- | --: |
| 256 squash — **what the models were trained on** | **82.47%** |
| 512 letterbox | 85.67% |
| 640×480 letterboxed to a 640 square | 86.31% |
| **native 640×480, no resize — the exact app path** | **85.51%** |

The Android app does no resize: `CameraScreen.kt` sets
`setTargetResolution(Size(640, 480))` on the ImageAnalysis use case, and
`HandTracker.kt` builds the MediaPipe image straight from
`imageProxy.toBitmap()` with no crop, scale or letterbox. So 85.51% is the
number that corresponds to what a user actually gets.

## Is the gap real?

Paired McNemar, **same 1,249 clips, squash256 vs the native app path**:

| | |
| --- | --- |
| squash256 only right | 52 clips |
| native 640×480 only right | 90 clips |
| both correct | 978 |
| neither correct | 129 |
| McNemar z | 3.10 |
| p (two-sided) | **0.0019** |

The app geometry wins by roughly 3 points, and it is not a sample-size artefact.

A separate comparison — squash256 vs *640×480 letterboxed to a square* (the
approximation I used before wiring the native path) — gives 49 vs 97, z=3.89,
p=0.0001. That is the number recorded in `results/geometry_pair.json` under
`app640x480`, and it belongs to the **approximation**, not to the native path.
Both tests agree on the direction and both are decisive; only the native one
describes what ships.

## Why this is the actionable finding

The app has been running at an operating point the models were never trained on,
and that operating point is *better* than the one they were trained on. That is
an unusual direction — normally deployment drift costs you accuracy. Here the
training data is the weaker representation.

Retraining at the app's geometry is therefore not a trade-off. It makes training
and deployment agree, and the measured ceiling at that geometry is about 85.5%
against the current 82.5%.

Two honest qualifications:

- **All of these numbers come from fresh extraction, not the shipped cache.**
  The cache scores 76.94% at 256 squash because its presence mask is wrong on
  29% of test clips (see [LANDMARK_CACHE_DISCREPANCY.md](LANDMARK_CACHE_DISCREPANCY.md)).
  The comparison above is internally consistent — every row uses fresh
  extraction — but it is not the number currently in `landmark_model_meta.json`.
- **This measures the models we have, re-fed at a different geometry.** It is
  not a measurement of a model retrained at that geometry, which should do
  better still. The 85.5% is a floor for the retrain, not a ceiling.

## The letterbox approximation was close

Before wiring the native path I approximated the app geometry as "640×480
letterboxed to a 640 square". That gave 86.31%; the true native path gives
85.51%. Within 0.8 points, so the earlier runs were not misleading — but the
direct measurement is the one to quote.

## Reproducing

```
python .scratch/app_path_eval.py        # native 640x480, the app path
python scripts/ml/geometry_pair.py --limit 0   # the three-way comparison
```

`read_frames(..., size=None)` is the native path. **The default is unchanged** —
`size` defaults to `FRAME_SIZE = 256` — so extraction, the cache and every
checkpoint are unaffected. Verified byte-identical after the change.