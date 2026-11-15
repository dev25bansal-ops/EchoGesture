# Pixel robustness — measured, corrected, and reproducible

Written 14 Nov 2026 by Dev. Two corrections are recorded here because I made
them after first writing this file. The current state is what stands.

## FINAL: full-split result, shipped aggregator

Measured on all 1,249 test clips, mean-logit aggregator, 512 letterbox,
6 workers (`scripts/ml/robustness_pixel_parallel.py`):

| corruption | top-1 | drop |
| --- | --: | --: |
| clean | 85.67% | — |
| blur_k5 | 84.55% | +1.12 |
| **low_light_40pct** | **81.18%** | **+4.48** |
| downscale_35pct | 83.75% | +1.92 |
| jpeg_q25 | 82.79% | +2.88 |

**Worst +4.48 against a 5.00 budget: PASS.** All four corruptions pass
individually.

Cross-check: `geometry_pair.py`, written independently, gives 85.61% clean at
the same geometry. The two agree to **0.06 points**, which is the evidence that
neither harness is quietly broken.

The 120-clip samples above pointed the same way but scattered more
(blur +2.50 vs +1.12, downscale −0.83 vs +1.92); at 1,249 clips the picture
settles. Low light remains the worst corruption, as the earlier work predicted,
but on the full split it lands **inside** the budget rather than over it.

## Status of the earlier sections

Everything below predates the full-split run and the aggregator correction. The
tables are kept because they show how the conclusion moved, but **the section
above is the number to quote**. In particular the "minimum +7.50, no cell passes"
conclusion was a 120-clip estimate scored with the rejected aggregator; both of
those changed.

One clip set, one code path, one harness invocation per cell:
`scripts/ml/robustness.py --level pixel --limit 120` with `--size` and
`--letterbox`. `load_split` sorts `.npz` filenames and takes `[:limit]`, and
neither flag touches it, so all four cells score the **identical 120 clips**.
Runs are deterministic — pass 1 of the matrix reproduced the session's first
measurement to the decimal.

| config | clean | blur | low_light | downscale | jpeg | worst |
| --- | --: | --: | --: | --: | --: | --: |
| 256 squash (baseline) | 85.83 | +2.50 | +5.83 | **+28.33** | +8.33 | **+28.33 FAIL** |
| 512 squash | 85.00 | −0.83 | +5.83 | −0.83 | +0.00 | **+5.83 FAIL** |
| 256 letterbox | 90.00 | +5.83 | +7.50 | **+32.50** | +11.67 | **+32.50 FAIL** |
| **512 letterbox** | 90.83 | +1.67 | +7.50 | **+0.00** | +2.50 | **+7.50 FAIL** |

Cell 4 reproduces `results/robustness_pixel.json` to the decimal, which is the
check that the matrix and the committed number are the same measurement.

**Three findings, all measured:**

1. **Resolution is the dominant lever.** Going 256 → 512 fixes downscale almost
   regardless of framing: +28.33 → −0.83 (squash) and +32.50 → +0.00 (letterbox).
2. **Letterboxing alone makes it worse.** At 256 it degrades every corruption
   (+28.33 → +32.50 on downscale, +8.33 → +11.67 on jpeg), exactly as the
   arithmetic predicted — it leaves 67 rows instead of 89.
3. **The two interact, but not in letterbox's favour.** 512-letterbox has the
   highest raw top-1 under every corruption *because* its clean accuracy is
   5.8 points higher (90.83 vs 85.00). Measured as a **drop from clean** — which
   is what the criterion scores — it is worse on all four. See the corrected
   table below.

There is no configuration in this matrix that passes overall. The worst case
moves between +5.83 and +7.50 depending on cell, and low light is the binding
corruption in three of four.

## What caused the 28.33, and what does not

Arithmetic by construction, not inference. The detector input is a 35% downscale
of whatever frame it is handed, so detail after corruption is:

| config | detector sees | after 35% downscale |
| --- | --- | ---: |
| 256 squash | 256×256 | 89 px |
| 256 letterbox | 192×256 | 67 px |
| 512 squash | 512×512 | 179 px |
| 512 letterbox | 384×512 | 134 px |

Two conclusions follow, and one of them corrects something I said earlier.

**Letterboxing is not the fix, and on its own it is harmful.** At 256 it leaves
*fewer* rows than squashing (67 vs 89), because preserving aspect ratio shrinks
the short side — and the matrix confirms it degrades every corruption at that
size. Measured, not predicted.

**Resolution reaching the detector is the fix.** Both 512 cells recover downscale
to roughly zero. Letterboxing at 512 does **not** add robustness on top of that —
see below.

## Honest reading of the four cells

No cell passes the 5.00 budget. The two 512 cells:

- 512 squash: worst +5.83, clean 85.00
- 512 letterbox: worst +7.50, clean 90.83

**Letterbox is worse on every corruption.** Comparing drops (not raw top-1,
where letterbox's higher clean baseline flatters it):

| corruption | 512 squash drop | 512 letterbox drop |
| --- | --: | --: |
| blur_k5 | −0.83 | +1.67 |
| low_light_40pct | +5.83 | +7.50 |
| downscale_35pct | −0.83 | +0.00 |
| jpeg_q25 | +0.00 | +2.50 |

Letterbox has the better clean accuracy (90.83 vs 85.00) and therefore the
better raw top-1 under every corruption, but a **larger drop** in all four
cases. An earlier version of this section claimed letterbox was better on
"three of four corruptions"; that was wrong — it compared raw accuracies, which
inherit the clean-accuracy difference.

Neither geometry is a robustness win over the other. The choice between them
should rest on clean accuracy, and at 512 the difference there is 5.8 points on
120 clips.

## Low light is a classifier limit, measured not inferred

Hand presence moves 48.8% → 46.9% of frames under `low_light_40pct`. Landmarks
are found; the classifier then misreads them. CLAHE and gamma 2.2 both *lowered*
detection further (45.8% and 43.0%). No front-end change fixes this. It needs
photometric augmentation and a retrain.

## The Android app

Verified in source: `CameraScreen.kt` calls
`setTargetResolution(android.util.Size(640, 480))` on the `ImageAnalysis` use
case, and `HandTracker.kt` builds the MediaPipe image straight from
`imageProxy.toBitmap()` with no resize. The app presents 640×480 landscape to
the detector. The Python default is 256×256 squashed, so the harness was
measuring a harder operating point than the app ships.

## The error this file replaces

The first version of this doc claimed the 28.33 "before" and 7.50 "after" runs
had scored different clip sets (12 of 120 in common). That was wrong, and it was
my error twice over:

1. The comparison I made was harness-to-harness, and `load_split` selects clips
   by filename order regardless of `--size`/`--letterbox`, so every harness run
   this session scored the **identical 120 clips**. The clip-set story was false.
2. What actually happened: the session's first run (256 squash, worst +28.33)
   wrote to `results/robustness_pixel_nine.json`. A later run, launched after
   `FRAME_SIZE` had been changed to 512, wrote to the **same filename**, and I
   then copied that 512-squash output into a file I named
   `robustness_pixel_before_letterbox.json` — mislabeling a 512-squash run as
   the pre-fix baseline. I then retracted the 28.33 result on the strength of my
   own file mismanagement. The matrix re-establishes it: pass 1 reproduced
   +28.33 exactly, and the session's first run had produced it too.

The lesson is the one in the header: track which file holds which
configuration, and re-derive a baseline from the harness rather than trusting a
copy whose provenance is unclear.

## Defaults are unchanged

`FRAME_SIZE` stays 256 and `letterbox` stays False. The landmark cache, every
checkpoint and the app's feature transform are downstream of extraction
settings, so changing them requires a retrain. The default path was verified
byte-identical after the refactor.

## Reproducing

```
python scripts/ml/robustness.py --level pixel --limit 120 --size 512
python scripts/ml/robustness.py --level pixel --limit 120 --size 512 --letterbox
```

`--size` and `--letterbox` were added for this; without them the harness
measures the 256-squash path. Output goes to `--out`, default
`results/robustness.json`.

**The four matrix cells were all computed with the mean-probability aggregator**
(see [AGGREGATOR_FINDING.md](AGGREGATOR_FINDING.md)), so they are not the shipped
operating point. They are kept in `results/pixel_matrix/` with that in the
filename — `256squash_meanprob.json`, `512squash_meanprob.json`,
`256letterbox_meanprob.json`, `512letterbox_120clips_meanprob.json` — so no
reader can mistake them for shipped numbers.

**These cells are all 120-clip samples.** An earlier full-split attempt was
silently truncated to 120 by a `--limit 0` bug in the harness (`args.limit or
120` treated "all" as falsy), now fixed.

**The full-split measurement does now exist** — `results/robustness_pixel_full.json`,
1,249 clips, mean-logit, 512 letterbox, worst +4.48 PASS. It is the table at the
top of this document and the row in `results.csv`. It was produced by
`scripts/ml/robustness_pixel_parallel.py`, because the serial harness projected
to ~9.6 hours.

Note for anyone opening `results/robustness_pixel.json`: that file also holds a
**mean-prob, 120-clip** result (clean 90.83) with no aggregator marker in its
name. Prefer `results/pixel_matrix/` where the filename states the aggregator and
the clip count, and prefer `robustness_pixel_full.json` for the shipped number.