# Aggregator discrepancy — every robustness number was at the wrong operating point

Written 14 Nov 2026 by Dev. This supersedes the robustness figures in
`results/results.csv` until they are re-run.

## What was wrong

`scripts/ml/robustness.py` averaged **per-member softmax probabilities**.
`scripts/ml/geometry_pair.py` did the same. What actually ships is different:

| file | aggregation |
| --- | --- |
| `src/main.py:208` | `np.mean(member_logits)` then one softmax |
| `LandmarkClassifier.kt:230` | `EnsembleAggregation.meanLogits` then one softmax |
| `robustness.py` (before today) | `torch.softmax(lg)` per member, then `np.mean` |

Probability-averaging is a different estimator. This project had already
measured it: `results/EXPERIMENTS.md` records mean-probability aggregation as
+0.9 test / −0.9 val against logit averaging, and it was **rejected**. So every
robustness number in the repo described a model we do not ship.

Both functions are now fixed to average logits.

## What the correction changes

**Clean accuracy is unaffected by this fix.** 76.94% under both aggregators, to
the decimal — so the aggregator bug did not move the headline number. That is
*not* the same as the figure being sound: the cache it is measured on is wrong
about presence on 29% of test clips, which is a separate and larger problem
(a second section below). 77.02% survives the aggregator correction; it does not
survive the cache correction.

**The robustness verdicts swap.** Using the real `robustness.CORRUPTIONS` and
the harness's own RNG seed, 1,249 test clips:

| corruption | mean-logit (shipped) | mean-prob (reported) | difference |
| --- | --: | --: | --: |
| clean | 76.94 | 76.94 | 0.00 |
| jitter_sigma0.01 | 75.98 | 75.98 | 0.00 |
| jitter_sigma0.03 | 73.26 | 73.18 | +0.08 |
| drop_landmarks_15pct | 75.02 | 75.10 | −0.08 |
| **drop_hand_25pct** | **71.58** | **72.54** | **−0.96** |
| half_framerate | 74.62 | 75.02 | −0.40 |
| camera_further_25pct | 75.74 | 76.06 | −0.32 |
| **worst drop** | **+5.36 FAIL** | **+4.40 PASS** | |

The landmark criterion **FAILS by 0.36 points** under the aggregator the app
uses, and passed only under the one it does not.

Pixel robustness moves the other way, at 512-letterbox on 120 clips:

| | mean-logit | mean-prob |
| --- | --: | --: |
| worst case | **+5.00 PASS** | +7.50 FAIL |

So the two criteria have swapped relative to what was recorded. Two independent
runs of the pixel cell produced byte-identical JSON, so this is stable, not
noise. It is still a 120-clip estimate — the full split has not been run through
the pixel harness.

## A second, larger finding: the shipped 77.02% understates the model

Scoring the same 1,249 clips from the **landmark cache** gives 76.94%, but
re-extracting from the videos and scoring identically gives **82.47%** — a 5.52
point gap with the same models, same features, same aggregator.

The decomposition is decisive. Splitting by whether the two extractors agree on
which frames contain hands:

| group | n | cached | fresh | gap |
| --- | --: | --: | --: | --: |
| presence agrees | 885 (70.9%) | 82.49% | 82.49% | **+0.00** |
| presence differs | 364 (29.1%) | 63.46% | 82.42% | **+18.96** |

On the 71% of clips where presence matches, the two score **identically**. The
entire gap is a presence-mask error on 29% of clips, where the cache loses 19
points. The error has a specific shape — the cache holds a detection open across
the back half of the clip (frames 15-31, or 11-31) where a fresh run finds no
hand at all.

That makes it consequential beyond reporting: the models' attention masking reads
exactly this signal, so the shipped cache feeds a known-wrong mask to both
training and evaluation. Full detail in
[LANDMARK_CACHE_DISCREPANCY.md](LANDMARK_CACHE_DISCREPANCY.md).

**Practical consequence:** every accuracy figure in this repo that comes from
the cache — including 77.02% — is measured on a derived artefact of a superseded
extractor. Re-extracting before further training is no longer optional.

I also nearly filed a false bug against `extract_parallel.py`; that correction
is recorded in the same document.

## Geometry: the app runs at a better operating point than its training data

Full 1,249 clips, logit aggregator, paired McNemar:

| comparison | accuracy | McNemar |
| --- | --- | --- |
| squash256 (training geometry) | 82.47% | — |
| letterbox512 | 85.67% | z=3.34, p=0.0008 |
| app640x480 | **86.31%** | z=3.89, p=0.0001 |

letterbox512 vs app640x480 is z=1.17, p=0.24 — not distinguishable.

The Android app already feeds MediaPipe 640×480 (`CameraScreen.kt`
`setTargetResolution`, no resize in `HandTracker.kt`). So the app has been
running at a geometry that beats the one the models were trained on, by a
statistically significant margin. Retraining at the app's geometry is the
actionable item, and it is worth more than the robustness tuning that started
this thread.

## Open

- Full-split pixel run with the corrected aggregator (the 120-clip PASS needs
  confirming).
- The cache-vs-fresh cause.
- `results.csv` rows need re-issuing from corrected runs; they currently carry
  the mean-prob numbers.