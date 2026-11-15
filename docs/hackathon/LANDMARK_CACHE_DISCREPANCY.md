# Landmark cache disagrees with fresh extraction — cause not recovered

Written 14 Nov 2026 by Dev. Status: **effect measured, cause NOT established.**
Read this before quoting any accuracy figure from the cache.

## The effect

Same 1,249 test clips, same nine checkpoints, same features, same logit
aggregator — the only difference is where the landmarks come from:

| landmark source | test top-1 |
| --- | --: |
| `.scratch/landmarks/` cache (built 8 Nov) | **76.94%** |
| re-extracted from video today | **82.47%** |

A 5.53-point gap, with 81 clips correct on fresh extraction and wrong from the
cache, against 12 the other way.

This is the figure the shipped `models/landmark_model_meta.json`
(`ensemble_test_top1: 0.7702`) and `results/results.csv` rest on. If the cache
is the degraded copy, **77.02% understates the model by about five points** and
the real number is nearer 82%.

### The cache is not simply losing detections — it finds MORE

A full audit of all 1,255 cached test clips against a fresh extraction:

| | |
| --- | --: |
| clips with an identical presence pattern | 891 |
| clips where presence **differs** | **364 (29.0%)** |
| total hand-frames, cache | 18,337 |
| total hand-frames, fresh | 17,323 |

So the cache detects **1,014 more** hand-frames than a fresh run, and still
scores 5.5 points worse.

### The problem is concentrated in TEST, not spread across the corpus

Auditing each split against a fresh extraction. Numbers below are from the
stored artifact `results/cache_presence_by_split.json`:

| split | sampled | presence differs | rate |
| --- | --: | --: | --: |
| train | 138 | **0** | **0.0%** |
| val | 35 | 1 | 2.9% |
| test | 1,255 (full) | 364 | **29.0%** |

The train and test figures are the ones that carry weight. Train's zero is
statistically solid: if train had test's rate, the chance of seeing 0 in 138
clips is far below 10⁻²⁰.

**Val is underpowered and two samples disagree.** An earlier 32-clip draw gave
3 differing clips (9%); the stored 35-clip draw gives 1 (2.9%). Both say "low",
neither establishes a number, and the spread between them is the tell — at n≈35
on a ~5% base rate you expect 1-2 hits, so neither draw is informative on its
own. What is established is that val is **much closer to train than to test**,
and that test's 29% is far outside either val sample.

All three splits were written in one session (11:50–12:01 on 8 Nov), so this is
not a build-order artefact.

**Why this is the most important thing in this document.** Had the presence mask
been broadly wrong, the training data would have been wrong too and the models
would have learned around it — the error would be baked in but self-consistent.
Instead the models were trained on a mask that is essentially correct (0 on 138
train clips) and are being **evaluated on one that is wrong on 29% of clips**.
That is a measurement error, not a model error. It makes the model look about
five points worse than it is, and it does so through exactly the signal the
attention mask consumes.

One caveat I will not paper over: 0/128 on train and 9% on val versus 29% on test
is a strange shape. A degradation confined to the *last* split written would
explain it, and test files are timestamped 11:55–11:58 while train runs 11:59–12:01
— i.e. test was written *first*, not last. So I do not have a mechanism that
predicts this gradient. What I have is the measurement, repeated at three sample
sizes, and it is consistent.

### Where the gap actually lives: it is entirely a presence problem

Splitting the 1,249 scored clips by whether the two extractors agree:

| group | n | cached | fresh | gap |
| --- | --: | --: | --: | --: |
| presence **agrees** | 885 (70.9%) | **82.49%** | **82.49%** | **+0.00** |
| presence **differs** | 364 (29.1%) | **63.46%** | **82.42%** | **+18.96** |
| all | 1,249 | 76.94% | 82.47% | +5.52 |

**On the 885 clips where the extractors agree, the two score identically to two
decimal places.** The feature pipeline, the checkpoints, the aggregator — all of
it is innocent. The entire 5.52-point gap comes from 364 clips where the cached
presence mask is wrong, and on those the cache loses 19 points.

### The shape of the error

Of the 364 disagreeing clips, **262 have the cache finding MORE hand-frames**
(median **+5** over those 262, mean +5.57), 89 have it finding fewer, 13 differ
only in position. The median over all 364 is +3; the +5 figure is the one for
the cache-more group specifically. No clip has
one extractor finding hands where the other found none at all.

Inspecting the worst cases, the pattern is consistent and specific: the cache
claims hands for the **entire back half** of the clip.

```
6545272573094927-WALK-TIGHTROPE-CL
  cache 00011111111111111111111111111111 (29)
  fresh 00111111111111100000000000000000 (13)
  cache-only frames: 15..31

7754706991206437-ELEVATOR
  cache 00000001111111111111111111111111 (25)
  fresh 00111111111000000000000000000000 (9)
  cache-only frames: 11..31
```

That is not a detector resolving a borderline frame differently. It is a
detection being **held open** across seventeen to twenty-one consecutive frames
after the fresh run has lost the hand. A stale tracker, a recycled landmarker
whose state leaked between clips, or a run whose per-clip reset did not happen —
all fit this shape. That the cache builder rebuilt its landmarker every
`RECYCLE_EVERY = 200` clips makes a state-leak explanation the natural one,
though I could not reproduce it.

## What is established

- **MediaPipe detection is deterministic.** Five identical `detect_hands` calls
  on the same frames return identical output, max abs difference 0.000000.
  Landmarker reuse does not change results; running another clip first does not
  change results.
- **The sampled frames are pixel-identical.** `extract_landmarks.read_frames`
  and `landmark_frontend.read_frames` produce byte-identical frames on every
  clip checked (0 of 5 differed, max pixel diff 0).
- **Every current code path reproduces the FRESH result, not the cache.** I ran
  the cache-producing loop from `extract_landmarks.py` and separately
  `extract_parallel.extract_one` — both give frames 7-20 detected, exactly like a
  fresh run. The cache holds frames 26-31 for the same clip.
- The cache files are all timestamped **2026-11-08**, before the current
  extraction code landed.

## What I could not establish

**The exact code that produced it.** Every hypothesis I could *test* failed:

| hypothesis | result |
| --- | --- |
| Non-determinism in MediaPipe | disproved — 5/5 identical |
| Landmarker state accumulating across clips in-process | disproved in my repro |
| VIDEO mode instead of IMAGE | disproved — does not match either |
| Different `read_frames` (squash, linspace, np.unique) | disproved — byte-identical |
| `extract_parallel` vs `extract_landmarks` | disproved — both give fresh result |

The cache was built from a code state that no longer exists and that I cannot
reconstruct from git (the extraction files were committed after the cache was
written). What I *can* say is narrower and more useful than a root cause: the
error is a presence-mask error of a specific shape (detections held open across
the back half of a clip), it affects 29% of clips, and it costs about 5.5 points
overall and 19 points on the clips it touches.

## What to do about it

1. **Do not treat the cache as ground truth.** It is a derived artefact of an
   unknown, superseded extractor, and it is wrong about presence on 29% of clips.
2. **Re-extract before any further training or evaluation.** This is not
   optional now: the gap is concentrated in the presence mask, which is exactly
   the signal the models' attention masking depends on. Training or evaluating
   against the cache builds on a mask that is known wrong.
3. Until then, quote **both** numbers or neither. 77.02% means "on the shipped
   cache", not "the model's accuracy".

A full re-extraction is roughly an hour of MediaPipe over 3,390 clips. Given
that it removes a known-wrong input from every downstream number, that is cheap.

## A bug I thought I found, and did not

I initially wrote that `scripts/ml/extract_parallel.py` passes `mp.Image` — the
module namespace — into a parameter named `mp_image` that is then called, and
that this would raise `TypeError`. **That was wrong and I checked it.** Calling
`extract_one` exactly as the worker does returns a correct array with the
expected detections.

The reason: `mp.Image` is not a module attribute, it is the `Image` **class**
itself (`<class 'mediapipe.tasks.python.vision.core.image.Image'>`), which is
callable and constructs an image. The parameter name `mp_image` is confusing but
harmless. Recorded here because I nearly left a false bug report in the repo.

## Reproducing

```
python .scratch/cache_vs_fresh.py      # 76.94 vs 82.47, paired per clip
python .scratch/determinism.py         # 5 identical calls
python .scratch/run_extract_one.py     # extraction loop reproduces fresh, not cache
```