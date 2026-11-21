# Why the live camera misreads signs — 2026-11-21

Supersedes `CAMERA_FRAME_DUMP_2026-11-20.md` (which blamed frame count alone)
and `WHY_CAMERA_MISSES_2026-11-19.md` (superseded already).

**There is no hardcoding.** The nine shipped members score **85.45% (1,069/1,251)**
on the full test set through this machine, and the most over-predicted gloss
occurs 16 times against 11 true (ratio 1.42) — ordinary class imbalance, not a
magnet class. `LandmarkClassifier.kt:243` does a real per-member argmax over 100
logits.

## What was measured

Every number below is over the shipped nine FP16 TFLite members, loaded through
`src/main.py`'s own `_init_model()`. Corpus numbers use `.scratch/landmarks_native`
(real MediaPipe output on native geometry, i.e. the app's own path).

| condition (on 248 test clips) | top-1 | mean margin | member agreement |
|---|---|---|---|
| baseline | 85.08% | 0.615 | 7.85/9 |
| portrait aspect warp | 82.66% | 0.533 | 7.32/9 |
| motion ÷3 (8 of 24 frames) | 72.58% | 0.517 | — |
| z zeroed | 75.00% | 0.504 | — |
| all combined | 67.74% | 0.439 | 6.46/9 |
| **real camera capture** | **15.1%** | **0.061** | **1/9** |

So the three simulated effects explain about 17 points. The live capture sits far
below even the combined condition, which means something outside that model of
the failure was also wrong — and one candidate was then found and fixed.

## The defect that was found and fixed

MediaPipe normalises landmark **x by the frame's own width and y by its own
height**. The corpus is 640×480; the app is pinned to portrait
(`AndroidManifest.xml` `screenOrientation="portrait"`) so the detector receives
**480×640**. Every hand therefore arrived squashed in x by 4/3 and stretched in
y by 3/4 — a **1.78× anisotropy** — before the feature transform ever ran.

Fix: `x *= W/640`, `y *= H/480` in both `HandTracker.toHands()` (camera) and
`ClipAnalyzer.toHands()` (file picker), so a clip classifies identically either
way. z is untouched; it is relative depth in units of the hand, not the frame.

Measured on the real 32-frame capture:

```
before fix   1/9 members agreed   top-1 margin 0.061   WHATFOR1 0.151
after  fix   5/9 members agreed   top-1 margin 0.202   WHATFOR1 0.290
```

No regression: **0 of 21 sampled clips changed top-1**, including two 960×540
(16:9) clips. 640×480 is an exact identity, so the corpus path is unchanged.
Committed `44b7fc2`.

## The second lever: analyser throughput

The app captures a **3.7 s** window (`CAPTURE_SPAN_MS`) and resamples it to 32
points, so the analyser's real rate sets the temporal resolution. Simulating a
3× coarser window costs **12.5 points** — the single largest simulated effect,
because `features.py` computes velocity as a plain per-frame difference with no
`dt`, so sparse sampling scales the motion features down.

A representative capture measured **0.20 hand-scales/frame against 0.72 in the
corpus** — 3.6× less motion reaching the model.

`4d5d045` adds a second line to the result card reporting the profiler's own
numbers (effective fps, delivered frames, per-frame analysis time). The existing
"Captured 32 frames" line could never show this: the model always receives 32
points, so a starved capture and a full one read identically.

## On-device verification — 2026-11-21, Vivo V2404 (Android 16)

Device reconnected; the fixed build was installed (`lastUpdateTime 09:05:58`)
and driven with `adb shell input` + `uiautomator dump`.

**Clip path — all correct, held-out clips included:**

| clip | result | score | gate | infer |
|---|---|---|---|---|
| DARK | DARK1 | 73.4% | accepted | 65 ms |
| MECHANIC | MECHANIC1 | 89.3% | accepted | 32 ms |
| VOICE | VOICE | 96.9% | accepted | 19 ms |

**Camera path — three consecutive real recordings, one held fist:**

| capture | fps (kept) | model fps | hand frames | result | score | margin | agree |
|---|---|---|---|---|---|---|---|
| 1 | 5.1 | — | 32/32 | RAZOR2 | 0.506 | 0.249 | 6/9 |
| 2 | 4.6 | — | 30/32 | RAZOR2 | 0.794 | 0.733 | 9/9 |
| 3 | 8.1 | — | 27/32 | RAZOR2 | 0.802 | 0.728 | 9/9 |

Before the fix the same setup produced 1/9 member agreement at a 0.061 margin.
The camera path now reaches 9/9 at 0.73. The sign was a fist, not a lexicon
sign, and the model answered RAZOR2 confidently — the known "a held hand that is
not signing still gets a named gloss" limitation, not a regression.

**Throughput — the analyser, not the camera, is the limit.** The profiler
measures the real rate directly rather than inferring it from frame counts:

```
profile fps=16.1 interval_p50=62 analyse_p50=59 convert_p50=1
        detect_p50=58 detect_p95=70 delivered=7302 dropped=4379 limit=ANALYSER
profile fps=17.5 interval_p50=57 analyse_p50=51 convert_p50=1
        detect_p50=50 detect_p95=68 delivered=7522 dropped=4425 limit=ANALYSER
```

The detector takes 50–70 ms per frame on the GPU delegate and the camera delivers
every ~62 ms, so `limit=ANALYSER`: the camera produces frames faster than the
detector consumes them, and 4,379–4,425 of them are discarded by
`STRATEGY_KEEP_ONLY_LATEST`. `convert_p50=1 ms` rules out the YUV→Bitmap copy.

Why 5 fps kept against 16 fps analysed: `SignSegmenter` trims the recording to
the motion, and the kept interval is shorter than the span the frames were
spread over. The 16 fps figure is the analyser's own rate; the 5 fps figure is
the kept window after trimming.

The open lever is therefore the detector cadence, not resolution or the copy. At
~16 fps analysed the window already samples close to the corpus's ~8.7 fps, so
raising it further buys little; **lowering** `detect` cost would.

## What is NOT yet verified

- ~~No post-fix device run.~~ Verified on device 2026-11-21; see above.
- Analyser throughput has not been raised. The measurement above says the
  detector cadence is the limit (`limit=ANALYSER`, 4,400 frames dropped per
  capture), not resolution and not the YUV copy.
- Live framing is still selfie-portrait with the face filling much of the frame.
  The corpus is landscape, third-person, full signing space. No code change
  addresses this; it needs either a landscape capture mode or demo guidance.

## Reproducing

```
.scratch/camera_effect_ablation.py 250   # per-effect table
.scratch/combined_sim.py 250             # all effects together
.scratch/full_ablation.py                # 85.45% + magnet-class check
.scratch/verify_fix.py                   # before/after on the real capture
.scratch/aspect_regression.py            # no-regression check on clips
```