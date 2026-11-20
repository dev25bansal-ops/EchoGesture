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

## What is NOT yet verified

- **No post-fix device run.** Wireless debugging was unreachable for the whole
  session (`adb devices` empty through repeated server restarts and mDNS scans).
  The aspect fix is verified offline end-to-end through the shipped members and
  by 150 unit tests; it has not been run on the phone.
- Analyser throughput has not been raised. The card now measures it; the fix
  (lower analysis resolution, or `STRATEGY_KEEP_ONLY_LATEST` plus fewer frames
  analysed) is a separate change that needs that measurement first.
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