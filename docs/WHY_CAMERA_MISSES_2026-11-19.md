# Why the camera misses what the clip gets

Written 2026-11-19, in answer to: *"when uploading the clip or video it is
detecting correct, so why can't it predict when we are using the camera?"*

> **SUPERSEDED 2026-11-20 by `CAMERA_FRAME_DUMP_2026-11-20.md`.**
>
> This document concluded that the cause was domain shift in the image — the
> hand's scale, angle and background. **That conclusion is wrong.** A real
> camera frame dump was classified through the shipped pipeline and scored
> **VOICE at 0.9678 with 9/9 member agreement**, from a hand in front of a
> laptop screen.
>
> The camera's pixels are fine. What fails is **frames per capture**: one
> measured capture delivered 4 frames over 1.5 s at 1.9 fps, and another showed
> the hand present for only 11 of 32 frames with the analyser dropping 29%.
>
> The refutations below (§2a–§2d) still stand — frame rate, frame count,
> timestamps and tracking mode genuinely do not change the answer on clip
> pixels. What does not stand is the jump from those refutations to "the pixels
> are the problem". Read this document for the refutations, not for its
> conclusion.

Short answer: **the model is fine and the camera path is fine. What differs is
the pixels, and specifically how big and how posed the hand is.** Everything
that sounded like a plausible mechanism was tested and refuted. This document
records each refutation so nobody re-runs them.

## 1. The facts to explain

| input | result |
|---|---|
| NOON clip through `predict()` | NOON1, 0.9731 |
| NOON clip through the app's clip picker | NOON1, 0.957 |
| NOON signed live at the camera | FOREIGNER1, 0.51–0.60, NOON1 ranked 2nd/4th |

The camera is not blind: the failing captures had `hand_frames=32/32` and
`30/32` — denser hand coverage than the corpus median (14/32). The detector
found the hand in every frame. So the loss is downstream of detection.

## 2. Four candidate mechanisms, all refuted

### 2a. The analyser frame rate — REFUTED

The captures ran at 6.8 and 16.3 fps against the clip's 30 fps. Resampling the
clip to those rates through the real pipeline:

| input | prediction | confidence |
|---|---|---|
| native 30 fps | NOON1 | 0.9731 |
| 15 fps | NOON1 | 0.9629 |
| 7.5 fps | NOON1 | 0.9549 |
| every 3rd frame (25 frames) | NOON1 | 0.9786 |
| every 7th frame (11 frames) | NOON1 | 0.9800 |

**Eleven frames still scores NOON1 at 0.98.** The model is essentially
insensitive to frame rate on this sign.

### 2b. The frame count — REFUTED

First 32 frames of a 74-frame clip: NOON1 0.9236. All 74: NOON1 0.9789.
32 frames spread across the whole clip: NOON1 0.9772. Any sampling keeps it.

### 2c. The camera's synthetic 33 ms timestamp — REFUTED

`HandTracker.analyse()` calls `detectForVideo` at `nextFrameTs.getAndAdd(33)`,
a fixed 33 ms step, even when the analyser is really delivering a frame every
59–147 ms. The hypothesis was that MediaPipe's tracker uses that timestamp for
motion propagation, so the camera describes a hand moving 3–4× faster than it
really did.

Tested by running identical pixels through `detectForVideo` with different
timestamp steps:

| pixels | told step | result |
|---|---|---|
| 30 fps | 33 ms | NOON1 0.9236 |
| 30 fps | 143 ms | NOON1 **0.9236** |
| 10 fps | 33 ms | NOON1 **0.7792** |
| 10 fps | 100 ms | NOON1 **0.7792** |

**The timestamp provably does not matter** — rows 1/2 and 3/4 are bit-identical.
MediaPipe 0.10.14's per-frame detector does not use the timestamp for motion on
this path. (The 10 fps rows do score lower, but §2a shows that is the shorter
window, not the rate: 11 frames spread across the whole clip scores 0.98.)

### 2d. VIDEO-mode tracking vs IMAGE-mode detection — REFUTED

The camera uses `RunningMode.VIDEO` with a persistent tracker, so landmarks are
propagated frame to frame and can drift; `predict()` uses IMAGE mode and
redetects each frame. On the same 32 frames:

| mode | result |
|---|---|
| VIDEO (camera) | NOON1 0.9236 |
| IMAGE (clip path) | NOON1 0.9120 |

Landmarks differ (mean 0.037, p95 0.173 in normalised units) but the answer does
not change. Not the cause.

## 3. What is actually left: the hand in the picture

Four mechanisms eliminated leaves the pixels. The refutations above all hold the
*clip's* pixels fixed — a well-framed hand filling a sensible portion of a
640×480 frame, shot at the corpus's own distance and angle. The camera has none
of that:

- **Scale.** The frame guide invites a hand into a box that is a fraction of the
  preview. The feature transform normalises by the wrist→middle-MCP distance, so
  absolute size cancels — but MediaPipe's *detection quality* degrades with a
  small hand, and its landmark noise does not cancel with it.
- **Pose and angle.** The corpus is shot by a fixed camera with the signer facing
  it. A phone held in one hand, signing with the other, is a different viewing
  geometry. Landmarks are 3-D but the z is very noisy at short range.
- **Background and lighting.** Normalised coordinates discard background, but
  detection confidence and landmark precision do not.
- **What is in frame.** A corpus clip contains the signer's torso and the signing
  space consistently. A phone camera pointed at a hand sees whatever else is
  around it.

That is the honest answer: **this is a domain shift in the input, not a bug in
the code.** The clip path is the same model on the distribution it was trained
and measured on; the camera is the same model on a distribution nobody measured.

## 4. What would settle it, and what it would take

The measurement that would make this precise is a **camera-frame dump**: have
the app save the exact `ImageProxy` bitmaps from one capture, then run those
pixels through `predict()` offline and compare the landmarks to a corpus clip's.
That isolates "the detector produced different landmarks" from "the classifier
misread the same landmarks", which is the only remaining fork.

Partially built and deliberately not enabled: this adds a debug-only write of
raw camera frames to disk, which is exactly the thing an accessibility tool
should not ship enabled, and it needs the phone connected to drive a capture.

`docs/ANALYSER_RATE_2026-11-19.md` §"What would settle it" already proposed the
same instrument; this document is the third independent route to the same
conclusion, which is itself evidence that no cheaper measurement exists.

## 5. What this means for the submission, honestly

- **Do not claim camera performance.** The measured 85.45% is corpus test-split
  accuracy. No live-camera accuracy has been measured, and the two data points
  that exist (FOREIGNER1 for a NOON sign) say it is worse.
- **Do claim the clip path.** It is verified end to end on the phone: four clips,
  correct, through the shipped nine-model ensemble.
- **Say the limitation out loud.** The demo script's Beat 6 already carries the
  weak-sign table. The camera/clip gap is the same class of finding — a known,
  measured limit — and stating it is stronger than a judge discovering it.

## Reproduce

```
python .scratch/timestamp_test.py
```
(Fresh landmarker per timestamp regime; the cached one rejects a timestamp reset
with `Input timestamp must be monotonically increasing`, which is a harness
constraint, not a finding.)
