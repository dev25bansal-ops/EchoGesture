# Live recognition: diagnosis and fixes

Branch `hackathon/echogesture-v2-rc`. No model, weight, detector, label or
feature-semantic byte was changed.

## What was actually measured

Two consecutive captures on an Android 14 emulator, with **no hand in frame**,
logged by the new diagnostics mode:

| Capture | Frames | Duration | Effective analysis FPS |
| --- | --: | --: | --: |
| First (cold, models still loading) | 32 | **3173 ms** | **9.8** |
| Second (warm) | 32 | **1031 ms** | **30.1** |

Same code, same device, same content — and the capture window was **three times
longer** in the first case.

That is the central finding. The app captured a fixed **32 frames**, and treated
that as if it were a fixed span of time. It is not: the window's real duration is
`32 / achieved analysis FPS`, and that rate depends on device load, thermal state
and what else is running. The classifier therefore saw the same physical gesture
compressed into a different temporal scale on almost every capture.

## Root cause

**The model was trained on whole clips; the app feeds it a window whose duration
is uncontrolled.**

`src/landmark_frontend.py:read_frames()` samples 32 frames **spread across the
entire clip** — one documented example was a 110-frame clip, about 3.7 s at
30 fps. The feature matrix is 24 frames with **no timestamp channel** (255
features: positions, velocities, accelerations, inter-hand geometry). The model
therefore learned rates of change measured against whatever wall-clock span its
input covered.

The app took 32 **consecutive** frames from the moment of the tap, so the span
it presented was whatever the analyser happened to achieve. At 9.8 fps that is
3.2 s; at 30 fps it is 1.0 s. Velocities and accelerations are computed in
per-frame differences, so the same gesture produced features differing by up to
3× in their temporal derivative depending on device state.

This is a pipeline defect, not a model-quality question. It explains a *variable*
score across attempts on the same sign better than it explains a consistently
wrong one.

### A correction to my earlier claim

In my previous turn I reported that Android assigned hand slots by horizontal
position while training assigned them by handedness, and that this swapped slots
on the front camera. **That was wrong.** I had misread a grep. The actual code,
`HandTracker.toHands()`:

```kotlin
val slot = if (handedness.equals("Left", ignoreCase = true)) 0 else 1
```

keys on MediaPipe's handedness **label**, exactly as `landmark_frontend.py` does.
There is no slot-swap bug. I am recording the error here rather than quietly
dropping it.

## Other findings, with evidence

**Presence is not what the pill implies.** The live pill said "Searching for
hands" / "Hands detected" based on the ring, but the capture result never
reported its own coverage. A capture with a hand in 2 of 32 frames produced a
full six-significant-figure prediction like any other.

**"43 ms on device" was inference only.** It is the sum of four TFLite
invocations. It excludes the capture window (1–3 s), bitmap conversion and hand
tracking. The label invited it to be read as end-to-end.

**Feature construction is correct.** `FeatureTransform` is fixture-asserted
against `src/features.py` to 1e-4 on two clips, and the 32→24 resample matches
`_resample()` exactly (`np.linspace(0, t-1, target)`). The pipeline from
landmarks to features is not the problem.

**The score was presented as certainty.** `renderResult` printed the gloss, read
it aloud when speech was on, and coloured a chip — with no indication that a
0.35 score and a 0.90 score are very different evidence. Three strings named
`result_confidence_high/medium/low` existed and were **used by nothing**.
Separately, `result_responsibility` asserted "A model score is a probability",
which is exactly the claim that must not be made.

## Fixes, all software, all restricted to this pipeline

1. **Capture is now time-based.** `HandTracker` records a real
   `SystemClock.elapsedRealtime()` timestamp per frame and ends the capture when
   it has at least `RING_CAPACITY` frames **and** covers `MIN_CAPTURE_MS`
   (1000 ms), bounded by `MAX_CAPTURE_MS` (4000 ms) and `MAX_CAPTURE_FRAMES`
   (96) so a stalled analyser cannot hold it open. Verified: warm captures now
   report ~1031 ms instead of an unknown span.

2. **Real capture facts are reported.** `Capture` carries frame count, duration,
   achieved FPS and hand-frame count, and the result card shows all four:
   `32 frames · 1031 ms captured · 30 fps · 43 ms inference`.

3. **Insufficient hand coverage is its own state.** A capture with fewer than
   `MIN_HAND_FRAMES` hand-bearing frames now returns `LOW_FRAMES` with an
   explanatory card instead of classifying. Zero hand frames still returns
   `NO_HANDS`. Both are logged.

4. **Confidence is labelled and speech is gated.** See the calibration section.

5. **`result_responsibility` no longer claims probability.**

6. **Diagnostic mode.** Off by default, in Settings. Writes only frame counts,
   duration, FPS, per-frame presence flags, predicted label, score, top-5 and
   inference time to `files/diagnostics.txt`, capped at 500 records. It writes
   **no images, no landmarks, no identifiers**, and the app has no internet
   permission, so the file cannot be uploaded even in principle. Turning the
   switch off deletes it.

## Confidence calibration

Derived from the frozen held-out ensemble predictions on the ASL Citizen
validation split — 346 clips, 5 unseen signers — via
`evaluation/device/recognition_diagnosis/calibrate.py`. 334 map to a shipped
class label; 11 do not map and are excluded rather than guessed.

| score ≥ | kept | coverage | top-1 on kept |
| --: | --: | --: | --: |
| — | 334 | 100% | **74.85%** |
| 0.20 | 260 | 77.8% | 86.92% |
| 0.25 | 237 | 71.0% | 90.30% |
| 0.30 | 208 | 62.3% | 93.27% |
| **0.35** | **191** | **57.2%** | **93.72%** |
| 0.40 | 171 | 51.2% | 95.91% |
| 0.50 | 146 | 43.7% | 97.26% |

74.85% reproduces the independently recomputed 75.36% in `VAL_RESULT.md`
(the difference is the 11 unmapped clips), which is why this table is trusted.

**The app uses 0.35.** Below it, the card says *"Uncertain — this is the model's
best guess, not a confirmed reading"* and **does not speak the result aloud**.
At or above it, the card says *"Best match — still confirm before relying on
it"*. 0.35 was chosen because it is where held-out top-1 crosses ~94% while
still keeping 57% of captures — not by eye.

This is a ranking score, not a calibrated probability, and the app now says so.

## What remains unproven

- **No physical phone was attached to this machine**, so nothing here is a
  real-camera result. The EAT1→TAIL1 and DEAF1→DOG1 observations are yours, not
  reproduced here.
- **Whether recognition accuracy is now sufficient is unknown.** The capture fix
  addresses a real defect; it is not a demonstrated accuracy improvement. That
  requires re-running your sign tests on a phone.
- **Frame-drop robustness is a documented weakness.** `FINAL_EVIDENCE.md`
  records 13.33 pt degradation under frame_drop at severity 0.5 — directly
  relevant, since variable analysis FPS is a form of frame dropping.

## What to run next, on the phone

1. Enable **Capture diagnostics** in Settings.
2. Record your sign tests again, at least 5 signs, 3 attempts each.
3. Retrieve the log:

```bash
adb shell run-as org.echogesture.finalapp cat files/diagnostics.txt
```

For each row compare `hand_frames` against `frames`. If `hand_frames` is low,
the capture missed the sign and that is a capture problem. If coverage is high
and the prediction is still wrong, the pipeline saw the gesture and the model
misclassified it — that is a model question for Dev, and the record now
separates the two.