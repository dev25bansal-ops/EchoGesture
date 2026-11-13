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

**The model was trained on whole clips; the app feeds it a window whose temporal
step is uncontrolled — and, even at a controlled rate, roughly 3.5× too fine.**

`src/landmark_frontend.py:read_frames()` samples 32 frames **spread across the
entire clip** — `np.linspace(0, total - 1, 32)`. Its own docstring records a
typical ASL Citizen clip as **110 frames, about 3.7 s at 30 fps**. So training's
consecutive samples are about **115 ms apart**.

`src/features.py:160-162` computes motion as a plain per-frame difference with no
time term:

```python
vel[1:] = norm[1:] - norm[:-1]
acc[2:] = norm[2:] - norm[:-2]
```

The step between samples *is* the time base. There is no `dt`, and the feature
matrix (255 dims: pos, vel, acc, geometry) has no timestamp channel, so nothing
downstream can correct the scale.

The app took 32 **consecutive** camera frames — about **33 ms apart** — covering
only ~1.0 s of the sign. Two defects followed from one mistake:

1. **Wrong time base.** Every velocity was ~3.5× smaller than training saw and
   every acceleration ~12× smaller (ratio of a 115 ms step to a 33 ms step).
   The model was shown a third of the sign played at unit time.
2. **Unstable time base.** Because the window ended at a frame *count*, its real
   span was whatever the analyser achieved. Measured on-device: the same 32
   frames took **1031 ms** warm and **3173 ms** cold — a 3× swing between
   captures of the same code.

This is a pipeline defect, not a model-quality question. It explains a
*variable* score across attempts on the same sign better than it explains a
consistently wrong one.

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

1. **Capture is now a span of real time.** `HandTracker` records a real
   `SystemClock.elapsedRealtime()` timestamp per frame, ends the capture after
   `CAPTURE_SPAN_MS` (3700 ms, matching the documented training clip), and then
   selects `RING_CAPACITY` (32) frames **evenly across that span by timestamp**
   — `evenlySpacedIndices`, the same `linspace` selection `read_frames()` makes
   over a clip. Every selected frame is a real analysed frame; nothing is
   interpolated or invented. That puts the sample step back near training's
   ~115 ms instead of the analyser's ~33 ms, and makes the span independent of
   device speed.

   Verified on the **final shipped APK** (clean build `9bba9191…`, installed on
   the Android 14 emulator): a capture now reports **`dur_ms=3695` at 10.0 fps**.
   Before the fix the same 32-frame count covered **1031 ms** warm and
   **3173 ms** cold on the same code.

   The ring keeps the whole span (bounded at `MAX_CAPTURE_FRAMES = 200` purely
   as a memory backstop); the frame count no longer decides when capture ends.

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
  reproduced here, and they were made on the pre-fix build.
- **Whether recognition accuracy is now sufficient is unknown.** The capture fix
  addresses a real defect; it is not a demonstrated accuracy improvement. That
  requires re-running your sign tests on a phone.
- **Robustness is uneven, and the worst axis is crop, not frame drop.**
  `submission_final/RELEASE_MANIFEST.json` (30 clips, 5 signers, seed 42) records
  crop **10.0 pt** and occlusion **6.67 pt** degradation, while frame_drop and
  motion_blur measure **0.0** and low_light **−3.33** (i.e. it improves).
  An earlier revision of this document quoted 13.33 pt for frame_drop from the
  older `FINAL_EVIDENCE.md`; that figure is superseded and was wrong to repeat.

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

## Sign-test table

**Cannot be filled here.** These columns require a physical phone running your
signs; the machine this was built on has no camera and no phone attached. No row
below is a measurement — the table is the template, and I will not invent values
into it.

| # | Expected label | Predicted label | Top-5 | Score | `hand_frames`/`frames` | `dur_ms` | Verdict |
| --: | --- | --- | --- | --: | ---: | --: | --- |
| 1 | EAT1 | — | — | — | — | — | *not run on hardware* |
| 2 | DEAF1 | — | — | — | — | — | *not run on hardware* |

Your two earlier observations cannot be used as before/after rows either: they
were recorded on the pre-fix build, which had no diagnostics, so there is no
`dur_ms` or coverage figure for them.

## Recommendations for Dev

The capture fix removes a real pipeline defect. It does **not** establish that
the model generalises to live phone footage, and two known weaknesses make a
remaining accuracy gap plausible:

1. **Training and inference see different sampling rates.** Training samples
   across a whole clip; the app now samples across a fixed 3.7 s window. Signs
   performed faster or slower than the training distribution will still land on
   the wrong time base. **Suggested:** train with random temporal-span
   augmentation (`linspace` over a randomly truncated span, plus resampling to
   a random step in ~60–200 ms) so the model stops depending on one step size.
2. **Crop and occlusion are the measured weak axes** — 10.0 pt and 6.67 pt
   (`submission_final/RELEASE_MANIFEST.json`). Frame drop measures 0.0, so
   variable-rate sampling is *not* the priority the earlier draft of this
   document implied. **Suggested:** augmentation that changes framing and
   self-occlusion, and re-measurement on the held-out split rather than only
   the synthetic corruption suite.
3. **Per-signer generalisation is the dominant error source** if the failures
   are on a signer unlike the training set. **Suggested:** report top-1 by
   signer on the validation split; if your two misclassifications are
   consistent, they point at signer- or camera-domain shift rather than noise.
4. **Camera domain shift.** ASL Citizen is recorded with a fixed setup; a
   handheld or laptop camera has different framing, lighting and lens. Nothing
   in the pipeline corrects for it and I have not changed any model feature.
   **Suggested:** if the table shows high coverage with consistently wrong
   labels, add a small amount of phone-camera-style augmentation (perspective,
   illumination, white balance) and re-evaluate — a training change, which needs
   your and Dev's approval before I touch anything.

Until the table above is filled on a phone, the honest statement is: the
capture defect is fixed and measured; recognition accuracy on your phone is
**not yet established**.
---

# Round 2 — no-retrain investigation

Branch `hackathon/echogesture-v2-rc`. No model, weight, detector, label,
vocabulary, feature dimension, ensemble order or feature-semantic byte changed.

## Android VIDEO mode vs Python IMAGE mode — parity is good

Python's `detect_hands(mode="video")` runs the same MediaPipe HandLandmarker in
VIDEO mode (a tracker carried across calls); Android runs VIDEO mode on every
camera frame. To test whether that difference costs accuracy, the same
authorized ASL Citizen clips were decoded once and pushed through both modes in
**separate processes** (the shared `_LANDMARKER` singleton silently ignores its
`mode` argument on later calls, which produced a false "VIDEO finds 0 hands"
result in a first attempt — recorded here so it is not repeated).

| Clip | Hands image / video | Top-1 image | Top-1 video | Score image / video |
| --- | --- | --- | --- | --- |
| EAT | 29 / 29 (of 75) | EAT1 | EAT1 | 0.651 / 0.676 |
| WHAT FOR | 29 / 28 (of 116) | WHATFOR1 | WHATFOR1 | 0.681 / 0.586 |
| DOG | 34 / 35 (of 96) | DOG1 | DOG1 | 0.211 / 0.208 |
| BASKETBALL | 40 / 40 (of 90) | BASKETBALL1 | BASKETBALL1 | 0.258 / 0.328 |

**Identical top-1 in 4/4 clips, hand counts within ±1.** Feature tensors differ
(mean |Δ| 0.017, max 0.92 on a normalized scale) but not enough to move the
decision. **Android's tracker is not the cause of the phone failures.**

Left/right slots key on MediaPipe's handedness label in both code paths
(`HandTracker.toHands()` and `landmark_frontend.detect_hands()`), never swapped,
and neither pipeline mirrors. Rotation is applied exactly once, via
`ImageProcessingOptions.setRotationDegrees`.

## The model generalizes on authorized footage

`src/main.py`'s public `predict()` was run on 12 in-vocabulary ASL Citizen
validation clips (the split contains 10,304 clips, of which only **346** map to
the 100 shipped labels — the other 9,958 are out of vocabulary and cannot be
scored, which invalidated an earlier 26-clip sample that was 96% out-of-vocab):

| Expected | Predicted | Score |
| --- | --- | ---: |
| EAT1 | **EAT1** | 0.529 |
| DEAF1 | **DEAF1** | 0.457 |
| WHATFOR1 | **WHATFOR1** | 0.625 |
| DOG1 | **DOG1** | 0.195 |
| BASKETBALL1 | **BASKETBALL1** | 0.310 |
| plus 7 more (PATIENT2, GUESS1, RIVER1, SHAVE1 ×2, ELEVATOR1 ×2) | all correct | 0.101–0.755 |

**12/12 correct; 6/6 correct above the 0.35 cut.** Both signs the phone got
wrong — EAT1 and DEAF1 — classify correctly here, and DEAF1's top-5 does **not**
contain DOG1.

This is the decisive result: the pipeline is right and the model works on
in-domain footage. The phone failures are a domain gap, not a code defect.

## What was actually fixed this round

1. **Stale frames excluded from the window.** A frame is now timestamped when it
   *arrives*, and a frame that arrived before the tap is dropped even if its
   detection finished after the window opened. Previously `capturedAtMs` was
   taken after `detectForVideo()` returned, so an in-flight frame analysed
   across the tap was credited to the capture.
2. **Dead stall bound removed.** The old "too long" guard was computed from the
   frame cap (200 × 33 ms = 6.6 s) and OR'd with the 3.7 s span check, so it
   could never fire first. Capture now ends on the measured span.
3. **Coverage floor is measured, not guessed.** `MIN_HAND_FRAMES = 3`
   (9% of a 32-frame window) was replaced by `MIN_HAND_FRACTION = 0.15`. On
   correctly classified clips coverage ranged 7/32 (22%) to 21/32 (44%); the old
   floor admitted windows far emptier than any that worked. A fraction is used
   because the sampled window can be shorter than 32 on a slow device.
4. **Capture duration and classifier latency are separate lines.** The old
   single line ("32 frames · 1031 ms captured · 30 fps · 43 ms inference")
   invited the 43 ms to be read as end-to-end. They are now two labelled
   measurements.
5. **Live capture countdown.** The capture is a fixed 3.7 s span; the status
   line now counts it down so the signer knows how long to keep performing.

## What is still unproven

- **The phone failures are not reproduced here.** No phone was attached. That
  the model is correct on clip footage does **not** prove it is correct through
  a handheld camera; the evidence points to a domain gap (framing, distance,
  lighting, lens, handheld motion, mirroring of the preview) rather than to a
  pipeline bug.
- **No accuracy improvement is claimed.** The fixes above are correctness and
  honesty fixes. Whether they change your phone results is unknown until you
  re-run the same signs.
