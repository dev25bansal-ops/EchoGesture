# 04 — The live-only risk, and the measurement that now exists

## First measurement of the real analyser rate — it does NOT support the hypothesis

A real analyser-rate record has now been produced (on the emulator, since no
phone was available). One capture, one `profile` line:

```
frames=32 dur_ms=12061 fps=29.5 hand_frames=0 ...
profile fps=29.4 interval_p50=34 interval_p95=35 analyse_p50=13
        convert_p50=1 detect_p50=12 detect_p95=22 delivered=684 dropped=0
        limit=CAMERA
```

Reading it:

| Quantity | Value | Meaning |
| --- | --- | --- |
| `fps` | **29.4** | The analyser is running at roughly camera rate |
| `interval_p50` | 34 ms | Frames arrive every ~34 ms, i.e. ~29 fps |
| `analyse_p50` | 13 ms | Total per-frame work is 13 ms of a 34 ms budget |
| `convert_p50` | 1 ms | YUV→Bitmap is not the cost |
| `detect_p50` | 12 ms | MediaPipe is not the cost |
| `dropped` | 0 | Nothing was discarded |
| `limit` | **CAMERA** | The camera, not the app, sets the pace |

**At 29 fps a median 2.2 s sign yields ~64 frames, not 20.** That is well above
the 32-frame budget, so the pass-through path below is *not* taken and the
timestamp sampler runs normally.

This **weakens the leading live hypothesis** rather than confirming it. On a
device that analyses at 30 fps, the frame-budget theory does not bite.

Two honest caveats: this is an **emulator**, whose synthetic camera may hand
frames over differently from a real sensor; and the emulator never contains a
hand, so the detector cost here is for empty frames and is a floor, not a
realistic figure. A phone running MediaPipe on real hands may be slower. The
record above is a real measurement of this device, not of the user's.

**It does not close the question.** It means the question must be asked again on
a phone. It no longer licenses the conclusion that a low frame budget is the
cause.

---

## The defect itself — unchanged and still unpatched

`RecordingFeatures.sample()` (RecordingFeatures.kt:40-49):

```kotlin
val frames = recording.frames.subList(seg.startIndex, seg.endIndex)
if (frames.size <= SAMPLED) return frames          // <- pass-through
val idx = HandTracker.evenlySpacedIndices(
    LongArray(frames.size) { frames[it].capturedAtMs }, SAMPLED)
return idx.map { frames[it] }
```

Above 32 frames, selection is **by timestamp**, so irregular arrival is
absorbed. At or below 32 the frames are returned as they are, and the caller's
linear resample to 24 steps (FeatureTransform.kt:81-102) then treats
irregularly-timed frames as if they were evenly spaced.

A dropped analyser frame therefore makes one model step span 80 ms of motion
and the next 40 ms. On a live phone this is normal; on a constant-rate training
clip it never happened.

Note this is **not** a divergence from Python. `landmark_frontend.read_frames`
also returns short clips unpadded and lets the same linear resample stretch
them. The Android code is faithful here. The exposure is a property of live
capture, not of the port.

## How much it matters depends on one unknown

Whether a live capture falls below the threshold depends entirely on the real
analyser rate, which was never measured. Using the real durations of the
validation clips:

| Clip duration (median 2.17 s) | At 10 fps | At 15 fps | At 25 fps |
| --- | --- | --- | --- |
| Frames captured | **~20** | ~33 | ~54 |
| Below the 32-frame pass-through? | **yes** | borderline | no |

Across ten sampled clips:

| Analyser rate | Clips taking the unresampled path |
| --- | --- |
| 10 fps | **10 / 10** |
| 15 fps | 5 / 10 |
| 25 fps | 0 / 10 |

## What was measured instead

The one thing measurable without a phone is how detection degrades as the frame
budget shrinks. Replaying real clips through the real detector at a reduced
budget:

| Clip | hands @32 | @16 | @10 | @6 |
| --- | --- | --- | --- | --- |
| BEE | 13 | 6 | 3 | 2 |
| CALENDAR | 15 | 7 | 6 | 2 |
| DEAF | 18 | 8 | 5 | 3 |
| DEMAND | 20 | 9 | 6 | 3 |
| DOWNSIZE | 17 | 8 | 6 | 3 |
| ELEVATOR | 21 | 10 | 6 | 3 |
| FINE | 15 | 7 | 4 | 3 |

Hand-bearing frames fall roughly in proportion to the budget. At a 10-frame
budget only 3–6 of 24 model steps carry a real observation; the rest are
interpolation. That is a large loss of signal, and it happens **before** the
model is asked anything.

## Why no code change was made

The brief is explicit: *"Do not apply five speculative fixes simultaneously"*
and *"Each confirmed defect requires a reproduction case."*

I can show the pass-through exists and show that a low frame budget costs
detections. I **cannot** show that a real phone actually operates below the
threshold, because that requires measuring the analyser rate on a phone. Adding
a timestamp resample below the budget would change behaviour for every capture
without evidence that it helps — and because it would change the input
distribution relative to training, it could make a working path worse.

So it is documented, not patched.

## Why the profiler wrote nothing — three separate causes

Chasing this produced three findings, and the third is the one that mattered.

**Cause 1: it was never called.** An earlier commit ported `AnalysisProfiler`
but never invoked it. It compiled, it appeared in the source tree, and it
recorded nothing.

**Cause 2: the wrong function was instrumented.** `logDiagnostic()` looked like
the obvious place to flush a snapshot, and it is a real function that really
writes capture records. But the no-hand capture path that fires on this hardware
goes through `logRecordingUnusable()`, which calls `diagnostics.record()`
**directly** and never enters `logDiagnostic()` at all. Traces added to
`logDiagnostic()` could not fire — and did not.

**Cause 3: Gradle's build cache.** `./gradlew :app:assembleDebug` reported
`compileDebugKotlin FROM-CACHE` after source edits, so an APK was built from
stale classes. A change that "had no effect" had in fact never been compiled.
`--no-build-cache` is required for a trustworthy build after editing Kotlin.

The fix is in `logRecordingUnusable()`: it now flushes the profile as well, and
records an explicit `profile_unavailable` note when the tracker is gone rather
than leaving a silent gap.

Verified end to end on the device — the `profile` line in the section above is
the output of that fix.

## The experiment that would settle it

With a phone attached:

1. Settings → enable **Capture diagnostics** (the toggle is real and writes
   `diagnostics = true`; verified on the emulator).
2. Perform one sign.
3. Read the record:

```bash
adb shell run-as org.echogesture.finalapp cat files/diagnostics.txt
```

The `profile` line gives:

```
profile fps= interval_p50= interval_p95= analyse_p50= convert_p50= detect_p50=
        detect_p95= delivered= dropped= limit=
```

If `fps` reads at or above ~20 and `dropped` is low, this risk is closed and
recognition problems lie elsewhere. If `fps` reads near 10, this becomes the
leading explanation and the fix is to resample by timestamp below the budget as
well — with the measured before/after that a change of that size requires.

`limit=` names which side is the constraint, so the answer is direct rather than
inferred: `ANALYSER` means the app is too slow, `CAMERA` means the sensor is
delivering slowly and no app-side change would help.