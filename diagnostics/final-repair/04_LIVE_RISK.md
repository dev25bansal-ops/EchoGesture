# 04 — The live-only risk I could quantify, and the one I could not

## The defect

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

## The experiment that would settle it

With a phone attached:

1. Enable capture diagnostics.
2. Perform one sign.
3. Read the `profile` record written by `DiagnosticsLog.recordProfiler`:
   `fps= interval_p50= analyse_p50= convert_p50= detect_p50= delivered= dropped=`.

If `fps` reads at or above ~20 and `dropped` is low, this risk is closed and
recognition problems lie elsewhere. If `fps` reads near 10, this becomes the
leading explanation and the fix is to resample by timestamp below the budget as
well — with the measured before/after that a change of that size requires.