# AUTO_CAPTURE_IMPLEMENTATION

## The state machine

`AutoSegmenter` (398 lines, pure Kotlin, no Android types, caller supplies each
frame with its real arrival time):

```
IDLE -> ARMED -> SIGNING -> PROCESSING -> RESULT / RETRY
```

Observed sub-states: ONSET is confirmed while still ARMED, and END is confirmed
by an internal candidate window with hysteresis, so the observable states the UI
renders are ARMED and SIGNING.

The brief's finer split (ONSET_CANDIDATE, END_CANDIDATE, FINALIZING) exists as
internal timing, not as separate UI states. That is deliberate: each extra screen
state is a place for the button label and the live status to disagree, and the
user needs to know only "waiting" or "signing".

## Timing (measured values, not tuned to one clip)

| Parameter | Value | Why |
| --- | --- | --- |
| `acquireMs` | 300 ms | one noisy detection must not arm a capture |
| `poseHoldMs` | 450 ms | a held handshape counts as a sign start |
| `quietHoldMs` | ~700 ms | end must be sustained, not one still frame |
| pre-roll | ~200 ms | the sign's first handshape is not cut off |
| `maxSignMs` | 10 s | bounded attempt |
| `MAX_BUFFERED_FRAMES` | 600 | hard memory bound (~20 s at 30 fps) |

All are **milliseconds**, never frame counts: the analyser rate varies by device.
A 5 fps stream and a 30 fps stream both behave correctly — pinned by
`slow 5 fps input still segments without a frame-count trigger`.

## The motion bar is relative, not absolute

Landmarks are normalised by hand size, so a fixed displacement threshold means a
different physical movement depending on camera distance. The noise floor is
measured **while the signer is waiting** and frozen at sign start. Measuring it
during the sign would be self-defeating: a signer moving steadily would raise the
floor and their own sign would look quiet.

## Static signs

A held pose with little movement arms a sign via `poseHoldMs`. A motion-only
onset would never fire on a handshape sign.

## Two-part signs

The end test requires quiet **continuously**. Any motion above the floor
restarts the timer, so the internal pause in a two-stage sign does not end it.

## Double-trim decision — measured, and the answer was NO

§8 of the brief warned that re-running the motion trim over an already-selected
interval could remove the sign's start and end handshapes.

I first "fixed" this by skipping the trim on the one-tap path, based on a Python
stand-in that showed 0.900 vs 0.817. **That was wrong.** Measured through the
real Kotlin `SignSegmenter` on 30 labelled clips:

| Sampling | Accuracy |
| --- | --- |
| With the motion trim | **0.867** |
| As-given, no trim | 0.833 |

The trim **earns its place**, because `SignSegmenter`'s threshold is relative to
the recording's own peak motion and therefore adapts. The change was reverted,
and `alreadySelectedIntervalIsNotHarmedByAReTrim` now pins the direction the
evidence supports so nobody repeats my mistake.

**The lesson is recorded because it cost time:** a crude reimplementation of an
algorithm is not a measurement of that algorithm.

## Buffer cap

`buffer` had no size limit. The 10 s timeout bounded it in time, but a device
delivering faster than expected would grow the list in proportion to frame rate.
Now capped at 600 frames, dropping oldest — safe because only the 32 evenly
spaced samples survive anyway.

## Known limitation

**None of this has been exercised by a person signing into a real phone camera.**
The state machine is proven by 15 JVM tests and 9 instrumentation tests driving
the real `HandTracker`, but the thresholds have never met a live signer's motion.
