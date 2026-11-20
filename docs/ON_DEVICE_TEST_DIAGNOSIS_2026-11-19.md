# Why the on-device replay tests "crashed with no output"

Diagnosed 2026-11-19. Three test classes failed the same way on the Vivo V2404 —
`INSTRUMENTATION_RESULT: shortMsg=Process crashed`, no output, several 30-minute
hangs — and the cause turned out to be three separate things stacked, only one
of which was a real bug.

## What was NOT the problem

Each of these was suspected and then ruled out by a direct probe:

| suspect | probe | result |
|---|---|---|
| MediaPipe broken under instrumentation | create + close a `HandLandmarker` on CPU and GPU, nothing else | **passes in 0.86 s** |
| the 9-member classifier | `ModelContract.load` + `LandmarkClassifier.create` (nine interpreters, `setNumThreads(4)`, XNNPACK) | **48 ms, clean** |
| the decoder deadlocking | bounded `VideoFrames.sample` with progress lines | **works, 24 frames, `completed=true`** |
| test classes missing from the APK | scan the dex for `Lorg/echogesture/...` | **present** |

## What WAS the problem

### 1. Instrumentation stdout is discarded on this device

This is the big one, and it explains the "no output" in every failure report. A
probe that **passes in 0.86 s** and prints four `println` lines produces *zero*
visible output — neither through `am instrument -w` nor in logcat.

`am instrument -w -r` reveals why the test summary is all you get:

```
INSTRUMENTATION_STATUS: stream=
org.echogesture.finalapp.MediaPipeProbeTest:
INSTRUMENTATION_STATUS: stream=.
INSTRUMENTATION_RESULT: stream=
Time: 0.856
OK (1 test)
```

The `stream=` fields carry only the progress dots. Everything the test printed
was dropped. **Consequence: "crashed with no output" was never evidence that the
test produced no output** — it was evidence that this device does not forward it.
Every conclusion drawn from empty output this session was therefore built on a
bad signal.

Fix: tests now write results to `files/<name>.txt` via `run-as`, which survives
regardless of the stdout channel. That is how the classifier probe above was
read.

### 2. The pushed corpus contained clips with no hands in them

`ClipReplayAccuracyTest` and my first parity run both reported `h=0` (zero
frames with a detected hand) on the first clip. That looked like a detector
failure. It was not:

```
009924818599383212-COMB.mp4: python native extraction also finds 0 hands,
                             and the clip decodes to only 5 frames
```

The 40-clip set was chosen as "the 40 smallest `.mp4` files that resolve to a
vocabulary key" — a size filter, not a content filter. Small files are
frequently short or hand-less.

Fix: select by `hand_frac > 0.9` from `probs_test.npz`, giving 10 clips that
certainly contain hands.

### 3. `VideoFrames.sample` pads by repeating frames

When a clip has fewer frames than the 24 requested, the target loop keeps
emitting the *same* decoded frame until it hits the count. So a 5-frame clip
reports `frames=24` — real frames, just duplicated. That inflates the apparent
hand coverage and would quietly corrupt any accuracy number taken through it.

Worth knowing before trusting a rate from this path. The parity test is
therefore run only on clips long enough to fill 24 slots.

## The real bug that was found along the way

The original per-class decode loops had no iteration bound:

```kotlin
while (!sawOutputEos && picked.size < target) { ... }
```

A codec that never emits end-of-stream never exits. That is a genuine hang, and
it is now capped in the shared `VideoFrames` helper, which reports `completed`
so an incomplete decode fails loudly instead of silently.

## The GPU delegate crashes in IMAGE mode — but the app is unaffected

This is the most interesting result, and it needed a second measurement to get
right.

Calling `detect()` on a GPU-delegated `HandLandmarker` in `RunningMode.IMAGE`
takes a hard native crash:

```
Fatal signal 11 (SIGSEGV), code 1 (SEGV_MAPERR), fault addr 0x18
  Java_com_google_mediapipe_framework_Graph_nativeMovePacketToInputStream+60
  Graph.addConsumablePacketToInputStream
  TaskRunner.process
  BaseVisionTaskApi.processImageData
  HandLandmarker.detect
```

Creating the landmarker succeeds; only `detect()` faults. The **CPU** delegate
in the same test context works fine (`DET detect delegate=CPU OK hands=0`).

**But the shipped app does not crash.** Driving a real capture on the device:

```
I HandTracker: HandLandmarker delegate=GPU
fatal_count: 0
pid 14237 still alive
```

The difference is the running mode. The app uses `RunningMode.VIDEO` with its
own GL context; the test used `RunningMode.IMAGE`. So:

- **the shipped app's GPU path is sound** — this probe would have "found" a
  crash that affects nothing if it had stopped at the test result
- **MediaPipe 0.10.14's GPU + IMAGE mode is unusable on this device**, which is
  a limitation of that combination, not of the product

Consequence for testing: a GPU-vs-CPU parity test **cannot** be written with
`RunningMode.IMAGE` on this device. It has to use VIDEO mode, feeding one
timestamped frame at a time — which is exactly what the app does, and therefore
the more faithful comparison anyway.

## Method note

This is a case where an empty result was read as a negative result four times
before the measurement channel itself was checked. The right first move —
before probing MediaPipe, the classifier, or the decoder — was to prove that a
*known-passing* test's output could be observed at all. The second lesson is on
the other side: a crash reproduced in a test harness is not automatically a
product defect, and the app was still healthy throughout.