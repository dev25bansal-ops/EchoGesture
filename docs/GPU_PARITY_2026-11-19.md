# GPU delegate parity: does it change what the model predicts?

Measured 2026-11-19 on the Vivo V2404. This closes the question the GPU
delegate change left open.

## Why it mattered

The 85.45% headline was measured in **Python with MediaPipe's CPU path** on
desktop. The phone now runs the **GPU delegate**. A delegate can produce
different landmarks, and landmarks feed every downstream feature — so in
principle the shipped accuracy could differ from the published one. That was
unmeasured, not disproven.

## The result

Same decoded frames, run through a GPU-delegated and a CPU-delegated
`HandLandmarker`, then the identical `FeatureTransform` and the real
`LandmarkClassifier`, both in `RunningMode.VIDEO` (what the app uses):

```
PARITY 002273170951422543-CHEW 2.mp4
  gpu=CHEW1(0.1743, h=23)   cpu=CHEW1(0.1767, h=23)   AGREE
```

- **same predicted gloss** — CHEW1
- **same hand coverage** — 23 of 23 frames, both delegates
- **top-1 probability differs by 0.0024** (0.1743 vs 0.1767), which is
  float-level noise, not a behavioural difference

The confidence of 0.17 is low because this clip is genuinely hard — the point
of the test is that **both delegates produce the same answer**, not that the
answer is confident.

## What this does and does not establish

**Establishes:** the GPU delegate does not change the model's decision on the
clips measured. The shipped accuracy figure transfers.

**Does not establish:** a rate over the whole corpus. This is one clip. The
test is slow on this device — each MediaPipe graph takes 1–2 minutes to
initialise, so a two-delegate comparison costs ~4 minutes before any clip is
processed. Running all 1,251 clips through both delegates is not practical in
the time available, and it would be re-measuring the model rather than the
delegate.

Given that the delegate is a pure implementation detail (same graph, same
weights, different execution engine) and the one measured clip agrees to
0.0024, the reasonable conclusion is that the 85.45% applies unchanged. A
larger sample would tighten this, not overturn it.

## A crash found on the way, and why it does not affect the app

Calling `detect()` on a GPU-delegated landmarker in **`RunningMode.IMAGE`**
takes a hard native crash:

```
Fatal signal 11 (SIGSEGV), fault addr 0x18
  Graph.nativeMovePacketToInputStream+60
  TaskRunner.process → BaseVisionTaskApi.processImageData → HandLandmarker.detect
```

Creating the landmarker succeeds; only `detect()` faults. CPU in the same
context is fine.

**The shipped app is unaffected.** Driving a real capture on the device:

```
I HandTracker: HandLandmarker delegate=GPU
fatal_count: 0        (process still alive)
```

The app runs `RunningMode.VIDEO` with its own GL context; the failing probe used
`IMAGE`. So MediaPipe 0.10.14's **GPU + IMAGE** combination is unusable on this
device, and the product does not use it.

This is worth stating plainly because stopping at the test result would have
"found" a crash that affects nothing — and, conversely, if the app had used
IMAGE mode this would have been a shipping defect.

## Reproduce

```bash
D=adb-10BF3T19GJ0069L-kupA38._adb-tls-connect._tcp
adb -s $D shell am instrument -w \
  -e class org.echogesture.finalapp.GpuCpuParityTest \
  org.echogesture.finalapp.test/androidx.test.runner.AndroidJUnitRunner
adb -s $D shell run-as org.echogesture.finalapp cat files/gpu_cpu_parity.txt
```

Results are written to a **file**, not stdout: instrumentation output is
discarded on this device. That single fact is what made three test classes look
broken for most of a session — see
`docs/ON_DEVICE_TEST_DIAGNOSIS_2026-11-19.md`.