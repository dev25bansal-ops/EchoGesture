# How on-device latency could be measured, and what blocks each route

Written 16 Nov 2026 by Dev. On-device latency, power and thermal are 20% of
the rubric and are `NOT_MEASURED`. This records every route I tried, so nobody
repeats them.

## Route 1: the emulator — BLOCKED, hard

Google's `com.google.mediapipe:tasks-vision:0.10.14` AAR ships native libraries
for **arm64-v8a, armeabi-v7a and x86 only**. There is **no x86_64 slice**. I
downloaded the AAR from Google's Maven and opened it to confirm.

The local AVD `echogesture34` is `abi.type = x86_64` with an empty
`ro.product.cpu.abilist32`, so the 32-bit `x86` slice cannot be used either.

Result on-device: `UnsatisfiedLinkError: dlopen failed: library
"libmediapipe_tasks_vision_jni.so" not found`, thrown at
`HandTracker.create()` before any inference.

A 32-bit x86 AVD would work, but it would measure an emulated 32-bit CPU with
no camera, no thermal behaviour and no battery — not a number anyone should
submit.

## Route 2: JVM unit tests — BLOCKED, documented by the app's own author

The shared branch has ten JVM tests including `EnsembleRuntimeParityTest`. I
checked whether they could time the classifier off-device.

They cannot, and the reason is written in that file's own header: *"the
tensorflow-lite AAR ships Android ABIs only and the JVM unit-test runtime fails
with `UnsatisfiedLinkError: no tensorflowlite_jni` (that is also why the
on-device test `TfliteEnsembleInferenceTest` exists)"*.

So the JVM tests pin the arithmetic between raw logits and the displayed result
— dequantize, mean, softmax, top-k — but never run TFLite. They cannot produce
a latency number.

## Route 3: a physical phone — WORKS IN PRINCIPLE, not completed

This is the only real route. Attempted on a **Vivo V2404, Android 16,
arm64-v8a** (which has the arm64 slice MediaPipe needs).

The APK install stalled part-way through an 83 MB transfer. `adb shell echo`
then timed out at 20 s with no output, the device went `offline` after an adb
server restart, and finally disappeared entirely (`no devices/emulators found`).
That needs a physical replug and possibly accepting a USB-debugging prompt on
the handset.

**The APK to install is verified**: `.scratch/apkcheck/echogesture-v4.3-masked.apk`,
SHA-256 `3db9d94a…1380896`, matching the artifact Abhijeet published.

### Once a phone is attached

```bash
adb install -r -d .scratch/apkcheck/echogesture-v4.3-masked.apk
adb shell am start -n org.echogesture.finalapp/.MainActivity
```

Then the numbers to capture, in the order the rubric weights them:

1. **Per-frame latency** — the app records `frameStatus` and per-member timings;
   `Settings → Diagnostics` exposes them. Target ≤ 50 ms.
2. **Power** — `adb shell dumpsys battery` before/after a fixed workload, or a
   proper USB meter. Target ≤ 500 mW.
3. **Thermal** — a 10-minute continuous run with `adb shell dumpsys thermalservice`
   sampled. Target ≤ 20% drift.

None of these can be estimated from desktop numbers: the phone's detector uses
`detectForVideo` with tracker reuse at native 640×480, a different path from
anything measured here.

## What the desktop numbers do and do not tell us

| | desktop x86 Python |
| --- | --: |
| per-frame MediaPipe | p50 133 ms (min 69) |
| classifier, 9 members | 4.2 ms/clip |
| end-to-end warm | ~4,650 ms/clip |

The classifier is **0.15%** of end-to-end cost, so the rubric's latency
criterion is entirely about the detector — and the detector on a phone with
tracker reuse and hardware acceleration is plausibly much faster than this
Python path. But "plausibly" is not a measurement, and the submission should
say so.
