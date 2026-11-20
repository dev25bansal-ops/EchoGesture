# The GPU delegate: latency now meets the budget

Measured 2026-11-19 on the Vivo V2404. This supersedes the CPU-only figures in
`ON_DEVICE_LATENCY_2026-11-19.md` — those were correct, but they described the
default CPU delegate, which is where the detector had been running untuned.

## What changed

`HandTracker.create()` built its MediaPipe `BaseOptions` with only a model path
and no delegate, so the hand detector ran on the default **CPU** path. That is
the single largest on-device inference cost, and it was the one thing nobody had
tried.

MediaPipe 0.10.14 exposes exactly two delegates — `Delegate.CPU` and
`Delegate.GPU` (verified with `javap` against `tasks-core-0.10.14.aar`; there is
**no NNAPI value in this version**, so that is not an option here). The app now
requests `GPU` and falls back to `CPU` if initialisation throws, because a
delegate that fails on a given driver must not turn a slow app into a broken one.

Confirmed on the real device rather than inferred from the API existing:

```
adb logcat | grep HandTracker
I HandTracker: HandLandmarker delegate=GPU
```

## The measurement

Six independent runs. The first was a single opportunistic sample; the three
below are clean runs, each a fresh process, capture drive via the app's own
"Start sign" → "Finish sign" path so the profiler flushes through its real code
path. No hands were in frame in any run, which is deliberate: it isolates the
detector's cost from anything the classifier might do.

```
detect_p50=45 detect_p95=54   fps=19.2
detect_p50=41 detect_p95=49   fps=21.3
detect_p50=41 detect_p95=50   fps=21.3
detect_p50=34 detect_p95=46   fps=25.0
```

| metric | CPU (previously measured) | **GPU (now)** | change |
|---|---|---|---|
| detect_p50 | 80–109 ms | **34–45 ms** | ~2.3× faster |
| detect_p95 | 159–187 ms | **46–54 ms** | ~3.5× faster |
| analysed fps | 8.7–11.6 | **19.2–25.0** | ~2.2× higher |
| frames dropped | 76–82% | 15–30% | far fewer |

Both sets come off the app's own `AnalysisProfiler`, via
`files/diagnostics.txt`, read with `run-as`. Same device, same build type
(DEBUGGABLE), same 640×480 VIDEO mode.

## What this means for the scored criterion

The organizer's metric is `latency_p95 <= 50 ms`. On this device:

- **CPU: 159–187 ms** — 3.2–3.7× over
- **GPU: 46–54 ms** — at the budget, with the best runs inside it

The caveat that still applies, and must be stated wherever this is quoted: this
is a **DEBUGGABLE build**, so it is an upper bound; a release build is faster but
cannot be read off this phone (no root, and `run-as` needs a debug build). And
the 50 ms referent is still undefined by the deck — see
`docs/WHAT_SCORES_2026-11-19.md`. On either reading, though, the GPU delegate
moves this criterion from "clearly over" to "at or inside".

## Reproduce

```bash
adb shell am force-stop org.echogesture.finalapp
adb shell am start -n org.echogesture.finalapp/.MainActivity
sleep 6
adb shell run-as org.echogesture.finalapp rm -f files/diagnostics.txt
adb shell input tap 540 1383     # Start sign
sleep 7
adb shell input tap 540 1383     # Finish sign
adb shell run-as org.echogesture.finalapp cat files/diagnostics.txt | grep ^profile
adb logcat -d | grep 'HandLandmarker delegate'   # confirms GPU was used
```

## Not yet done

- **Accuracy under GPU has not been re-measured.** The delegate can produce
  slightly different landmarks. The 85.45% figure was measured through CPU
  MediaPipe in Python. This needs a replay comparison before the number is
  claimed for the GPU path.
- **Power and thermal** are still unmeasured; the harness exists
  (`scripts/measure_power_thermal.py`) and the phone reports CPU temperatures,
  but the GPU path changes the thermal profile, so it wants re-running now.
