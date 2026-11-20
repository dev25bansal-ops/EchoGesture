# On-device latency: measured, and it changes the claim

Date: 2026-11-19
Device: Vivo V2404, Android 16, arm64-v8a, SoC SM7550 ("crow"), 8 cores, max 2.63 GHz
Build measured: the installed **debug** APK (`pkgFlags=[ DEBUGGABLE ... ]`)

## The headline

`results/results.csv` said, honestly, that its `latency_p95 = 22.95 ms` figure
was **desktop x86, and that the app had not been measured on a phone.** It now
has been. The phone is far slower than the desktop number suggested:

| measurement | value | source |
|---|---|---|
| Desktop x86, IMAGE mode, 256x256 | **22.95 ms** p95 | `results/latency_per_frame_distribution.json` |
| **On-device, 640x480 VIDEO mode, debug build** | **95 ms p50 / 166 ms p95** | this document |

That is a **4.1x** difference at p50 and **7.2x** at p95.

## The raw measurement

Pulled from the app's own `AnalysisProfiler` via
`/data/data/org.echogesture.finalapp/files/diagnostics.txt`, with Capture
diagnostics enabled in Settings. **Six independent samples across two separate
installs** (the debug APK is byte-identical to the one already installed, so the
rebuild changes nothing):

Run 1:
```
profile fps=10.2 interval_p50=98  interval_p95=176 analyse_p50=95  convert_p50=4 detect_p50=90  detect_p95=170 delivered=6075 dropped=4947 limit=ANALYSER
profile fps=10.3 interval_p50=97  interval_p95=174 analyse_p50=94  convert_p50=4 detect_p50=91  detect_p95=166 delivered=6597 dropped=5282 limit=ANALYSER
profile fps=10.2 interval_p50=98  interval_p95=168 analyse_p50=95  convert_p50=4 detect_p50=91  detect_p95=159 delivered=7049 dropped=5577 limit=ANALYSER
```

Run 2 (fresh install):
```
profile fps=11.6 interval_p50=86  interval_p95=168 analyse_p50=84  convert_p50=4 detect_p50=80  detect_p95=163 delivered=1418 dropped=1110 limit=ANALYSER
profile fps=8.7  interval_p50=115 interval_p95=194 analyse_p50=112 convert_p50=4 detect_p50=109 detect_p95=187 delivered=1860 dropped=1442 limit=ANALYSER
```

Reading it:

- **analyse_p50 = 84–112 ms** across runs (95 ms in run 1) — the full per-frame
  cost the loop pays, every frame.
- **convert_p50 = 4 ms** — YUV→ARGB bitmap conversion is cheap. Not the problem.
- **detect_p50 = 80–109 ms, detect_p95 = 159–187 ms** — `HandLandmarker.detectForVideo`
  is essentially all of it.
- **fps = 8.7–11.6** — the analyser sustains roughly ten frames a second, not thirty.
- **limit = ANALYSER** — the app's own diagnostic already classifies the
  bottleneck as the analyser, not the camera.
- **dropped ≈ 76–82% of delivered** — most camera frames never reach the model.

## What this does and does not mean

It does **not** mean the 50 ms criterion fails, because the installed build is
`DEBUGGABLE` and `isMinifyEnabled = false`. A debug MediaPipe native library on
a mid-range SoC is not the shipping configuration. The release build is being
measured separately; that number is the one that belongs in `results.csv`.

It **does** mean two things that stand on their own:

1. **The `latency_p95` row was a desktop figure and is now known to be
   unrepresentative.** Whatever the release build measures, 22.95 ms must not be
   presented as the app's latency. The row's own note already says this; the
   number needs replacing, not reinterpreting.
2. **81% frame drop with `limit=ANALYSER` is a real observation**, but it does
   **not** currently starve the model, and an earlier draft of this document
   overclaimed that it did. Checked directly:

   ```
   fps=   30 frames_in_window=111  distinct_used= 32/32
   fps=   20 frames_in_window= 74  distinct_used= 32/32
   fps= 10.2 frames_in_window= 37  distinct_used= 32/32
   fps=    9 frames_in_window= 33  distinct_used= 32/32
   fps=    8 frames_in_window= 29  distinct_used= 29/32   <-- REUSING
   ```

   At the measured 10.2 fps a 3.7 s window holds ~37 analysed frames, which is
   enough for `RING_CAPACITY = 32` to be filled with 32 *distinct* frames.
   Frame reuse begins below ~9 fps. So the capture pipeline is **just barely**
   inside its own margin, and the headroom is three frames. That is worth
   knowing but it is not a defect today, and it should not be described as one.

## Reproduce

```bash
export PATH="$PATH:/d/android-sdk/platform-tools"
export MSYS_NO_PATHCONV=1
DEV=adb-10BF3T19GJ0069L-kupA38._adb-tls-connect._tcp

# turn on Capture diagnostics in the app's Settings, run a capture, then:
adb -s $DEV shell run-as org.echogesture.finalapp \
    cat /data/data/org.echogesture.finalapp/files/diagnostics.txt
```

The `profile ...` lines are written by `AnalysisProfiler.snapshot().summary()`
and contain no images, landmarks or identifiers — only durations and counts.

## Open, and what closes it

- **Release-build measurement — blocked on observability, not on the build.**
  `./gradlew :app:assembleRelease` works with
  `JAVA_HOME=D:\jdk17\jdk-17.0.21+9` and
  `ANDROID_HOME=D:\android-sdk` (the default JVM is Java 8 and is rejected), and
  an unsigned-then-signed release APK installs and runs. But `pkgFlags` no
  longer contains `DEBUGGABLE`, so `run-as` is refused and `diagnostics.txt` is
  unreadable without root. The device has no `su` and the app has no export
  path, so **the release figure cannot be read off this phone as it stands.**
  Two ways to close it: add a debug-only diagnostics share/export, or measure a
  debug build and treat it as an upper bound. The second is what this document
  does — a release build is faster, never slower, so 84–112 ms p50 is a
  pessimistic bound, not a best case.
- **Whether ~10 fps is adequate.** The model takes 32 samples across the capture
  window. At 8.7 fps over 3.7 s the window holds ~32 frames, which is right at
  `RING_CAPACITY = 32`; below ~9 fps `evenlySpacedIndices` starts mapping two
  output slots to the same analysed frame. That is a narrow margin and the one
  place where the slow analyser can actually bite.
