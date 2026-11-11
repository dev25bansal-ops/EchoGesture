# Android device test — EchoGesture four-model ensemble APK

Status: **DEVICE STEPS NOT RUN.** No physical Android phone has been attached to
this machine (`adb devices -l` returns an empty list). Everything below that is
marked *device* is still open and must be run on real hardware.

What *has* been run is on the `eg34` emulator (Android 14 / API 34 /
arm64-v8a), and is labelled emulator-only wherever it appears. **An emulator is
not a phone:** its virtual camera, timing, power and thermal numbers are not
target-hardware results and must never be reported as such.

APK under test: `app/android_final/app/build/outputs/apk/debug/app-debug.apk`
(debug-signed, v2 scheme only, `org.echogesture.finalapp` 1.0, minSdk 24 /
targetSdk 34). Rebuild first: see `docs/demo/DEMO_RUNBOOK.md` for the toolchain
exports (`JAVA_HOME`, `ANDROID_HOME`).

Current build (from this source tree, **not** a pinned reproducible artifact —
see below): 75,072,706 B, SHA-256
`e5ed0a3b21e5f7ddfba140512252a2509acf92928f9132c3b25370b1895ea547`,
`apksigner` v2 verified, `zipalign -c -v 4` successful.

### Reproducibility — corrected claim

`evaluation/device/android_four_model_evidence.md` records a bit-identical
SHA-256 (`2ec087e5…`). That is **not reproducible from this build** and the
claim should be treated as unverified until re-established. Measured here:

- three consecutive `:app:assembleDebug --rerun-tasks` builds of the
  **unmodified** `HEAD` source produced three different APK hashes;
- a full entry-by-entry comparison shows **only `classes3.dex` differs**, and a
  `dexdump -d` disassembly diff of the two builds is empty — the varying bytes
  are the synthetic-lambda name hashes that D8 writes for
  `MainActivity$$ExternalSyntheticLambda*` (`"37fa68ae3"`, `"9c79600fa"`, …),
  plus the dex header checksum/signature that follows from them;
- with `--no-daemon --no-build-cache` two consecutive builds *were* identical,
  so the variation is tied to how D8 is invoked in this environment, not to
  source. It reproduces with the unmodified tree, so it is not caused by the
  camera-path change in this branch.

Until D8 is pinned (e.g. a fixed `-Dorg.gradle.workers.max` plus a
reproducibility check in CI), quote the size and the per-member model hashes
(which **are** pinned and checked at load time), not a whole-APK SHA.

## What the app actually does (matches this procedure)

- **Camera:** binds `DEFAULT_BACK_CAMERA` — the organizer's scenario is an
  exocentric third-person view (deck p.10), someone filming the signer. If the
  device has no back camera it falls back to `DEFAULT_FRONT_CAMERA`. Neither
  path mirrors the image: the sensor frame reaches MediaPipe unmirrored and its
  Left/Right handedness label is used as-is (no handedness swap).
- **Capture window:** the most recent **32 analysed frames** (0.5–1.5 s at
  camera rate). It is a *sliding* window, so sign, then Capture immediately;
  waiting leaves later frames in the window and progressively displaces the
  sign.
- **States:** `Initializing…` → `Waiting for hands` → `Processing…` →
  `Caption`, plus `No hands detected` and `Something went wrong`. `Retry` is
  shown for the last three; `Grant camera` is shown only while permission is
  missing.
- **No confidence gate:** every capture that contains at least one hand frame
  produces the ensemble's top-1 and its confidence. A low-confidence result is
  reported as-is, not suppressed.

## Install

```bash
ADB="$ANDROID_HOME/platform-tools/adb"
"$ADB" install -r app/android_final/app/build/outputs/apk/debug/app-debug.apk
"$ADB" logcat -c
"$ADB" shell am start -n org.echogesture.finalapp/.MainActivity
```

## Procedure (verify each, record pass/fail)

Each step is marked **[device]** (real phone only) or **[emulator]** (already
run or runnable on `eg34`). Emulator results do not satisfy the device steps.

1. **[emulator] Permission, deny then grant.** First launch shows "Camera
   permission required" with a `Grant camera` button. Deny: the same state
   persists, `Grant camera` re-requests, and `Capture & translate` stays
   disabled (a capture without the camera can only see an empty ring, so it must
   not be able to report a misleading "No hands detected"). Grant: preview starts
   without a relaunch and Capture enables. *Emulator: verified.*
2. **[device] Preview orientation.** Rear-camera preview fills the view in
   portrait with no rotation, stretch or mirroring. The emulator has only a
   virtual sensor, so this needs a phone. *Open.*
3. **[device] Handedness.** Sign with the right hand, then the left. The
   caption must match the sign, not its mirror. This is the check that would
   catch an accidental sensor mirror or a swapped Left/Right slot; nothing short
   of a real camera and a real sign exercises it. *Open.*
4. **[emulator] Hand detection.** `frameStatus` reads `N/32 frames · M with
   hands`; `M` rises while a hand is in frame. *Emulator: verified the counter
   (0 with no hand in view).* With a hand in frame: **[device].**
5. **[emulator] Capture.** Tap `Capture & translate`; the button disables and
   `statusTitle` becomes "Processing…", then a caption or a no-hands result.
   *Emulator: verified the no-hands transition.*
6. **[device] Caption + top-5.** A word appears in `captionText`,
   `confidenceText` shows "Confidence NN%", `alternativesText` lists
   "Also considered: …" (the other four of the top-5), `latencyText` shows
   "Inference N ms · total N ms". *Open — needs a hand and a known sign.*
7. **[emulator] No-hand path.** With no hands in view, capture shows "No hands
   detected" / "Keep hands in frame and try again." with `Retry` enabled; `Retry`
   returns to "Waiting for hands". *Emulator: verified.*
8. **[emulator] Capture-window reset.** The window is the last 32 frames, not a
   segment starting at the last Capture, and `Retry` clears the caption *and*
   the ring (`tracker.clear()` then back to "Waiting for hands"). So even after
   Retry the window refills within about a second on a live camera — always sign
   and capture promptly rather than lingering, which would let the sign's frames
   be displaced. *Emulator: verified the ring fills (`32/32 frames`, from the
   virtual camera's blank scene) and `Retry` returns to "Waiting for hands".*
9. **[emulator] Lifecycle restart.** Press Home then return: preview resumes,
   the HUD still shows the four-member identity line, capture works again.
   Rotating is disabled (`screenOrientation="portrait"`). *Emulator: verified
   the activity can be destroyed mid-analysis with no crash.*
10. **[emulator] Repeated capture.** Capture 5× in a row: no crash, no OOM,
    each result replaces the previous. *Emulator: verified — Dalvik heap
    10.2 MB → 11.5 MB live across 5 captures, native heap flat (~85 MB), no
    `FATAL`/`OOM` in logcat. Emulator memory is not a phone memory figure.*
11. **[device] Offline.** Enable airplane mode: `OFFLINE` stays visible and
    capture/caption still work. The APK declares no INTERNET or
    ACCESS_NETWORK_STATE permission, so it cannot open a socket; confirm on
    device with `adb shell dumpsys package org.echogesture.finalapp | grep -i
    permission`. *Open.*
12. **[device] Identity.** `modelIdentity` shows `4-model ensemble · float16 ·
    100 classes · SHA-256 <prefix>… · 4 members` and must not reference the old
    single classifier. *Emulator: verified this exact string.*
13. **[device] Cold start and timing.** Record cold start (`am start -W`) and
    the on-screen `Inference N ms · total N ms` for a real capture. These are
    the numbers a jury may quote and only a phone can produce them.

## Device-only measurements (still open)

These cannot be measured on the emulator and must not be reported from it:

- camera capture latency and preview-to-analysis lag on the target phone;
- inference time on the target SoC (Delegate/CPU, thermal state);
- power draw and thermal throttling over repeated captures;
- rear-camera orientation and handedness on real optics (steps 2, 3, 6);
- cold start on a real device.

Emulator timing, power and camera behaviour are not target-hardware results.

## Evidence to capture on the device run

- `adb logcat -d > logcat.txt` (keep the full buffer, including any
  `FATAL EXCEPTION`, `tflite`, `mediapipe`, `OOM`, or `HandTracker` warnings).
- Screenshots of: the permission state, preview with hands, a caption with
  top-5, and the no-hands retry.
- The on-screen `Inference N ms · total N ms` and `frameStatus` values for each
  capture.
- `adb shell getprop ro.product.model`, `ro.build.version.release`,
  `ro.product.cpu.abi`.

## Automated coverage that already exists

- `./gradlew --offline testDebugUnitTest` — 20 JVM tests (feature parity,
  ensemble parity, model contract).
- `./gradlew --offline connectedDebugAndroidTest` — 6 tests on a device or
  emulator: four members allocate and reproduce the Python reference, plus
  `CameraFeaturePathTest` (camera rotation reported and accepted by the
  tracker; missing-hand frames keep their temporal position). On `eg34` these
  pass 6/6, with the virtual sensor reporting rotation 90.