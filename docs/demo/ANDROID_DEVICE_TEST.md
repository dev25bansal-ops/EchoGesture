# Android device test — EchoGesture four-model ensemble APK

Status: **NOT RUN.** No Android device is attached to this machine (`adb devices`
returns an empty list) and `adb` is not on `PATH` (only under the local
toolchain). Nothing here is a device result; run it on real hardware.

APK under test: `app/android_final/app/build/outputs/apk/debug/app-debug.apk`
(debug-signed, v2 scheme only, `org.echogesture.finalapp` 1.0, minSdk 24 /
targetSdk 34). Rebuild first: see `docs/demo/DEMO_RUNBOOK.md` for the toolchain
exports (`JAVA_HOME`, `ANDROID_HOME`).

## Install

```bash
ADB="$ANDROID_HOME/platform-tools/adb"
"$ADB" install -r app/android_final/app/build/outputs/apk/debug/app-debug.apk
"$ADB" logcat -c
"$ADB" shell am start -n org.echogesture.finalapp/.MainActivity
```

## Procedure (verify each, record pass/fail)

1. **Permission.** First launch shows "Camera permission required" with a
   `Grant camera` button. Grant it; preview must start without a relaunch.
   Deny once, confirm the button re-requests.
2. **Preview.** Rear camera preview fills the view, portrait, no distortion.
3. **Hand detection.** `frameStatus` reads `N/32 frames · M with hands` and `M`
   rises while a hand is in frame.
4. **Capture.** Tap `Capture & translate`; button disables and
   `statusTitle` becomes "Processing…".
5. **Caption + top-5.** A word appears in `captionText`, `confidenceText` shows
   "Confidence NN%", `alternativesText` lists "Also considered: …" (top-5
   alternatives), `latencyText` shows "Inference N ms · total N ms".
6. **No-hand retry.** Point away from hands and capture: expect "No hands
   detected" / "Keep hands in frame and try again." with `Retry` enabled; retry
   with a hand returns a caption.
7. **Lifecycle restart.** Press Home then return: preview resumes, HUD still
   shows the four-member identity line, capture works again.
8. **Repeated capture.** Capture 5× in a row: no crash, no OOM, latency stays
   bounded, each result replaces the previous.
9. **Offline.** Enable airplane mode (or revoke network): `OFFLINE` stays
   visible and capture/caption still work end to end.
10. **Identity.** `modelIdentity` shows `float16 · SHA-256 <prefix>… · 4 members`
    and must not reference the old single classifier.

## Evidence to capture

- `adb logcat -d > logcat.txt` (app has no `Log.*` tags; keep the full buffer,
  including any `FATAL EXCEPTION`, `tflite`, `mediapipe`, or `OOM` lines).
- Screenshots of: permission prompt, preview with hands, a caption with top-5,
  the no-hands retry.
- The on-screen `Inference N ms · total N ms` and `frameStatus` values for each
  capture.
- Device model, Android version, and `adb shell getprop ro.product.model`.