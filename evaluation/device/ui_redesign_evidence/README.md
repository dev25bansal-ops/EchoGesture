# Android UI redesign — captured evidence

All eleven screenshots were taken on the Android emulator `eg34`
(sdk_gphone64_arm64, android-34, 1080×2400, density 420) from a debug APK built
on this branch, via `adb exec-out screencap -p`. Nothing here is a mockup, a
render, or a re-creation of the UI in another tool.

They were captured on the build at `974f610`. The two commits after it —
`f8d4643` (a test file) and `5d90063` (removal of five unreferenced strings) —
change no rendered pixel; the current APK is the one those commits build to.

## What each shot shows, and what it does not

| File | Screen | How it was reached |
| --- | --- | --- |
| `01_camera.png` | Camera, idle | Launched the app, camera running, no hands in view (emulator synthetic camera) |
| `02_permission.png` | Permission-needed card | `pm revoke CAMERA` → force-stop → relaunch → tapped **Don't allow** |
| `03_capturing.png` | CAPTURING | Tapped **Recognise sign** |
| `04_processing.png` | PROCESSING | `ProcessingStateRenderTest`, which calls the production `render()` at PROCESSING and asserts the button is disabled and reads "Reading…" |
| `05_result.png` | RESULT | `ResultCardRenderTest`, real four-model FP16 ensemble over the pinned Python-parity feature vector |
| `06_no_hands.png` | NO_HANDS | Tapped **Recognise sign** with the synthetic camera showing no hand |
| `07_glossary.png` | Glossary | Bottom navigation |
| `08_history.png` | History, one entry | Written by the `ResultCardRenderTest` result, then opened from the bottom navigation |
| `09_settings.png` | Settings | Bottom navigation |
| `10_help.png` | Help sheet | **How to use EchoGesture** on the camera screen |
| `11_gloss_detail.png` | Gloss detail sheet | Tapped a Glossary row |

### `05_result.png` is not a live-camera prediction

The emulator has no hands in front of it. The result shown is the real output of
the four FP16 TFLite interpreters — four mean-pooled `[1,100]` logit vectors
averaged, then a single softmax — run over the pinned feature vector that the
Python and Kotlin sides agree on. It is a genuine inference result, but it is
**not** a recognition of anything the emulator's camera saw, and it is never
presented in the app or here as one.

`ResultCardRenderTest` states the same constraint in its own doc comment and
asserts the caption, score, alternatives, and latency before it holds the
screen for the capture.

### PROCESSING and the 70 ms window

Inference finishes in roughly 70 ms, and the PROCESSING state is written to the
UI thread and cleared again inside that window — too briefly for a screencap
round trip, and `kill -STOP` is refused on Android 14. `04_processing.png` is
therefore produced by `ProcessingStateRenderTest`, which invokes the same
private `render()` the capture path calls at its PROCESSING step. The test
asserts both state-specific affordances before holding the screen, so the image
is backed by an assertion rather than by a visual guess.

## Reproducing

```bash
# from app/android_final, with the emulator running
./gradlew --offline assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb shell pm grant org.echogesture.finalapp android.permission.CAMERA
adb shell am start -n org.echogesture.finalapp/.MainActivity
adb exec-out screencap -p > 01_camera.png
```

The two render tests use a marker-file handshake
(`eg_result_ready`/`eg_result_shot_done`, `eg_processing_ready`/
`eg_processing_shot_done`) under the app's external files directory so the
capture and the test stay in step without depending on logcat.