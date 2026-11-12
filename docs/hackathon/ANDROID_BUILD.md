# Android build environment

Everything needed to build, test and run `app/android_final` on a Windows
machine. Verified working 12 Nov 2026 — the app built, installed on an emulator,
and rendered its camera UI.

## What is installed, and where

| Component | Version | Path | Size |
| --- | --- | --- | --- |
| JDK (Temurin) | **17.0.21** | `D:\jdk17\jdk-17.0.21+9` | 300 MB |
| Android SDK | — | `D:\android-sdk` | 4.3 GB |
| SDK Platform | android-34 | `platforms/android-34` | |
| Build-Tools | 34.0.0 | `build-tools/34.0.0` | |
| Platform-Tools | 37.0.1 | `platform-tools` (`adb`) | |
| Emulator | 37.2.13 | `emulator` | |
| System image | android-34 google_apis x86_64 | `system-images/android-34/google_apis/x86_64` | |
| CMake | 3.22.1 | `cmake/3.22.1` | |
| NDK | 26.1.10909125 | `ndk/26.1.10909125` | |

**JDK 17 is required, not optional.** The Android Gradle Plugin 8.5.2 rejects
JDK 21+ with a bare `26.0.1` as the entire error message. The JDK 26 on this
machine does not work; neither does the Java 8 that is first on `PATH`.

The SDK lives on `D:` because `E:` was at 97% when this was set up. It can move —
point `local.properties` at wherever it ends up.

An AVD named **`echogesture34`** is created (Pixel 6, API 34, x86_64).

## Build

`app/android_final/local.properties` must point at the SDK. This file is
gitignored; create it per checkout:

```bash
sdk.dir=D:/android-sdk
```

Forward slashes. Backslashes fail: AGP's `validateSdkPath` throws
`java.io.IOException: The filename, directory name, or volume label syntax is
incorrect` for `D:\android-sdk`.

Assets must be present or the build refuses to run:

```bash
mkdir -p app/android_final/app/src/main/assets
cp models/hand_landmarker.task \
   models/landmark_labels.json \
   models/landmark_model_meta.json \
   models/landmark_ens_tcn21_fp16.tflite \
   models/landmark_ens_tcnm61_fp16.tflite \
   models/landmark_ens_tcnm62_fp16.tflite \
   models/landmark_ens_rob71_fp16.tflite \
   app/android_final/app/src/main/assets/
```

Then:

```bash
cd app/android_final
export JAVA_HOME="D:\\jdk17\\jdk-17.0.21+9"
./gradlew :app:assembleDebug        # -> app/build/outputs/apk/debug/app-debug.apk
./gradlew :app:testDebugUnitTest    # 30 tests
```

Measured: debug build **47 s**, APK **75 MB**, 30 unit tests.

## Run on the emulator

```bash
export ANDROID_HOME="D:\\android-sdk"
"$ANDROID_HOME/emulator/emulator.exe" -avd echogesture34 -no-snapshot -no-audio \
    -gpu swiftshader_indirect &

# wait for sys.boot_completed == 1, then:
"$ANDROID_HOME/platform-tools/adb.exe" install -r \
    app/android_final/app/build/outputs/apk/debug/app-debug.apk
"$ANDROID_HOME/platform-tools/adb.exe" shell pm grant \
    org.echogesture.finalapp android.permission.CAMERA
"$ANDROID_HOME/platform-tools/adb.exe" shell am start \
    -n org.echogesture.finalapp/.MainActivity
```

**Observed on first launch:** the HUD briefly showed
`Something went wrong / Model load failed: com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker`
while the camera permission dialog was still open. After granting CAMERA and
relaunching it came up normally: live camera preview, `OFFLINE` badge,
"Waiting for hands", the capture button correctly disabled. So the failure was
a first-run race between permission grant and model init, not a broken model —
but if it recurs on a device with permissions already granted, that is a real
bug worth chasing.

Keyboard note: the emulator accepts `adb shell input keyevent` and `input text`.
Its `uiautomator dump` writes to a Windows-translated path, so read the XML with
`adb shell cat /sdcard/ui.xml` rather than the path it prints.

## Known issues

- **MediaPipe 0.10.14 has no x86_64 library, so the app cannot run on an
  x86_64 emulator.** The AAR ships `arm64-v8a`, `armeabi-v7a` and `x86` only.
  On an x86_64 AVD the APK contains no `libmediapipe_tasks_vision_jni.so` for
  that ABI and the app fails at load with:

  ```
  Model load failed: dlopen failed:
  library "libmediapipe_tasks_vision_jni.so" not found
  ```

  **Fix: upgrade to `com.google.mediapipe:tasks-vision:1.2.3`**, which ships
  `jni/x86_64/libmediapipe_tasks_jni.so` (note the library name change — the
  `_vision` infix is gone). The Kotlin API used here is source-compatible; it
  compiled unchanged. Physical devices are all ARM, so this only affects
  emulator testing.

  No 32-bit x86 system image exists for any modern API level, so dropping back
  to `x86` is not an option — upgrading MediaPipe is the way.

- **`extraProvenanceFieldsDoNotChangeWhatLoads` fails against the shipped
  metadata.** The test appends `"ensemble_test_top1"` to whatever meta file it
  finds — but `models/landmark_model_meta.json` from `hackathon/dev-model`
  already contains that key, so `org.json` rejects the result with
  `Duplicate key "ensemble_test_top1"`. It passes on this repo's own meta, which
  omits the key. Fix: only append keys not already present, or regenerate this
  repo's meta from the shipped one. Found 12 Nov 2026, not yet fixed upstream.
- Two builds are needed if you are switching branches: `abhijeet-public` before
  `9afe975` fails at the asset-hash gate on Windows (CRLF). Use
  `hackathon/abhijeet-pr2-fix` or later.

## Feeding a sign to the emulator

`-camera-back "imagefile:<path>"` loads a still image as the back camera, and
the app prefers the back camera. This is enough to exercise the whole pipeline:

```bash
"$ANDROID_HOME/emulator/emulator.exe" -avd echogesture34 -no-snapshot -no-audio \
    -gpu swiftshader_indirect \
    -camera-back "imagefile:E:\path\to\frame.png" -camera-front none
```

Measured with one frame of a PONDER clip: `32/32 frames · 32 with hands`, hand
detection and all four members ran, prediction `NOON1` at 34.1% confidence in
240 ms inference / 256 ms total.

**A still image cannot test accuracy.** The classifier is temporal — 24 frames
of motion. Given the same clip's real landmarks it predicts **PONDER at 59.1%**,
which is correct; from a frozen frame it produces a confident-looking wrong
answer. Use the still only to prove the plumbing, and evaluate accuracy through
`scripts/ml/evaluate.py`, never through the emulator camera.

## What is not verified

- **No physical device.** Everything above ran on the x86_64 emulator with a
  synthetic camera. Real-device camera behaviour, thermal load and battery use
  are unmeasured.
- **No live hand tracking test.** The emulator's virtual scene has no hands, so
  the app correctly sat at "Waiting for hands". The MediaPipe detector itself
  was exercised (it loaded), but no prediction was produced on-device.
