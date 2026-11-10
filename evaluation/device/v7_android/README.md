# EchoGesture V7 — Android edge slice (A09)

Status: **BLOCKED for build/device; VERIFIED for host parity & contract.**
No APK, device, camera-frame, latency, or thermal claim is made anywhere in
this directory. Everything claimed below was executed on this host.

## 1. Environment actually observed (re-probe, 2026-11-08)

Full raw log: `evaluation/device/v7_android/env_probe.txt`. Summary:

| Probe | Result |
| --- | --- |
| Host | Darwin arm64 (macOS 25.3.0) |
| Java on PATH (`java -version`) | **absent** ("Unable to locate a Java Runtime") |
| Android Studio bundled JBR | OpenJDK **25.0.4** at `/Applications/Android Studio.app/Contents/jbr/Contents/Home` (launchable, but not on PATH / no `JAVA_HOME`) |
| `adb` / `gradle` / `sdkmanager` / `emulator` / `kotlinc` | all **not found** on PATH |
| `ANDROID_HOME` / `ANDROID_SDK_ROOT` / `JAVA_HOME` | all **unset** |
| `~/Library/Android/sdk` | **does not exist** (no `platforms/`, no `build-tools/`) — Android Studio's internal `android.jar` is a Studio tooling jar, not an installable SDK tree |
| USB Android device | `system_profiler SPUSBDataType` reports **no** Samsung/Pixel/Android device |
| Camera probe (OpenCV `VideoCapture(0)`) | `opened False` — "not authorized to capture video"; no live frames obtainable headlessly |
| Python host runtime | `$REPO/.venv/bin/python` with `ai-edge-litert`, `mediapipe`, `cv2`, `numpy` importable (this is what made the host verification below possible) |

This matches the prior V4 finding (`evaluation/device/v4_hardware/README.md`):
a shipping Android Studio does not imply a prepared SDK.

## 2. Frozen model contract (exact, no substitution)

Candidate (FP16, **not** INT8 — never relabelled as such):

- `models/landmark_classifier_fp16.tflite` — SHA-256 `a012f2b5f7a1110961079f19eb0e28c9abacb95e4d0787352e4157ebca7044a4`, 832,888 bytes, input `[1,24,255]` float32, output `[1,100]` float32 **logits**, tensor census: 192 tensors (117 float32, **38 float16**, 0 int8), quantization classification `fp16_float32_io`. Verified in this run by `evaluation/device/model_contract.py::validate_candidate_bundle(interpreter=…)` → `status=compatible`.
- `models/landmark_labels.json` — 100 ordered glosses, index = class id (`DOG1`=0 … `LONGLINE`=99).
- `models/landmark_model_meta.json` — `{frames:24, feature_dim:255, classes:100, quantization:"float16"}`.
- `models/hand_landmarker.task` — SHA-256 `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1` (MediaPipe HandLandmarker, VIDEO mode, 2 hands, thresholds 0.3).

Rejected alternative (present, never selected): INT8 classifier `9d24640a…` — a different artifact; the FP16 candidate must not silently become INT8.

Feature layout (identical in Python `src/features.py` and Kotlin `FeatureTransform.kt`):
`2 hands × 21 landmarks × 2 coords × 3 channels + 3` = **255**, per-frame layout `pos[84] | vel(lag-1)[84] | acc(lag-2)[84] | geom[3]`; geom = `[wrist distance, dx, dy]` on **raw** landmarks before centring, zero unless both hands present; shared scale = mean wrist→middleMCP distance over valid hands (1.0 if none); absent hands are explicit zeros; linear resample to 24 frames. Kotlin mirror: `app/v7_android/src/main/java/org/echogesture/v7/FeatureTransform.kt`.

Classifier output: stable softmax + top-5, same as `src/main.py`.

## 3. Host verification actually executed

```bash
$REPO/.venv/bin/python -I \
  evaluation/device/v7_android/test_parity.py
```

Result: candidate compatible (tensor census + logits probe), both .npz fixtures regenerate with 0.0 error, `CHEW 2` clip → top1 `CHEW1` (argmax 59, logit 5.67), synthetic → `SHAVE1`. This is the same clip `predict()` scores on the host, now also covered by a Kotlin parity test consuming the identical fixtures (`app/v7_android/src/test/java/org/echogesture/v7/FeatureParityTest.kt`, JSON fixtures in `src/test/resources/`).

## 4. Unblock: exact setup + acceptance commands

```bash
export JAVA_HOME='/Applications/Android Studio.app/Contents/jbr/Contents/Home'
export ANDROID_HOME="$HOME/Library/Android/sdk"
mkdir -p "$ANDROID_HOME/cmdline-tools"
# Download cmdline-tools (latest) from https://developer.android.com/studio#command-line-tools-only
# and extract so that $ANDROID_HOME/cmdline-tools/latest/bin/sdkmanager exists, then:
"$ANDROID_HOME/cmdline-tools/latest/bin/sdkmanager" --sdk_root="$ANDROID_HOME" \
  --install 'platform-tools' 'platforms;android-35' 'build-tools;35.0.0'
"$ANDROID_HOME/platform-tools/adb" devices -l          # must show a real device/emulator
cd $TMP/echogesture-.../A09/app/v7_android
# Gradle wrapper jar: download gradle-8.10.2 distribution (see
# gradle/wrapper/gradle-wrapper.properties) and run the wrapper once, e.g.
#   gradle wrapper --gradle-version 8.10.2   (with a Gradle on PATH), or
#   unzip the distribution and put gradle-wrapper.jar in gradle/wrapper/
./gradlew --no-daemon :app:testDebugUnitTest            # acceptance 1: parity
./gradlew --no-daemon :app:assembleDebug                # acceptance 2: APK builds
"$ANDROID_HOME/platform-tools/adb" install -r app/build/outputs/apk/debug/app-debug.apk
"$ANDROID_HOME/platform-tools/adb" shell am start -n org.echogesture.v7/.MainActivity
# acceptance 3 (on device): grant CAMERA, point at a hand performing CHEW,
# tap Capture, expect prediction CHEW1 within the in-app top-5, and record
# device model / Android version / APK SHA-256 / app logs in
# evaluation/device/v7_android/README.md. No claim until then.
```

## 5. Layout

- `app/v7_android/` — Android app slice: `settings.gradle.kts`, `app/build.gradle.kts`
  (CameraX 1.3.4, MediaPipe Tasks Vision 0.10.14, TFLite 2.16.1, pinned),
  `AndroidManifest.xml` (CAMERA only — **no network permission**),
  Kotlin: `FeatureTransform.kt`, `Classifier.kt`, `MainActivity.kt`,
  assets: `landmark_classifier_fp16.tflite`, `hand_landmarker.task`,
  `labels.json` (ordered vocab), `model_contract.json`.
- `evaluation/device/v7_android/` — evidence: `env_probe.txt`,
  `environment.json`, `parity_reference.json`, `parity_synthetic.npz`,
  `parity_chew2.npz`, `test_parity.py` (host verifier, runnable now).

Assets were copied from `models/` into the app by a plain file copy in this
session; the Gradle build must re-verify SHA-256 before packaging (pins in
`model_contract.json`) — that automated assertion is part of the post-SDK
build step, not claimed here.

## 6. What is NOT claimed

No APK exists. No compile, unit-test, install, camera-frame, on-device
latency, power, or thermal result is claimed. `MainActivity.kt` is the
smallest genuine offline path (CameraX → MediaPipe → features → LiteRT,
all on-device, frames never persisted); it has not been compiled here.
