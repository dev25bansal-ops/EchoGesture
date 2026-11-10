# EchoGesture Android V4

This directory holds the smallest genuine offline Android path for EchoGesture V4:
camera preview frames stay on device, hand landmarks are extracted with the
MediaPipe Hand Landmarker task, the shared 255-feature transform is applied,
and the existing `[1,24,255] -> [1,100]` TFLite classifier is run with LiteRT.

It does **not** vote on, relabel, or train a model. Bundled artifacts are copied
from the repository only when Gradle runs a verified release build:

- `models/hand_landmarker.task` = `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1`
- `models/landmark_classifier_fp16.tflite` = `a012f2b5f7a1110961079f19eb0e28c9abacb95e4d0787352e4157ebca7044a4`
- `models/landmark_classifier_int8.tflite` = `9d24640a23244a3c...` (full hash recorded in `evaluation/device/v4_hardware/provenance.json`)
- `models/landmark_labels.json`, `models/landmark_model_meta.json`

## Honest status

- JVM: `/Applications/Android Studio.app/Contents/jbr/Contents/Home/bin/java` reports
  OpenJDK 25.0.4. This satisfies a Java launch check; no Gradle project in this copy was
  used to imply a phone run.
- The Gradle wrapper is gradle-8.10.2; run with `./gradlew --no-daemon -q :app:assembleDebug`
  from `app/android_v4/` after installing the Android SDK in the next section.
- No physical Android device, emulator, `adb` executable, SDK `platforms/`, or SDK
  `build-tools/` was found in this agent copy. Therefore no APK install, camera-frame
  capture, phone latency, power, or thermal claim is made here.
- The selected classifier and bundled detector are FP16, not INT8. The separate
  classifier artifact is tensor-verified as INT8 only in the audit; it is not loaded by
  the current `src/main.py` default and is not named as INT8 headroom in the app.

## Unblock commands

Install the SDK platform and build tools, then connect a phone or start an emulator:

```bash
export JAVA_HOME='/Applications/Android Studio.app/Contents/jbr/Contents/Home'
export ANDROID_HOME="$HOME/Library/Android/sdk"
mkdir -p "$ANDROID_HOME/cmdline-tools"
# Extract a command-line tools release into $ANDROID_HOME/cmdline-tools/latest, then:
"$ANDROID_HOME/cmdline-tools/latest/bin/sdkmanager" --sdk_root="$ANDROID_HOME" \
  --install 'platform-tools' 'platforms;android-35' 'build-tools;35.0.0'
"$ANDROID_HOME/platform-tools/adb" devices -l
cd app/android_v4
./gradlew --no-daemon :app:assembleDebug
"$ANDROID_HOME/platform-tools/adb" install -r app/build/outputs/apk/debug/app-debug.apk
"$ANDROID_HOME/platform-tools/adb" shell am start -n org.echogesture.v4/.MainActivity
```

Once a phone or emulator is visible in `adb devices`, run the same build/install
commands on that target device and record the device model, Android version, runtime
delegate, APK SHA-256, clean-extraction log, and the exact app command in
`evaluation/device/v4_hardware/README.md`.

## Layout

- `settings.gradle.kts`, `build.gradle.kts`, `gradle/libs.versions.toml`
- `gradle/wrapper/gradle-wrapper.jar` and checked-in 8.10.2 wrapper metadata
- `app/src/main/`
  - `AndroidManifest.xml` (camera permission; no network permission requested)
  - `java/org/echogesture/v4/FeatureTransform.kt` (shared transform in Kotlin)
  - `java/org/echogesture/v4/Classifier.kt` (TFLite adapter, no graph math)
  - `java/org/echogesture/v4/MainActivity.kt` (CameraX preview and capture flow)

Build verification of the APK itself remains blocked until the Android SDK pieces
above are installed and a target device or emulator is available.
