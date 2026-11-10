# EchoGesture Final — Offline Android caption app

This is a buildable Android project for the frozen EchoGesture landmark model.
It contains no model binaries and makes no APK, device, power, thermal, or
camera-permission verification claims on this host.

## Offline pipeline

CameraX front-camera preview and frame analysis -> MediaPipe HandLandmarker
(VIDEO mode, 2 hands, confidence thresholds 0.3) -> exact Kotlin port of
`src/features.py` -> `landmark_classifier_fp16.tflite` via LiteRT/TFLite ->
stable softmax/top-5 -> caption HUD.

The capture window retains 32 analyzed frames. Tapping **Capture & translate**
resamples those frames linearly to the model's exact 24-frame, 255-feature
contract and runs inference on a background executor. Frames and results are
never uploaded or saved to video files.

## UI states

- **Waiting for hands**: live camera and model are ready, but a capture window
  contains no hands.
- **Processing…**: frame collection or model inference is active.
- **Caption**: predicted gloss, confidence, top alternatives, inference and
  total post-capture latency.
- **No hands detected — Retry**: a capture completed without hands.
- **Error**: actionable model/camera/detector failure; Retry is available.

The HUD permanently shows `OFFLINE`, the exact FP16 model identity and its
SHA-256 prefix, capture frame count, hand-frame count, and timings. The app
requests CAMERA only; there is no INTERNET or network-state permission.

## Prepare model assets (local copy; do not commit)

From the repository root:

```bash
mkdir -p app/android_final/app/src/main/assets
cp models/hand_landmarker.task \
   models/landmark_classifier_fp16.tflite \
   models/landmark_labels.json \
   models/landmark_model_meta.json \
   app/android_final/app/src/main/assets/
```

The Android asset names are fixed by `ModelContract.kt`. Gradle validates all
four copied files and the two binary SHA-256 hashes before configuration or
assembly. Model bytes stay in the existing root `models/` directory and are
excluded by `app/android_final/.gitignore`; no second model copy belongs in Git.

Expected hashes:

- classifier: `a012f2b5f7a1110961079f19eb0e28c9abacb95e4d0787352e4157ebca7044a4`
- detector: `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1`

## Build locally

Use JDK 17 and Android SDK platform/build-tools 34. The existing local
toolchain can be selected without installing anything:

```bash
export JAVA_HOME=$HOME/AppData/ActionMesh-fork/android-toolchain/jdk
export ANDROID_HOME=$HOME/AppData/ActionMesh-fork/android-toolchain/sdk
export PATH="$JAVA_HOME/bin:$ANDROID_HOME/platform-tools:$PATH"
cd $REPO/app/android_final
```

A Gradle 8.10.2 executable/wrapper is required. Once Gradle is available:

```bash
gradle :app:testDebugUnitTest :app:assembleDebug
# or, if the wrapper jar/scripts have been generated:
./gradlew --no-daemon :app:testDebugUnitTest :app:assembleDebug
```

APK output: `app/build/outputs/apk/debug/app-debug.apk`.

## Manual device install/test — requires user confirmation

No device was attached during development and no installation was attempted.
After explicitly choosing to connect and use a device:

```bash
adb devices -l
adb install -r $REPO/app/android_final/app/build/outputs/apk/debug/app-debug.apk
adb shell am start -n org.echogesture.finalapp/.MainActivity
```

On the phone, grant CAMERA, allow detector/model initialization, hold one or
both hands in view for at least 32 analyzed frames, tap Capture & translate,
and compare the caption/top-5 with the host reference. For CHEW validation,
compare against the existing CHEW2 host fixture whose top-1 is CHEW1.

Record phone model, Android version, model SHA-256, APK SHA-256, inference and
end-to-end timings, camera permission behavior, retry/error behavior, and
extended-run power/thermal observations. These remain **unverified** here.

## Source map

- `MainActivity.kt`: real CameraX preview/analysis, permissions, state machine,
  capture lifecycle and HUD.
- `HandTracker.kt`: RGB conversion, correct camera rotation, MediaPipe VIDEO
  tracking, canonical Left/Right hand slots, bounded frame ring.
- `FeatureTransform.kt`: direct port of `src/features.py` including linear
  resampling, shared scale, root centering, lag-1/lag-2 channels, and raw
  pre-centering inter-hand geometry.
- `LandmarkClassifier.kt`: memory-mapped FP16 model, strict float32 shape and
  tensor checks, stable softmax/top-5.
- `ModelContract.kt`: asset paths, exact model identity, label-map loading.
- `FeatureTransformTest.kt`: synthetic edge-case JVM tests; fixture parity
  tests consume the verified v7 JSON fixtures without model binaries.

## Verification status

Measured on this host (2026-11-10), with the toolchain paths above:

| Check | Result |
| --- | --- |
| Kotlin feature-parity JUnit (5 tests) | **OK (5 tests)** — compiled with `kotlinc 2.4.0` (Studio JBR/JDK 25) against the committed `src/test/resources/parity_*.txt` fixtures; fixtures byte-match `evaluation/device/v7_android/parity_*.npz` (max abs diff 5e-07, float32 rounding only) |
| Full app Kotlin compile (`MainActivity`, `HandTracker`, `LandmarkClassifier`, `ModelContract`, `FeatureTransform`) | **0 errors** against `android-34/android.jar` + real AndroidX/MediaPipe/TFLite class trees |
| Resource compile + link (`aapt2` 34.0.0) | **OK** — app + transitive AAR resources link to a 696 KB `resources.apk`; all `@string`/`@style` references resolve |
| DEX (`d8` via R8 9.5.29 from Android Studio) | **OK** — `classes.dex` (8.5 MB) + `classes2.dex` (0.5 MB) |
| APK assembled manually (`aapt2` + `d8` + `zipalign` + `apksigner`) | **Build succeeded** by hand using the pre-staged dependency class trees; `apksigner verify` passes (v1/v2/v3). This proves the app's code, resources and assets package; it is **not** an AGP/Gradle `assembleDebug` and is not installable-proof (no device). |
| Packaged asset identity | classifier/detector SHA-256 inside the APK match the pins (`a012f2b5…`, `fbc2a300…`) |
| Gradle/AGP `:app:assembleDebug`, on-device install, live camera, latency/power/thermal | **not run** — no Gradle distribution on disk (see below) |

The manual APK from this run is at `/tmp/a3_apk/echogesture-debug.apk`
(SHA-256 `4e80e9435b84a97bd768fc00bc8509e26326b3a87be6bfab9b05ce2a17bd9ce2`,
17,072,935 bytes; arm64-v8a native libs, CAMERA-only manifest, `versionCode=1`,
`minSdk 24`, `targetSdk 34`, launchable `org.echogesture.finalapp.MainActivity`).
It was produced with the class trees the earlier investigation left in
`$HOME/.cache/mpverify`; a clean build must go through Gradle (below) so
dependencies are resolved from Maven rather than inherited from a temp
directory. Without a connected device it is **compiled, packaged and signed
only** — not installed or run.

## Why Gradle is still blocked (and the exact unblock)

`gradle`, `./gradlew`, Android Gradle Plugin and a Gradle distribution are all
absent: no `~/.gradle`, no `~/.m2`, no `gradle-wrapper.jar` in
`gradle/wrapper/` (only `gradle-wrapper.properties`). Android Studio does ship
the Gradle **wrapper** classes (`org.gradle.wrapper.*` inside its
`gradle-plugin/lib/gradle-api-9.6.0.jar`) but that jar cannot bootstrap a
distribution. A real Gradle build therefore needs a one-time download you must
approve:

```bash
cd $REPO/app/android_final
# 1) obtain Gradle 8.10.2 (see gradle/wrapper/gradle-wrapper.properties)
#    e.g. download https://services.gradle.org/distributions/gradle-8.10.2-bin.zip,
#    unzip somewhere outside the repo, then either put `gradle` on PATH:
export GRADLE=/path/to/gradle-8.10.2/bin/gradle   # network fetch required

# 2) generate the wrapper jar from the real distribution:
"$GRADLE" wrapper --gradle-version 8.10.2

# 3) build (JDK 17 + ANDROID_HOME from above; `sdk.dir` in local.properties
#    or ANDROID_HOME both work — local.properties is git-ignored):
./gradlew --no-daemon :app:testDebugUnitTest :app:assembleDebug
```

`settings.gradle.kts` (pluginManagement → google/mavenCentral/gradlePluginPortal;
`dependencyResolutionManagement` FAIL_ON_PROJECT_REPOS) and
`app/build.gradle.kts` (AGP 8.5.2, Kotlin 2.0.20, compileSdk/targetSdk 34,
minSdk 24, Java/Kotlin jvmTarget 17, pinned CameraX 1.3.4 / MediaPipe 0.10.14 /
TFLite 2.16.1) are coherent and point at real artifacts. AGP 8.5.2 runs on
JDK 17; use the ActionMesh JDK 17 for Gradle, not the Studio JDK 25.

## Manual (no-Gradle) packaging recipe

This is the exact wiring used for the measured manual APK, for anyone without
Gradle. It reuses whatever local dependency class trees are available (`$M`);
it does not replace a real Gradle build.

```bash
SDK=$HOME/AppData/ActionMesh-fork/android-toolchain
export JAVA_HOME="$SDK/jdk"; export PATH="$JAVA_HOME/bin:$PATH"
BT="$SDK/sdk/build-tools/34.0.0"; AJ="$SDK/sdk/platforms/android-34/android.jar"
K="/Applications/Android Studio.app/Contents/plugins/Kotlin/kotlinc"
R8="/Applications/Android Studio.app/Contents/plugins/android/lib/r8.jar"
M=$HOME/.cache/mpverify                       # extracted AAR class trees + jars
CP="$AJ:$M/guava.jar:$M/tfliteapiroot/classes.jar:$M/x/tflite2/classes.jar:$M/org.json.jar"  # + androidx jars

# 1) resources: compile app res, and every dependency AAR's res/, then link.
"$BT/aapt2" compile --dir app/src/main/res -o /tmp/res_flat
for a in "$M"/*.aar; do unzip -q -o "$a" 'res/*' -d /tmp/aarres/$(basename "$a" .aar); done
"$BT/aapt2" compile --dir /tmp/aarres/<dep>/res -o /tmp/res_flat/<dep>   # each dependency
# aapt2 does not write `package`, minSdk, targetSdk or versionCode from
# build.gradle.kts — inject them into a copy of the manifest first:
sed -e 's|<manifest \(.*\)>|<manifest \1 android:versionCode="1" android:versionName="1.0">|' \
    -e 's|</manifest>|<uses-sdk android:minSdkVersion="24" android:targetSdkVersion="34"/></manifest>|' \
    -e 's|<manifest |<manifest package="org.echogesture.finalapp" |' \
    app/src/main/AndroidManifest.xml > /tmp/manifest.aapt2.xml
"$BT/aapt2" link -o /tmp/resources.apk -I "$AJ" --manifest /tmp/manifest.aapt2.xml \
  --auto-add-overlay --java /tmp/gen $(find /tmp/res_flat -name '*.flat')

# 2) Kotlin compile (real R.java on the classpath)
"$JAVA_HOME/bin/javac" -source 17 -target 17 -cp "$AJ" -d /tmp/classes /tmp/gen/org/echogesture/finalapp/R.java
JAVACMD="$JAVA_HOME/bin/java" "$K/bin/kotlinc" app/src/main/java/org/echogesture/finalapp/*.kt \
  -classpath "$CP:/tmp/classes" -d /tmp/classes -jvm-target 17

# 3) DEX — use Studio's newer R8; build-tools 34's d8 (R8 8.2.2) crashes on
#    androidx.appcompat classes.
"$JAVA_HOME/bin/java" -cp "$R8" com.android.tools.r8.D8 --min-api 24 --lib "$AJ" \
  --output /tmp/dex --classpath <each dependency jar> $(find /tmp/classes -name '*.class')

# 4) package resources + dex + lib/<abi>/*.so + assets/ into base.apk, then
#    5) align and 6) sign with the debug keystore:
"$BT/zipalign" -f -p 4 /tmp/base.apk /tmp/aligned.apk
"$BT/apksigner" sign --ks "$SDK/debug.keystore" --ks-pass pass:android \
  --ks-key-alias androiddebugkey --key-pass pass:android \
  --out echogesture-debug.apk /tmp/aligned.apk
"$BT/apksigner" verify echogesture-debug.apk
```

Native libraries are the arm64-v8a `jni/*.so` from `tasks-vision.aar`,
`tflite2.aar` and `camera-core.aar`; the four model assets come from the
`Prepare model assets` step above.
