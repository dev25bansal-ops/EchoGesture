# EchoGesture Final — Offline Android caption app

This is a buildable Android project for the frozen EchoGesture landmark
ENSEMBLE (4 members). It contains no model binaries and makes no APK, device,
power, thermal, or camera-permission verification claims on this host.

## Offline pipeline

CameraX front-camera preview and frame analysis -> MediaPipe HandLandmarker
(VIDEO mode, 2 hands, confidence thresholds 0.3) -> exact Kotlin port of
`src/features.py` -> **all four** FP16 ensemble members via LiteRT/TFLite ->
element-wise logit averaging -> one stable softmax -> top-5 -> caption HUD.

The ordered member list is read from `landmark_model_meta.json` at load time;
there is no hardcoded member count. Each member is loaded as its own
`Interpreter` and validated independently. **Each member's own input/output
tensor indices are used** — the shipped members genuinely differ (Python
observes `out_idx` 224 for `tcn21`/`rob71` and 191 for `tcnm61`/`tcnm62`, and
`in_idx` 0 for all). Reusing one member's index for another reads the wrong
tensor and silently changes predictions; this port never shares them.

An incomplete asset set is a hard error: a member listed in the meta sidecar
that is missing from `assets/`, unpinned, or hash-mismatched fails model load,
and the HUD shows the error instead of running a smaller ensemble.

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

The HUD permanently shows `OFFLINE`, the ensemble identity (member count,
quantization, class count), the combined SHA-256 prefix, capture frame count,
hand-frame count, and timings. The app requests CAMERA only; there is no
INTERNET or network-state permission.

## Prepare model assets (local copy; do not commit)

Copy **four `.tflite` members** plus the labels, meta and detector. The member
names are exactly those listed in `models/landmark_model_meta.json`:

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

That is 7 files in total: 4 ensemble members + labels + meta + detector
(3,544,080 bytes of model). The Android asset names are fixed by
`ModelContract.kt`. Gradle validates every copied file and all five binary
SHA-256 hashes (four members + detector) before configuration or assembly.
Model bytes stay in the root `models/` directory and are excluded by
`app/android_final/.gitignore`; no second model copy belongs in Git.

Expected hashes:

| Asset | SHA-256 |
| --- | --- |
| `landmark_ens_tcn21_fp16.tflite` | `112279bc86090470b82b20b510746a946220322c25ec84cb3706a538c73dc22a` |
| `landmark_ens_tcnm61_fp16.tflite` | `01a261d8eb6c2c4d8b501f0ce1a36db338365d17fddabc63528a7941b1390f63` |
| `landmark_ens_tcnm62_fp16.tflite` | `27058a85e032bee3af444fb1538ff0bf32fd38fa2b01ce01a2f25c3a428d20b3` |
| `landmark_ens_rob71_fp16.tflite` | `08417e72da59150ac9adf749e6324c30fb001d4a401394e32f125d3276d1ab21` |
| detector `hand_landmarker.task` | `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1` |

The combined identity shown in the HUD is the SHA-256 of the four members
concatenated in meta order:
`0e4c392732e7da7e77632620dc72e4afce8b1a5aa00234dbf7a13a1049a7ad93`.

### Label mapping

`ModelContract.parseLabels` maps `{label: index}` to an index-ordered list with
`uppercase()`, exactly like `src/main.py` (`k.upper()`). All keys in
`landmark_labels.json` are already uppercase, so the two paths produce an
identical 100-label vocabulary — including the one compound label,
`HURDLE/TRIP1`. No label translation is needed for a caption comparison.

### Parity reference for the reference clip

On `evaluation/device/v5_accuracy/data/val_30/0283886564670357-WHAT FOR.mp4`,
the Python reference (`src/main.py::predict`, direct read) returns
`WHATFOR1`, confidence `0.8821`, top-5
`[WHATFOR1, OFFEND, KNIGHT1, SCROLLDOWN, DECIDE1]`. The v7 demo CLI, which
re-encodes sampled frames to a temporary clip before predicting, returns
`WHATFOR1`, confidence `0.8643`, top-5
`[WHATFOR1, OFFEND, KNIGHT1, DECIDE1, PARTY1]`. The difference is the
re-encode, not the ensemble. Use the direct `predict` values for on-device
comparison.

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

On the phone, grant CAMERA, allow detector/model initialization (the HUD shows
`4-model ensemble · float16 · 100 classes · SHA-256 <combined>… · 4 members`),
hold one or both hands in view for at least 32 analyzed frames, tap Capture &
translate, and compare the caption/top-5 with the host reference above. For
CHEW validation, compare against the existing CHEW2 host fixture whose top-1
is CHEW1.

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
- `LandmarkClassifier.kt`: one `Interpreter` per ensemble member, strict shape
  and dtype checks per member, element-wise logit averaging, one stable
  softmax, top-5. Rejects an empty or duplicate-member set at construction.
- `EnsembleAggregation.kt`: pure mean-logits / softmax / top-k math with no
  Android or TFLite types, so JVM unit tests can pin it to the Python reference.
- `ModelContract.kt`: meta-driven member list, per-member pinned SHA-256,
  combined-ensemble hash, asset paths, label-map loading. Rejects an
  incomplete, unpinned or hash-mismatched member set.
- `FeatureTransformTest.kt`: synthetic edge-case JVM tests; fixture parity
  tests consume the verified v7 JSON fixtures without model binaries.
- `EnsembleParityTest.kt`: aggregation parity against real per-member logits
  captured from the Python reference (aggregation math only; no interpreter).

## Verification status

Measured on this host (2026-11-11) unless noted, with the toolchain paths above:

| Check | Result |
| --- | --- |
| Kotlin JVM tests (20 tests: 4 ensemble, 3 ensemble-runtime, 8 model-contract, 5 feature) | **OK (20/20 pass)** via `:app:testDebugUnitTest` (JDK 17, Gradle 8.10.2, offline). `EnsembleParityTest` consumes real 4x100 member logits captured from `src/main.py::_run_ensemble` on the reference clip; mean/softmax/top-5 reproduce Python to `<1e-5` / `<1e-6` and top-1 `0.8821`. This is **aggregation math only** — no TFLite interpreter runs in a JVM test. |
| Typecheck of changed non-CameraX code (`LandmarkClassifier`, `ModelContract`, `EnsembleAggregation`, `FeatureTransform`) | **OK** — full `:app:compileDebugKotlin` against the real CameraX 1.3.4 / tasks-vision 0.10.14 / TFLite 2.16.1 AARs. |
| `MainActivity.kt`, `HandTracker.kt` full compile | **OK** — `:app:compileDebugKotlin` succeeds against the real dependency class trees (CameraX 1.3.4, tasks-vision 0.10.14). |
| Kotlin feature-parity JUnit (5 tests) | **OK (5 tests)** — fixtures byte-match `evaluation/device/v7_android/parity_*.npz` (max abs diff 5e-07, float32 rounding only) |
| Resource compile + link (`aapt2` 34.0.0) | **OK** — every `@string`/`@style` reference resolves via `:app:assembleDebug`. |
| Packaged APK | **OK** — `:app:assembleDebug` produced `app/build/outputs/apk/debug/app-debug.apk`; `aapt2 dump permissions` shows only `CAMERA` (plus the auto-generated dynamic-receiver permission — no `INTERNET`/`ACCESS_NETWORK_STATE`), and all seven assets (4 members + labels + meta + detector) are present at the pinned sizes. |
| Gradle/AGP `:app:assembleDebug`, on-device install, live camera, latency/power/thermal, **ensemble parity on device** | Build **OK** offline. On-device install, live camera, latency/power/thermal and ensemble parity on a real phone remain **not run** — no device attached. |

### Android blockers (exact)

1. **Gradle**: resolved — Gradle 8.10.2 and the ActionMesh JDK 17 are present on
   this host; `./gradlew --offline :app:testDebugUnitTest :app:assembleDebug`
   runs against the cached dependency tree (offline Gradle needs
   `GRADLE_USER_HOME=~/.gradle`, which holds the JUnit/org.json test deps that
   the toolchain cache lacks).
2. **Dependency class trees**: resolved — the CameraX 1.3.4, tasks-vision/
   tasks-core 0.10.14 and TFLite 2.16.1 AARs are in the Gradle cache; the
   module compiles against them.
3. **No device / no adb target**: on-device inference, the ensemble tensor
   index behavior under Android's TFLite build, latency, power and thermal were
   **not** measured. An APK was built but not installed. Any on-device parity
   claim is unverified.

## Rationale: reproducing the offline build

The build now runs offline on this host with the ActionMesh toolchain (below).
This section is kept as the reproduction recipe; on a machine without a Gradle
distribution or cached dependencies the same steps apply after a one-time,
approved download.

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

This is the exact wiring used for an earlier single-model manual APK, for
anyone without Gradle. It reuses whatever local dependency class trees are
available (`$M`); it does not replace a real Gradle build. On this host `$M`
(`~/.cache/mpverify`) is now **absent**, so this recipe cannot be re-run here
as written; it is kept as reference only.

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
`tflite2.aar` and `camera-core.aar`; all seven prepared assets (four ensemble
members + labels + meta + detector) come from the `Prepare model assets` step
above.
