# Android APK build evidence

Measured 2026-11-11T05:08:25Z.

| | |
| --- | --- |
| Path | `app/android_final/app/build/outputs/apk/debug/app-debug.apk` |
| Bytes | 75072782 |
| SHA-256 | `041e163fba2665bd33678a5463b0373c14d62b6b0c4cff28e67632aecc895b45` |
| Signature | apksigner verify exit 0, v2 scheme, CN=Android Debug |
| Alignment | zipalign -c 4 exit 0 |
| Unit tests | Gradle `:app:testDebugUnitTest` — 9/9 (4 ensemble + 5 feature) |

## Toolchain (actually used)

| Component | Version / path |
| --- | --- |
| Gradle | 8.10.2 full distribution (`ActionMesh-fork/android-toolchain/gradle-dist`) |
| JDK | Temurin 17.0.21 (`android-toolchain/jdk`) |
| Android SDK | platform android-34, build-tools 34.0.0 |
| AGP / Kotlin | 8.5.2 / 2.0.20 |

## Embedded assets (verified with `unzip -l`)

All four ensemble members plus labels, metadata and detector:

```text
7819105  assets/hand_landmarker.task
 939152  assets/landmark_ens_rob71_fp16.tflite
 939152  assets/landmark_ens_tcn21_fp16.tflite
 832888  assets/landmark_ens_tcnm61_fp16.tflite
 832888  assets/landmark_ens_tcnm62_fp16.tflite
   1492  assets/landmark_labels.json
    296  assets/landmark_model_meta.json
```

## Not verified

- **No device attached** (`adb devices` empty): no install, no camera, no on-device
  inference, no latency/power/thermal measurement.
- Assets and the APK are gitignored; neither is published.
