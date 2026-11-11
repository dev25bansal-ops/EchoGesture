# Android four-model verification

Measured 2026-11-11 on the build machine. Everything here is read from the
built APK, not from the source folder.

## Build

```
JAVA_HOME=<toolchain>/jdk  ANDROID_HOME=<toolchain>/sdk
./gradlew --offline testDebugUnitTest assembleDebug
```

JDK 17.0.21 · Android SDK 34 · build-tools 34.0.0 · Gradle 8.10.2 · AGP 8.5.2 ·
Kotlin 2.0.20. `BUILD SUCCESSFUL`.

## APK

| | |
| --- | --- |
| Path | `app/android_final/app/build/outputs/apk/debug/app-debug.apk` |
| Size | 78,880,127 B |
| SHA-256 | `67c836f52dc5c168bc98ed970fcd25cbf2da54af74adc4ec4998effb1682eeb0` |
| Signature | APK Signature Scheme v2, `CN=Android Debug`, RSA-2048 |
| Zipalign | verified, all `.so` stored and 4-byte aligned |
| minSdk / targetSdk | 24 / 34 |

## Permissions — fixed

The source manifest declared camera only, but the merged APK also carried
`INTERNET` and `ACCESS_NETWORK_STATE`, injected transitively by
`com.google.android.datatransport` via `com.google.mediapipe:tasks-core`. The
shipped app could therefore open a socket, which contradicts the stated privacy
property.

`AndroidManifest.xml` now removes both with `tools:node="remove"`. Read back from
the built APK:

```
uses-permission: android.permission.CAMERA
uses-permission: org.echogesture.finalapp.DYNAMIC_RECEIVER_NOT_EXPORTED_PERMISSION
```

No network permission remains. The offline claim is now true of the artifact,
not only of the source.

## Embedded assets — hashes verified

| Asset | SHA-256 (from inside the APK) | Pinned | |
| --- | --- | --- | --- |
| `hand_landmarker.task` | `fbc2a30080c3c557…` | `fbc2a300…c7de1` | match |
| `landmark_ens_tcn21_fp16.tflite` | `112279bc86090470…` | `112279bc…dc22a` | match |
| `landmark_ens_tcnm61_fp16.tflite` | `01a261d8eb6c2c4d…` | `01a261d8…390f63` | match |
| `landmark_ens_tcnm62_fp16.tflite` | `27058a85e032bee3…` | `27058a85…8d20b3` | match |
| `landmark_ens_rob71_fp16.tflite` | `08417e72da59150a…` | `08417e72…1ab21` | match |
| `landmark_labels.json` | `e78f5028e97e2eaf…` | `e78f5028…19a4a` | match |
| `landmark_model_meta.json` | `06f655bbec9c7518…` | `06f655bb…2ddbc` | match |

`landmark_classifier_fp16.tflite` — the obsolete single classifier — is **not**
in the APK and is not referenced by the dex.

## Tests

`testDebugUnitTest`: **20 run, 0 failed, 0 errors, 0 skipped**
(`ModelContractTest` 8, `EnsembleParityTest` 4, `EnsembleRuntimeParityTest` 3,
`FeatureTransformTest` 5).

These run on the JVM and therefore cannot load TFLite: `tensorflow-lite:2.16.1`
ships only Android `.so` files, with no macOS or Linux dylib. `EnsembleParityTest`
is explicit about this — it verifies mean-logit → single softmax → top-5 against
frozen logits produced by the Python reference. It is parity of *aggregation*,
not proof the app runs four interpreters.

`connectedDebugAndroidTest` — **NOT RUN.** No device is attached and no emulator
or system image is installed on this machine. `EnsembleInferenceTest` is written
and compiles into the test APK; it allocates all four real interpreters from the
packaged assets, asserts each tensor contract, runs inference, and compares
top-1, top-5 and confidence against the Python fixture. It requires a device.

## What is proven and what is not

## Build-time validation

`build.gradle.kts` fails the build when an asset is missing, when any member or
the detector or either JSON does not match its pinned SHA-256, when the members
listed in `landmark_model_meta.json` are not exactly the pinned member set, or
when the metadata's `quantization` is not `float16`.

Both semantic guards were tested by deliberately breaking the tree and
confirming the build failed with the intended message:

| Sabotage | Result |
| --- | --- |
| `quantization` set to `int8` | build failed at the metadata hash check |
| member list cut to three, hash re-pinned to isolate the set check | build failed: `Metadata members [...] do not match the pinned set [...]` |

The final tree builds clean and the APK is byte-identical to the one measured
above, so these checks add no artifact change — they only prevent a bad one.

**Proven from the artifact:**
- All four ensemble members are packaged, byte-identical to the pinned hashes.
- The metadata declares exactly those four, in that order.
- The dex contains `ModelContract`, `LandmarkClassifier` and
  `EnsembleAggregation` — the four-model code path, not the single-classifier one.
- No network permission.
- The app builds, signs, aligns, and its unit tests pass.

**Not proven without hardware:**
- That four interpreters allocate and run on device.
- On-device camera latency, capture-to-caption time, memory, power, thermal.
- Real hand detection against a live camera.

This gap is why the verdict is `ANDROID_FOUR_MODEL_VERIFIED_IN_RUNTIME_TESTS_DEVICE_PENDING`
at best, and not device-verified. See `ANDROID_DEVICE_TEST.md` for the procedure
that closes it.