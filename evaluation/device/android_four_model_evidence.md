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
| Size | 75,072,706 B |
| SHA-256 | `2ec087e543c2065d609ecea43056a390e9ce7248c7e2a52e8dc24e1282d44fcb` |
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

`connectedDebugAndroidTest` — **RUN AND PASSING** on an Android emulator.

Device: Android 14 (API 34), arm64-v8a, AVD `eg34`, headless.
`./gradlew connectedDebugAndroidTest` → **4 tests, 0 failures, 0 errors, 0 skipped, 0.659 s**.

| Test | Time |
| --- | --: |
| `allFourMembersAllocateWithTheFrozenTensorContract` | 0.008 s |
| `productionClassifierReproducesPythonOnFrozenFeatures` | 0.167 s |
| `attentionToTensorIndexResolutionPerMember` | 0.091 s |
| `fourRealInterpretersMatchThePythonReference` | 0.090 s |

These allocate all four real `Interpreter` instances from the packaged assets,
assert each member's tensor contract, run the fixture's real 24x255 feature
vector through every member, and compare top-1, top-5 and confidence against the
Python reference (`WHATFOR1`, 0.882111).

Two defects in the instrumentation tests themselves were found and fixed by
running them:

- The feature vector was fed as an all-zero tensor, which runs the interpreters
  but lands on `BREAKFAST1` — a meaningless result that a passing assertion
  would have hidden. It now replays the fixture's real features.
- The feature line index was hardcoded as `7 + 1 + 1` for a 4-member fixture, and
  the logits fixture (12 lines, no features) was read instead of the runtime
  fixture (16 lines, real features). Both corrected.

An emulator is not target hardware: it validates that four interpreters allocate,
execute and aggregate correctly under the Android runtime, but not camera
capture, sensor behaviour or real-device latency.

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

## Reproducibility — corrected

An earlier revision of this document claimed the APK reproduced **bit-for-bit**
across independent trees. That claim is wrong and is withdrawn.

Measured:

| Check | Result |
| --- | --- |
| Two clean builds in the **same** tree | identical SHA-256 |
| Clean builds in **two different** trees | **differ** (`e5ed0a3b…` vs `4ebe0774…`) |
| Entries differing between trees | 1 of 480 — `classes3.dex` (90,344 vs 90,288 B) |
| Model assets, detector, labels, metadata | **byte-identical** (CRC matches) |
| Entry count / central directory | identical (480) |

The variance is in D8's synthetic-lambda naming: the compiler derives
identifiers from inputs that differ across build environments, so the dex bytes
differ while `dexdump -d` disassembly and predictions are identical. Nothing in
the shipped model varies.

**What a reviewer should take from this:** the APK is reproducible in *content*,
not bit-reproducible across machines. The identity guarantee that actually
matters — four members, detector, labels, metadata — is enforced at load time by
`ModelContract` against pinned SHA-256 values, so a substituted model cannot run
whatever dex bytes were emitted. Quote the asset hashes, not a single APK
SHA-256.

An earlier build also reported 78,880,127 B. That artifact was defective: it
carried 540 local headers for 478 real entries, so ~3.8 MB of orphaned local
headers had accumulated across incremental Gradle builds. Deleting
`app/build/outputs/apk` and rebuilding produces the clean 480/480 archive. The
78,880,127 B figure should not be quoted.

**Proven from the artifact:**
- All four ensemble members are packaged, byte-identical to the pinned hashes.
- The metadata declares exactly those four, in that order.
- The dex contains `ModelContract`, `LandmarkClassifier` and
  `EnsembleAggregation` — the four-model code path, not the single-classifier one.
- No network permission.
- The app builds, signs, aligns, and its unit tests pass.

**Not proven without physical hardware:**
- On-device camera latency, capture-to-caption time, power, thermal.
- Real hand detection against a live camera.
- Camera rotation and handedness on a physical sensor (code-derived only).

Real inference is now verified on an Android runtime. What remains unverified is
physical hardware specifically: real camera capture, sensor rotation and
handedness on a device, and on-device latency, power and thermal. See
`ANDROID_DEVICE_TEST.md`.