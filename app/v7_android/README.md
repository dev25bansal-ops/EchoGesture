# EchoGesture V7 — app slice (A09)

See `evaluation/device/v7_android/README.md` for the full evidence log, the
verified frozen model contract, and the exact setup / acceptance commands.

## What this is

The smallest genuine offline Android path for the frozen FP16 candidate
(`a012f2b5…7044a4`, `[1,24,255] float32 → [1,100] float32 logits`, 100
classes): CameraX preview → MediaPipe HandLandmarker (VIDEO, 2 hands,
thresholds 0.3) → Kotlin port of `src/features.py` (255 features) → TFLite
classifier (LiteRT) with the same stable softmax + top-5 as `src/main.py`.

Manifest requests **CAMERA only** — no network permission, no video
persistence; frames stay on the device.

## Layout

- `settings.gradle.kts`, `app/build.gradle.kts` (pinned: CameraX 1.3.4,
  MediaPipe Tasks Vision 0.10.14, TFLite 2.16.1, AGP 8.5.2, Kotlin 2.0.20)
- `gradle/wrapper/gradle-wrapper.properties` (gradle-8.10.2 — wrapper jar
  must be generated after SDK install; see README §4)
- `src/main/AndroidManifest.xml`
- `src/main/java/org/echogesture/v7/`
  - `FeatureTransform.kt` — exact port of `src/features.py` (resample,
    shared-scale root-centring, pos/vel/acc + pre-centring inter-hand geom)
  - `Classifier.kt` — strict shape/dtype checks, softmax/top-5 adapter only
  - `MainActivity.kt` — CameraX + MediaPipe VIDEO + capture-to-predict flow
- `src/main/assets/` — `landmark_classifier_fp16.tflite`, `hand_landmarker.task`,
  `labels.json` (ordered vocab), `model_contract.json`
- `src/test/` — `FeatureParityTest.kt` + `parity_synthetic.json` /
  `parity_chew2.json` (identical numbers to the Python fixtures in
  `evaluation/device/v7_android/`)

## Status

Host-verified: candidate strictly compatible, fixtures regenerate exactly,
CHEW 2 clip top1 = CHEW1 (see `evaluation/device/v7_android/test_parity.py`).

**Not built / not run:** no Android SDK, adb, Gradle, emulator, or device is
available on this host (evidence: `evaluation/device/v7_android/env_probe.txt`),
so there is no APK, compile, unit-test, install, or on-device result to
report. Unblock commands are in `evaluation/device/v7_android/README.md` §4.
