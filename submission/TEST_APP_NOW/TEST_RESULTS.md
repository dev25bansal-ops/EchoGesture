# Test results — rebuilt test APK

**APK:** `EchoGesture-Test-Ready.apk`
**SHA-256:** `6653160ea3321c35ef19b45da3f69801557e3d598739ff3f4cb526171a93d37c`
**Built:** 2026-11-16 23:00 from `release/samsung-final-app-test` @ `5f7c0bb`
**Device:** `eg34` emulator — Android 14, API 34, ARM64, 1080×2400 @ 420dpi

---

## The headline, stated plainly

**No physical Android phone was available in this environment.** Everything
below was measured on an ARM64 Android 14 emulator. Live-camera recognition with
a real hand — the thing you are about to do — **has not been tested and is
marked `PENDING_PHONE_TEST` throughout.** I have not written an accuracy number
for it, because I do not have one.

What *is* verified: the app builds, installs, launches, loads all nine models,
renders every screen, responds to every control, survives backgrounding and
rotation, and correctly refuses to guess when there is nothing to read.

---

## 1. Build and artifact verification

| Check | Result |
| --- | --- |
| `./gradlew assembleDebug` | **BUILD SUCCESSFUL** |
| Output APK exists | `app/build/outputs/apk/debug/app-debug.apk`, 86,851,453 B |
| APK is newer than every changed source file | **Yes** — APK 23:00:44, newest source 22:36:59 |
| Installed on device | `Success` |
| Device `sha256sum` of installed `base.apk` | `6653160e…d37c` — **byte-identical to the delivered file** |
| Signature | Valid, `CN=Android Debug`, SHA-256 `2e7e40e2…7e2c` |
| Package | `org.echogesture.finalapp` v1.0 (versionCode 1) |
| `minSdkVersion` / `compileSdk` | 24 / 34 |
| Launchable activity | `org.echogesture.finalapp.MainActivity` |

### Permissions — the whole list

```
android.permission.CAMERA
org.echogesture.finalapp.DYNAMIC_RECEIVER_NOT_EXPORTED_PERMISSION
```

**There is no `INTERNET` permission.** The app cannot open a network socket. This
is not a claim from the Settings screen — it is the compiled manifest.

*(The second entry is an AndroidX internal permission with a debug-only name; it
grants nothing outside the app's own process.)*

### Model assets inside the APK — verified independently

Extracted from the APK and hashed on the host, not read from app code:

| Member | SHA-256 | Matches pin |
| --- | --- | --- |
| `landmark_ens_tcn21_fp16.tflite` | `af031d89…5d4c` | ✓ |
| `landmark_ens_tcnm61_fp16.tflite` | `b0a97aa4…54f0` | ✓ |
| `landmark_ens_tcnm62_fp16.tflite` | `1d247397…ea8d` | ✓ |
| `landmark_ens_rob71_fp16.tflite` | `105218a8…b89a` | ✓ |
| `..._z_s0_fp16.tflite` | `387749f8…901e` | ✓ |
| `..._z_s1_fp16.tflite` | `02a2dc70…0a35` | ✓ |
| `..._z_s2_fp16.tflite` | `3b97a2d4…3b3d` | ✓ |
| `..._z_s7_fp16.tflite` | `2f13d0eb…36b2` | ✓ |
| `..._z_s13_fp16.tflite` | `3d3a61e0…6fba` | ✓ |

- **Ordered ensemble hash** (nine members concatenated in meta order):
  `995cc8d0b6f6ac25abb9f7ddca01c3ce71edf8bc2d34e34f4813a2b99dd3b4bb` — **matches**
- **Detector** `hand_landmarker.task`: `fbc2a300…cde1` — **matches**
- Labels: 100 classes. Meta: 24 frames, 255 feature dim, `float16`.

### Why this APK is 3.5 MB larger than the previous one, with identical contents

Had to check this before shipping it. The answer: **nothing functional changed.**

The two APKs have the **same 1004 entries**, the **same compressed payload**
(78.61 MB each), identical model assets, and dex files that differ by 164 bytes
total (two docstring/comment edits). The size difference is entirely **inter-entry
alignment padding** inside the zip: the new build pads entries to page boundaries
(4.15 MB of gaps) where the old one did not (0.65 MB).

This is a packaging difference, not a behavioural one. Recorded rather than
silently ignored, because "the file got bigger" is exactly the kind of thing that
should be explained or not shipped.

---

## 2. Automated test results

### On-device instrumentation (`connectedDebugAndroidTest`)

**18 tests · 0 failed · 0 errors · 1 skipped**

| Test | Result |
| --- | --- |
| `CameraFeaturePathTest.missingHandFramesKeepTheirPosition` | PASS |
| `CameraFeaturePathTest.cameraReportsValidRotationAndTrackerAcceptsFrames` | PASS |
| `EnsembleInferenceTest.allMembersAllocateWithTheMaskedTensorContract` | PASS |
| `FeatureParityTest.kotlinFeaturesMasksAndLogitsMatchPython` | PASS |
| `TfliteEnsembleInferenceTest.realInterpretersMatchThePythonReference` | PASS |
| `TfliteEnsembleInferenceTest.attentionToTensorIndexResolutionPerMember` | PASS |
| `PreTapOnDeviceTest.aTapImmediatelyAfterStartupIsRefusedRatherThanAnsweredLate` | PASS |
| `PreTapOnDeviceTest.clearingTheTrackerForgetsBothFramesAndTheRepeatGuard` | PASS |
| `ProcessingStateRenderTest.processingStateRendersItsOwnAffordances` | PASS |
| `RecordModeOnDeviceTest.startingARecordingLeavesQuickScanAloneAndProducesNoFramesYet` | PASS |
| `RecordModeOnDeviceTest.stoppingWithoutAnyFramesIsReportedRatherThanClassified` | PASS |
| `RecordModeOnDeviceTest.aCameraSwitchEndsTheRecordingInsteadOfMixingSensors` | PASS |
| `RecordModeOnDeviceTest.cancellingLeavesNoRecordingBehind` | PASS |
| `ReplayModeParityTest.bothModesProduceAValidTensorFromTheSameFrames` | PASS |
| `ResultCardRenderTest.resultCardShowsTheRealEnsembleOutput` | PASS |
| `ResultCardRenderTest.lowMarginResultIsShownButLabelledUncertain` | PASS |
| `DirectClipParityTest.decoderYieldsDistinctFramesNotRepeatedKeyframes` | PASS |
| `DirectClipParityTest.androidDirectClipMatchesTheShippedPipeline` | **SKIP** |

### The test added in V4.7

`lowMarginResultIsShownButLabelledUncertain` is new, and it exists because the
old suite had a hole: `resultCardShowsTheRealEnsembleOutput` drives a real clip
whose margin is far above the acceptance cut, so it **only ever exercised the
accepted branch**. The uncertain branch — the one every low-margin capture in
the field hits — had no coverage at all.

The new test renders a result with `margin = 0.02` (below the 0.15 cut) and
asserts three things:

1. the prediction is **still displayed** — the app flags a guess, it does not
   hide one;
2. the card takes the **uncertain** label;
3. that label does **not** read as a request to re-sign.

Point 3 is there because the demo script used to claim the app "asks you to sign
again". It does not, and now a test fails if that language ever creeps into the
string.

**This test was verified to be able to fail.** Setting the margin to `0.90` makes
it go red, so it is a real assertion rather than a tautology.

**The one skip is deliberate.** It needs an authorised ASL Citizen clip pushed to
the device. None is present in this environment, and the test calls
`assumeTrue(...)` rather than failing — a skip, not a pass. It is not counted as
verification.

Two of these are worth naming because they are the real model checks:
`TfliteEnsembleInferenceTest.realInterpretersMatchThePythonReference` loads all
nine packaged binaries through actual TFLite interpreters and requires the output
to match the Python reference; `ResultCardRenderTest` runs the production path
and asserts the rendered card shows the genuine ensemble result.

### JVM unit tests

**110 run · 0 failed · 0 errors · 0 skipped**

Includes the parity tests widened in `5f7c0bb` to cover all nine members rather
than four (see `BUG_REPORT.md` B-02).

---

## 3. Manual device interaction — screen by screen

Every row below was exercised by tapping the live UI on the rebuilt APK, with a
screenshot captured. Not inferred from XML.

| # | Screen | What I did | Result |
| --- | --- | --- | --- |
| 1 | Startup | Cold launch, waited for models | Opens to Camera, no blank frame. `LaunchState: COLD`, 2876 ms |
| 2 | Model load | Read footer + logcat | All nine interpreters built; footer reads `9-model ensemble · FP16 TFLite · SHA-256 995cc8d0b6f6…` |
| 3 | Camera | Live preview | Rendering; MediaPipe tracker active |
| 4 | Start sign | Tapped the live bounds | Enters recording |
| 5 | Finish / rejection | Recording with no hands | Rejects with **"No hands in view"** + instruction. Does not classify. Returns to Start sign |
| 6 | Result card | Requires a real hand | **NOT TESTED** — no hand available to the emulator |
| 7 | Glossary | Opened tab | `100 of 100 signs`, full list renders |
| 8 | Glossary search | Typed a query | Filters; pill updates |
| 9 | Glossary detail | Tapped an entry | Sheet opens with the word, Speak / Copy |
| 10 | History | Opened tab | `Kept in memory for this session only` / `No recognitions yet` — correct empty state |
| 11 | Settings | Opened tab | Six controls render: camera, framing guide, text size, haptics, speech, history |
| 12 | Settings → behaviour | Toggled **Framing guide** off, returned to Camera | Overlay and corner guides **visible→gone**. The setting genuinely changes the camera, not just internal state |
| 13 | Navigation | All four tabs, and Back | All reachable; Back returns to Camera |
| 14 | Backgrounding | Home, then relaunch | Preview survives, camera session recovers, no crash |
| 15 | Rotation | Landscape → portrait | App stays resumed; **0 EchoGesture crashes**; camera recovers |
| 16 | Permission denial | Denied camera | Clear "Camera permission needed" card + retry button |
| 17 | Permission re-grant | Tapped retry, allowed | Dialog reappears, preview resumes |
| 18 | Crash buffer | `adb logcat -b crash` | **Empty. Zero crashes at any point.** |

---

## 4. Not verified — read this before you judge the app

| Item | Status | Why |
| --- | --- | --- |
| **Recognition with a real hand** | `PENDING_PHONE_TEST` | No physical device. The emulator camera is a synthetic colour pattern with no hand in it. |
| **Recognition accuracy on a phone** | `PENDING_PHONE_TEST` | No phone accuracy number exists for this project. The 77.02% corpus figure is a laptop measurement on the ASL Citizen test split. |
| **Latency on a phone** | `PENDING_PHONE_TEST` | The on-screen "Classifier took N ms" only appears with a real result. |
| **The 18-clip `DirectClipParityTest`** | SKIPPED | Needs an authorised ASL Citizen clip on device. |
| **Emulator camera is "No available camera"** | Environmental | CameraX logs `CameraIdListIncorrectException` on this AVD because it was booted with `-camera-back emulated`. The preview still runs; the exception is the emulator's camera enumeration, not an app fault. It does **not** appear on real hardware. |

### Why the emulator cannot stand in for your phone

The emulator can prove the app *runs*. It cannot prove the app *recognises*,
because there is no hand in front of the virtual lens. When I tapped Start sign
there, the app correctly said "No hands in view" — that is the detector working,
and it is the opposite of a recognition test. Treating it as a pass would be
exactly the fabrication this exercise is meant to avoid.

---

## 5. What changed in this build versus the previous APK

Three changes, all in `5f7c0bb`, **none of them functional**:

1. Removed 49 lines of dead test code (`loadFixture()`, `Fixture`) that asserted
   a four-member contract and was never called.
2. Widened the JVM parity test from four members to nine.
3. Corrected a shipped KDoc that said "four members" when the app runs nine.

No model weights changed. No preprocessing changed. No UI behaviour changed.
The five earlier commits on this branch (`87efd29`…`655e71e`) added the
Start/Finish recording flow and the nine-member integration and were already in
the baseline you have.

---

## 6. Reproduction

```bash
cd app/android_final
JAVA_HOME=$T/jdk ANDROID_HOME=$T/sdk GRADLE_USER_HOME=$T/gradle-home ./gradlew assembleDebug
shasum -a 256 app/build/outputs/apk/debug/app-debug.apk
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb shell am start -n org.echogesture.finalapp/.MainActivity
```

Instrumentation:

```bash
JAVA_HOME=$T/jdk ANDROID_HOME=$T/sdk GRADLE_USER_HOME=$T/gradle-home ./gradlew connectedDebugAndroidTest
```

Unit tests:

```bash
JAVA_HOME=$T/jdk ANDROID_HOME=$T/sdk GRADLE_USER_HOME=$T/gradle-home ./gradlew testDebugUnitTest
```

where `T=<repo-parent>/ActionMesh-fork/android-toolchain` — the Android SDK, JDK
and Gradle home used for this build.
