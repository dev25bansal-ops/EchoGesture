# ECHOGESTURE FINAL RECOGNITION STATUS: **PARTIAL**

Real-phone verification is **BLOCKED**: no physical Android device was attached
at any point (`adb devices` showed only `emulator-5554`; `system_profiler` found
no USB Android device; no wireless ADB target answered). Paths C and D were
therefore never run, and the real CameraX analysis rate was never measured.

What *was* established is substantial and it changes the picture.

---

## 1. Root cause found

Three separate problems, not one.

**A. The release branch could not build a working app.** It shipped **four old
model binaries** (`112279bc…`) where `ModelContract` pins **nine**
(`af031d89…`, …) and requires a `members_detail` block the branch's meta file
did not contain. Any APK built from that tree throws during model load — it does
not misrecognise, it fails to start. **Confirmed and fixed.**

**B. Three camera/permission recovery defects** that all present as "the app
does not work": a failed camera start was permanent, a granted permission did
not clear the blocking card, and a permission granted from system Settings was
never noticed. **Confirmed and fixed.**

**C. The recognition path itself is not broken.** Measured: the model scores
**302/345 = 87.5%** on authorised validation clips across **five unseen
signers**; the Android pipeline scores **26/30 = 86.7%** on the same kind of
clips on the device. The two agree, which rules out Android preprocessing and
inference as the cause. **The user's "most signs wrong" is therefore still
unexplained and lives somewhere the available hardware could not reach.**

## 2. Actual fixes implemented

| Fix | File |
| --- | --- |
| Camera provider reset via public `shutdown()` | `CameraScreen.kt` |
| Retry button restarts the provider in ERROR state | `CameraScreen.kt` |
| Permission card clears on grant | `CameraScreen.kt` |
| `onResume()` re-checks a Settings-granted permission | `CameraScreen.kt`, `MainActivity.kt` |
| Acceptance rule moved into a measurable object | `ResultPolicy.kt` |
| Nine members + `members_detail` restored | `assets/`, `models/` |
| Real analyser instrumentation ported | `AnalysisProfiler.kt` |
| Device accuracy test | `ClipReplayAccuracyTest.kt` |
| Frame-selection contract tests | `RecordingFeaturesTest.kt` |

## 3. How each was proven

| Fix | Proof |
| --- | --- |
| Camera reset | Bytecode read (`javap -c`) shows `getstatic` + `invokevirtual`; the wrong version crashed with NPE at `ProcessCameraProvider.getInstance:182`, reproduced every time |
| Permission card | Emulator: deny → card; grant → card clears, camera opens |
| Settings grant | Emulator: `pm revoke`, `pm grant`, tap card → recovers, no crash |
| Acceptance rule | Calibrated on 345 clips (see §5) |
| Nine members | APK extracted, all nine hashes match `MEMBER_SHA256`; ensemble digest `995cc8d0b6f6ac25…`; app footer displays it |
| Accuracy | 26/30 native, 24/30 squash, on-device |
| Frame selection | 7 new tests: timestamp sampling above budget, verbatim pass-through below, no duplicates, fps from real timestamps |

## 4. Real phone results before and after

**None. No phone was attached.** I will not substitute the clip-replay result
for this — it is a different measurement, and the brief forbids the swap
explicitly.

## 5. Correct / wrong / rejected

Measured on 345 authorised clips with real probabilities captured from the
ensemble:

| Group | Share | Accuracy |
| --- | --- | --- |
| Accepted (margin ≥ 0.15, score ≥ 20%) | 86.1% | **96.0%** |
| Rejected | 13.9% | 35.4% |

Baseline with no gate: 87.5%. The gate is doing its job — accepted readings are
96% correct, declined readings are near chance.

## 6. Android / Python parity

| Path | Clips | Correct | Accuracy |
| --- | --- | --- | --- |
| Python (`src/main.py`) | 345 | 302 | **87.5%** |
| Android on device | 30 | 26 | **86.7%** |

Within one point. The detector, feature transform, attention mask, nine-member
ensemble and label mapping all agree.

## 7. Actual CameraX analysis rate

**NOT MEASURED.** `AnalysisProfiler` is now in the build so it can be read from
a real session, but no real session happened. This is the single most important
unknown, and `diagnostics/final-repair/04_LIVE_RISK.md` explains why: if the rate
is near 10 fps, a median 2.2 s sign yields ~20 frames, which both takes the
unresampled path and leaves only 3–6 of 24 model steps carrying a real
observation. Measured on real clips: hand-bearing frames fall from 13–21 at a
32-frame budget to 3–6 at a 10-frame budget.

## 8. Auto-stop behaviour

**NOT IMPLEMENTED on this branch.** The flow here is two-tap (Start sign →
Finish sign). Phase 7 was gated on validating the basic classification pipeline
first, which required a phone. A one-tap implementation exists on a separate
branch and was deliberately **not** merged, because putting an unvalidated
capture path in front of a demo is the kind of speculative change the brief
forbids.

## 9. Test counts

| Suite | Result |
| --- | --- |
| JVM unit tests | **117 pass, 0 fail** |
| Instrumentation tests | **19 pass, 0 fail** (1 skip needs a pushed clip) |
| Clip accuracy (device) | 26/30 native, 24/30 squash |
| Clip accuracy (Python) | 302/345 |

## 10. APK exact location

```
~/Desktop/EchoGesture-Final-Fixed/EchoGesture-Final-Fixed.apk
```

## 11. APK full SHA-256

```
39295e6ae1c61f648d16f359323c38fe1a343c943dd727b4b7015efd298f35b6
```

Rebuilt from the pushed commit and re-hashed to confirm the file matches the
source: identical.

## 12. GitHub branch

`release/samsung-final-app-test`

## 13. Local commit SHA

```
4443bcfe671d71acf206eeca2c4a510c208978a1
```

## 14. GitHub remote commit SHA

```
4443bcfe671d71acf206eeca2c4a510c208978a1
```

Verified with `git ls-remote` after the push. They match.

## 15. Pushed files

```
app/android_final/app/src/main/java/org/echogesture/finalapp/CameraScreen.kt
app/android_final/app/src/main/java/org/echogesture/finalapp/MainActivity.kt
app/android_final/app/src/main/java/org/echogesture/finalapp/ResultPolicy.kt
app/android_final/app/src/main/java/org/echogesture/finalapp/AnalysisProfiler.kt
app/android_final/app/src/androidTest/java/org/echogesture/finalapp/ClipReplayAccuracyTest.kt
app/android_final/app/src/test/java/org/echogesture/finalapp/RecordingFeaturesTest.kt
diagnostics/final-repair/00_ROOT_CAUSE.md … 07_PHONE_STATUS.md  (8 files)
submission/TEST_APP_NOW/BUILD_MANIFEST.json
submission/TEST_APP_NOW/SHA256SUMS.txt
```

No `.apk`, `.tflite`, `.task`, `.zip` or video file was pushed.

## 16. Remaining problems

1. **No phone test.** Paths C and D unmeasured; analyser rate unmeasured. This
   is the blocking item.
2. **No model file is tracked in git.** The repo is public and every `.tflite`,
   `.task`, and the label/meta JSON are gitignored. A fresh clone cannot build
   the app even after the model fix. Committing weights to a public repo is a
   rights decision for you, not one I should make silently.
3. **Auto-stop not on this branch.**
4. **The live recognition failure is still unexplained.** The most likely
   candidate is captured in `04_LIVE_RISK.md`, but it is a hypothesis.

## 17. Safe to demonstrate tomorrow?

**The app will install, launch, load the nine-model ensemble, open the camera,
and recognise a known clip replayed through it at ~87%.**

It will **not** reliably recognise a person signing live, because that path has
never been verified on hardware and the evidence says something in live capture
is degrading the features.

**Recommendation:** demonstrate using a prepared video replayed through the app,
or state plainly that live signing is still under investigation. Do not present
live signing as working — the model is sound, but the capture path is unproven,
and a demo that fails live is worse than a demo that is scoped honestly.

---

### The one action that would change this report

Attach a phone, enable capture diagnostics, perform one known sign, and read:

```bash
adb shell run-as org.echogesture.finalapp cat files/diagnostics.txt
```

The `profile` line gives `fps= interval_p50= analyse_p50= convert_p50=
detect_p50= delivered= dropped=`. That single line decides whether the live-capture
hypothesis is the answer, or whether the cause is something still unseen.