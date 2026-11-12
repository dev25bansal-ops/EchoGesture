# Unified release candidate — report

Branch: `hackathon/echogesture-v2-rc`, commit `1a7601b`, pushed.
Base: `origin/hackathon/abhijeet-integration` @ `d67d468`.
Corrections from `hackathon/abhijeet-pr2-fix` @ `ff13119`.

## Verdict

| Claim | State |
| --- | --- |
| `UNIFIED_BUILD_VERIFIED` | **yes** |
| `EMULATOR_VERIFIED` | **yes** |
| `PHYSICAL_CAMERA_VERIFIED` | **pending — no phone attached** |
| `SUBMISSION_READY` | **no** |
| `BLOCKED` | no |

The build and the emulator evidence are real and reproducible. Submission is not
ready, for two independent reasons: the physical-camera acceptance has not run,
and the repository-level weight exposure recorded in
`docs/release/PUBLIC_WEIGHT_EXPOSURE.md` is still an unsatisfied release gate.
Passing tests does not discharge either.

---

## 1. What was created

New branch `hackathon/echogesture-v2-rc` cut from the premium V2 UI branch, with
the four PR #2 corrections applied on top. One commit, `1a7601b`.

| Branch | Role | State |
| --- | --- | --- |
| `hackathon/abhijeet-integration` | premium V2 UI (`d67d468`) | untouched |
| `hackathon/abhijeet-pr2-fix` | verified PR #2 corrections (`ff13119`) | untouched |
| `hackathon/echogesture-v2-rc` | the union | 1 commit ahead of base |

Nothing was merged, nothing force-pushed, `hackathon/dev-model` is untouched at
`3f16d4b`, and PR #1 and PR #2 are untouched and open.

## 2. Delta applied — four files

| File | Lines | Why |
| --- | --- | --- |
| `app/android_final/app/build.gradle.kts` | +179 / −46 | contract validation block, hand-integrated |
| `app/android_final/app/src/test/.../ModelContractTest.kt` | +264 / −59 | COMPLETE / ABSENT / BROKEN asset classification |
| `app/android_final/app/src/test/.../EnsembleRuntimeParityTest.kt` | +66 / −16 | weight-free vocabulary tests |
| `models/landmark_model_meta.json` | −2 | canonical five-field sidecar |

**The Gradle file was not copied wholesale.** V2 and PR #2 both edit
`build.gradle.kts`, and they edit different parts: PR #2 rewrites the validation
block, V2 adds four Material/RecyclerView/dynamicanimation/fragment dependencies
for the redesign. Overwriting with PR #2's file would have broken the UI. The
validation block was transposed onto V2's file and the result verified:

```
=== dependency block: base vs HEAD ===
IDENTICAL
```

V2-only dependencies confirmed present: `material:1.12.0`,
`recyclerview:1.3.2`, `dynamicanimation:1.0.0`, `fragment-ktx:1.5.4`.

The two test files were byte-identical between V2 and `abhijeet-public`, so they
transferred unchanged.

## 3. Premium UI preserved

```
=== V2 UI files touched by commit? ===
none
```

No layout, drawable, color, string, navigation, `MainActivity`, `CameraScreen`,
`GlossaryScreen`, `HistoryScreen`, `SettingsScreen`, `Speaker`, `HandTracker` or
`LandmarkClassifier` file is in the commit. The entire Material 3 redesign, the
four-destination single-activity shell, the offline speech path, the frame guide
and the capture flow are exactly as proven at `d67d468`.

All four screens were then launched from the new APK and captured:

- Camera — premium layout, offline chip, frame guide, hand-detection pill
- Glossary — 100 signs, search, detail sheet with iconed actions
- History — empty state and its explanation
- Settings — every control rendered; text size, haptics, speech, privacy
  checklist, clear-history action

## 4. Test results

### JVM unit tests — complete asset set

```
UNIT: tests=30 failed=0 skipped=0
```

`ModelContractTest` 17/17, `EnsembleParityTest` 4, `EnsembleRuntimeParityTest` 4,
`FeatureTransformTest` 5.

### JVM unit tests — source-only checkout (no weights)

```
TOTAL tests=30 failed=0 skipped=11
```

Exactly the 11 weight tests skip, each naming why. `./gradlew help` and
`testDebugUnitTest` both configure and run without weights; `assembleDebug`
fails with `Missing model asset …hand_landmarker.task`, which is the intended
gate.

### Instrumentation — emulator `eg34` (Android 14 / API 34)

```
8 tests, 0 failed, 0 skipped
```

`CameraFeaturePathTest` ×2 · `EnsembleInferenceTest` ×2 ·
`TfliteEnsembleInferenceTest` ×2 · `ProcessingStateRenderTest` ·
`ResultCardRenderTest`.

### Python CI

Run `43237550262` on the branch: **success**, 363 passed / 0 failed / 119
skipped. This is the Python suite only; it is not Android coverage and is not
offered as such.

## 5. APK and model identity

| | |
| --- | --- |
| Size | 77,708,379 bytes |
| SHA-256 | `a6e1a00e4dfa9dcc88baf1dcb312b48a3e82441c2caf2f81875fd321d390cdf6` |
| Signer | `C=US, O=Android, CN=Android Debug` |
| Cert SHA-256 | `2e7e40e2…be37e2c` |
| Signature | verifies |

Assets extracted from the built APK and re-hashed — all six match their pins:

| Asset | Matches |
| --- | --- |
| `landmark_ens_tcn21_fp16.tflite` | yes |
| `landmark_ens_tcnm61_fp16.tflite` | yes |
| `landmark_ens_tcnm62_fp16.tflite` | yes |
| `landmark_ens_rob71_fp16.tflite` | yes |
| `hand_landmarker.task` | yes |
| `landmark_labels.json` | yes |

The packaged sidecar is byte-identical to `models/landmark_model_meta.json`
(260 bytes, `945150…`), so the APK carries the canonical contract, not a stale
copy. No model byte, no weight, no member order and no feature semantic was
changed.

### Permissions

```
uses-permission: android.permission.CAMERA
uses-permission: org.echogesture.finalapp.DYNAMIC_RECEIVER_NOT_EXPORTED_PERMISSION
```

`INTERNET` and `ACCESS_NETWORK_STATE` are removed from the merged manifest by
`tools:node="remove"`. No analytics or tracking SDK is present. The second
permission is an app-private signature permission generated by androidx that
grants nothing outside the app.

## 6. Exact commands run

```bash
cd app/android_final
./gradlew clean testDebugUnitTest assembleDebug connectedDebugAndroidTest
```

`JAVA_HOME`, `ANDROID_HOME` and `GRADLE_USER_HOME` pointed at the local JDK 17
and SDK 34 installations.

Toolchain: JDK 17.0.21 Temurin, Android SDK 34 / build-tools 34.0.0,
Gradle 8.10.2.

## 7. What was published, and what was not

The push carried **five text files and no binary**:

```
modified 179+ 46-  app/android_final/app/build.gradle.kts
modified  66+ 16-  .../EnsembleRuntimeParityTest.kt
modified 264+ 59-  .../ModelContractTest.kt
added    127+  0-  docs/release/RC_PHONE_ACCEPTANCE.md
modified   0+  2-  models/landmark_model_meta.json
```

Verified against the remote: `git ls-remote` reports base `1a7601bb…`, and
`gh api .../commits/hackathon/echogesture-v2-rc` lists exactly those five files.
No `.tflite`, `.task`, `.apk`, `.zip`, `.mp4` or `.mov` was added. No local
filesystem path, credential or secret appears in the diff.

The APK — which contains the restricted ensemble weights — was **not**
published. It stays local, recorded in
`evaluation/device/rc_unified/RESULT.md` together with its hashes, and must not
be redistributed without the owner's authorization.

## 8. Pre-existing exposure this did not change

Two `.tflite` files are tracked on this branch:
`models/ptq_100_int8.tflite` and `models/s3d_conv1d_qat_int8.tflite`. Both were
added in the initial commit `c47c5c9` and are already present in the base
branch; they are not part of this change. They are inventoried in
`docs/release/PUBLIC_WEIGHT_EXPOSURE.md` §2 as unresolved third-party
WLASL-derived weights.

That document also records that the four FP16 ensemble members are
byte-identical to files reachable on the public remote via
`hackathon/dev-model`. **This branch does not fix that**, and no code change on a
branch can: the blobs are already in public history. It remains an
unsatisfied release gate and is the main reason the verdict is not
`SUBMISSION_READY`. Resolving it needs the repository owner's decision
(visibility, history, or licensing) — it is not a merge question.

## 9. Physical-device acceptance — pending

`adb devices` showed only `emulator-5554`; no physical phone was attached. Live
recognition, real-speaker speech and real-device latency are therefore
**unverified**.

A 30-check acceptance run is written and committed at
`docs/release/RC_PHONE_ACCEPTANCE.md`: permissions, preview rotation, no-hands
behaviour, repeated recognition, history persistence, offline speech in airplane
mode, settings persistence and a 20-recognition stability pass. It explicitly
notes which rows the emulator already covers so the phone run can concentrate
on what the emulator cannot do.

The emulator has no hands in its virtual scene, so no claim of a working
live-camera prediction is made anywhere in this report.

## 10. PR #2 disposition — recommendation

**Recommend merging PR #2** (`abhijeet-app-fix` → `abhijeet-public`) as the
reviewed record of the contract corrections, then merging this branch on top.

Reasoning:

- PR #2 is the reviewed change on its own base and should not be silently
  bypassed; its branch `abhijeet-pr2-fix` is the verified commit `ff13119`.
- Every correction in it is demonstrably present in `1a7601b`, so merging this
  branch is not dependent on PR #2 being merged first.
- The one caveat on PR #2 is unchanged and unchanged by this work: no native
  Windows Gradle build was run; the CRLF fix is proven by controlled test, not
  by execution on Windows.

This branch is **not** a PR yet, and none will be opened without approval. The
merge into `hackathon/two-person-sprint` is the owner's call.

## 11. Blockers and next actions

| # | Blocker | Owner | Effect |
| --- | --- | --- | --- |
| 1 | No physical phone attached | Abhijeet | `PHYSICAL_CAMERA_VERIFIED` stays pending |
| 2 | Public weight exposure unresolved | Owner / Dev | `SUBMISSION_READY` stays no |
| 3 | No native Windows Gradle build | Dev | CRLF fix proven by test only |

Next highest-value action: run the 30-check acceptance on a real phone and
record the gloss, confidence and latency for a handful of signs. That closes
blocker 1 and produces the only evidence the emulator fundamentally cannot
supply.

---

## Files changed by this work

On the branch: the four above plus `docs/release/RC_PHONE_ACCEPTANCE.md`.

Local and private only, not published:
`evaluation/device/rc_unified/` (four screenshots and `RESULT.md`).