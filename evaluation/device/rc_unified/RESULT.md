# Unified release-candidate build record (private)

Private. Not for publication. The APK described here contains the restricted
ensemble weights and must not be distributed.

## Source

| | |
| --- | --- |
| Branch | `hackathon/echogesture-v2-rc` |
| Base | `origin/hackathon/abhijeet-integration` @ `d67d468` |
| Worktree | `/tmp/eg-rc` (temporary; the branch is the durable artifact) |
| Corrections applied | four files, +509 / −123, from `hackathon/abhijeet-pr2-fix` @ `ff13119` |

The four changed files:

| File | Change |
| --- | --- |
| `app/android_final/app/build.gradle.kts` | PR #2 validation block hand-integrated onto V2's dependency block. **No dependency changed.** |
| `app/android_final/app/src/test/.../ModelContractTest.kt` | COMPLETE / ABSENT / BROKEN asset classification; 17 tests |
| `app/android_final/app/src/test/.../EnsembleRuntimeParityTest.kt` | weight-free vocabulary tests replacing the skipped one |
| `models/landmark_model_meta.json` | canonical five-field sidecar; `dim` and `backbone` removed |

## Toolchain

| | |
| --- | --- |
| JDK | 17.0.21 Temurin (`ActionMesh-fork/android-toolchain/jdk`) |
| Android SDK | API 34, build-tools 34.0.0 |
| Gradle | 8.10.2 |
| Device | `emulator-5554`, `sdk_gphone64_arm64`, Android 14 / API 34, 1080×2400, density 420 |

## Commands that produced the verified state

```bash
export JAVA_HOME=/Users/abhijeetmiskin/AppData/ActionMesh-fork/android-toolchain/jdk
export ANDROID_HOME=/Users/abhijeetmiskin/AppData/ActionMesh-fork/android-toolchain/sdk
export GRADLE_USER_HOME=/Users/abhijeetmiskin/AppData/ActionMesh-fork/gradle-home
cd /tmp/eg-rc/app/android_final && ./gradlew clean testDebugUnitTest assembleDebug
```

## Test results

| Suite | Result |
| --- | --- |
| JVM, complete assets | 30 tests, 0 failed, 0 skipped (`--rerun-tasks`) |
| JVM, source-only checkout | 30 tests, 0 failed, 11 skipped (all weight tests, each reporting why) |
| Instrumentation on `eg34` | 8 tests, 0 failed, 0 skipped — including with an app instance deliberately left running |
| Python suite | green in CI on the branch |

Instrumentation tests: `CameraFeaturePathTest` ×2, `EnsembleInferenceTest` ×2,
`TfliteEnsembleInferenceTest` ×2, `ProcessingStateRenderTest`,
`ResultCardRenderTest`.

### APK reproducibility

Two clean builds of identical source produce byte-identical APKs. The current
artifact is `9bba9191…`, reproduced twice from clean. An **incremental**
build of the same source yields a different hash (`73d2e805…`) because the
debug APK embeds a build timestamp — the difference is metadata, not code:
every model asset hashes identically in both. The earlier `a6e1a00e…` was
likewise an incremental build. Compare clean builds when checking
reproducibility; compare the asset digests below when identifying the model.

## APK

| | |
| --- | --- |
| Path | `app/android_final/app/build/outputs/apk/debug/app-debug.apk` |
| Size | 77,710,351 bytes |
| SHA-256 | `8d2e175edf2f3927885656a302ce22c4d493012c091f6d7eccf35efcd4b1cc90` |
| Commit | `a6cf722` (`hackathon/echogesture-v2-rc`) |
| Signer | `C=US, O=Android, CN=Android Debug` |
| Cert SHA-256 | `2e7e40e24ccff12d99457a4267a377cb5b9ae14c00750ba4b406e8df6be37e2c` |
| Signature | verifies (`apksigner verify --print-certs`) |

### Permissions in the packaged manifest

```
uses-permission: android.permission.CAMERA
permission:      org.echogesture.finalapp.DYNAMIC_RECEIVER_NOT_EXPORTED_PERMISSION
uses-permission: org.echogesture.finalapp.DYNAMIC_RECEIVER_NOT_EXPORTED_PERMISSION
```

`INTERNET` and `ACCESS_NETWORK_STATE` are removed from the merged manifest by
`tools:node="remove"`. The second entry is an app-private signature permission
that androidx generates; it grants nothing outside the app.

### Assets extracted from the APK and re-hashed

| Asset | SHA-256 | Matches pin |
| --- | --- | --- |
| `landmark_ens_tcn21_fp16.tflite` | `112279bc86090470b82b20b510746a946220322c25ec84cb3706a538c73dc22a` | yes |
| `landmark_ens_tcnm61_fp16.tflite` | `01a261d8eb6c2c4d8b501f0ce1a36db338365d17fddabc63528a7941b1390f63` | yes |
| `landmark_ens_tcnm62_fp16.tflite` | `27058a85e032bee3af444fb1538ff0bf32fd38fa2b01ce01a2f25c3a428d20b3` | yes |
| `landmark_ens_rob71_fp16.tflite` | `08417e72da59150ac9adf749e6324c30fb001d4a401394e32f125d3276d1ab21` | yes |
| `hand_landmarker.task` | `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1` | yes |
| `landmark_labels.json` | `e78f5028e97e2eaf86aff65af403af65061eebef010f0fbd41012293e8019a4a` | yes |
| `landmark_model_meta.json` | `945150120f4a8eecec406ca1d2012fb405b39072b8e9c6f42e5b7295ee9a380d` | canonical bytes |

The sidecar is not byte-pinned by design; its content is validated semantically
at configuration time and again by `verifyPinnedAssets`. The packaged copy is
byte-identical to `models/landmark_model_meta.json` (260 bytes, same digest), so
the APK carries the canonical sidecar rather than a stale one.

## Screens verified on this APK

All four V2 screens launched and rendered from this exact APK:

- `eg-rc-camera.png` — premium camera screen with the offline chip, frame guide
  and hand-detection pill
- `eg-rc-glossary.png` — 100 signs, search field, alphabetical list
- `eg-rc-history2.png` — History empty state with its explanation
- `eg-rc-settings.png` — every control rendered; result text size, haptics,
  speech, privacy checklist, clear-history danger action

The images live in this private directory. They were captured on the emulator,
so they show no live hand and are not evidence of a live-camera prediction.

## Not verified

- **No physical phone was attached.** Live-camera recognition, speech through a
  real speaker and real-device latency are unverified. See
  `docs/release/RC_PHONE_ACCEPTANCE.md`.
- **No native Windows build.** The CRLF normalization is covered by controlled
  test only.