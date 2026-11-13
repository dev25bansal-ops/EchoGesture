# Phone build identity

Every phone test result is only meaningful against an exact build. This file
says which build is which, so no measurement can be attributed to the wrong one.

## Builds in order

| # | APK | SHA-256 | Branch | Commit | What changed | Phone-tested? |
| --: | --- | --- | --- | --- | --- | --- |
| 1 | `dist/echogesture-v2-cc00f6a.apk` | `530f5ec4…` | `hackathon/echogesture-v2-rc` | `cc00f6a` | V2 release candidate | No |
| 2 | `dist/echogesture-diag-87e3835.apk` | `abe6b24b…` | `diag/training-clip-repro` | `87e3835` | Diagnostics log, latency split | No |
| 3 | `dist/echogesture-rc-a2c0a6c.apk` | `abe6b24ba3f0cca9860ce08cb1562fb0312e020261027988fba522b10aa43bdd` | `diag/training-clip-repro` | `a2c0a6c` | **Byte-identical to #2**: the MediaCodec decoder fix lives in the test harness, not the app | **No physical device attached** |
| 4 | `dist/echogesture-phone-test.apk` | `429f7c4d713bdc7ca59da6a7b6cabacd3b0b60cb74c1ea5ec8db2eddd2d30f90` | `diag/training-clip-repro` | `12feeef` | Pre-tap rolling window: the tap now classifies the 3.7 s **ending** at the tap instead of starting there | **No physical device attached** |
| 5 | `dist/echogesture-v3-12feeef-pretap.apk` | `ce83cf4c0e080b210f3b3cb160d650177cf759f29e190b8951730dcbbdda8aa4` | `diag/live-phone-recognition-v3` | `12feeef` + working tree | Strict capture contract: no waiting path, stale/short history rejected with a message, repeat-tap guard | **No physical device attached** |

Build 5 is the current candidate. It is the only build that contains the
hardening described in `docs/release/V3_PHONE_ACCURACY_REPORT.md`.

Builds 4 and 5 both derive from commit `12feeef`. Build 4 is the pre-tap fix
alone; build 5 adds the strict contract, the repeat guard and the removal of
the capture-and-wait path. **They are not the same binary** — build 5's SHA-256
differs from build 4's.

## What is verified about each

| Claim | Build 4 | Build 5 | Evidence |
| --- | --- | --- | --- |
| Ensemble SHA-256 `0e4c3927…` | verified | verified | `ModelContract.ENSEMBLE_SHA256`, recomputed over the four members in meta order |
| Vocabulary 100 labels | unchanged | unchanged | `models/landmark_labels.json`, untouched on this branch |
| Capture window ends at tap | yes | yes | `HandTracker.selectCapture`, covered by `PreTapCaptureContractTest` |
| No waiting path after tap | **no — it had one** | yes | Bug A below |
| Stale-ring fallback removed | **no — it had one** | yes | Bug B below |
| JVM unit tests | 62 | 64 | `./gradlew :app:testDebugUnitTest` |
| Instrumentation tests | 2 | 5 | `am instrument` on `emulator-5554` |
| Ran on a physical phone | **no** | **no** | `adb devices -l` shows only `emulator-5554` |

## The two defects this build fixes

**A. The waiting path re-introduced the original bug.** `beginCapture(onComplete)`
began by clearing the ring and only completed immediately when at least 32
frames were buffered with sufficient history. When that check failed it set
`capturing = true` and waited — so the classification happened 3.7 s *after*
the tap, which is exactly the defect `12feeef` had just removed, reachable
through a different door. Build 5 has no waiting path at all: `beginCapture()`
returns synchronously and either yields a window or a reason.

**B. The stale-ring fallback could classify old frames.** `captureSnapshot()`
filtered the ring to the trailing span and, if that filter came back empty,
fell back to `ring.toList()` — the entire buffer, including frames far older
than the span. A stalled analyser would therefore be classified as if it had
just delivered. Build 5 has no fallback: `buildWindow()` filters to the span
and reports what it has.

## How to tell which build a phone is running

The model card at the bottom of the Camera screen prints the ensemble SHA-256
prefix. That identifies the **models**, not the app build — all five builds
above carry the same models. To identify the app build on a device:

```bash
adb shell dumpsys package org.echogesture.finalapp | grep -E "versionName|lastUpdateTime"
```

| Build | versionName |
| --- | --- |
| 1–4 | see `app/build.gradle.kts` `versionName` at that commit |
| 5 | same `versionName` — the distinguishing signal is `lastUpdateTime` plus the APK hash you installed |

Record the APK's SHA-256 before installing and keep it with the test notes. A
result without a build identity is not evidence.