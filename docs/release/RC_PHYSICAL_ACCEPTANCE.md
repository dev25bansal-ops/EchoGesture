# Physical acceptance and release freeze

Date: 2026-11-12
Branch: `hackathon/echogesture-v2-rc`
HEAD: `d1a01debf1a2ecbafef69af30d3fbdc1884124a5`

## Verdict

| Claim | State |
| --- | --- |
| `BUILD_VERIFIED` | **yes** |
| `EMULATOR_VERIFIED` | **yes** |
| `PHONE_CAMERA_VERIFIED` | **NO — no physical device was attached** |
| `DEMO_RECORDED` | **NO — no demo video exists** |
| `SUBMISSION_READY` | **NO** |

Every phase that needs a physical phone stopped at the device-detection step.
The work that does not need a phone was completed, and one genuine test defect
was found and fixed.

## 1. Physical device identity

**No physical Android device was attached.** This was established four ways:

```
adb devices -l
  -> emulator-5554  product:sdk_gphone64_arm64 model:sdk_gphone64_arm64

adb devices | awk '$2=="device"' | grep -v '^emulator-'
  -> NONE

ioreg -c IOUSBHostDevice | grep -c '"USB Product Name"'
  -> 0        (no USB device of any kind on the bus)

adb kill-server && adb start-server && adb devices -l
  -> emulator-5554 only
```

No model, OS version, ABI or camera capability of a phone can be reported,
because no phone was present. **Nothing in this document is a physical-device
result.** The emulator was not substituted for one.

An iPhone cannot install an Android APK, and no attempt was made to treat one as
a test device.

The exact Mac-to-Android USB debugging steps are in
`docs/release/RC_PHONE_ACCEPTANCE.md` — unlock, Build number ×7, USB debugging,
data cable, File transfer mode, Allow USB debugging, `adb devices -l`.

## 2. APK hash and installation

| | |
| --- | --- |
| Size | 77,708,379 bytes |
| SHA-256 (current) | `527b70b2d79e9736a1024014129fd7fb4c6053dadba0c0e87e929e3c761fda4d` |
| SHA-256 (previously recorded) | `a6e1a00e4dfa9dcc88baf1dcb312b48a3e82441c2caf2f81875fd321d390cdf6` |
| Signature | verifies; `C=US, O=Android, CN=Android Debug` |
| Cert SHA-256 | `2e7e40e24ccff12d99457a4267a377cb5b9ae14c00750ba4b406e8df6be37e2c` |

**The recorded hash did not match the rebuilt APK, and that discrepancy is
explained rather than excused.** Two clean builds of identical source now
produce byte-identical APKs, and rebuilding commit `1a7601b` — the commit the
`a6e1a00e` hash was recorded against — also produces `527b70b2`. So
`a6e1a00e` came from an incremental build of the same source, not from different
code, and it is not reproducible. `527b70b2` is the hash to quote from now on.

What actually identifies the model is not the APK hash but the asset digests,
and those are unchanged. Extracted from the current APK and re-hashed:

| Asset | SHA-256 | Matches pin |
| --- | --- | --- |
| `landmark_ens_tcn21_fp16.tflite` | `112279bc…c73dc22a` | yes |
| `landmark_ens_tcnm61_fp16.tflite` | `01a261d8…b1390f63` | yes |
| `landmark_ens_tcnm62_fp16.tflite` | `27058a85…428d20b3` | yes |
| `landmark_ens_rob71_fp16.tflite` | `08417e72…276d1ab21` | yes |
| `hand_landmarker.task` | `fbc2a300…3607cde1` | yes |
| `landmark_labels.json` | `e78f5028…8019a4a` | yes |
| `landmark_model_meta.json` | `945150120…e9a380d` | canonical |

All four expected members, the detector, the labels and the metadata are
present and correct.

**Installation on a physical device: NOT DONE.** The APK was installed on
`emulator-5554` only, where it launches and runs (`pid` present, zero crashes in
the crash buffer). It was not published or distributed; redistribution rights
remain unresolved.

## 3. Feature test results

`NOT_TESTED` means exactly that: no physical device existed to test on. It is
not a pass and not a failure.

| # | Feature | Phone | Basis |
| --- | --- | --- | --- |
| 2.1 | Camera permission prompt | NOT_TESTED | |
| 2.2 | Permission denial leaves app usable | NOT_TESTED | |
| 2.3 | Live camera preview | NOT_TESTED | emulator preview is synthetic |
| 2.4 | Rotation correctness | NOT_TESTED | |
| 2.5 | Front/rear switching | NOT_TESTED | code path exists; `switchCamera()` clears the frame ring |
| 2.6 | Switch back | NOT_TESTED | |
| 2.7 | Hand detection | NOT_TESTED | emulator scene has no hands |
| 2.8 | Torch | NOT_TESTED | |
| 3.1 | No-hands state | EMULATOR PASS | correct state, no fabricated prediction |
| 3.2 | Isolated sign capture | NOT_TESTED | |
| 3.3 | Prediction through all four models | EMULATOR PASS | 4 interpreters, frozen parity vector |
| 3.4 | Genuine caption and top-5 | EMULATOR PASS | from the real ensemble, not live camera |
| 3.5 | Copy | NOT_TESTED | |
| 3.6 | Offline speech | NOT_TESTED | |
| 3.7 | Recognise again | NOT_TESTED | wired to `beginCapture()` |
| 3.8 | No-hands recovery | NOT_TESTED | |
| 3.9 | Capture → caption latency (stopwatch) | NOT_TESTED | |
| 3.10 | Repeated captures, several signs | NOT_TESTED | |
| 4.1–4.5 | History population and clearing | EMULATOR PARTIAL | empty state verified |
| 5.1–5.3 | Speech, including airplane mode | NOT_TESTED | |
| 6.1–6.3 | Settings persistence | NOT_TESTED | rendering verified on emulator |
| 7.1–7.5 | Glossary search and detail | EMULATOR PASS | 100 signs, search, detail sheet |
| 8.1–8.5 | Stability, memory, thermal | NOT_TESTED | |
| 1.x | Launch, tabs, permissions | EMULATOR PASS | see §5 |

**Zero real recognition samples exist**, because no hand has ever been seen by
this app on a real camera. None are reported. No false recognition was observed
or invented.

**Zero latency measurements exist** for capture-to-caption, for the same reason.
The on-device inference latency measured in the parity tests is a different
figure and is not presented as the user-facing number.

## 4. Bugs found and repaired

One genuine defect, found by running the instrumentation suite.

**Symptom.** `ProcessingStateRenderTest` and `ResultCardRenderTest` failed with
`AssertionError: MainActivity never settled into a waiting state` after the
40-second timeout.

**Root cause.** Both tests wait for the camera screen to reach `WAITING`. If an
activity from the same package was already alive — left open by a developer, or
by an earlier test — that instance held the camera. The new activity's CameraX
bind failed, the screen never settled, and both tests timed out. The failure
depended on prior device state, not on the code under test: the same build
passes 8/8 when nothing is running first.

**Fix.** A new `ActivityLaunch.fresh()` reaps any stale activity before
launching. It uses `am kill`, not `am force-stop`: the instrumentation is
injected into the same package as the activity, so force-stop terminates the
test runner too. That was demonstrated — the first attempt at this fix failed in
1 ms with `Process crashed`, which is the test runner being killed.

**Verification.** An instance was deliberately left running (pid confirmed
present), then the suite was run: **8 tests, 0 failed, 0 skipped**. JVM suite
rerun with `--rerun-tasks`: **30 tests, 0 failed, 0 skipped**, including all
four-member parity tests.

**No model was changed.** No weight, detector, label or feature-semantic byte was
touched. The premium V2 UI was not modified. The change is confined to two
androidTest files plus one new test helper.

## 5. Tests

| Suite | Result |
| --- | --- |
| JVM unit tests (complete assets, `--rerun-tasks`) | 30 tests, 0 failed, 0 skipped |
| JVM unit tests (source-only) | 30 tests, 0 failed, 11 skipped, each naming why |
| Instrumentation on `eg34` | 8 tests, 0 failed, 0 skipped |
| Python suite (CI) | success |

Instrumentation: `CameraFeaturePathTest` ×2, `EnsembleInferenceTest` ×2,
`TfliteEnsembleInferenceTest` ×2, `ProcessingStateRenderTest`,
`ResultCardRenderTest`.

The parity tests are the model-integrity check: the four real FP16 interpreters
reproduce the Python reference output on frozen features.

## 6. Git status and CI

Branch `hackathon/echogesture-v2-rc` is at `d1a01de`, pushed and level with
`origin`. Three commits this phase:

| Commit | Purpose |
| --- | --- |
| `dc193a8` | ignore `*.apk` / `*.aab` repository-wide |
| `d1a01de` | fix the render-test isolation defect |

CI: run `43247489689` on `d1a01de` — **success**. All four runs on this branch
are green.

Unchanged, as required:

| Ref | SHA |
| --- | --- |
| `hackathon/dev-model` | `3f16d4b` — untouched |
| `hackathon/abhijeet-integration` | `d67d468` |
| `hackathon/abhijeet-pr2-fix` | `ff13119` |
| PR #1 | open, untouched |
| PR #2 | open, untouched |

Nothing was force-pushed. No tracked file was matched by the new ignore rules.

Outgoing delta across the whole branch: seven text files, no `.tflite`, `.task`,
`.apk`, `.aab`, `.zip` or video. Verified against the remote, not just locally.

## 7. PR #2 disposition

PR #2 (`abhijeet-app-fix` → `abhijeet-public`) is a **2-file, 65-line** patch. Its
two test files are **byte-identical** in the RC. Its Gradle changes are present
in the RC too — a line-by-line comparison finds **zero PR #2 code lines missing**
from the RC. The only differences are comment wording and the four Material /
RecyclerView / dynamicanimation / fragment dependencies the V2 UI added.

The RC does not contain PR #2's commits, because the RC is a separate line cut
from `abhijeet-integration` and `abhijeet-public` is not an ancestor of it. The
*content* is fully carried.

**Recommendation: merge PR #2, do not merge the RC into it.**

- PR #2 is an accurate, self-contained, reviewed fix for its own base. Merging
  it is honest history: those corrections were written, reviewed and verified
  there.
- Merging it duplicates no code, because duplicate content in two branches is
  normal; duplicate *commits* are the thing to avoid, and this creates none.
- Its Windows caveat stands on its own and is unaffected by the RC.
- Closing PR #2 unmerged and pointing at the RC would lose the review record for
  work Dev's base needs independently of the UI.

Then merge the RC on top. Neither is done here — both need approval.

## 8. Remaining blockers

| # | Blocker | Owner | Blocks |
| --- | --- | --- | --- |
| 1 | No physical phone attached | Abhijeet | `PHONE_CAMERA_VERIFIED`, `DEMO_RECORDED` |
| 2 | Redistribution rights for the weights unresolved | Owner / Dev | publishing the APK, `SUBMISSION_READY` |
| 3 | Public weight exposure in `dev-model` history | Owner / Dev | `SUBMISSION_READY` |
| 4 | Organizer delivery requirements not finalised | Abhijeet | `SUBMISSION_READY` |
| 5 | No native Windows Gradle build | Dev | closes PR #2's only caveat |

## 9. Exact final submission actions

In order. Steps 1–3 need a phone; step 4 needs an approval; step 5 needs rights.

1. **Connect the phone** using the steps in `RC_PHONE_ACCEPTANCE.md`, then
   `adb devices -l` and record model, `ro.build.version.release`,
   `ro.product.cpu.abi`, and camera features.
2. **Install and run the acceptance checklist** (sections 2–8), recording
   PASS/FAIL per row, plus at least five gloss samples (signed / predicted /
   confidence) and three stopwatch capture-to-caption timings. Any failure goes
   back through reproduce → root cause → smallest fix → retest.
3. **Record the demo** only if step 2 passes: genuine offline recognition,
   camera and result on screen, Glossary, History, Settings, real latency. The
   narration must say the model recognises 100 isolated ASL glosses and does not
   translate continuous signing. Do not claim an MP4 exists until one has been
   recorded and played back.
4. **Get approval** to merge PR #2, then the RC, into `hackathon/two-person-sprint`.
5. **Resolve redistribution rights** for the weights before the APK leaves this
   machine, and get the owner's decision on the public exposure in `dev-model`.
6. **Package the submission** against the organizer's requirements, with the
   APK hashes above and the asset digests, and state clearly what remains
   unverified.

Do not mark this submission ready until 1–5 are closed. A green build and a
passing emulator suite are not a physical-camera result, and they say nothing
about redistribution rights.