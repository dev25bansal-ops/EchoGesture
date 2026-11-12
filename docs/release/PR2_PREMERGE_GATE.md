# PR #2 pre-merge gate

Branch under review: `hackathon/abhijeet-app-fix` (Dev) → `hackathon/abhijeet-public`.
Fix commit: `b90e3ae` on `hackathon/abhijeet-pr2-fix`.
Not merged. Nothing was force-pushed, and `hackathon/dev-model` was not touched.

## 1. Metadata differences and the canonical decision

| Source | `frames` | `feature_dim` | `classes` | `quantization` | members | extra |
| --- | --- | --- | --- | --- | --- | --- |
| `dev-model:models/landmark_model_meta.json` | 24 | 255 | 100 | float16 | 4, pinned order | `ensemble_test_top1: 0.7278` |
| `abhijeet-public:models/…` | 24 | 255 | 100 | float16 | same 4, same order | `dim: 192`, `backbone: "conv"` |
| `src/main.py` reads | frames | feature_dim | — | — | members | ignores the rest |
| `ModelContract.kt` reads | frames | feature_dim | classes | quantization | members | ignores the rest |
| `build.gradle.kts` | rejects ≠24 | rejects ≠255 | rejects ≠100 | rejects ≠float16 | exact ordered list | ignores the rest |

Every field the runtime actually reads already agreed. Only the optional keys
differed.

**Decision — the canonical sidecar is the five runtime fields:**

```json
{ "frames": 24, "feature_dim": 255, "classes": 100,
  "quantization": "float16",
  "members": [tcn21, tcnm61, tcnm62, rob71] }
```

`dim` and `backbone` were removed from `models/landmark_model_meta.json`.
`ensemble_test_top1` was not added.

**Why, having checked for readers first:**

- `dim` and `backbone` are written by `scripts/ml/export_tflite_tf.py` from a
  training checkpoint. Nothing reads them back out of the meta file — not
  `src/main.py`, not `ModelContract.kt`, not `model_contract.py`, not the
  release builder. They describe the exporter's input, not the runtime.
- `ensemble_test_top1` is an evaluation metric. The file is read at inference
  time by both `src/main.py` and `ModelContract.load()`. A metric in runtime
  metadata is a number the runtime cannot verify; it belongs in evaluation
  evidence.
- Removing them changes no reader: no reader exists. It changes no model, no
  weight, and no member order.

The Gradle gate still **allows** such keys — an extra provenance field is not a
reason to refuse a build. It is `test 12` below that proves a meta carrying
`ensemble_test_top1`, `evaluated_on` and `notes` loads to an identical identity.

## 2. Code changes

Four files, 511 insertions / 123 deletions across the two commits.

**`models/landmark_model_meta.json`** — now the five runtime fields only.

**`app/android_final/app/build.gradle.kts`**

- `expectedMembers` pins the ordered member list as a list, not a set.
- `validateMetadata(metaFile)` performs the semantic gate: frames 24,
  feature_dim 255, classes 100, quantization float16 (case-insensitive),
  exactly four members, exact order, no duplicates, plain basenames, no `..`.
  Integer checks reject `"24"` and `24.5` as well as `32`.
- `requirePinnedAssets()` keeps Dev's CRLF/LF-safe text hashing and byte-level
  SHA-256 for the four members and the detector.
- `verifyPinnedAssets` task runs the sidecar gate **and** the byte gate; the
  packaging and install tasks depend on it.
- The sidecar is still validated at configuration time **when it is present**.

**`EnsembleRuntimeParityTest.kt`** — `fixtureVocabularyMatchesTheShippedLabels`
(chained to model weights) is replaced by two weight-free tests:
`shippedVocabularyMatchesTheFrozenClassAxis` and
`vocabularyWithADuplicateOrMissingIndexIsRejected`.

**`ModelContractTest.kt`** — split into semantic tests that always run and
weight tests that classify the asset set as COMPLETE / ABSENT / BROKEN.

## 3. Corrected validation logic

The previous gate compared `declared.toSet() != pinned`. A reordering passed:
aggregation is a mean over members **in that order** and the combined hash
concatenates in it, so a reordered list is a different ensemble. It is now an
ordered list comparison, with separate messages for "reordered" and "different
set".

The previous gate also ran at configuration time, so `./gradlew help` failed on
a checkout with no weights. `verifyPinnedAssets` now carries the byte gate and
only packaging depends on it. **No APK can be assembled, packaged or installed
without the complete pinned set** — verified below at cases B, C and D.

## 4. Negative test results

All run on this machine. Gradle cases use `./gradlew help` (configuration-time
sidecar gate) or `assembleDebug` (packaging gate).

| # | Case | Result |
| --- | --- | --- |
| 1 | Missing metadata, no weights | `help` configures; `assembleDebug` → "Missing …/landmark_model_meta.json" |
| 2 | Three members | `declares 3 members, expected 4` |
| 3 | Duplicate members | `lists a member more than once … Averaging the same member twice is not the pinned ensemble` |
| 4 | Reordered members | `member order [...] does not match the pinned order [...]` |
| 5 | Unknown member | `members [...] do not match the pinned set [...]` |
| 6 | Wrong frame count (32) | `'frames' is 32; expected 24` |
| 7 | Wrong feature dim (225) | `'feature_dim' is 225; expected 255` |
| 8 | Wrong class count (99) | `'classes' is 99; expected 100` |
| 9 | INT8 substituted | `declares quantization='int8'; the shipped ensemble is FP16` |
| 10 | Unsafe basename | `has unsafe member name '../escape.tflite'` |
| 11 | Corrupted label mapping | JVM: `Duplicate label index 9` from both vocabulary tests |
| 12 | Missing binary asset | `assembleDebug` → "Missing model asset …", skips nothing |
| 13 | Modified member binary | `verifyPinnedAssets` → `landmark_ens_tcn21_fp16.tflite SHA-256 is d16e98…; expected 112279…` |

Positive controls:

| # | Case | Result |
| --- | --- | --- |
| P1 | Labels file rewritten CRLF (1492 → 1593 bytes) | `BUILD SUCCESSFUL` — the LF-normalized pin still matches |
| P2 | Metadata sidecar rewritten CRLF | `BUILD SUCCESSFUL` |
| P3 | Extra provenance keys (`ensemble_test_top1`, `evaluated_on`, `notes`) | `BUILD SUCCESSFUL`; `extraProvenanceFieldsDoNotChangeWhatLoads` asserts identical identity, member names and labels |
| P4 | Reordered labels in `models/` (DOG1 ↔ HURDLE/TRIP1) | Fails: `shipped labels must equal the frozen class axis, in order` |

## 5. Source-only test status

No weights present (the state of a clean public checkout):

```
EnsembleParityTest             tests=  4 fail=0 skip=0
EnsembleRuntimeParityTest      tests=  4 fail=0 skip=0
FeatureTransformTest           tests=  5 fail=0 skip=0
ModelContractTest              tests= 17 fail=0 skip=11
TOTAL tests=30 failed=0 skipped=11
```

The 11 skips are exactly the weight tests, each reporting
`weight-dependent tests SKIPPED: … has the tracked labels/sidecar but none of
the restricted weights`. The six semantic tests, both vocabulary tests and all
13 fixture-backed parity tests run. `./gradlew help` and `testDebugUnitTest`
both configure and run without the weights.

The vocabulary test can no longer silently pass: both failure modes above
(changed label order, duplicate index) fail it, and a missing labels file throws
an `AssertionError` naming the path rather than skipping.

## 6. Asset-equipped test status

With the complete verified set:

```
UNIT: tests=30 failed=0 skipped=0
```

`ModelContractTest` 17/17, including missing member, corrupt member, 3-member,
reordered, unpinned member, wrong frames/feature_dim/classes, INT8, corrupt
labels, extra provenance and build/runtime hash agreement.

Instrumentation on `eg34` (Android 14, API 34) — **6 tests, 0 failed, 0 skipped**:

- `TfliteEnsembleInferenceTest.fourRealInterpretersMatchThePythonReference`
- `TfliteEnsembleInferenceTest.attentionToTensorIndexResolutionPerMember`
- `EnsembleInferenceTest.productionClassifierReproducesPythonOnFrozenFeatures`
- `EnsembleInferenceTest.allFourMembersAllocateWithTheFrozenTensorContract`
- `CameraFeaturePathTest.cameraReportsValidRotationAndTrackerAcceptsFrames`
- `CameraFeaturePathTest.missingHandFramesKeepTheirPosition`

Reference prediction parity is covered by the first three.

## 7. Windows test status

**A native Windows Gradle build was NOT executed.** This machine is macOS and
has no Wine. What was executed instead is a controlled CRLF test: both tracked
text assets were rewritten with CRLF line endings and the build re-run,
covering the failure Dev's commit fixes. Both passed (P1, P2).

That is evidence the normalization works for the files it normalizes. It is not
a substitute for a Windows build, and this report does not claim one.

CI on `hackathon/abhijeet-pr2-fix` (run `43221154980`, success) is **Python
only**: 376 passed, 0 failed, 134 skipped. It is not Android test coverage and
is not presented as such.

## 8. APK and model identity

Built with JDK 17.0.21 (Temurin), Android SDK 34 / build-tools 34.0.0,
Gradle 8.10.2.

| | |
| --- | --- |
| Path | `app/build/outputs/apk/debug/app-debug.apk` |
| Size | 75,072,694 bytes |
| SHA-256 | `8147e2e040f80db8481b8dc8df557c1b8a27d4aacada2c01ef858939d30bd95e` |
| Signature | verifies; debug cert `2e7e40e2…` (`C=US, O=Android, CN=Android Debug`) |

Assets extracted from the APK and re-hashed:

| Asset | SHA-256 | Matches pin |
| --- | --- | --- |
| `landmark_ens_tcn21_fp16.tflite` | `112279bc…3dc22a` | yes |
| `landmark_ens_tcnm61_fp16.tflite` | `01a261d8…390f63` | yes |
| `landmark_ens_tcnm62_fp16.tflite` | `27058a85…8d20b3` | yes |
| `landmark_ens_rob71_fp16.tflite` | `08417e72…6d1ab21` | yes |
| `hand_landmarker.task` | `fbc2a300…07cde1` | yes |
| `landmark_labels.json` | `e78f5028…19a4a` | yes |

This APK contains restricted weights and must not be published.

## 9. Regression impact

- **No model change.** No weight, detector or label byte was touched; member
  order is unchanged.
- **No runtime code change.** `ModelContract.kt`, `src/main.py` and the release
  builder are untouched. The meta edit removes only keys nothing reads.
- **Python suite:** 376 passed, 0 failed (CI). `test_model_contract.py` asserts
  frames, feature_dim, classes and quantization — all retained.
- **Build:** `clean testDebugUnitTest assembleDebug` succeeds.
- **UI work:** the premium UI on `hackathon/abhijeet-integration` (`d67d468`) is
  untouched; all work happened in a separate worktree at `/tmp/eg-pr2-fix`.
  None of the four changed files is a layout, drawable, navigation or camera
  file.
- **Dev's branches:** `hackathon/dev-model` stays at `3f16d4b`,
  `hackathon/abhijeet-app-fix` stays at `9afe975`.
- **Outgoing objects:** two commits, four files, no `.tflite`, `.task`, `.apk`
  or `.zip`.

## 10. Verdict

**MERGE READY — with one caveat that is not a blocker.**

Every contract and test issue in the brief is fixed and demonstrated by a
failing negative test and a passing positive control. The build refuses an APK
with any incomplete, reordered, duplicated, unpinned, mis-sized, INT8 or
corrupted asset set; the JVM suite runs and fails correctly without any
weights; and the vocabulary test can finally fail.

The caveat: **no native Windows build was run**, so the CRLF fix is verified by
controlled test rather than by execution on Windows. If a Windows machine is
available before merge, `./gradlew testDebugUnitTest` there would close it.

PR #2 itself is not modified — `b90e3ae` sits on `hackathon/abhijeet-pr2-fix`.
Merging is the owner's call; this gate does not merge.