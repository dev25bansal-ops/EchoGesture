# V4.3 — release status

Branch `integration/v4-3-masked-ensemble`, commit `5251bf8`.
Date 2026-11-14. Deadline 2026-11-18.

## Final decision

| Question | Answer |
| --- | --- |
| Four-model masked baseline works | **YES** — parity + structural tests pass |
| Nine-model masked ensemble works | **YES** — parity + structural tests pass |
| Android/Python numerical parity | **PASS** — features 0.000000, masks exact, logits 5e-6, top-1 6/6 |
| Accuracy improvement | **MEASURED** — C 79.3% vs A 63.6% aggregated over 184 evaluations |
| Physical-phone recognition | **NOT TESTED** — no device attached |
| Model rights | **BLOCKED** — see §6 |
| INT8 compliance | **UNRESOLVED** — FP16 ships |
| Recommended final APK | **V4.3** `dist/echogesture-v4.3-masked.apk` |

## 1. APK identity

| | |
| --- | --- |
| Path | `dist/echogesture-v4.3-masked.apk` |
| Full SHA-256 | `3db9d94ad761a76bdcf5dd9b3eaeff7e91dd14a51eaee2eca44acb98f1380896` |
| Bytes | 83,350,522 |
| Source commit | `5251bf8` |
| Differs from V4.2 (`5490e029…`) | yes |
| Ensemble members | **9** |
| Feature dimensions | 255 ×4 members, 381 ×5 members |
| Attention mask | hand-presence over RESAMPLED presence; supplied to every member by name |
| Model size (members only) | 8.11 MB |
| Detector-inclusive | 15.57 MB |
| Permissions | CAMERA only (plus the framework's dynamic-receiver permission) |
| Internet | absent |

Verified inside the shipped APK: all nine member hashes match the pins, combined
`995cc8d0b6f6ac25abb9f7ddca01c3ce71edf8bc2d34e34f4813a2b99dd3b4bb`, detector
`fbc2a300…`, 100 labels.

Installed and launched on `emulator-5554`: the app runs, Start → Finish
classifies, an empty capture is refused with "No hands in view" rather than a
fabricated prediction, and no crash appears in the window.

**Fallback preserved:** `dist/echogesture-v4.2.apk` is untouched at
`5490e029…`, and its four member binaries were extracted and hash-verified for
rollback.

## 2. Tests

| Suite | Result |
| --- | --- |
| Kotlin JVM | **110 passed, 0 failed** |
| Instrumentation | **11 passed, 0 failed** across 7 classes |
| `FeatureParityTest` | PASS — 6 clips, 5 signers, 9 members, both widths |
| Python A/B/C study | 184 evaluations across 2 splits × 3 conditions |
| Build-time asset gate | PASS — 9 pinned members, order and hashes checked at configure time |

Instrumentation detail: `RecordModeOnDeviceTest` 4, `CameraFeaturePathTest` 2,
`EnsembleInferenceTest` 1, `ResultCardRenderTest` 1, `ProcessingStateRenderTest` 1,
`ReplayModeParityTest` 1, `FeatureParityTest` 1.

### Tests changed or removed, and why

- `EnsembleInferenceTest.productionClassifierReproducesPythonOnFrozenFeatures`
  **removed**. Its fixture was recorded from the old four member binaries and its
  format cannot represent nine. `FeatureParityTest` supersedes it against the
  models actually shipped.
- `ResultCardRenderTest` no longer asserts a recorded confidence from the old
  weights. It now asserts the card displays a real vocabulary label, a
  confidence in range, and a populated top-5. Pinning a stale constant would tie
  the UI test to a model that is not shipped.
- `EnsembleInferenceTest` structural test now resolves tensors **by name**. It
  originally indexed input 0 and, on the new bundle, read the mask where it
  expected features — the exact hazard the brief warned about, caught by the
  test rather than by production.

## 3. What was NOT measured

- **Any physical Android phone.** Only `emulator-5554` was attached.
- **On-device latency.** The 0.5 ms ensemble figure is a laptop CPU measurement.
  Dev's 6.56 ms is likewise a laptop result. Neither is an Android number.
- **Dev's 77.02% / 74.06% splits.** Not reproduced; different clips, and the
  historical test split was used during model development.
- **The 1,249-clip test population.** This work used 72 authorised clips.
- **A real clip with hands absent for the whole recording** (all-zero mask).

## 4. Dev's result artifacts — what each actually measures

The brief flagged stale files. Here is what each is, verified against its
contents:

| Artifact | Reports | Which model | Verdict |
| --- | --- | --- | --- |
| `results/results.csv` | 77.02% test, 87.25% val, 92.79% top-5 | **nine-member masked** | Current and internally consistent |
| `results/eval_summary_test.json` | 72.78% test, 87.51% top-5 | **four-member, pre-mask** — the file itself says `"ensemble": "tcn21+tcnm61+tcnm62+rob71 (4x fp16 tflite)"` | **STALE** |
| `results/robustness.json` | clean 73.66%, worst drop 6.65 (drop_hand_25pct) | **four-member, pre-mask** — no hash field, numbers match `eval_summary_test.json` | **STALE** |
| `results/results.csv` robustness row | clean 76.94%, worst drop 4.40 | claims "9-member masked" | **INCONSISTENT with the JSON on disk** |

**None of the three JSON artifacts records a model hash.** That is the root
cause: an artifact cannot be attributed to a model, so a stale one looks as
authoritative as a current one.

**The 4.40 figure is from the nine-member ensemble** if `results.csv` is
believed — clean 76.94%, worst case `drop_hand_25pct`. The number floating in
`robustness.json` (6.65) is the old pre-mask model and must not be quoted as the
current robustness.

**I did not reproduce Dev's 4.40**, because the checkpoints and harness state it
was measured with are not all available here, and re-running a robustness sweep
against a different code state would produce a number that merely looks
comparable.

### Landmark-level is not pixel-level, and neither is a camera

All six of Dev's robustness conditions are **landmark-level corruptions**:
jitter on landmark coordinates, dropping landmarks, dropping a hand, halving the
frame rate, and a synthetic 25% camera-further scaling of coordinates.

They say nothing about:

- **pixel-level corruption** — compression, motion blur, low light, occlusion by
  a real object;
- **real-camera environmental robustness** — a cluttered background, a window
  behind the signer, a different lens, rolling shutter.

The synthetic "camera further 25%" condition scales landmark coordinates. It
does not move a camera. **A pass on it is not evidence about real cameras.**

## 5. Samsung compliance

| Requirement | Status |
| --- | --- |
| Reproducible `predict(video_path)` | **PASS** — unchanged interface |
| Fixed vocabulary + output format | **PASS** — 100 labels, 5-key dict |
| Python 3.11 | **PASS** |
| Results and model evidence | **PASS** — with the artifact caveats in §4 |
| **Quantization requirement** | **FAIL** — nine-member ensemble is FP16, not INT8 |
| **≤50 ms edge latency** | **FAIL** — inference ~276 ms p95 desktop (V4.2 measurement, unchanged direction); capture is seconds by design |
| **Robustness** | **PARTIAL** — landmark-level only; pixel and real-camera unmeasured |
| README / source package | PASS |
| Docker / setup | PASS |
| **3-minute demo video** | **BLOCKED** — needs filming |
| Presentation | BLOCKED — depends on the video |
| **Exact team/college naming** | **BLOCKED** — string not supplied |
| **Rights and licences** | **BLOCKED** — §6 |
| **Final submission form** | **BLOCKED** — needs explicit approval |

### On the INT8 question

The nine-member bundle is still FP16. It is **not** INT8 and must not be
described as such. Raw sizes are reported separately above (8.11 MB members,
15.57 MB including the 7.46 MB detector) precisely so "model size" is never
quoted as if it included the detector, or vice versa.

### On 3D

Five members consume MediaPipe's **relative z** coordinate. That is a per-frame
depth estimate in the detector's own normalised space, not metric 3D pose, and
nothing here tested invariance to camera viewpoint. **Do not claim 3D pose
invariance** on the strength of the z channel.

## 6. Model rights — BLOCKED, and not mine to resolve

Nine model binaries are now in the working tree's asset directory and were
needed to build this APK. They are **gitignored** (`app/.../assets/*.tflite` is
matched by `.gitignore`), so they are not in this commit and cannot be pushed.

Separately, and predating this work: `models/ptq_100_int8.tflite` and
`models/s3d_conv1d_qat_int8.tflite` **are** committed on `origin/main` and every
other remote branch, and `hackathon/dev-model` carries the FP16 members. I have
not added or removed them; removing them means rewriting history, which is
forbidden here and would disturb other workstreams.

**If these artifacts are restricted, the repository must not be made public
until the owner rewrites that history.** This blocks submission, not the build.

## 7. Exact human actions remaining

1. **Attach a physical Android phone** and install `dist/echogesture-v4.3-masked.apk`.
   Nothing else moves the primary criterion.
2. **Decide the model-rights question** in §6.
3. **Film the ≤3-minute demo video** (consent from anyone appearing).
4. **Supply the exact `CollegeName_TeamName`** string.
5. **Approve or reject** the Google Form submission.

## 8. Proven / hypothesis / untested

**Proven**
- The nine-member masked ensemble loads, runs, and is numerically identical to
  Python's pipeline on the same landmark input.
- The attention mask is computed the way training computed it, and mismatches
  zero times in 144 steps.
- A real defect existed in the z-path scale and is fixed; the test that found it
  now has tight enough bounds to find the next one.
- On 184 evaluations, the masked ensembles substantially beat the old one.

**Hypothesis**
- That this improves recognition on a real phone. Every off-phone measurement
  supports it; none of it was taken on a phone.
- That the mask's benefit on padded recordings will translate to live
  recordings. It is strongly suggested by the +8 and +14 clip gains, but idle
  was simulated, not recorded.

**Untested**
- Any physical phone, any live signer, any camera condition.
- Pixel-level and real-camera robustness.
- All-zero-mask recordings on a real clip.