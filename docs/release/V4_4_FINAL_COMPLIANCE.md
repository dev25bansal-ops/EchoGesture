# V4.4 — final compliance

Checked against the actual artifact, not against the plan.

## Organizer requirements

| Requirement | Status | Evidence |
| --- | --- | --- |
| Reproducible `predict(video_path)` | **PASS** | Runs on real MP4s; verified from a freshly extracted bundle and inside Docker |
| Exact output schema | **PASS** | Five keys, types and ranges validated (§ contract check) |
| Fixed 100-class vocabulary | **PASS** | Labels unchanged, 100 entries, order preserved |
| Python 3.11 | **PASS** | Host venv 3.11.18; Docker image `python:3.11-slim` |
| `requirements.txt` closure | **PASS** | Present in bundle; fresh-env install verified |
| Docker setup | **PASS** | `docker build` + real inference succeeded |
| **Quantization (INT8)** | **FAIL** | Nine FP16 members. Not INT8, and not described as INT8 |
| **≤50 ms edge latency** | **FAIL** | Inference ~276 ms-class on laptop; capture is seconds by design |
| **Robustness** | **PARTIAL** | Landmark-level only; pixel and real-camera unmeasured |
| Results files | **PASS (provisional)** | Versioned, attributable; organizer manifest absent so official build stays blocked |
| **Demo video ≤3 min** | **BLOCKED** | Not filmed |
| Presentation | **BLOCKED** | Depends on the video |
| **Exact team/college naming** | **BLOCKED** | String not supplied |
| **Organizer vocabulary/manifest** | **BLOCKED** | Not supplied; none invented |
| **Rights and licences** | **BLOCKED** | §Rights |
| **Google Form** | **BLOCKED** | Requires explicit approval |

## Measured compliance detail

### Model size — reported separately, never conflated

| Component | Size |
| --- | --- |
| Nine FP16 members | 8.11 MB |
| MediaPipe detector | 7.46 MB |
| **Detector-inclusive** | **15.57 MB** |
| APK | 83.35 MB |

Quoting "3.52 MB" for the shipped bundle (as `results.csv` did for the old
four-member set) would understate the detector-inclusive footprint by more than
four times.

### On INT8

`results.csv` records INT8 losing 6.37 points and two TFLite runtimes disagreeing
on the same file. No new INT8 export was attempted. The bundle is FP16 and is
labelled FP16 everywhere.

### On 3D

Five members consume MediaPipe **relative z** — a per-frame depth estimate in the
detector's normalised space. **This is not 3D pose and no viewpoint invariance is
claimed.** Nothing tested rotation or camera-angle invariance.

### Power and thermal

**NOT TESTED.** No target hardware. `results.csv` correctly records
`avg_power` and `thermal_drift_10min` as `NOT_MEASURED`.

## Privacy

| Check | Status |
| --- | --- |
| CAMERA permission only | **PASS** — verified in the shipped APK manifest |
| No INTERNET permission | **PASS** |
| No hidden uploads | **PASS** — no network permission exists to upload with |
| No raw-video persistence without consent | **PASS** — landmarks only, in memory |
| No private model assets in source commits | **PASS for this branch** — `.tflite` is gitignored and none were added |

## Rights — BLOCKED, and pre-existing

### What may be restricted

| Path | Concern | Documented rights | Status |
| --- | --- | --- | --- |
| `models/ptq_100_int8.tflite` | Committed on `origin/main` and every remote branch since the initial commit | **None found in-repo** | **UNCLEAR** |
| `models/s3d_conv1d_qat_int8.tflite` | Same | **None found** | **UNCLEAR** |
| `hackathon/dev-model` → four FP16 members | Trained from ASL Citizen (research licence) | ASL Citizen terms described as research-use | **UNCLEAR for redistribution** |
| Nine masked members (this work) | Same training corpus | Same | **UNCLEAR** |
| ASL Citizen videos, landmark caches | Research data, never committed | Research licence | **Must stay out of Git** |
| `evaluation/asl_citizen_predictions.csv` | Derived predictions | Possibly fine | **Verify before public release** |
| `results/*.csv` | Aggregate metrics only | Likely fine | **Low risk** |

### What is safe to publish

- Source code (`src/`, `app/`, `scripts/`, `evaluation/` tooling)
- Reports and aggregate metrics
- The APK **only if** the model rights are cleared — it embeds weights

### What requires authorised private distribution

- The nine `.tflite` members and the detector
- Any APK that bundles them
- Any fixture containing landmark arrays

### Honest statement on the existing exposure

Restricted-derived weights **may already be in public Git history** on
`origin/main` and other branches. **Deleting them in a new commit does not remove
them from history.** Remediation requires a history rewrite, which is the
repository owner's decision and was explicitly not performed here.

**No rights are claimed as cleared merely because the repository is already
public.**

### Owner approval required before:

1. Making the repository public or announcing it.
2. Publishing any APK.
3. Rewriting history to remove weights.
4. Submitting the Google Form.

## APK

No production Android change was made in V4.4: the V4.4 work is Python, parity
tooling, results provenance and release process. The validated Android artifact
is therefore **V4.3**, unchanged. An identical rebuild was deliberately not
produced — see `V4_4_FINAL_RELEASE_STATUS.md`.