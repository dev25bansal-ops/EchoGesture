# Track 2 compliance matrix — verified against the organizer PDF

Source: `XR_Hackathon_2026_Consolidated_Slides_for_sharing_v1.pdf` (28 pages).
Track 2 is pages 9–12; general submission rules are 24–27. Every quote below is
transcribed from that PDF's text layer, not from an earlier repo summary.

**States:** `PASS` · `FAIL` · `BLOCKED_EXTERNALLY` · `NOT_TESTED` · `FIXED_AND_VERIFIED`

Compiled 2026-11-12. Where the deck contradicts itself the row says so rather
than picking a side.

---

## 1. Scoring rubric (p.11, 100-point jury matrix)

| # | Requirement | PDF wording | Implementation | Evidence | State |
| --: | --- | --- | --- | --- | --- |
| R1 | Model quantized | `Model Quantization (20%): Compilation to TFLite/ONNX <20MB.` | 4 × FP16 TFLite, 3,544,080 B | `models/landmark_model_meta.json` | **PASS** (format), see R8 |
| R2 | Edge latency | `Edge Latency (20%): Measured inference time <50ms.` | full clip-to-JSON p95 274.2 / 154.8 / 99.8 ms | `results/results.csv` | **FAIL** |
| R3 | Top-1 accuracy | `Top-1 Accuracy (35%): Precision on held-out test videos.` | 909/1249 = 72.78% | `evaluation/device/ensemble_evidence/` | **PASS** (≥70% target) |
| R4 | Robustness & HUD | `Robustness & Delivery (25%): Occlusions & HUD clarity.` | low-light/blur/frame-drop PASS; crop −10.00 pp, occlusion −6.67 pp FAIL | `robustness_ensemble/RESULTS.md` | **FAIL** (2 of 5 conditions) |
| R4b | Results-sheet template | p.11 table: `metric target measured unit device runtime pass` with rows `top1_accuracy >=70 %`, `latency_p95 <=50 ms`, **`model_size <=25 MB`**, `accuracy_drop_robustness <=5 pt`, `avg_power <=500 mW`, `thermal_drift_10min <=20 %` | `results/results.csv` matches the template exactly (8 columns, 6 rows, this order) | `results/results.csv` | **PASS**. Note: the sheet's own target is **25 MB**, not the 20 MB of p.9/p.10; the row carries `<=25` and the 11.37 MB measurement passes both readings. |

## 2. Hard budgets (p.9)

| # | Requirement | PDF wording | Implementation | Evidence | State |
| --: | --- | --- | --- | --- | --- |
| R5 | Model size | `Budgets: ≤ 20 MB quantized model (INT8)` | 11,363,185 B total incl. detector | `submission_final/RELEASE_MANIFEST.json` | **PASS** |
| R6 | Latency budget | `≤ 50 ms latency` | see R2 | — | **FAIL** |
| R7 | Zero cloud | `zero cloud streaming` | APK requests CAMERA only; INTERNET and ACCESS_NETWORK_STATE removed from the merged manifest | `evaluation/device/android_four_model_evidence.md` | **FIXED_AND_VERIFIED** |
| R8 | INT8 specifically | p.9 `quantized model (INT8)`; p.10 `< 20 MB INT8`; p.11 rubric says only `TFLite/ONNX <20MB` with no INT8 | shipped artifact is **FP16** | `models/landmark_model_meta.json` → `"quantization": "float16"` | **BLOCKED_EXTERNALLY** — organizer clarification drafted, not sent |

## 3. Architecture constraints (p.10)

| # | Requirement | PDF wording | Implementation | Evidence | State |
| --: | --- | --- | --- | --- | --- |
| R9 | Exocentric third-person | `Exocentric third-person view: moving heads, angled signers, low light` | Android bound `DEFAULT_FRONT_CAMERA` (selfie view), contradicting the stated scenario | `MainActivity.kt` | **FIXED_AND_VERIFIED** — now prefers `DEFAULT_BACK_CAMERA` with front fallback, no handedness swap |
| R10 | **3D hand features** | `Pose-invariant 3D hand features` | `build_features(..., use_z=False)`; `landmarks[..., :2]` discards the MediaPipe z channel | `src/features.py:151` | **FAIL (literal reading)** — see §7 |
| R11 | Local processing | `strictly local processing, zero cloud streaming` | same as R7 | — | **PASS** |
| R12 | Captions | `Captions on display or audio output for two-way conversation` | on-screen caption + top-5 on the Android HUD | `MainActivity.kt` | **PASS** |

## 4. Required deliverables (p.10, p.11)

| # | Requirement | PDF wording | Implementation | Evidence | State |
| --: | --- | --- | --- | --- | --- |
| R13 | `predict()` | `GitHub repo: predict() video → text, reproducible setup`; `the predict() in main.py` | `predict(video_path, *, clip_id=None, profile=False) -> dict` | `src/main.py:434` | **PASS** |
| R14 | Predictions file | p.10 `predictions.jsonl (one JSON per clip)`; p.11 `Teams submit a single predictions.jsonl file, one JSON object per line, one line per test clip`; but p.11 package bullet says `predictions.json and results.csv in results folder` and p.12 layout says `results/predictions.json` | `results/predictions.jsonl`, 1,249 lines, p.11 schema; builder emits **both** `.jsonl` and `.json` with identical JSON-Lines content | `results/predictions.jsonl` | **PASS** (deck contradicts itself; both spellings shipped) |
| R15 | Results sheet | p.11 `A CSV with one row per metric. Each row has a measured value, a target, and a pass/fail column.` plus the template table | 6 rows, exact p.11 column names and metric names and order | `results/results.csv` | **PASS** |
| R16 | Return object | p.11 `{"clip_id","prediction","top5","confidence","latency_ms"}` | exact five keys | `src/main.py:552` | **PASS** |
| R17 | Manifest IDs | `clip_id must match the test manifest exactly` | no organizer manifest was supplied anywhere in the repo; builder refuses to fabricate one | `scripts/release/build_bundle.py` | **BLOCKED_EXTERNALLY** |
| R18 | Conversion script | p.10 `Quantized model + conversion script (< 20 MB INT8)`; p.11 `Quantized model file and conversion script` | `scripts/ml/export_tflite_tf.py` is present in the built release ZIP at `scripts/ml/export_tflite_tf.py` | `evaluation/device/release_final2/echogesture_final2_ensemble.zip` listing | **PASS** |
| R19 | Demo video | p.10 `Demo video (2–3 min) or deployed app URL`; p.24 `Prototype code, 5-min demo video and deck due`; p.25 `Demo video, max 3 minutes (YouTube or Drive link)`; p.27 `a 5-minute demo video by 18th November` | **none recorded**; recording guide targets a 2:50 cut, which satisfies all three readings | `docs/demo/DEMO_RECORDING_GUIDE.md` | **NOT_TESTED** (no video file exists in the repo or `submission_final/demo/`) |

## 5. Submission rules (p.24–27)

| # | Requirement | PDF wording | Implementation | Evidence | State |
| --: | --- | --- | --- | --- | --- |
| R20 | Deadline | `FINAL SUBMISSION · Last date 18 Nov 2026, 11:59 PM` | — | — | **BLOCKED_EXTERNALLY** (timezone unstated in 28 pages) |
| R21 | Private repo OK | `Working prototype code — public or shared GitHub repo` | public source-only branch | `hackathon/abhijeet-public` | **PASS** |
| R22 | Docker | `README with reproducible setup instructions, Docker files & other requirements` | root `Dockerfile`, Python 3.11, entrypoint fixed | `Dockerfile` | **FIXED_AND_VERIFIED** (entrypoint and `--help` verified by a fresh local build of the current Dockerfile); **FAIL** on real inference — see §9 |
| R23 | Deck naming | `Presentation file (PPT or PDF) — CollegeName_TeamName` | 11-slide deck, placeholders unfilled | `docs/submission/EchoGesture_Submission.pptx` | **BLOCKED_EXTERNALLY** (team/college strings) |
| R24 | ZIP layout | p.12 `team_name.zip` → `src/{requirements.txt,main.py,README.md}` + `results/{predictions.json,results.csv}` | builder produces this; adds `models/`, `manifest.json`, `release.json` | `build_bundle.py` | **PASS** |
| R25 | Python 3.11 | p.12 `Python 3.11 – all scripts are written for Python 3.11` | `FROM python:3.11-slim`; local venv is CPython 3.11.18 | `Dockerfile`, `.venv/bin/python -V` | **PASS** |
| R26 | Disqualification | `Teams NOT following the submission guideline would lead to direct disqualification` | naming and template rules still unfilled | — | **BLOCKED_EXTERNALLY** |

---

## 6. The two contradictions in the organizer deck

Recorded rather than resolved, because we cannot resolve them from inside:

1. **INT8.** p.9 and p.10 say `INT8`; the p.11 scored rubric says only
   `Compilation to TFLite/ONNX <20MB`. The 20-point quantization score is
   defined on p.11, which does not require INT8. We ship FP16 and state it
   everywhere. Clarification drafted in `docs/submission/INT8_CLARIFICATION.md`.
2. **Predictions filename.** p.10/p.11 say `predictions.jsonl`; the p.12 bundle
   diagram says `predictions.json`. The builder now emits both with identical
   JSON-Lines content so either reading resolves.

## 7. The 3D-feature gap (R10) — stated plainly

`src/features.py` builds 255 features per frame from
`landmarks[..., : 3 if use_z else 2]`, and `use_z` defaults to `False`. So the
MediaPipe **z** channel is discarded; the model consumes x/y plus derived
velocity, acceleration, inter-hand geometry and presence masks.

- **Literal reading of p.10:** features are 2D, so this is a scoring risk on the
  architecture description.
- **Mitigating facts:** depth from a single monocular camera is itself
  estimated, and the shipped ensemble reaches 72.78% × 35% of the rubric
  weighting on xy features. The trained weights expect 255 values; enabling
  `use_z` would change `FEATURE_DIM` and invalidate every model.
- **Decision:** do **not** flip `use_z`. It requires retraining, which is Dev's
  call, not a change to make against a frozen, measured candidate.
- **Dev proposal:** `docs/release/DEV_3D_FEATURE_PROPOSAL.md` — benchmark an
  xy+z variant against the same signer-disjoint split before any change.

## 8. What is genuinely not done

| Item | Owner | Why |
| --- | --- | --- |
| Physical-phone camera, power, thermal | Abhijeet | no device attached |
| Demo video | Abhijeet | not recorded |
| Team/college strings, deck rename | Abhijeet | registration details not in repo |
| INT8 answer | Organizer | clarification unsent |
| Organizer manifest + vocabulary | Organizer | never supplied |
| Weight redistribution | Dev | no documentary permission |
| 50 ms latency | — | tracking is 90% of runtime; model change cannot fix it |

---

## 9. Corrections made on 2026-11-12 by Agent C (independent PDF re-extraction)

Rows below were **wrong or unverifiable as written** and have been corrected
above. Quotes are re-extracted from the PDF's own text layer with a second,
independent decoder (zlib + content-stream token scan), not copied from
`_slides_text.txt`.

| # | Was | Why it was wrong | Corrected to |
| --- | --- | --- | --- |
| R15 | implied the sheet's target for size is 20 MB | p.11's own table says **`model_size <=25 MB`**; only p.9/p.10 say 20 MB | `results/results.csv` now carries `<=25` as the p.11 template value; the note records that 11.37 MB passes both |
| R19 | cited only p.10 and p.25 | p.24 says **`Prototype code, 5-min demo video and deck due`** and p.27 says **`a 5-minute demo video by 18th November`**. There are three lengths: 2–3 min (p.10), 5-min (p.24, p.27), max 3 min (p.25) | row now lists all four citations; the 2:50 cut satisfies all three readings |
| R22 | `NOT_TESTED` (no Docker daemon here) | a Docker daemon **is** present and a fresh build of the current `Dockerfile` succeeds | entrypoint/`--help` **verified**; end-to-end `predict()` **FAILS** inside the image — see §9.1 |
| R25 | cited `runtime.txt` as evidence | `runtime.txt` **does not exist** in the repo | evidence is `Dockerfile` + `.venv/bin/python -V` (3.11.18) |
| R18 | said "`build_bundle.py` closure" | the claim is verifiable directly | the built ZIP contains `scripts/ml/export_tflite_tf.py` (5,582 B) |
| R14 | p.11 attribution was partly wrong | the `predictions.json` spelling is on **p.11 too** (`predictions.json and results.csv in results folder`), not only p.12 | row now quotes both p.11 bullets |

### 9.1 Docker: new, genuine FAIL

A fresh `docker build` of the current `Dockerfile` succeeds, and
`docker run --rm <image> --help` prints the contract. But running a real
prediction with weights and a clip mounted fails:

```
File "/app/src/landmark_frontend.py", line 74, in get_landmarker
OSError: libGLESv2.so.2: cannot open shared object file: No such file or directory
```

`python:3.11-slim` does not carry the EGL/GLES shared objects MediaPipe 1.2.2
loads at `hand_landmarker` creation. `src/requirements.txt` correctly moved to
`opencv-python-headless`, which fixes the **cv2** import, but `mediapipe` itself
still needs `libegl1 libgl1 libgles2`. With those three packages installed the
same container returns the correct reference prediction:

```
"clip_id": "0283886564670357-WHAT FOR", "prediction": "WHATFOR1", "confidence": 0.8818
```

The fix is one `apt-get install` line in `Dockerfile`. **This file does not own
`Dockerfile`** (`scripts/release/` and packaging belong to another workstream),
so it is recorded here as a defect with the exact remediation, not applied.

### 9.2 Two pre-existing Docker images in the local daemon are stale

`echogesture:rc` and `echogesture:rc2` both fail at `import cv2` with
`libxcb.so.1`. Both were built before the headless-OpenCV change and must not
be used to evidence Docker compliance. The fresh `eg-verify-c` build above is
the only accurate test.

### 9.3 Claims verified exactly (no change needed)

| Claim | Verification |
| --- | --- |
| 4 × FP16, 3,544,080 B; detector 7,819,105 B; total 11,363,185 B | `stat` on all five files; all four member SHA-256 match `models/README.md` |
| `"quantization": "float16"` | `models/landmark_model_meta.json` |
| 909/1249 = 72.78%, top-5 87.51%, 6 eligible no-hand clips | `results/predictions.jsonl` has 1,249 rows, exact five keys, no empty predictions |
| median p95 274.2 / 154.8 / 99.8 ms | recomputed from `performance_final/results_ensemble/benchmark.json`; raw warm p95 is 274.18 / 154.77 / 99.81 |
| crop −10.00 pp, occlusion −6.67 pp | `robustness_ensemble/RESULTS.md` |
| `use_z=False` discards the z channel | `src/features.py:151` |
| Android merged manifest has CAMERA only | read back from `merged_manifest/debug/.../AndroidManifest.xml`; no INTERNET, no ACCESS_NETWORK_STATE |
| Back camera preferred, front fallback | `MainActivity.kt:281-284` |
| `predict()` returns the five p.11 keys | `src/main.py:552-557` |
| 100-gloss vocabulary, all uppercase | `models/landmark_labels.json`, 100 keys, all `.isupper()` |
| Python 3.11 | 3.11.18 |

### 9.4 Discrepancies noted but belonging to other owners

- **`results.csv` latency row drift.** The working tree briefly carried
  `275.8 / 154.8 / 93.9` sourced only from a `latency_remeasured` block in
  `submission_final/RELEASE_MANIFEST.json`; no raw benchmark with those
  numbers exists in the tree. Restored to the 274.2 / 154.8 / 99.8 values that
  `performance_final/results_ensemble/benchmark.json` actually contains. If a
  newer run is authoritative, the raw JSON must land in the tree first.
- **`RELEASE_MANIFEST.json` is internally inconsistent.** `performance` still
  says 274.2 / 154.8 / 99.8 while `release_candidate.latency_remeasured` says
  275.8 / 152.5 / 93.9, and `release_candidate.gates.docker` points at
  `evaluation/device/DOCKER_ACCEPTANCE.md`, **which does not exist**. The
  release ZIP also predates `src/main.py` (22,149 B in ZIP vs 23,946 B on disk)
  and lacks `src/parallel_tracker.py`. Owner: packaging workstream.
- **APK size disagreement.** `evaluation/device/android_apk_evidence.md` says
  75,072,782 B / SHA `041e163f…`; `android_four_model_evidence.md` says
  75,072,706 B / SHA `2ec087e5…`; the APK on disk is 75,072,706 B but hashes to
  `dd774e9d…`. Three different values. Owner: Android workstream.
- **Unit-test count.** `docs/submission/*` and slide 9 of the deck say "9 unit
  tests"; the actual `app/src/test` tree contains **20** `@Test` methods
  (8 ModelContract, 4 EnsembleParity, 3 EnsembleRuntimeParity, 5
  FeatureTransform), which is what `android_four_model_evidence.md` and
  `app/android_final/README.md` report. Slide 9 corrected to 20.
- **Deck slide 9 parity figure.** "1.3e-7 drift" has no artifact in the repo.
  The parity tests pin `1e-5` (logits), `1e-6` (softmax) and `1e-3`
  (confidence) tolerances. Corrected to the repo's recorded `<1e-5` / `<1e-6`.