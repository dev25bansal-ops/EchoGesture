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
| R14 | Predictions file | p.10 `predictions.jsonl (one JSON per clip)`; p.11 `single predictions.jsonl file, one JSON object per line`; p.12 layout `results/predictions.json` | `results/predictions.jsonl`, 1,249 lines, p.11 schema; builder emits **both** `.jsonl` and `.json` | `results/predictions.jsonl` | **PASS** (deck contradicts itself; both spellings shipped) |
| R15 | Results sheet | `results.csv`; `A CSV with one row per metric. Each row has a measured value, a target, and a pass/fail column.` | 6 rows, exact p.11 template order and metric names | `results/results.csv` | **PASS** |
| R16 | Return object | p.11 `{"clip_id","prediction","top5","confidence","latency_ms"}` | exact five keys | `src/main.py` | **PASS** |
| R17 | Manifest IDs | `clip_id must match the test manifest exactly` | no organizer manifest was supplied; builder refuses to fabricate one | `scripts/release/build_bundle.py` | **BLOCKED_EXTERNALLY** |
| R18 | Conversion script | `Quantized model + conversion script (< 20 MB INT8)` | `scripts/ml/export_tflite_tf.py`, staged into the bundle | `build_bundle.py` closure | **PASS** |
| R19 | Demo video | p.10 `Demo video (2–3 min) or deployed app URL`; p.25 `max 3 minutes` | **none recorded**; recording guide exists | `docs/demo/DEMO_RECORDING_GUIDE.md` | **NOT_TESTED** |

## 5. Submission rules (p.24–27)

| # | Requirement | PDF wording | Implementation | Evidence | State |
| --: | --- | --- | --- | --- | --- |
| R20 | Deadline | `FINAL SUBMISSION · Last date 18 Nov 2026, 11:59 PM` | — | — | **BLOCKED_EXTERNALLY** (timezone unstated in 28 pages) |
| R21 | Private repo OK | `Working prototype code — public or shared GitHub repo` | public source-only branch | `hackathon/abhijeet-public` | **PASS** |
| R22 | Docker | `README with reproducible setup instructions, Docker files & other requirements` | root `Dockerfile`, Python 3.11, entrypoint fixed | `Dockerfile` | **FIXED_AND_VERIFIED** (entrypoint); **NOT_TESTED** (no Docker daemon here) |
| R23 | Deck naming | `Presentation file (PPT or PDF) — CollegeName_TeamName` | 11-slide deck, placeholders unfilled | `docs/submission/EchoGesture_Submission.pptx` | **BLOCKED_EXTERNALLY** (team/college strings) |
| R24 | ZIP layout | p.12 `team_name.zip` → `src/{requirements.txt,main.py,README.md}` + `results/{predictions.json,results.csv}` | builder produces this; adds `models/`, `manifest.json`, `release.json` | `build_bundle.py` | **PASS** |
| R25 | Python 3.11 | `Python3.11 – all scripts are written for Python3.11` | `FROM python:3.11-slim`; venv is 3.11 | `Dockerfile`, `runtime.txt` | **PASS** |
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