# Final release evidence

Status: **NOT release-certified**. This file is filled from final measured
artifacts, not copied from the old S3D baseline or a smoke run. The
submission checklist's targets are internal working targets until the
organizer's starter kit and final form settle any difference — no
competition template has been received in this checkout and none is invented
here. [Submission dashboard](SUBMISSION_DASHBOARD.md) is the judge-facing
summary of this file.

**Held-out headline (independently recomputed here): 62.61% top-1 / 80.46%
top-5 on 1,249 signer-disjoint test clips from 11 unseen participants — below
the ≥70% internal target.**

## Candidate (verified in this copy)

| Item | Value |
|---|---|
| Default backend | `ECHOGESTURE_MODEL=landmark` (no silent fallback; `s3d` is a separate baseline) |
| Classifier | `models/landmark_classifier_fp16.tflite`, 832,888 bytes, float16 weights / float32 I/O |
| Classifier SHA-256 | `a012f2b5f7a1110961079f19eb0e28c9abacb95e4d0787352e4157ebca7044a4` |
| Detector | `models/hand_landmarker.task`, 7,819,105 bytes, SHA-256 `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1` |
| Labels | `models/landmark_labels.json`, 100 classes, SHA-256 `e78f5028e97e2eaf86aff65af403af65061eebef010f0fbd41012293e8019a4a` |
| Feature config | `src/features.py`, SHA-256 `baf28d05bbd36e2783ea7ef6f78e793b89ed832d0eaad5dffed23d35c775aa84`, 255 features, 24 frames |
| Frontend | `src/landmark_frontend.py`, SHA-256 `c73ef64b72f21a86638c4a126ad642b64c4ee29851cd2f8fd8ea4ed516fa97b0` |
| Total shipped model bytes | **8,651,993** (classifier + detector; the INT8 alternative at 509,720 bytes is **unselected**) |
| Quantization | **FP16 — not INT8.** INT8 compliance is explicitly not established. |
| Organizer vocabulary/manifest | **Pending** — not received; the 100-label vocabulary is project-defined (`models/landmark_labels.json`) and the S3D `label100.json` is a different, older vocabulary |

Code commit: pending an official release commit (current copies are
uncommitted integration).

## Held-out test result (recomputed in this checkout)

Independently recomputed here by joining Dev's 1,249 test-split predictions
(`origin/hackathon/dev-model:results/predictions.jsonl`, bound to the shipped
FP16 classifier `a012f2b5…7044a4`) against the official signer-disjoint test
ground truth in `evaluation/device/v5_accuracy/splits/test.csv`, restricted to
the 100-class vocabulary. 1,249/1,249 predictions matched a ground-truth row;
no unmatched or out-of-vocabulary prediction.

| Split | N | Signers | Top-1 | Top-5 | Per-signer mean (min / max) |
| --- | --: | --: | --: | --: | --- |
| **Test (held-out, signer-disjoint)** | 1,249 | 11 unseen | **782/1249 = 62.61%** | 1005/1249 = 80.46% | 63.36% (45.77% / 80.41%) |
| Validation | 346 | 5 | 272/346 = 78.61% | 309/346 = 89.31% | see §5 of the V8 report |

The test split is 11 unseen participants, so 62.61% is the honest
generalization figure and is **below the ≥70% internal target**. Validation
(78.61%) is optimistic because the checkpoint was selected on it. Do not
report the validation rate as if it were the held-out score.

## Measurements

| Metric | Target | Evidence and result | Status |
| --- | --- | --- | --- |
| Held-out top-1 (test) | ≥70% | **62.61%** (782/1249), 11 unseen signers, recomputed in this checkout from Dev's test predictions + official split truth | BELOW TARGET |
| Validation top-1 | ≥70% | ASL Citizen validation subset, not organizer test data: FP16 272/346 = 78.61% top-1, INT8 254/346 = 73.41% top-1; FP16 top-5 89.31%, INT8 top-5 86.13%; 5 signers, 100 glosses; paired exact McNemar p = 0.007916. Official train/val/test signer and clip-ID overlaps verified **zero**. | PASS (optimistic; checkpoint selected on val) |
| Clip end-to-end p95 | <50 ms | Same 30-clip validation run: 236.44 ms p50, 315.94 ms p95; prior regeneration 327.63 ms (sequential laptop variability); target device not used | NOT_MEASURED for target hardware |
| Per-frame p95 (streaming) | <50 ms | Dev-reported on Windows x86: MediaPipe 15.7 ms/frame p50 + 0.55 ms classifier ≈ **32.80 ms p95** (`origin/hackathon/dev-model:results/results.csv`). Team measurement, **not reproduced in this checkout** and not on target hardware. | TEAM-REPORTED (unreproduced here) |
| Robustness drop | ≤5 pt | ASL Citizen validation batch N=30, 5 signers, all five corruptions paired at severity 0.5, seed 42, zero failed pairs: low_light **3.33 pt PASS**; crop **6.67 pt FAIL**; occlusion **13.33 pt FAIL**; motion_blur **13.33 pt FAIL**; frame_drop **13.33 pt FAIL** | PASS only for low_light; other conditions FAIL |
| Average power | ≤500 mW | No physical-device method yet | NOT_MEASURED |
| Thermal drift | ≤20% | No sustained 10-minute run yet | NOT_MEASURED |
| INT8 compliance | If required | Selected classifier is FP16; INT8 alternative unselected and documented as worse | NOT established |

## Verified integration smoke evidence

See `../device/CANDIDATE_INTEGRATION.md` for the single real ASL Citizen
clip, stage profile, provisional bundle verification, offline-demo worker/HUD
render, and the **161-test** integration suite. The 30-clip validation batch
and five paired robustness reports are real measured evidence but remain
limited, laptop-only and validation-only, not organizer test results.

## External blockers

- Official vocabulary, manifest, evaluator details, and any template.
- Compliant final model/candidate agreement with Dev (INT8 decision, or
  written confirmation it is not required).
- Official train/validation/test signer and clip-ID overlaps are **verified zero**
  in `evaluation/device/v5_accuracy/results/split_audit.json`; Dev training
  manifest has now been supplied and audited. The train CSV itself is kept under
  `evaluation/device/v5_accuracy/splits/train.csv` so the audit remains runnable.
- This verifies exact ID and signer disjointness, not near-duplicate video content.
- Physical Android device and camera permission for the demo.
- Registration, submission form, and receipt.
