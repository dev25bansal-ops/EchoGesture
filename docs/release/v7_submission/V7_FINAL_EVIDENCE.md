# EchoGesture V7 — Final Judge-Facing Evidence

Status: **NOT release-certified. Provisional evidence package only.**
Every number below is copied from a measured artifact in this isolated copy
(`$TMP/echogesture-.../A16`). No organizer template, manifest,
vocabulary, or target-device result has been received or invented.
Companion: [V7 dashboard](V7_DASHBOARD.md) · [Architecture](V7_ARCHITECTURE.md) ·
[Reproduction](V7_REPRODUCTION.md) · [Checklist](V7_FINAL_CHECKLIST.md) ·
[Handoff](V7_HANDOFF.md)

---

## 1. Candidate identity (verified in this copy)

| Item | Value | Verified |
|---|---|---|
| Default backend | `ECHOGESTURE_MODEL=landmark` in `src/main.py` (no silent fallback; `s3d` is a separate legacy pixel baseline) | Read `src/main.py:52` |
| Selected classifier | `models/landmark_classifier_fp16.tflite`, 832,888 bytes. Float16 weights, float32 I/O. **FP16 — not INT8.** Model meta: `models/landmark_model_meta.json` (`{"frames":24,"feature_dim":255,"classes":100,"dim":192,"backbone":"conv","quantization":"float16"}`) | SHA-256 `a012f2b5f7a1110961079f19eb0e28c9abacb95e4d0787352e4157ebca7044a4` (matches `v5_accuracy/results/val_346/summary.json`, `v6_release/release_report.json`) |
| Selected classifier input/output | `[1,24,255]` float32 in, `[1,100]` logits out. Enforced at init: meta frames/feature_dim must match tensor shape, dtypes must be float32, feature dim must equal `src/features.py:FEATURE_DIM` | `src/main.py:120-130` |
| Detector | `models/hand_landmarker.task`, 7,819,105 bytes (Google MediaPipe HandLandmarker model blob) | SHA-256 `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1` (matches `val_346/summary.json`, `release_report.json`) |
| Labels/vocabulary | `models/landmark_labels.json`, 100 classes, SHA-256 `e78f5028e97e2eaf86aff65af403af65061eebef010f0fbd41012293e8019a4a`. **Project-defined vocabulary** — the organizer vocabulary has not been supplied. | 100 entries verified (`len(json.load(...))==100`) |
| Feature config | `src/features.py`, SHA-256 `baf28d05bbd36e2783ea7ef6f78e793b89ed832d0eaad5dffed23d35c775aa84`, `FEATURE_DIM=255` (2 hands × 21 landmarks × 2 coords × 3 channels + 3 inter-hand features), 24 frames | `src/features.py:38-41`, `val_346/summary.json:features_sha256` |
| Frontend | `src/landmark_frontend.py`, SHA-256 `c73ef64b72f21a86638c4a126ad642b64c4ee29851cd2f8fd8ea4ed516fa97b0`. Detector mode IMAGE (offline clips), up to 32 frames sampled evenly across the whole clip, resized to 256×256. | `landmark_frontend.py:read_frames/detect_hands`, `val_346/summary.json:frontend_sha256` |
| Total shipped model bytes | **8,651,993** (832,888 classifier + 7,819,105 detector). Under the internal <20 MB working cap. Whether the organizer counts the detector toward the cap is unresolved — reported separately everywhere. | `docs/release/FINAL_EVIDENCE.md`, `release_report.json:model_selection.classifier_and_detector_bytes` |
| Unselected INT8 alternative | `models/landmark_classifier_int8.tflite`, 509,720 bytes, SHA-256 `9d24640a23244a3cd00c9fa62257733819f86ed1da5a10194c9b1cb64323700b`. **Not selected by `src/main.py`** — `_model_paths()` hardcodes the FP16 filename for `landmark`. | `src/main.py:57`, `int8_lab/README.md` note |
| Quantization compliance | **FP16 fixed default. INT8 compliance NOT established.** Detector is FP16 (`hand_landmarker.task` ships FP16; see `landmark_frontend.py` docstring noting MediaPipe ships FP32/FP16 — the verified fact is the selected classifier is non-INT8 and whole-pipeline INT8 is unresolved). | `FINAL_EVIDENCE.md`, `release_report.json:model_selection.int8_compliance=NOT_ESTABLISHED`, `remaining_blockers.json:int8_criterion` |

Code commit: **pending** — this isolated copy is not a git repository
(`git log` fails with "not a git repository"). No commit SHA is claimed for
V7. `v6_release/baseline/v6_baseline_latency.json` and robustness JSONs
record `"code_commit": null` for the same reason.

---

## 2. Accuracy evidence (measured, validation only)

### 2.1 Primary: 346-clip ASL Citizen validation (paired FP16 vs INT8)

Source: `evaluation/device/v5_accuracy/results/val_346/summary.json`
(plus `predictions.json`, `predictions.csv` in the same directory).
Protocol: "paired FP16 vs INT8 on identical feature tensors; signer-disjoint
ASL Citizen official val split; no tuning."
Dataset: "ASL Citizen permitted validation subset (NOT locked test data)."
Manifest: `evaluation/device/v5_accuracy/data/val_346/clips/fetched_manifest.csv`,
SHA-256 `3f06798af00e69314da87414bf0c39bb2ec49e578f560c111754fa00966f6c82`.
Detector mode IMAGE. Build features once per clip, both models invoked on the
same `[1,24,255]` tensor. Failures count as incorrect (scope caveat in JSON).

| Model | Top-1 (measured) | Wilson 95% | Top-5 | N |
|---|---|---|---|---|
| **FP16 (selected)** | **272/346 = 78.61%** | [73.99%, 82.61%] | 309/346 = 89.31% | 346 clips, 5 signers, 100 glosses |
| INT8 (unselected) | **254/346 = 73.41%** | [68.52%, 77.79%] | 298/346 = 86.13% | same 346 (paired) |

Successful predictions 345, failed predictions 1. The one failure (signer_2f68a231 /
INTUITIVE — no hand detected in any frame) counts as incorrect for both
models. Paired analysis (n_ok=345): argmax agreement 281/345 = 81.4%;
both correct 242, FP16-only 30, INT8-only 12, both wrong 61;
FP16 − INT8 = +5.20 pp. Mean top-1 confidence FP16 0.47, INT8 0.26.

Per-signer top-1 (FP16 / INT8, N per signer):

| Signer | N | FP16 | INT8 |
|---|---|---|---|
| signer_9123b627 | 24 | 13/24 = 54.2% | 8/24 = 33.3% |
| signer_8c1eb12a | 104 | 79/104 = 76.0% | 74/104 = 71.2% |
| signer_1393dac4 | 101 | 82/101 = 81.2% | 80/101 = 79.2% |
| signer_feb2f4a2 | 103 | 92/103 = 89.3% | 87/103 = 84.5% |
| signer_2f68a231 | 14 | 6/14 = 42.9% | 5/14 = 35.7% |

(1 signer_2f68a231 failure is included in the denominators; per-signer `failed` = 1 for signer_2f68a231.)
Performance is uneven across signers (42.9%–89.3% FP16); judges should read
the aggregate with the per-signer spread, not instead of it. The README
records per-class FP16 range 33.3% (ANYONE, FAIL, FLIP, KNIGHT1, MYSELF,
PULLCONVINCE, SCROLLDOWN) to 100% on many classes.

**Qualification (must travel with the number):** This is ASL Citizen
**validation** data, not organizer held-out test data. The team cannot claim
the organizer ≥70% held-out target from this alone — it is a signer-disjoint
validation measurement on a project-defined 100-label vocabulary. Signer
overlap between training and val/test was audited from the supplied Dev
training manifest and is zero (see §2.3), but organizer accuracy remains
PROVISIONAL until the organizer supplies its test set.

### 2.2 Secondary: 30-clip subset (subset of the same 346)

Source: `evaluation/device/v5_accuracy/results/val_30/summary.json` and
`predictions.json`. Same 5 signers; 26 glosses; unbalanced per signer.
FP16 22/30 = 73.33% (Wilson [55.6%, 85.8%]), top-5 25/30 = 83.33%;
INT8 21/30 = 70.0%, top-5 24/30 = 80.0%. Cross-run check recorded in the
README: re-extracting the same 30 clips from `val_346/predictions.json`
gives identical predictions and correctness (30/30 match), confirming the
paired pipeline is deterministic. The 30-clip set is a small subset of
§2.1 — do not quote 73.33% as a separate study; quote 78.61% (N=346) with
the per-signer table.

### 2.3 Split / leakage audit (independently verified disjoint)

Source: `evaluation/device/v5_accuracy/results/split_audit.json`.
Inputs: official `val.csv` (10,304 rows, SHA `6ac78c9d…`),
`test.csv` (32,941 rows, SHA `b1dee8f8…`).

- Official val signers: signer_2f68a231, signer_9123b627, signer_98b59cb6, signer_8c1eb12a, signer_1393dac4, signer_feb2f4a2 (6).
  Official test signers: signer_750f02ef, signer_80e03e2d, signer_e0f4874f, signer_5f5b9d6a, signer_eb1b45d9, signer_32821c08, signer_3e884348, signer_f7e60c2f, signer_03455f34, signer_e82b74bd, signer_12b28cd6 (11).
  **Official val↔test signer overlap: empty (`overlap: []`).**
- Vocab-eligible (gloss in the 100-label vocabulary): val 346 clips from
  5 signers (signer_2f68a231:14, signer_9123b627:24, signer_8c1eb12a:104, signer_1393dac4:101, signer_feb2f4a2:103); test 1,255 clips
  from 11 signers. Computed counts match Dev branch's reported
  "train 1,789 (32 signers), val 346 (5), test 1,255 (11)" exactly —
  all four `match` flags `True` (dev claim cross-check).
- 30-clip manifest: all 30 video files are rows in official val.csv,
  gloss matches val.csv gloss, all glosses in candidate vocab,
  **none in test.csv** (`none_in_test_csv: true`).
- Requires-Dev-split section (supplied manifest
  `$TMP/echogesture-.../train.csv`, SHA `87a10564…`, 40,154 rows,
  35 train signers, 1,789 vocab-eligible): `train_signers_overlap_val: []`,
  `train_signers_overlap_test: []`, `train_val_clip_overlap: 0`,
  `train_test_clip_overlap: 0`, `duplicate_clip_ids: 0`,
  `train_val_signer_disjoint: true`, `train_test_signer_disjoint: true`,
  `train_val_disjoint: true`, `train_test_disjoint: true`.
  → **The training split is verified disjoint from both val and test at
  signer and clip level, from the supplied Dev manifest.**

Limitations recorded in the audit itself: official val/test CSVs contain no
train rows; clip-level cross-split duplication could not be checked beyond
the supplied train manifest; the audit's `requires_dev_split` closes the
signer-overlap gap only because the Dev manifest was supplied and hashed.

### 2.4 Known failure cases (measured, not conjectured as causes)

- No-detection failure: 1/346 (signer_2f68a231 INTUITIVE) — no hand in any frame.
- Degenerate low-hand-count clips: CHEEK 3/32 frames with hands, BEE1 5/32
  (from `v5_release/README.md` diagnosis). Features built from mostly-absent
  hands; cause beyond the measured presence counts is not claimed.
- Per-class FP16 spread 33.3%–100% (see README §val-346).

---

## 3. Latency evidence (laptop only; target device NOT_MEASURED)

### 3.1 The number the task brief cites (236.44 / 315.94 ms)

`docs/release/FINAL_EVIDENCE.md:34` and `SUBMISSION_DASHBOARD.md:35` state:
"Same 30-clip validation run: **236.44 ms p50, 315.94 ms p95**; prior
regeneration 327.63 ms (sequential laptop variability); target device not used."
These are doc-level aggregates of sequential laptop runs; the underlying
machine-readable profile in this copy is the V6 controlled baseline (§3.2).
Both agree on the conclusion: full clip-to-JSON latency is **6× the 50 ms
target** on laptop, dominated by landmark tracking. Target-device p95 is
**NOT_MEASURED** (no Android SDK/ADB/device; see §7).

### 3.2 Controlled baseline (V6, machine-readable)

Source: `evaluation/device/v6_release/baseline/v6_baseline_latency.json`
and `v6_release/release_report.json:controlled_latency`.
Method: "External wall time around one production `predict(profile=True)` call
per listed clip; nearest-rank percentiles." Manifest: the 30-clip validation
manifest, SHA-256 `004e5effb7e6c19b515fdd00b048459fff236d8b35c6b7f7b49e91a1efa5d618`
(30 clips, 5 signers). Path: `src/main.py` `ECHOGESTURE_MODEL=landmark`,
32-frame IMAGE, 256×256, 24×255 features, FP16 classifier.
Environment: Python 3.11.18, macOS 26.3 arm64, ai-edge-litert 2.3.0,
MediaPipe 1.2.2, OpenCV 5.0.0. `target_device: false`.

| Metric | Measured |
|---|---|
| All 30 calls p50 | **231.48 ms** |
| **All 30 calls p95** | **306.55 ms** |
| Warm 29 calls p95 | 284.05 ms |
| Cold / first call | 483.00 ms |
| Video decode p95 | 27.14 ms |
| **Landmark tracking p95** | **274.34 ms** |
| Feature preparation p95 | 0.24 ms |
| **Classifier inference p95** | **0.12 ms (negligible)** |
| Output postprocessing p95 | 0.04 ms |

All 30 calls succeeded; validation top-1 in the same run 22/30 = 73.33%
(same 30-clip accuracy as §2.2). The file itself qualifies the measurement:
`representative_performance_measured: false`, "Local validation timing
only. Target-device performance is NOT_MEASURED and p95 cannot be used as
target-device evidence." Feasibility field: `"infeasible_on_this_32_frame_path"`.

The earlier profiled runs recorded alongside (235.21 ms p50 / 309.75 ms p95,
and the 327.63 ms regeneration in `CANDIDATE_INTEGRATION.md`) are the same
measurement at different times — laptop thermal/contention variance, not
different results. All are laptop-only.

### 3.3 Bottleneck statement (honest)

Landmark tracking is 89% of end-to-end time (274.34 ms of 306.55 ms p95).
The classifier is negligible (0.12 ms p95). Per-frame (~23 ms/frame at
256×256 in `landmark_frontend.py` docstring) or classifier-only timings
must **not** be substituted for the full clip-to-JSON p95 — the V4/V5/V6
reports explicitly reject that substitution, and the dashboard states it.

### 3.4 Optimizations NOT adopted (measured rejections, A01 latency lab)

Source: `evaluation/device/latency_lab/LATENCY_REPORT.md` and
`benchmark.json` (30-clip, image vs video strategies).
Fresh-per-clip MediaPipe VIDEO: 501 ms slower per clip on average.
Stateful cached VIDEO: 12.6 ms faster on average but changed top-1 on
12/60 pairs and top-5 on 52/60 pairs — rejected (state carried across clips).
`image_cpu`: 568 ms slower per clip. The existing IMAGE path is the only
tested strategy with no observed runtime or prediction regression. No
frame-reduction, ROI/crop, or INT8 substitution is selected as a safe speedup.

---

## 4. Robustness evidence (N=30, five conditions)

Sources: `evaluation/device/v5_release/paired/<condition>/fp16/robustness.json`
(FP16 = selected) and `../summary.json`; V6 copies the same five
`robustness.json` reports into `release_report.json:validation_scope.robustness`.
Protocol (all five): manifest SHA `004e5eff…618` (30 clips, 5 signers),
corruptions v2 (`evaluation/device/corruptions.py`), severity 0.5, seed 42,
per-clip seeds via `robustness_lab.config.derive_clip_seed`, paired clean
vs corrupted through the real pipeline, `split: validation`,
`representative_robustness_measured: true`, 0 failed pairs, clean top-1
22/30 = 0.7333 in every FP16 report (same clean baseline).

| Corruption (FP16, selected) | Clean top-1 | Corrupted top-1 | Drop | Verdict vs ≤5 pt |
|---|---|---|---|---|
| **low_light** | 0.7333 | 0.7000 (21/30) | **3.33 pt** | **PASS** |
| crop | 0.7333 | 0.6667 (20/30) | 6.67 pt | FAIL |
| occlusion | 0.7333 | 0.6000 (18/30) | 13.33 pt | FAIL |
| motion_blur | 0.7333 | 0.6000 (18/30) | 13.33 pt | FAIL |
| frame_drop | 0.7333 | 0.6000 (18/30) | 13.33 pt | FAIL |

Pair detail per condition (FP16): low_light 2 flips (2 clean→corrupt wrong,
1 wrong→correct); crop 4 flips (4/2); occlusion 5 flips (5/1);
motion_blur 4 flips (4/0); frame_drop 4 flips (4/0).
Every condition has `affected_signer_count: 5`, per-signer breakdowns in the
JSON. Wilson 95% for the 30-clip clean top-1 is [55.6%, 85.8%] in every
report — drops move in 3.33 pt steps (1 clip), so small differences are
not resolved at this N.

**Only low_light ≤5 pt passes. The other four conditions fail.**
INT8 (unselected, same protocol, clean 0.700) drops: low_light 0.00,
crop 0.00, occlusion 13.33, motion_blur 6.67, frame_drop 13.33. The
`v5_release/README.md` caution travels with these: at N=30 FP16-vs-INT8
robustness is not statistically resolved; no single model dominates.
INT8 is in any case unselected, so only the FP16 column is the candidate's
evidence.

What the diagnostics measure (not overclaimed): hand presence survives
corruption on flipped clips (frames-with-hand same range or higher vs clean),
so failures are downstream of detection; feature MAE on flipped clips
0.05–0.24 (crop/occlusion/frame_drop/low_light), 0.60 for motion_blur.
Clean-wrong set (8 clips) is identical across corruptions — baseline
errors, not corruption-induced. These are diagnoses of where the drift is,
not proven single causes.

---

## 5. Targets table (judge-facing, honest)

Internal working targets from the submission checklist (slides), until the
organizer starter kit settles any difference. "Target" is the checklist
target, not an organizer-confirmed threshold.

| Metric | Working target | Measured | Status |
|---|---|---|---|
| Held-out top-1 | ≥70% | 78.61% FP16 (272/346), 73.41% INT8 (254/346) on **ASL Citizen validation** (signer-disjoint, leakage-audited); **not organizer held-out test** | PROVISIONAL (validation only) |
| Clip end-to-end p95 | ≤50 ms (slides p95 <50 ms; see checklist) | Laptop full clip-to-JSON 306.55 ms p95 (N=30, V6 controlled, doc aggregate 315.94 ms); target device | **FAIL on laptop; target-device NOT_MEASURED** |
| Robustness drop | ≤5 pt | low_light 3.33 pt PASS; crop 6.67 pt, occlusion 13.33 pt, motion_blur 13.33 pt, frame_drop 13.33 pt (N=30 each) | **PASS only low_light** |
| Average power | ≤500 mW | No physical-device method | NOT_MEASURED |
| Thermal drift (10 min) | ≤20% | No sustained run | NOT_MEASURED |
| Model size | <20 MB (rubric) vs ≤25 MB (sample sheet) — unresolved | Selected classifier+detector 8,651,993 bytes (8.25 MiB); classifier alone 832,888 | Under either; detector-accounting unresolved |
| INT8 compliance | If required | Selected classifier FP16; detector FP16; INT8 alternative unselected and worse (−5.2 pp on §2.1) | NOT established |
| Android | Install/run on physical device | No SDK/ADB/Gradle/device | NOT available |
| Organizer vocabulary/manifest/template | Supplied | Not received | BLOCKED |

No compliance, privacy, or licensing approval is claimed anywhere in this
package (see §8). Where a target is not measurable, the status says so
rather than rounding a smoke result into a pass.

---

## 6. Judge-facing architecture summary (actual shipped path)

```
video_path
  → sampled decode            up to 32 frames spread evenly across the whole
                              clip (grab-skip, never the opening frames only),
                              resized to 256×256 RGB
  → MediaPipe HandLandmarker  2 hands, 21 landmarks each, 256×256 frames,
                              IMAGE mode, thresholds 0.3 (hand_landmarker.task)
  → feature transform         per-hand root-centre on wrist, SHARED body scale
                              (wrist→middle-MCP of the larger present hand) so
                              inter-hand geometry survives; 3 inter-hand
                              features computed BEFORE centring; pos + lag-1
                              velocity + lag-2 acceleration; z dropped
                              → [24, 255] float32, presence mask carried
  → TFLite classifier (FP16)  [1,24,255] → [1,100] float32 logits
  → softmax ranking           prediction, top-5, confidence, latency_ms
```

`src/main.py:predict()` implements exactly this. `_init_model()` enforces the
input contract (shape `[1,frames,feature_dim]` from
`landmark_model_meta.json`, float32 I/O, 100 outputs matching
`landmark_labels.json`, `FEATURE_DIM==255`) and raises on mismatch rather
than silently falling back. The legacy `ECHOGESTURE_MODEL=s3d` pixel path
remains in the same file as a separate research baseline: it scored
1.47% (PTQ) / 1.00% (QAT) signer-disjoint vs 1.02% chance and 100%
in-distribution — the finding that motivated the landmark architecture.
No S3D number describes the landmark candidate.

Key design points (as the judges need them, all verifiable in code):
landmarks discard signer appearance/background (the documented S3D failure
mode); per-clip normalization makes framing/body size signer-invariant;
missing hands are explicit `present` flags, not zero landmarks;
inter-hand geometry is measured pre-centring (a post-centring computation
would be identically zero — the bug `src/features.py` documents);
the feature module is one shared contract for training and inference.

Full synthesis (with provenance categories, third-party evidence, and
unmeasured items) stays in `ARCHITECTURE_R8.md`. The V7 extraction is
[V7_ARCHITECTURE.md](V7_ARCHITECTURE.md).

---

## 7. What is NOT claimed (explicit)

- **No target-device result.** No Android SDK, ADB, Gradle, physical
  device, or emulator in this environment (`release_report.json:android_status`,
  `remaining_blockers.json:android_environment`). No APK built or installed.
  FPS, power, thermal, and target-device p95 are all NOT_MEASURED.
- **No organizer held-out accuracy.** §2 is validation data.
- **No organizer vocabulary compliance.** Project-defined 100 labels until
  the organizer supplies its vocabulary.
- **No organizer manifest / predictions / template.** The official bundle
  builder fails by design without organizer `predictions.jsonl`
  (`v6_release/README.md` §verify step 5; `remaining_blockers.json:organizer_manifest`,
  `organizer_predictions`).
- **No registration/submission/receipt.** (`remaining_blockers.json:registration_submission`.)
- **No privacy/licensing approval.** See §8.
- **No conversion replay.** The export script ships
  (`scripts/ml/export_tflite_tf.py`) but training inputs
  (`.scratch/runs/model.pt`, `.scratch/landmarks/val`, `scripts/ml/tf_model.py`)
  are absent, so conversion provenance is source-present, not replay-verified.
- **No competition template was invented.** `results/results.csv` columns
  follow `build_bundle.py`'s required schema
  (`metric,target,measured,unit,device,runtime,pass,notes`), not an
  organizer template.
- **Detector FP16 / whole-pipeline INT8 is unresolved** — the selected
  classifier is FP16 with float32 I/O, and the detector is the stock
  MediaPipe FP16 blob. Whether the organizer requires INT8, and what "INT8"
  covers (classifier only vs whole pipeline), is unconfirmed.

---

## 8. Licensing, privacy, and redistribution status

Source: `docs/release/COMPLIANCE_AUDIT.md` (A15 safety audit). Verdict:
**NOT READY FOR SUBMISSION. Do not label this release compliant.**
That audit covers safety only — not accuracy/latency/INT8 — and its
licensing section is carried here in full because the submission package
must not overclaim:

- There is **no root `LICENSE`/`NOTICE`** in this copy. `README.md`'s
  "Our code and harnesses: MIT" has no licence text behind it.
- `models/hand_landmarker.task` (MediaPipe blob): needs Apache-2.0/NOTICE
  attribution + provenance note; verify MediaPipe terms at submission time.
- Dev landmark classifier + labels (ASL Citizen-derived, Microsoft
  research licence, non-commercial/no-redistribution): evaluation is what
  the licence contemplates; redistribution/shipping of trained weights
  requires explicit Dev + organizer written confirmation. Do not relabel
  as MIT/freely licensable.
- S3D weights (WLASL-derived, C-UDA academic/non-commercial): confirm the
  MIT grant covers redistribution; do not claim commercial shippability.
- Verified-safe subset (code evidence, not legal approval): `src/`+`app/`
  offline (no network imports); no hardcoded secrets; demo temp-clip
  deletion on success and on exception (D5 control). Known low/medium
  defects D1–D4, D6–D7 with proposed fixes in
  `evaluation/device/privacy_lab/fixes.py` (see `COMPLIANCE_AUDIT.md` §4).
- `evaluation/asl_citizen_predictions.csv` (1,705 rows, participant IDs)
  is tracked in Git — personal-adjacent metadata; confirm research-licence
  redistribution or strip/hash participant IDs before shipping.

Until those are resolved, the correct label for this release is
**not release-certified**, matching `FINAL_EVIDENCE.md`.

---

## 9. Evidence index (where every number lives)

| Claim | Artifact |
|---|---|
| 272/346 FP16, 254/346 INT8, Wilson CIs, per-signer, paired agreement | `evaluation/device/v5_accuracy/results/val_346/summary.json` + `predictions.json` + `predictions.csv` |
| 22/30 FP16 (30-clip subset), determinism cross-check | `evaluation/device/v5_accuracy/results/val_30/summary.json` + README |
| Training/val/test disjoint + Dev claim match | `evaluation/device/v5_accuracy/results/split_audit.json` |
| Latency p50 231.48 / p95 306.55 ms, stage breakdown, cold call | `evaluation/device/v6_release/baseline/v6_baseline_latency.json` + `benchmark.json` + `results.csv` |
| Doc-level 236.44 / 315.94 ms aggregate + 327.63 ms regeneration | `docs/release/FINAL_EVIDENCE.md:34`, `SUBMISSION_DASHBOARD.md:35`, `CANDIDATE_INTEGRATION.md:80` |
| Robustness 5 conditions × FP16 (and INT8) | `evaluation/device/v5_release/paired/*/fp16|int8/robustness.json` + `v5_release/summary.json` + `v5_release/README.md` |
| Model bytes, SHAs, INT8 NOT_ESTABLISHED, blockers | `evaluation/device/v6_release/release_report.json`, `remaining_blockers.json`, `package_audit.json`, `manifest.json` |
| Rejected speedups (cached VIDEO etc.) | `evaluation/device/latency_lab/LATENCY_REPORT.md`, `latency_lab/results/benchmark.json` |
| Provisional ZIP (13 members, SHA `41c5a938…`) | `evaluation/device/v6_release/provisional.zip` |
| Safety/licensing audit | `docs/release/COMPLIANCE_AUDIT.md`, `evaluation/device/privacy_lab/` |
| Architecture synthesis + provenance | `ARCHITECTURE_R8.md` |
| Submission checklist (targets, college restriction) | `docs/hackathon/SUBMISSION.md` |

All paths are relative to this isolated copy root. Reproduction commands
are in [V7_REPRODUCTION.md](V7_REPRODUCTION.md); the handoff to Dev and to
hardware is [V7_HANDOFF.md](V7_HANDOFF.md).
