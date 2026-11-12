# EchoGesture V7 — Dev / Hardware / Organizer Handoff

Owner: A16 (Submission and Demo Engineer). Work root:
`$TMP/echogesture-.../A16/docs/release/v7_submission/`.
Original checkout not edited; no installs, no nested agents.

This handoff states the measured state, what each side owns, and the exact
items that block a release-certified submission. Items are the
`remaining_blockers.json` gates from the V6 release report plus the
A15 safety audit — no new blockers invented, no blockers dropped.

## 1. Measured state (one paragraph)

The landmark candidate runs end-to-end on laptop through
`src/main.py:predict()`. On the 346-clip ASL Citizen validation set
(signer-disjoint, leakage-audited, project-defined 100-label vocabulary)
the FP16 selected model scores 272/346 = 78.61% top-1 and the unselected
INT8 scores 254/346 = 73.41%. Full clip-to-JSON latency on laptop
(N=30, V6 controlled) is 231.48 ms p50 / 306.55 ms p95, dominated by
MediaPipe tracking (274.34 ms p95); classifier inference is 0.12 ms p95.
Robustness on N=30 passes only low_light (3.33 pt); crop 6.67 pt and
occlusion/motion_blur/frame_drop 13.33 pt fail. Target-device latency,
power, thermal, Android, organizer vocabulary/manifest, and licensing /
redistribution are all unresolved (V7 evidence §§2–5, 7, 8).

## 2. Dev handoff (from Abhijeet → Dev)

Per `docs/hackathon/HANDOFF.md` ("Abhijeet → Dev · Evaluator inputs"):

| Item | Value |
|---|---|
| Starter-kit version / official vocabulary | **Not received.** Release uses project-defined `models/landmark_labels.json` (100 labels, SHA `e78f5028…01a4a`). Training was against this vocabulary; any change of label order requires a new config version + re-benchmark. |
| Manifest shape | Organizer manifest **not received**. Internal runner accepts `clip_id,path[,ground_truth,signer_id]` (see `src/README.md` §4). |
| End-to-end p95 + stage breakdown (to Dev) | N=30 laptop: total p50 231.48 / p95 306.55 ms. Stages p95: decode 27.14, **tracking 274.34**, features 0.24, inference 0.12, output 0.04 ms. Full JSON: `v6_release/baseline/v6_baseline_latency.json`. |
| Model SHA for every benchmark above | FP16 `a012f2b5…7044a4`; detector `fbc2a300…cde1`; features `baf28d05…aa84`; frontend `c73ef64b…97b0`; labels `e78f5028…01a4a` |
| Runtime / device | Python 3.11.18, macOS 26.3 arm64, ai-edge-litert 2.3.0, MediaPipe 1.2.2, OpenCV 5.0.0 |
| Parity status | `src/features.py` + `src/landmark_frontend.py` are the shared train/infer contract; bundled by SHA in every evidence JSON. Feature tensors are computed once and shared between FP16/INT8 in the paired evaluation (no drift between the two accuracy arms). |
| Request to Dev | (1) Written confirmation of training-split provenance (the split audit's `requires_dev_split` verified disjointness against the supplied manifest; keep that manifest hashed and available for the release SHA). (2) Decision or guidance on INT8: FP16 is the fixed default (78.61% vs 73.41% on N=346); the INT8 artifact exists (`9d24640a…`) but is unselected and `main.py` hardcodes FP16. If INT8 is required, a new benchmark of the actual INT8 candidate is needed before any switch. (3) Detector FP16 vs INT8-quantized detector is unmeasured — no proposal is made here. (4) Any retraining for robustness (occlusion/blur/crop/frame_drop at N=30) needs a paired rerun through `evaluation/device/robustness.py`. |

Dev → Abhijeet bundle items (per handoff) are all present in this copy
except the replayable training inputs (see §3, gate `conversion_inputs`).

## 3. Open gates (each blocks release certification)

From `evaluation/device/v6_release/remaining_blockers.json` (all BLOCKED)
plus the A15 audit. "Owner" is who can close the gate, not who reported it.

| # | Gate | Status | Owner / closer |
|---|---|---|---|
| 1 | Organizer manifest (`remaining_blockers:organizer_manifest`) | BLOCKED — official bundle fails by design | Organizer → Abhijeet feet-on-ground. |
| 2 | Organizer predictions (`organizer_predictions`) | BLOCKED — no `results/predictions.jsonl` exists; none fabricated | Organizer / evaluator run. |
| 3 | Official vocabulary (`official_vocabulary`) | BLOCKED — 100-label project vocabulary in use | Organizer; Dev retrains/relabels if it differs. |
| 4 | INT8 criterion (`int8_criterion`) | BLOCKED — FP16 selected, INT8 unselected, detector FP16, whole-pipeline INT8 unresolved | Organizer confirmation + Dev decision (see §2). |
| 5 | Android environment (`android_environment`) | BLOCKED — no SDK/ADB/Gradle/device; no APK; FPS/power/thermal NOT_MEASURED | Hardware owner (Abhijeet, per `ABHIJEET.md`). |
| 6 | Target timing boundary (`target_timing_boundary`) | BLOCKED — official boundary (clip-to-JSON vs per-frame vs streaming) unconfirmed | Organizer; re-measure on target device once known. |
| 7 | License/NOTICE (`license_notice`, also `COMPLIANCE_AUDIT.md` §2) | BLOCKED — no root LICENSE/NOTICE; MediaPipe blob + ASL Citizen-derived weights need attribution/confirmation | Abhijeet (package) + Dev (weight provenance) + organizer/licensor written confirmation. |
| 8 | Conversion replay (`conversion_inputs`) | BLOCKED — `.scratch/runs/model.pt`, `.scratch/landmarks/val`, `scripts/ml/tf_model.py` absent; export script ships but cannot replay | Dev. |
| 9 | Registration / template / form / receipt (`registration_submission`) | BLOCKED — none present in this copy | Abhijeet. |
| 10 | Safety defect fixes D1–D4, D6–D7 (`COMPLIANCE_AUDIT.md` §4) | Fixes proposed in `privacy_lab/fixes.py`, **not applied** (isolated copy) | Abhijeet — apply to owned files, rerun `test_privacy_lab` (7 tests). |
| 11 | Tracked research data with participant IDs (`COMPLIANCE_AUDIT.md` C5/D7) | `evaluation/asl_citizen_predictions.csv` tracked (1,705 rows) — confirm licence or strip/hash before shipping | Abhijeet. |
| 12 | College-dropdown registration check (`SUBMISSION.md`) | The live form lists exactly five colleges (VIT Vellore, VIT Chennai, SRM, Thapar, Manipal); email `prism@samsung.com` before submitting if neither teammate is from one (verified 6 Nov 2026) | Abhijeet — see V7 checklist. |

No gate is closed by anything in the V7 submission directory. The V7
evidence is explicitly "provisional" on every gate above.

## 4. Hardware handoff (what to measure, how, and to what standard)

- Device: none has been measured. Record device model, SoC, OS, runtime
  (LiteRT/MediaPipe versions), delegate (CPU/GPU/NNAPI), and thermal state
  with every number. Laptop numbers stay laptop numbers.
- Clip-path latency: run [V7_REPRODUCTION.md](V7_REPRODUCTION.md) §6
  equivalent on the target device against a labelled manifest.
  Report cold (first call) and warm separately, nearest-rank percentiles,
  per-stage breakdown — the stage table is the diagnostic if p95 lands
  above 50 ms (expect tracking to dominate).
- Power/thermal: no method has been run yet. Any claim needs a physical
  device, a sustained 10-minute workload, and the method stated
  (`release_report.json:device_metrics` is NOT_MEASURED across the board).
- Demo: follow [DEMO_SCRIPT.md](../../demo/DEMO_SCRIPT.md) +
  [DEMO_CHECKLIST.md](../../demo/DEMO_CHECKLIST.md). Record the offline
  clip-path version; the live camera path and Android remain blocked
  (macOS permission denied; no SDK/device).

## 5. What A16 delivered and where

Only `docs/release/v7_submission/` was written. No Dev model/training/
feature code, no `src/main.py`, no `evaluation/` evidence was edited.

| File | Contents |
|---|---|
| `V7_FINAL_EVIDENCE.md` | Measured evidence: candidate identity, 346-clip accuracy, leakage audit, latency, robustness, targets, non-claims, licensing status, evidence index |
| `V7_DASHBOARD.md` | Judge-facing scoreboard table with N/split/device on every row |
| `V7_ARCHITECTURE.md` | Shipped-path architecture + provenance of design decisions |
| `V7_REPRODUCTION.md` | Exact commands (tests, manifest runner, paired eval, split audit, V6 latency, robustness, bundle build/verify, safety lab) with expected outputs |
| `V7_FINAL_CHECKLIST.md` | Pre-submission checklist: verified, blocked-by-organizer, blocked-by-hardware, and must-not-claim items |
| `V7_HANDOFF.md` | This file |

Existing docs reused, not duplicated: `FINAL_EVIDENCE.md`,
`SUBMISSION_DASHBOARD.md`, `COMPLIANCE_AUDIT.md`, `SUBMISSION.md`,
`HANDOFF.md`, `DEMO_SCRIPT.md`/`DEMO_CHECKLIST.md`, `ARCHITECTURE_R8.md`,
`CANDIDATE_INTEGRATION.md`, and the `v5_accuracy` / `v5_release` /
`v6_release` / `latency_lab` evidence trees.
