# EchoGesture — submission dashboard (judge-facing summary)

**Status: NOT release-certified.** The integrated landmark candidate runs
end-to-end. A 30-clip ASL Citizen validation batch and five paired corruption
runs now provide measured, limited, laptop-only evidence. Target-device
latency, INT8 compliance, organizer contract, and physical-phone verification
remain blocked. Read the N, split, and device on every number.

Companion files: [demo script](../demo/DEMO_SCRIPT.md) ·
[shot list](../demo/SHOT_LIST.md) · [checklist](../demo/DEMO_CHECKLIST.md) ·
[final evidence](FINAL_EVIDENCE.md) · [candidate integration](../../evaluation/device/CANDIDATE_INTEGRATION.md)

---

## 1. What this is

EchoGesture recognizes a **fixed set of 100 isolated ASL signs** fully
on-device: sampled video frames → MediaPipe hand landmarks (two hands) →
shared 255-feature transform → TFLite classifier → top-1 gloss, top-5,
confidence, and latency as JSON. It is **not** continuous translation and
**not** an interpreter replacement.

The runtime entrypoint is `predict(video_path) -> dict` in
`src/main.py`, defaulting to `ECHOGESTURE_MODEL=landmark` (the FP16
landmark candidate). `ECHOGESTURE_MODEL=s3d` selects the older pixel
baseline; there is no silent fallback between them.

## 2. What actually works (candidate, measured)

| Claim | Result | N / evidence |
|---|---|---|
| `predict()` contract, JSON shape | Working: `clip_id`, uppercase `prediction`, `top5`, `confidence`, `latency_ms` | Real ASL Citizen clip via manifest runner |
| One real clip predicted correctly | `PULLCONVINCE`, confidence 0.7507 | **N=1** — smoke only, not accuracy |
| Full clip-to-JSON profile | 642.01 ms total: decode 27.34 ms, landmark tracking 612.15 ms, features 0.24 ms, classifier 0.17 ms | **N=1**, macOS 26.3 arm64, Python 3.11, LiteRT 2.3.0 |
| Full clip-to-JSON profile (validation batch) | 236.44 ms p50 / 315.94 ms p95; latest regeneration 327.63 ms p95 | **N=30**, 5 signers, ASL Citizen validation, macOS arm64; not target-device evidence |
| Validation accuracy (not organizer held-out test) | 22/30 = 73.33% top-1; 25/30 = 83.33% top-5 | **N=30**, 5 signers, 26 glosses; Wilson CI 55.6%–85.8%; provisional |
| Five paired robustness conditions (severity 0.5, seed 42) | low_light 3.33 pt PASS; crop 6.67 pt FAIL; occlusion 13.33 pt FAIL; motion_blur 13.33 pt FAIL; frame_drop 13.33 pt FAIL | **N=30 each**, 5 signers, zero failed pairs |
| Provisional ZIP bundle | Built, extracted, and a real `PULLCONVINCE` prediction executed from it | Verified in candidate integration |
| Offline desktop demo code | Worker, HUD states, cleanup; real-clip HUD render correct | Camera path **not** verified (see §5) |
| Test suite | **161 / 161 pass** (integration, evaluator, robustness, model-contract, release, demo, streaming, evidence) | Python 3.11 |

**Stage profile, full clip (N=1, 642.01 ms):**

| Stage | ms |
|---|---|
| Sampled decode | 27.34 |
| Landmark tracking (MediaPipe) | 612.15 |
| Feature preparation | 0.24 |
| Classifier inference | 0.17 |

Landmark tracking dominates. The classifier is not the bottleneck; the
front end is.

## 3. Model-size table (all deployed artifacts, bytes measured in this copy)

| Artifact | Bytes | SHA-256 (prefix) | Role / status |
|---|---:|---|---|
| `models/landmark_classifier_fp16.tflite` | 832,888 | `a012f2b5…` | **Selected classifier**, float16 weights, float32 I/O — **not INT8** |
| `models/hand_landmarker.task` | 7,819,105 | `fbc2a300…` | **Detector**, required by the landmark path |
| Candidate total (classifier + detector) | **8,651,993** | — | 8.25 MiB shipped model bytes |
| `models/landmark_classifier_int8.tflite` | 509,720 | `9d24640a…` | **Unselected** INT8 alternative (documented as worse; not the runtime default) |
| Old S3D baseline (QAT) `models/s3d_conv1d_qat_int8.tflite` | 901,200 | `107756f3…` | **Separate research baseline**, pixel path |
| Old S3D baseline (PTQ) `models/ptq_100_int8.tflite` | 904,256 | `067c743c…` | **Separate research baseline**, pixel path |

The internal submission checklist targets **<20 MB** built output, reporting
every deployed model and the total. The candidate total (8,651,993 bytes)
sits under that, **but** whether detector files count toward the size cap is
an unresolved organizer question — report classifier and detector separately
so either accounting works. Full SHA-256 digests are in
`../device/CANDIDATE_INTEGRATION.md`.

## 4. Unsupported targets — say so plainly

| Target | Status | What would close it |
|---|---|---|
| Held-out top-1 ≥70% | **PROVISIONAL / not organizer-held-out evidence** — 73.33% on the 30-clip ASL Citizen validation subset, with wide CI and no training-split audit | Full signer-disjoint validation or organizer held-out evaluation; Dev split and raw predictions |
| Clip end-to-end p95 <50 ms | **FAIL on laptop measurement** (309.75–327.63 ms); target device **NOT_MEASURED** | Representative p95 on target device; tracking remains the bottleneck |
| Robustness drop ≤5 pt | **FAIL overall** — only low_light 3.33 pt passes; crop 6.67 pt, occlusion 13.33 pt, motion_blur 13.33 pt, frame_drop 13.33 pt | Improve robustness for occlusion, blur, crop, and temporal drops; rerun paired validation |
| Average power ≤500 mW | **NOT_MEASURED** | Physical-device measurement method |
| Thermal drift ≤20% | **NOT_MEASURED** | Sustained 10-minute run on device |
| INT8 compliance | **NOT established** — selected classifier is FP16 | Compliant INT8 candidate + accuracy/runtime comparison, or written organizer confirmation that INT8 is not required |
| Live camera demo | **Blocked locally** (macOS denied camera permission) | Granted permission + verified preview/capture before filming |
| Android demo | **Not implemented** (no SDK/device) | Physical Android device + actual install/run |
| Official vocabulary, manifest, evaluator, template | **Not received in this checkout** | Organizer starter kit / final form; confirm vocabulary scope, timing boundary, size accounting, and any required template before filling those fields |

## 5. Architecture at a glance

```
video_path
  → sampled decode            up to 32 frames spread across the whole clip,
                              resized to 256×256
  → MediaPipe HandLandmarker  2 hands, 21 landmarks each, 256×256 frames
  → feature transform         root-centre per hand, shared body scale,
                              pos + lag-1 velocity + lag-2 acceleration,
                              3 inter-hand geometry features, z dropped
                              → [24, 255], then [1, 24, 255] float32
  → TFLite classifier (FP16)  → [1, 100] float32 logits
  → softmax ranking           prediction, top-5, confidence, latency_ms
```

Design points the judges actually need: landmarks discard signer appearance
and background; per-clip normalization makes framing/body size signer-
invariant; missing hands are explicit flags, not zeros; the feature module
is one shared contract for training and inference so they cannot drift.

The older S3D pixel baseline is a **separate** research path
(`ECHOGESTURE_MODEL=s3d`): it scored 1.47% (PTQ) / 1.00% (QAT) on
participant-disjoint ASL Citizen at a 1.02% chance floor, versus 100% on an
in-distribution control — the finding that motivated the landmark
architecture. No S3D number describes the candidate.

## 6. How to reproduce (Python 3.11, from the repo root)

```bash
pip install -r src/requirements.txt

# Test suite (39 tests, no model run required for the doc check)
.venv/bin/python -m unittest discover -s evaluation/device -p 'test_*.py' -v

# Real prediction on a manifest (clip_id,path; ground_truth/signer_id optional)
.venv/bin/python evaluation/device/run_manifest.py /absolute/path/to/manifest.csv \
  --output-dir /absolute/path/to/local-benchmark-output

# Profiled single-clip run (stage timings; still not a representative p95)
.venv/bin/python evaluation/device/run_manifest.py /absolute/path/to/manifest.csv \
  --output-dir /absolute/path/to/local-benchmark-output --profile

# Offline demo (requires camera permission; see limitations)
.venv/bin/python app/offline_demo.py
```

> A profiled run reports actual stage times and explicit failures; accuracy
> stays `null` unless every clip carries ground truth, and small samples are
> marked nonrepresentative even when correct. Laptop numbers are never
> target-device evidence.

## 7. How to read this dashboard

- **N=1** means one clip, one run. It proves the path executes; it proves
  nothing about accuracy or latency distributions.
- Every metric without an N, a split, and a device is a claim to distrust —
  including ours, so each row above names its denominator or its absence.
- If a target is not measurable yet, the dashboard says **NOT_MEASURED**
  rather than rounding a smoke result into a pass.
