# EchoGesture V7 — Judge-Facing Dashboard

Status: **NOT release-certified.** The integrated landmark candidate runs
end-to-end. A 346-clip ASL Citizen validation set, a leakage audit, and
five paired corruption runs provide measured, limited, laptop-only evidence.
Target-device latency, INT8 compliance, organizer contract, and
physical-phone verification remain blocked. Read the N, split, and device
on every number.

Companion files: [V7 evidence](V7_FINAL_EVIDENCE.md) ·
[Architecture](V7_ARCHITECTURE.md) · [Reproduction](V7_REPRODUCTION.md) ·
[Checklist](V7_FINAL_CHECKLIST.md) · [Handoff](V7_HANDOFF.md) ·
[Demo script](../../demo/DEMO_SCRIPT.md) ·
[Shot list](../../demo/SHOT_LIST.md) ·
[Demo checklist](../../demo/DEMO_CHECKLIST.md) ·
[Candidate integration](../../../evaluation/device/CANDIDATE_INTEGRATION.md)

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
| Full clip-to-JSON profile (single clip) | 642.01 ms total: decode 27.34 ms, landmark tracking 612.15 ms, features 0.24 ms, classifier 0.17 ms | **N=1**, macOS 26.3 arm64, Python 3.11, LiteRT 2.3.0 |
| Full clip-to-JSON profile (validation batch) | 231.48 ms p50 / 306.55 ms p95 (V6 controlled baseline); doc aggregate 236.44 ms p50 / 315.94 ms p95; regeneration 327.63 ms p95 | **N=30**, 5 signers, ASL Citizen validation, macOS arm64; not target-device evidence |
| Classifier inference alone | 0.12 ms p95 — negligible | N=30, same V6 run |
| Validation accuracy (not organizer held-out test) | **272/346 = 78.61%** FP16 top-1, 309/346 = 89.31% top-5; INT8 254/346 = 73.41%, top-5 298/346 = 86.13% | **N=346**, 5 signers (signer_2f68a231/signer_9123b627/signer_8c1eb12a/signer_1393dac4/signer_feb2f4a2), 100 glosses, signer-disjoint, leakage-audited; provisional |
| Training/val/test disjointness | Train (35 signers, 1,789 eligible) ∩ val ∩ test: zero signer overlap, zero clip overlap, zero duplicate clip IDs | `split_audit.json`; Dev claim cross-check all True |
| Five paired robustness conditions (severity 0.5, seed 42) | low_light 3.33 pt PASS; crop 6.67 pt FAIL; occlusion 13.33 pt FAIL; motion_blur 13.33 pt FAIL; frame_drop 13.33 pt FAIL | **N=30 each**, 5 signers, zero failed pairs |
| Provisional ZIP bundle | Built, extracted, and a real `WHATFOR1` prediction executed from it (also `PULLCONVINCE` in integration) | Verified in V6 release + candidate integration |
| Offline desktop demo code | Worker, HUD states, cleanup; real-clip HUD render correct | Camera path **not** verified (see §4) |
| Test suite | **161 / 161 pass** (integration, evaluator, robustness, model-contract, release, demo, streaming, evidence) | Python 3.11 |

**Stage profile, V6 controlled baseline (N=30, p95):**

| Stage | p95 ms |
|---|---|
| Video open/decode | 27.14 |
| Landmark tracking (MediaPipe) | 274.34 |
| Feature preparation | 0.24 |
| Classifier inference | 0.12 |
| Output postprocessing | 0.04 |

Landmark tracking dominates. The classifier is not the bottleneck; the
front end is.

## 3. Model-size table (all deployed artifacts, bytes measured in this copy)

| Artifact | Bytes | SHA-256 (full) | Role / status |
|---|---:|---|---|
| `models/landmark_classifier_fp16.tflite` | 832,888 | `a012f2b5f7a1110961079f19eb0e28c9abacb95e4d0787352e4157ebca7044a4` | **Selected classifier**, float16 weights, float32 I/O — **not INT8** |
| `models/hand_landmarker.task` | 7,819,105 | `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1` | **Detector**, required by the landmark path; stock MediaPipe FP16 blob |
| Candidate total (classifier + detector) | **8,651,993** | — | 8.25 MiB shipped model bytes |
| `models/landmark_classifier_int8.tflite` | 509,720 | `9d24640a23244a3cd00c9fa62257733819f86ed1da5a10194c9b1cb64323700b` | **Unselected** INT8 alternative (−5.2 pp on N=346; not the runtime default) |
| Old S3D baseline (QAT) | 901,200 | `107756f3…` | **Separate research baseline**, pixel path |
| Old S3D baseline (PTQ) | 904,256 | `067c743c…` | **Separate research baseline**, pixel path |

The internal submission checklist targets **<20 MB** built output, reporting
every deployed model and the total. The candidate total (8,651,993 bytes)
sits under that, **but** whether detector files count toward the size cap is
an unresolved organizer question — report classifier and detector separately
so either accounting works.

## 4. Unsupported targets — say so plainly

| Target | Status | What would close it |
|---|---|---|
| Held-out top-1 ≥70% | **PROVISIONAL / not organizer-held-out evidence** — 78.61% on the 346-clip ASL Citizen validation set (signer-disjoint, leakage-audited), project-defined vocabulary | Organizer held-out evaluation; official vocabulary |
| Clip end-to-end p95 <50 ms | **FAIL on laptop measurement** (306.55 ms controlled; doc 315.94 ms; regeneration 327.63 ms); target device **NOT_MEASURED** | Representative p95 on target device; tracking remains the bottleneck |
| Robustness drop ≤5 pt | **FAIL overall** — only low_light 3.33 pt passes; crop 6.67 pt, occlusion 13.33 pt, motion_blur 13.33 pt, frame_drop 13.33 pt (N=30 each) | Improve robustness for occlusion, blur, crop, temporal drops; rerun paired validation |
| Average power ≤500 mW | **NOT_MEASURED** | Physical-device measurement method |
| Thermal drift ≤20% | **NOT_MEASURED** | Sustained 10-minute run on device |
| INT8 compliance | **NOT established** — selected classifier FP16, detector FP16, whole-pipeline INT8 unresolved | Compliant INT8 candidate + accuracy/runtime comparison, or written organizer confirmation FP16 is accepted |
| Live camera demo | **Blocked locally** (macOS denied camera permission) | Granted permission + verified preview/capture before filming |
| Android demo | **Not implemented** (no SDK/device) | Physical Android device + actual install/run |
| Official vocabulary, manifest, evaluator, template | **Not received in this checkout** | Organizer starter kit / final form; confirm vocabulary scope, timing boundary, size accounting, and any required template before filling those fields |

## 5. Architecture at a glance

```
video_path
  → sampled decode            up to 32 frames spread evenly across the whole clip,
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

# Test suite (161 tests)
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

Full reproduction commands, including the exact V5/V6 harness invocations
and expected outputs, are in [V7_REPRODUCTION.md](V7_REPRODUCTION.md).

## 7. How to read this dashboard

- **N=1** means one clip, one run. It proves the path executes; it proves
  nothing about accuracy or latency distributions.
- **Every number carries its evidence.** If a number is validation-only,
  laptop-only, or N=30, the table says so in the same row. Nothing is
  restated from this dashboard without those qualifiers.
- If a target is not measurable yet, this dashboard says **NOT_MEASURED**
  rather than rounding a smoke result into a pass.
