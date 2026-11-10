# EchoGesture V7 — Architecture Summary (shipped path only)

This note extracts the architecture of the candidate that actually ships in
`src/main.py` (`ECHOGESTURE_MODEL=landmark`). It carries the provenance
discipline of `ARCHITECTURE_R8.md` — the full synthesis with third-party
evidence, GISLR analysis, and unmeasured items stays there and is not
duplicated here.

## 1. The shipped pipeline

```
video_path
  → [1] DECODE       read_frames() — up to 32 frames sampled evenly across
                     the WHOLE clip (linspace over frame count, cv2 grab-skip —
                     decodes nothing between targets), resized to 256×256 RGB.
                     Sampling the opening frames only is explicitly avoided:
                     one ASL Citizen clip yielded 0/32 hand detections on its
                     first 32 frames vs 11/32 when spread over all 110.
  → [2] LANDMARKS    MediaPipe HandLandmarker (models/hand_landmarker.task),
                     IMAGE mode for offline clips, num_hands=2,
                     min_hand_detection/presence/tracking_confidence=0.3.
                     Output canonical [Left, Right] slots (MediaPipe camera-
                     POV handedness passed through; no mirroring either in
                     data or demo). Missing hand = all-zero slot + present=0.
  → [3] FEATURES     src/features.py build_features → [24, 255] float32:
                     per-hand root-centre on wrist, SHARED scale (wrist→
                     middle-MCP distance of the larger present hand, so
                     inter-hand geometry survives), 3 inter-hand features
                     (wrist-wrist distance, signed x/y offset) computed from
                     RAW landmarks BEFORE centring, concat pos + lag-1
                     velocity + lag-2 acceleration, z DROPPED. Missing-hand
                     presence flags carried as an explicit mask.
  → [4] CLASSIFIER   models/landmark_classifier_fp16.tflite
                     (float16 weights, float32 I/O, dim=192 Conv1D+Transformer
                     per landmark_model_meta.json "backbone":"conv").
                     832,888 bytes, SHA a012f2b5…7044a4. Input [1,24,255].
  → [5] HEAD         softmax over 100 project-defined labels, top-5,
                     confidence, latency_ms → {"clip_id","prediction",
                     "top5","confidence","latency_ms"}.
```

Contract enforcement (`src/main.py:_init_model`): meta frames/feature_dim
vs tensor shape, float32 dtypes, 100-output vs 100-label match,
`FEATURE_DIM==255`. Any mismatch raises — there is no silent fallback to
another model, shape, or vocabulary. Model initialization is cached
(singleton) so repeat clips don't reload; `profile=True` records
per-stage `perf_counter_ns` timings used by the latency evidence.

Latency composition (V6 controlled, §3 of V7 evidence): tracking 274.34 ms
p95 of 306.55 ms total; features 0.24 ms; classifier 0.12 ms. The shipped
IMAGE path is the only strategy with no observed runtime/prediction
regression in the A01 paired comparison; cached VIDEO was measured
faster on average but changed top-1 on 12/60 pairs, and fresh VIDEO and
`image_cpu` were slower. No frame reduction, ROI/crop, or INT8 substitution
is shipped.

## 2. Design decisions (with provenance)

| Decision | Why | Provenance |
|---|---|---|
| Landmarks, not pixels | The shipped S3D pixel model scored 1.47% (PTQ) / 1.00% (QAT) signer-disjoint (1.02% chance) vs 100% in-distribution — it memorized signer appearance/background. Landmarks discard both. | **OURS, MEASURED** — S3D results in `evaluation/asl_citizen_{ptq,qat}_results.json` (see §4 below); corridor in `landmark_frontend.py` docstring |
| Shared (not per-hand) scale | Per-hand scaling would destroy hand-to-hand/hand-to-face geometry that separates minimal pairs (e.g. MOTHER vs FATHER). | Design decision documented in `src/features.py` docstring |
| Inter-hand geometry pre-centring | Root-centring puts both wrists at the origin; post-centring inter-hand distance is identically zero (a bug this module briefly shipped). | Documented in `src/features.py` docstring |
| z dropped | A third-party repo reports dropping MediaPipe z is worth +7.10 pp (transformer) / +11.34 pp (TCN) on WLASL-100 signer-disjoint, 3 seeds, 55.3%-complete, untuned corpus. **Not reproduced by us.** `use_z` flag keeps the ablation possible; default False on the strong prior. | **THIRD-PARTY — not ours.** Carries their caveats (smallest two-sided rank p at 3v3 is 0.1; dictionary video). See `ARCHITECTURE_R8.md` §3 |
| Per-sequence normalization | Mean/std per clip, not global constants, so a new signer's framing/body size drop out. | Matches GISLR 1st-place approach (third-party, above) |
| Explicit presence masks | An all-zero slot is distinguishable from a real detection at the origin. | `src/features.py` contract |
| 24 frames, uniform full-clip sampling | Train and inference use the same sampling rule (`landmark_frontend.py` is imported by the extractor; sampling lives there, not copied). | `landmark_frontend.py` docstring; `landmark_model_meta.json` |
| FP16 fixed default | The INT8 alternative is 5.2 pp worse on N=346 (78.61% vs 73.41%) and whole-pipeline INT8 (incl. FP16 detector) is unresolved. | **OURS, MEASURED** — V7 evidence §2.1 |

## 3. Measured constraints (not aspirations)

- Full clip-to-JSON p95 on laptop: 306.55 ms (V6 controlled, N=30).
  Tracking alone (274.34 ms p95) exceeds a 50 ms budget.
- Consequently the shipped path cannot meet a <50 ms end-to-end p95
  requirement on this hardware. Only a faster target device, an
  INT8-quantized detector, or a different front end could change that —
  none has been measured (see `ARCHITECTURE_R8.md` §2, levers 1–3 are
  explicitly unmeasured).
- Per-frame (~23 ms/frame at 256×256) and classifier-only (0.12 ms p95)
  timings exist in the codebase but are **not** clip p95 and are never
  presented as such.

## 4. The legacy S3D path (separate baseline, not the candidate)

`ECHOGESTURE_MODEL=s3d` selects raw-pixel S3D-Conv1D INT8 models
(`models/ptq_100_int8.tflite`, `models/s3d_conv1d_qat_int8.tflite`) with
`models/label100.json` — a different, older vocabulary. On
participant-disjoint ASL Citizen (N=1,705): PTQ 25/1705 = 1.47%,
QAT 17/1705 = 1.00% (sources: `evaluation/asl_citizen_ptq_results.json`,
`asl_citizen_qat_results.json`). It remains in the tree as a research
baseline only. No S3D number describes the landmark candidate, and no
landmark number describes S3D.

## 5. What this summary does not cover

Training recipe, augmentation, and GISLR winner analysis are in
`ARCHITECTURE_R8.md` (research draft, flagged as not independently
reproducible for its Round-8 artifacts). Data scope, vocabulary-selection
trade-offs, and the S3D-vs-landmark evidence are there too. This V7 note
covers only what ships and what was measured about it.
