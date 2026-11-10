# Quantization final audit and Dev handoff (A02)

Owner: A02 (`evaluation/device/quantization_final/` only). Branch
`hackathon/abhijeet-integration` at `f81dd30c3e6dc1fc035244c5ca9791e7a84da0c4`.
No model, feature, label, detector, or `src/main.py` file was edited. No
training, conversion, download, or benchmark loop was run. The selected FP16
model was not replaced; its baseline remains the default.

## Verdict (one line)

FP16 stays the default; the shipping INT8 classifier is a real quantized
graph but is a measured accuracy regression, the detector remains FP16, and
organizer INT8/detector scope is unresolved — so whole-pipeline INT8 is
NOT established and no improved artifact is selected.

## Artifacts verified (SHA-256, this checkout)

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `models/landmark_classifier_fp16.tflite` (selected) | 832,888 | `a012f2b5f7a1110961079f19eb0e28c9abacb95e4d0787352e4157ebca7044a4` |
| `models/landmark_classifier_int8.tflite` (candidate) | 509,720 | `9d24640a23244a3cd00c9fa62257733819f86ed1da5a10194c9b1cb64323700b` |
| `models/hand_landmarker.task` (detector) | 7,819,105 | `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1` |
| `models/landmark_labels.json` | 1,492 | `e78f5028e97e2eaf86aff65af403af65061eebef010f0fbd41012293e8019a4a` |
| `models/landmark_model_meta.json` | 123 | `7f189a16f4d175042b5481afb973217fc21410832ebd14c5bacf2f516473947e` |
| `src/features.py` (frozen) | 8,312 | `baf28d05bbd36e2783ea7ef6f78e793b89ed832d0eaad5dffed23d35c775aa84` |
| `src/landmark_frontend.py` (frozen) | 6,633 | `c73ef64b72f21a86638c4a126ad642b64c4ee29851cd2f8fd8ea4ed516fa97b0` |
| `src/main.py` (runtime) | 10,154 | `c695d78b1b1980e3b95b5357e0549385638eda3274a76eba479c6f3dcddf8141` |
| `scripts/ml/export_tflite_tf.py` | 5,582 | `8abdb6368748babdbf3a74351fe15b8fb8da3348de60dd1a387da51364e96e5f` |

## Classifier INT8 (tensor/operator facts)

From `results/audit.json` (FlatBuffer schema + LiteRT 2.3.0 interpreter IO):

- **FP16 selected**: 192 tensors (117 FLOAT32, 38 FLOAT16, 37 INT32),
  135 ops, 0 quantized tensors, 0 INT8-touching ops, 788,978 B weights.
  float32 IO, scale 0.0.
- **INT8 candidate**: 158 tensors (100 INT8, 54 INT32, 4 FLOAT32),
  101 ops, 86 INT8-touching ops, 117 quantized tensors, 402,777 B weights.
  float32 external IO (scale 0.0) with an internal QUANTIZE at the entry and
  DEQUANTIZE at the exit.
- Weight tensors: 21 INT8 + 35 INT32; no FLOAT16/FLOAT32 weights.
- All 117 quantized tensors have finite positive scales and in-range INT8 zero
  points; no scale/zero-point length mismatch; 35 tensors carry per-channel
  scales (the Conv1D / MatMul kernels use per-channel, `quantized_dimension`
  is 0 for dense kernels and 3 for the `[1,1,17,192]` depthwise kernels).
  All per-channel loss-points are zero; asymmetric zero points are confined to
  per-tensor activation tensors (e.g. `out_proj_1/BiasAdd` zp −10,
  `attn_scores_1/truediv` zp −128, `query_in_1/…` zp +19). This matches a
  per-channel-weight / per-tensor-activation QDQ quantization.

Full per-tensor scale/zero-point/dimension records are in `results/audit.json`
under `classifiers.int8.tensors`.

## Detector quantization

`hand_landmarker.task` members `hand_detector.tflite` (133 FLOAT16 weights)
and `hand_landmarks_detector.tflite` (102 FLOAT16 weights) are each FP16 with
zero quantized tensors. The landmark frontend is FP16 end to end, so any
whole-pipeline INT8 claim is false from the detector alone.

## Full runtime compliance

`results/runtime_boundary.json`:

- Default `_model_paths()` selects the FP16 landmark model (SHA
  `a012f2b5…7044a4`); the INT8 file is never chosen.
- Loading the INT8 file through an in-process `_model_paths()` override with
  **no loader change** succeeds: float32 `[1,24,255]` in / `[1,100]` out,
  100 labels, finite logits. The external quantisation branch in `predict()`
  is unused (`_INPUT_SCALE == 0`), so the INT8 file is an IO drop-in.
- Drop-in IO is **compatibility, not acceptance**. The joint acceptance gate
  (`acceptance_gate.py`) fails closed with 3 of 10 checks false:
  `existing_int8_no_regression`, `whole_pipeline_int8_established`,
  `official_interpretation_resolved`.

## Accuracy, size, latency evidence

- Existing paired 346-clip evidence verified live (`v5_accuracy/results/val_346`
  summary SHA `1ef28344…`, predictions SHA `8d980fbe…`, manifest SHA
  `3f06798a…`): FP16 **272/346 (78.61%)**, INT8 **254/346 (73.41%)**,
  345 successful extractions, 1 no-hand frontend failure (P5 INTUITIVE). FP16
  lead 5.20 pp. 64 top-1 disagreements (30 FP16-only, 12 INT8-only correct).
- Drift (345 clips): mean logit L2 5.355, max 10.209; mean logit max-abs
  1.742; prob L1 mean 0.584. FP16 mean top-1 confidence 0.47 vs INT8 0.26.
- Fresh real-clip inference reproduced the stored predictions exactly on three
  disagreement clips (`CHEEK`→RABBIT2, `ELEVATOR1`→8HOUR, `CHEEK`→KNIGHT1) and
  the no-hand clip.
- Size: FP16+detector 8,651,993 B; INT8+detector 8,328,825 B (saves 323,168 B,
  38.8% of the classifier). Under any 20 MiB / 20 MB interpretation.
- Latency is a prior-lab label; classifier time is sub-millisecond and not the
  bottleneck (MediaPipe tracking dominates ~241 ms mean / ~612 ms one-clip).
  No new timing loop was run here.

## Calibration / distribution / axis / missing-hand findings

- The shipped INT8 exporter (`export_tflite_tf.py`) calibrates on the first
  `X[:300]` rows of the official **val** split after `build_features` and it
  **skips** clips with zero hand frames. It is path-sorted, unshuffled, and has
  no per-signer/per-gloss/coverage stratification. A partly missing-hand clip
  is included only if its max presence > 0.5.
- Missing-hand handling is a real, load-bearing gap: `build_features` returns a
  mask, but `predict()` discards it. Velocity/acceleration are lagged
  differences, so an all-missing frame still carries the delta from the last
  present frame into the classifier (measured max-abs 2.20 on a held-hands-out
  probe) rather than being zeroed. This is not INT8-specific, but calibration
  and any QAT must include missing-hand and near-threshold frames.
- The INT8 graph already uses per-channel weight scales and per-tensor
  activation scales. The mixed float side path reported in earlier labs is
  `DEQUANTIZE/ZEROS_LIKE/QUANTIZE` around the query broadcast (the graph
  retains 4 FLOAT32 tensors: input 0, side-path 119/120, output 157); the main
  84-op path is INT8.
- Two exporters target the same output (`export_tflite_tf.py` via TensorFlow
  and `scripts/ml/export_int8.py` on `origin/hackathon/dev-model` via ONNX
  Runtime static QDQ). The shipped `landmark_model_meta.json` does not match
  either exporter's writer (it has `quantization:"float16"` and no
  `val_top1_int8`/`size_kb`). The checkpoint and calibration tensor list are
  absent from this checkout, so the shipped INT8 bytes cannot be reproduced or
  re-quantised here. Exact provenance is a Dev-side blocker.
- The accuracy gate in `export_tflite_tf.py` is dead: `drop = 0.0` cannot
  trigger the >0.15 rejection, and `int8_acc` (full X) vs `pt_acc` (first 200)
  are compared across different populations.

## Frozen feature and label contracts

`fixtures/provenance.json` pins the contract; `audit_quantization.feature_contract()`
re-checks it against `build_features`:

- Input `[1,24,255]` float32; output `[1,100]` float32 **logits** (softmax in
  the caller). Frames 24, feature dim 255, 100 classes, contiguous uppercase
  ordered labels.
- Feature SHA `baf28d05…aa84`; labels SHA `e78f5028…9a4a`. A change to label
  order or feature layout invalidates both artifacts and requires a new
  candidate + new paired benchmark.

## Exactly what Dev must change

1. **State the organizer INT8 criterion in writing** (does float32 IO with an
   INT8 interior count; do detector bytes count toward the cap). Until then,
   `official_interpretation` stays UNRESOLVED and whole-pipeline INT8 cannot be
   established. Record it in `docs/release/FINAL_EVIDENCE.md`.
2. **Provide the shipping INT8 provenance**: which script, which checkpoint
   SHA, which representative set, and why the meta file matches neither
   exporter. Without this the INT8 file is trusted-but-unverifiable.
3. **Re-export against the frozen contracts** with a representative set that
   (a) includes missing-hand and near-threshold frames per signer, (b) is
   stratified across glosses/signers and shuffled, and (c) is drawn without
   using the locked test split.
4. **Fix the dead accuracy gate** in `export_tflite_tf.py` (`drop` no longer
   hardcoded, and FP16/INT8 evaluated on the *same* population) before any
   re-export is accepted.
5. **Target the P12 gap and worst per-gloss regressions** (ADVERTISE,
   BASEBALLCAP, CAPTURE, CHEEK, ELEVATOR1, EMPTY2, FINE1 are 2-FP16-only each;
   P12 INT8 33% vs FP16 54%). If the criterion requires it, add a compliant
   detector or obtain organizer acceptance of the FP16 detector.
6. **Treat per-channel PTQ, QAT, and mixed precision as separate hypotheses**;
   record the float-operator list and operator/dtype map for each new
   candidate.

## Requested acceptance tests (already implemented, runnable)

- `test_quantization.py` (8 tests): graph classification, detector FP16,
  frozen IO contract, ordered labels, pinned feature hash, missing-hand
  feature behaviour, and live verification of the stored paired evidence.
- `acceptance_gate.py`: fails closed on wrong SHA, changed feature/label
  contract, non-INT8 candidate, accuracy regression, unresolved official
  interpretation. Current candidate exits 1 (three false checks by design).
- `runtime_boundary.py`: proves the INT8 file loads unmodified as an IO
  drop-in while the default remains FP16.

## Reproduction

```bash
P=/Users/abhijeetmiskin/AppData/EchoGesture
$P/.venv/bin/python -I -m unittest discover -s evaluation/device/quantization_final -p 'test_*.py' -v
$P/.venv/bin/python -I evaluation/device/quantization_final/audit_quantization.py \
  --output evaluation/device/quantization_final/results/audit.json
$P/.venv/bin/python -I evaluation/device/quantization_final/runtime_boundary.py \
  --output evaluation/device/quantization_final/results/runtime_boundary.json
$P/.venv/bin/python -I evaluation/device/quantization_final/acceptance_gate.py \
  --output evaluation/device/quantization_final/results/acceptance.json   # exit 1, expected
```

## Blockers

- Organizer INT8 definition and detector byte-counting are unresolved.
- No checkpoint / calibration list / improved INT8 artifact in this checkout,
  so no improved candidate can be verified and the shipped INT8 cannot be
  reproduced or replaced here.
- The older TensorFlow-vs-LiteRT divergence finding cannot be reproduced here
  (no `tensorflow` installed); it remains a documented risk from Dev's own
  `QUANTIZATION.md` and any new candidate must pass a LiteRT-only gate plus a
  second runtime if available.