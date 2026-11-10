# Dev handoff — quantization final (A02)

Date: 2026-11-10. Owner: A02. Model/export owner: Dev. Branch
`hackathon/abhijeet-integration` at `f81dd30c3e6dc1fc035244c5ca9791e7a84da0c4`.
No model, export, feature, or default-path code was changed.

## Verdict

INT8 candidate **NOT SELECTED**. Whole-pipeline INT8 **NOT_ESTABLISHED**.
FP16 stays the default. The existing INT8 file is a genuine quantized graph
and an IO drop-in, but it regresses accuracy and its provenance is unverifiable
in this checkout.

## Field table (HANDOFF.md)

| Field | Value |
| --- | --- |
| Candidate | not accepted as the default: `models/landmark_classifier_int8.tflite` SHA `9d24640a23244a3cd00c9fa62257733819f86ed1da5a10194c9b1cb64323700b` (509,720 B) |
| Selected default | `models/landmark_classifier_fp16.tflite` SHA `a012f2b5…7044a4` (832,888 B), float16 weights / float32 IO |
| Artifacts | model, ordered `landmark_labels.json` (100, SHA `e78f5028…9a4a`), `landmark_model_meta.json`, `hand_landmarker.task` (FP16) |
| Input contract | `[1,24,255]` float32, 24 frames, `build_features` (FEATURE_DIM 255) |
| Output contract | `[1,100]` float32 logits, axis = ordered label index, softmax in caller |
| Evidence | FP16 272/346 (78.61%) vs INT8 254/346 (73.41%) on official val; INT8 regresses 5.20 pp |
| Data scope | ASL Citizen official val split (5 signers); test split not opened |

## What A02 verified

- Both classifiers' exact tensor/operator census and all 117 INT8 scale/
  zero-point vectors (per-channel weights, per-tensor activations, asymmetric
  zero points on activation tensors only).
- Detector members are FP16 (0 quantized tensors).
- The INT8 file loads through the **unmodified** loader as float32 IO.
- Existing paired 346-clip evidence hashes out and its numbers re-verified.
- Fresh real-clip inference reproduces stored predictions on disagreement
  clips and the no-hand clip.

## What Dev must change

1. State the organizer INT8 criterion (float32 IO acceptance; detector bytes)
   in `docs/release/FINAL_EVIDENCE.md`.
2. Provide the shipping INT8 provenance: script, checkpoint SHA, representative
   set. The meta file matches neither `export_tflite_tf.py` nor
   `export_int8.py`; the checkpoint/calibration list are absent here.
3. Re-export with a shuffled, per-signer/per-gloss calibration set that
   includes missing-hand and near-threshold frames (all without test data).
4. Fix `export_tflite_tf.py`: `drop = 0.0` is a dead gate, and `int8_acc`
   (full X) vs `pt_acc` (first 200) compare different populations.
5. Address P12 (INT8 33% vs FP16 54%) and the 2-FP16-only regressions
   (ADVERTISE, BASEBALLCAP, CAPTURE, CHEEK, ELEVATOR1, EMPTY2, FINE1).
6. If the criterion covers detector weights, provide an INT8 detector or
   organizer acceptance of the FP16 detector.

## Acceptance tests Dev can rerun

```bash
P=/Users/abhijeetmiskin/AppData/EchoGesture
$P/.venv/bin/python -I -m unittest discover -s evaluation/device/quantization_final -p 'test_*.py' -v
$P/.venv/bin/python -I evaluation/device/quantization_final/acceptance_gate.py   # exit 1 expected
$P/.venv/bin/python -I evaluation/device/quantization_final/audit_quantization.py \
  --output evaluation/device/quantization_final/results/audit.json
```

A replacement candidate must: be INT8 by tensor census, keep float32 IO and the
frozen feature/label contract, be non-regressive against FP16 272/346 on the
same 345 successful val clips (plus the no-hand failure), keep the detector
compliant or explicitly acceptable, and ship a checkpoint SHA, representative
set, converter command, and operator/dtype map.

## Remaining blockers (unchanged)

- Organizer INT8/detector scope unresolved.
- No improved candidate or reproducibility inputs in this checkout.
- TensorFlow-runtime parity is unverified here (no `tensorflow` installed).