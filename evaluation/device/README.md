# INT8 lab — EchoGesture classifier compliance investigation (Agent A06)

Owner: A06 (INT8 Research Engineer), 8 Nov 2026. Isolated copy at branch
`hackathon/abhijeet-integration`. Nothing here retrains or overwrites weights;
`models/`, `scripts/ml/export*` and `src/main.py` are read-only inputs.

## One-line verdict

**The selected model is FP16, not INT8. The Dev INT8 alternative is a real
INT8 graph (verified by tensor inspection), but it is NOT selected, has a
large FP16-vs-INT8 logit divergence on synthetic inputs, has no measured
accuracy in this copy, and is incompatible with the current loader's
hardcoded FP16-path selection. The detector is FP16 too.**
**INT8 compliance status: NOT ESTABLISHED. This lab does not certify compliance.**

Filenames prove nothing. Every classification below comes from real tensor
dtypes/quantization parameters and operator inspection.

## Artifacts examined (hashes/sizes measured 8 Nov 2026)

| Artifact | Bytes | SHA-256 |
| --- | --- | --- |
| `models/landmark_classifier_fp16.tflite` (selected) | 832,888 | `a012f2b5…7044a4` |
| `models/landmark_classifier_int8.tflite` (Dev alternative) | 509,720 | `9d24640a…3700b` (matches brief) |
| `models/hand_landmarker.task` (detector, zip of 2 TFLite) | 7,819,105 | `fbc2a300…cde1` |
| `models/s3d_conv1d_qat_int8.tflite` (old pixel baseline) | 901,200 | `107756f3…cc0a` |
| `models/ptq_100_int8.tflite` (old pixel baseline) | 904,256 | `067c743c…027c3` |

## What tensor inspection actually shows

All results from `evaluation/device/int8_lab/inspect_tflite.py` (FlatBuffer
`schema_py_generated` + LiteRT Interpreter IO), run 8 Nov 2026.

### FP16 (selected): FLOAT16 weights, 0 quantized tensors → NOT INT8
- Interpreter IO: input `[1,24,255]` **float32**, scale 0.0; output `[1,100]`
  **float32**, scale 0.0.
- FlatBuffer: 192 tensors (117 FLOAT32, **38 FLOAT16**, 37 INT32); **0 tensors
  with a quantization scale**; 56 weight-bearing tensors = 38 FLOAT16 +
  18 INT32 shape constants; 135 ops, **0 touching INT8**.
- `models/landmark_model_meta.json` itself says `"quantization": "float16"`.
- `src/main.py` (line 127) explicitly enforces float32 IO for the FP16 path.
**Correct classification: FP16 / non-INT8. Any INT8 claim for this file is false.**

### INT8 alternative: genuine INT8 graph, float32 IO via internal QUANTIZE
- Interpreter IO: input `[1,24,255]` **float32**, scale 0.0; output `[1,100]`
  **float32**, scale 0.0 — same external contract as FP16, so it is a drop-in
  at the IO level.
- FlatBuffer: 158 tensors (100 INT8, 54 INT32, 4 FLOAT32); **117 tensors carry
  per-tensor/per-channel scales** (e.g. head MatMul INT8 `[100,192]` with
  scale ~0.00098, out_proj/v_proj/q_proj INT8 `[192,192]` with scales
  ~0.0014); 101 ops, **84 touching INT8**; opcodes include QUANTIZE +
  DEQUANTIZE (one at each graph edge, which is how float32 IO coexists with
  an INT8 interior); weight buffers 402,777 B vs FP16's 788,978 B.
**Correct classification: INT8 weights/activations, float32 IO.** Whether the
organizer's INT8 criterion accepts "INT8 interior + float32 IO via
QUANTIZE/DEQUANTIZE" is **unconfirmed** — ask, do not assume. Note the export
script sets `inference_input_type = tf.float32`, consistent with this.

### Detector (`hand_landmarker.task`): FP16, not INT8
Zip members: `hand_detector.tflite` (2,339,878 B, 133 FLOAT16 weight tensors,
0 quantized) and `hand_landmarks_detector.tflite` (5,478,949 B, 102 FLOAT16
weights, 0 quantized). Both have a single DEQUANTIZE (FP16→FP32) and no INT8.
**The landmark frontend is FP16 end to end.** `ARCHITECTURE_R8.md` §2 already
proposes INT8-quantising the MediaPipe models (predicted 1.5–2.5× CPU) — not
done in this copy.

### Old S3D baselines (context only, not the candidate)
Both `s3d_conv1d_qat_int8` and `ptq_100_int8` are full-INT8 (INT8 input/output,
scale 0.00392 / 0.00391), 17 INT8 weight tensors. They are a **different model**
(pixel `[1,24,64,64,1]`, `label100.json` lowercase vocab) — they prove nothing
about the landmark candidate and must not be cited as its INT8 evidence.
Their signer-disjoint ASL Citizen top-1 is 1.47% (PTQ) / 1.00% (QAT) vs 1.02%
chance (`evaluation/asl_citizen_*_results.json`) — quantization itself is not
their failure mode; signer appearance is.

## Combined footprint (classifier + detector)

| Combination | Bytes | MiB | vs 20 MiB | vs 20 MB (20e6) |
| --- | --- | --- | --- | --- |
| FP16 classifier + detector (current selection) | 8,651,993 | 8.251 | under by 12,319,527 | under by 11,348,007 |
| INT8 classifier + detector | 8,328,825 | 7.943 | under by 12,642,695 | under by 11,671,175 |
| INT8 + detector uncompressed members | 8,328,547 | 7.943 | similar | similar |

Size is **not** the blocker either way. The open question is whether the
organizer counts **detector** bytes toward the cap and whether "INT8" means
"classifier INT8" or "all shipped weights INT8" (**unconfirmed** — see
`docs/hackathon/SUBMISSION.md` module size row and `CANDIDATE_INTEGRATION.md`
"Remaining work" #3). FP16→INT8 saves 323,168 B (38.8%) on the classifier.

## FP16 vs INT8 behavior (measured, read the caveats)

`evaluation/device/int8_lab/compare_fp16_int8.py`, LiteRT 2.3.0, macOS arm64:

- **Synthetic feature rows (N=700, no labels — divergence only, NOT accuracy):**
  argmax agreement **48–50%** (re-runs vary with input stream); logit max-abs
  mean 0.41–0.42, max ~3.3–3.5; flattened logit correlation ~0.91; prob L1
  mean ~0.44–0.45; FP16 mean top-1 conf ~0.16 vs INT8 ~0.13. Zeros input
  disagrees (BREAKFAST1 vs ANYONE, max-abs 2.60).
- **Synthetic plausible-motion landmarks (N=30, via real `features.py`):**
  agreement 26/30 (86.7%) — better than white-noise rows, as expected, but
  still not parity. Feature std 1.17, range [-8.89, 8.20] in this probe.
- **Classifier invoke (warm, 200 reps, one row):** FP16 ~0.066–0.068 ms vs
  INT8 ~0.044 ms — both sub-millisecond; classifier time is not the latency
  bottleneck (MediaPipe tracking was 612 ms on the one real clip in
  `CANDIDATE_INTEGRATION.md`).
- **Both run and produce finite logits; INT8 runs through the current loader
  path unchanged** (IO contract identical, QUANTIZE is internal) — verified by
  pointing the unmodified loader at the INT8 file in-process (see test).

**Failure/compatibility analysis (no invented diagnoses):**
1. **Selection blocker (certain):** `src/main.py:_model_paths()` hardcodes
   `landmark_classifier_fp16.tflite` for `ECHOGESTURE_MODEL=landmark`; the INT8
   file is never selected. `run_manifest.py` defaults are the S3D QAT model +
   `label100.json` (wrong vocab for the landmark candidate — must pass
   `--model …_int8.tflite --labels models/landmark_labels.json`).
2. **Divergence risk (measured, cause unproven):** 50% argmax disagreement on
   off-manifold synthetic rows + 13% on plausible-motion rows shows INT8
   quantization materially moves outputs. **No accuracy number exists in this
   copy for either landmark model** — `FINAL_EVIDENCE.md` marks held-out top-1
   NOT_MEASURED. Do not infer real-clip accuracy from synthetic agreement.
3. **Conversion-script smell (read-only observation, not a proven bug):** in
   `scripts/ml/export_tflite_tf.py`, `drop = 0.0` is hardcoded so the
   "INT8 accuracy gate" can never trip; `int8_acc` is computed on the full `X`
   but compared against a subset `pt_acc`. The script also writes
   `landmark_model_meta.json` **without** a `"quantization"` field, overwriting
   the current FP16 meta. Dev should confirm what produced the shipped INT8
   file before it is trusted as a candidate.
4. **Unverified prerequisite:** no checkpoint (`.scratch/runs/model.pt`),
   calibration data or vocab source exists in this copy, so the conversion
   **cannot be reproduced here**. Reproduction is Dev-side only.
5. **Not blockers:** IO shape/dtype/vocab (100 classes, same labels) and
   20 MB size (see table) are compatible. Classifier latency is negligible.

## Metric comparison on valid clips

**Not run — no valid labelled clips exist in this copy.** `evaluation/` holds
only old-S3D predictions/results and eval scripts; there is no landmark-model
labelled manifest, no downloaded clips, and no `.scratch/landmarks/`. The
existing harnesses make this runnable once Dev/Abhijeet supply a permitted
batch: `evaluation/device/fetch_validation_batch.py` + `run_manifest.py`
(see "Reproduction commands" below). The single real clip in
`CANDIDATE_INTEGRATION.md` (`PULLCONVINCE`, FP16 only, N=1) cannot support a
metric.

## Exact Dev change request

1. **Decide + state the INT8 criterion in writing** with the organizer:
   does "INT8" accept INT8 weights/activations with float32 IO (internal
   QUANTIZE/DEQUANTIZE), and do detector bytes count? Record the answer in
   `docs/release/FINAL_EVIDENCE.md` before any selection change.
2. **Confirm the shipped INT8 file's provenance:** which script/command actually
   produced `landmark_classifier_int8.tflite` (SHA `9d24640a…`), from which
   checkpoint, with what calibration/representative dataset, and why it differs
   from `export_tflite_tf.py` as it now reads (gates as noted above). Put the
   exact command + checkpoint SHA in the handoff.
3. If INT8 is required: **select it deliberately** — change the landmark
   model path in `src/main.py:_model_paths()` (or add an explicit
   `ECHOGESTURE_MODEL=landmark-int8` value) rather than relying on the
   `@replace` style hack; update `landmark_model_meta.json` to record
   `"quantization": "int8 (float32 IO)"`, the INT8 SHA/size, and the conversion
   command; keep FP16 in the tree for A/B during validation.
4. **Provide a labelled signer-disjoint validation manifest** (or the fetch
   inputs) so FP16-vs-INT8 top-1 can be measured on identical clips — the
   `compare_fp16_int8.py --npz` path and `run_manifest.py` both need it.
5. If the criterion covers **all shipped weights**: provide an INT8 (or
   otherwise compliant) detector, or state explicitly that the current
   hand_landmarker.task ships FP16 and seek organizer acceptance.
6. Fix or explicitly supersede the dead accuracy gate (`drop = 0.0`) and the
   subset/full comparison in `export_tflite_tf.py` before re-exporting.

## Reproduction commands (from repo root, Python 3.11 `.venv`)

```bash
.venv/bin/python -m pip install -r src/requirements.txt
.venv/bin/python -I evaluation/device/int8_lab/inspect_tflite.py --json /tmp/int8_report.json
.venv/bin/python -I evaluation/device/int8_lab/inspect_tflite.py --strict   # exit 1: FP16 files are non-INT8, by design
.venv/bin/python -I evaluation/device/int8_lab/compare_fp16_int8.py --json /tmp/cmp.json
.venv/bin/python -I evaluation/device/int8_lab/inspect_detector.py --json /tmp/det.json
.venv/bin/python -m unittest evaluation/device/test_int8_lab.py -v
# After Dev supplies clips + manifest (Abhijeet's existing harness):
.venv/bin/python evaluation/device/run_manifest.py /abs/manifest.csv --output-dir /tmp/fp16_out \
  --model models/landmark_classifier_fp16.tflite --labels models/landmark_labels.json \
  --feature-config src/features.py --profile
.venv/bin/python evaluation/device/run_manifest.py /abs/manifest.csv --output-dir /tmp/int8_out \
  --model models/landmark_classifier_int8.tflite --labels models/landmark_labels.json \
  --feature-config src/features.py --profile
```

## Blockers (honest list)

- No labelled clips/manifest/checkpoint in this copy → no accuracy metric, no
  conversion reproduction, no parity gate possible here.
- Organizer INT8 criterion (float-IO acceptance, detector counting) **unconfirmed**.

