# V7 INT8 Dev Handoff (Agent A06)

Owner of this evidence: A06. Model/export owner: Dev. No model, frontend,
feature, or default-path code was changed in this bundle.

## Verdict

**INT8 candidate: NOT SELECTED. Whole-pipeline INT8: NOT_ESTABLISHED.**
Keep FP16 default. INT8 stays opt-in experimental.

## Why

1. INT8 classifier = INT8 weights/activations with **float32 external IO**
   (`[1,24,255]` in, `[1,100]` out, scale 0.0). Not full-INT8 IO.
2. Deployed detector is FP16 (133 + 102 FLOAT16 weights, 0 quantized tensors).
3. On 346 official val clips (345 successful extractions, 1 P5 no-hand
   frontend failure counted): FP16 272/346 (78.61%) vs INT8 254/346
   (73.41%) — FP16 lead 18 clips / 5.20 pp. Top-1 agreement 81.45%;
   both-correct 242, FP16-only 30, INT8-only 12, both-wrong 61.
4. Drift is large (logit L2 mean 5.355, max 10.209; prob L1 mean 0.584)
   and largest in FP16-only clips (L2 mean 6.211), but drift alone does not
   prove a quantization cause.
5. INT8 saves 323,168 B (38.80% of classifier) and is faster on this host
   (1-thread AUTO mean 0.0357 ms vs 0.0566 ms, classifier-only, fixed real
   tensor, n=300), but classifier time (0.058 ms mean) is negligible vs
   frontend (241 ms mean). Speed/size do not offset the accuracy regression
   for a default switch.

## Exact graph facts for re-quantization work

- FP16: 192 tensors (38 F16 + 18 I32 weights, 117 F32 acts), 135 ops,
  0 quantized tensors, 0 INT8-touching ops => `FP16`.
- INT8: 158 tensors (100 INT8, 54 INT32, 4 FLOAT32), 101 ops, 56 weights
  (21 INT8 + 35 INT32), 117 quantized tensors, 86/101 ops touch INT8/UINT8.
- Float operators (complete): entry `QUANTIZE[0]`, `DEQUANTIZE[62]`,
  `ZEROS_LIKE[63]`, `QUANTIZE[64]`, `DEQUANTIZE[100]`. Float tensors: 0
  (input), 119, 120 (side path), 157 (output). Any new candidate must report
  its float-operator list in the same form.
- Detector: `hand_detector.tflite` (413 tensors, 272 ops) and
  `hand_landmarks_detector.tflite` (269 tensors, 165 ops), both FP16.

## What Dev should do next (in order)

1. Resolve organizer INT8/detector-counting scope with the organizer before
   any default change; current FP16 detector blocks whole-pipeline INT8.
2. Investigate per-channel scales on extreme tensors / signer buckets
   (P12 INT8 33.3% vs FP16 54.2% is the worst signer gap) without test data.
   Do not assume calibration fixes it — no calibration was run here.
3. Add missing-hand / near-threshold frames (incl. the P5 no-hand clip
   `5989539775823847-INTUITIVE`) to the representative set per signer.
4. Treat per-channel PTQ, QAT, and mixed precision as separate hypotheses;
   record float ops for each candidate.
5. Gate any candidate on: same 346 val clips (345 + failure), identical
   feature hashes, per-signer/per-gloss/hand-band, Top-1/Top-5, agreement,
   logit/prob drift, and no accuracy regression vs FP16 (272/346 baseline).
6. Ship checkpoint SHA, representative-set identity, converter command,
   quantization settings, and operator/dtype map with each candidate.

## Reproduction

```bash
/Users/abhijeetmiskin/AppData/EchoGesture/.venv/bin/python -I evaluation/device/v7_int8/audit_tflite.py
/Users/abhijeetmiskin/AppData/EchoGesture/.venv/bin/python -I evaluation/device/v7_int8/run_v7_int8.py
/Users/abhijeetmiskin/AppData/EchoGesture/.venv/bin/python -I evaluation/device/v7_int8/benchmark.py --repeats 300 --warmup 50
/Users/abhijeetmiskin/AppData/EchoGesture/.venv/bin/python -I evaluation/device/v7_int8/test_v7_int8.py -v
```

Pinned hashes: FP16 `a012f2b5...7044a4`, INT8 `9d24640a...23700b` (full
hashes in README). Split: official val only; test never opened. No retraining,
no fabricated evidence, no installs, no nested agents.
