# EchoGesture V7 INT8 qualification (Agent A06)

Owned path only: `evaluation/device/v7_int8/` in `/tmp/echogesture-v7-agents/A06`.
No original-checkout, Dev model/export, training, or default-path code was edited.
No packages were installed and no nested agents were launched.

## Scope and split policy

This finished V7 run recovers the interrupted V6 partial work at
`evaluation/device/v6_int8/` and re-executes it end-to-end (fresh `tflite_audit.json`,
paired run, and fixed-input benchmark) inside V7. It is an actual paired audit on
the retained 346 official ASL Citizen **validation** clips (not the locked official
test). Each clip was decoded and landmark-tracked once; the resulting identical
`[1,24,255]` float32 tensor was supplied to both pinned classifiers. Full
100-class logits are retained for every successful clip. The detector is the
pinned FP16 MediaPipe task. No test clip was opened.

Candidate hashes (verified, not assumed):

- FP16 `a012f2b5f7a1110961079f19eb0e28c9abacb95e4d0787352e4157ebca7044a4` (832,888 B)
- INT8 `9d24640a23244a3cd00c9fa62257733819f86ed1da5a10194c9b1cb64323700b` (509,720 B)
- Detector `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1` (7,819,105 B)
- Labels `e78f5028e97e2eaf86aff65af403af65061eebef010f0fbd41012293e8019a4a`
- Manifest `3f06798af00e69314da87414bf0c39bb2ec49e578f560c111754fa00966f6c82`
- Frontend `c73ef64b72f21a86638c4a126ad642b64c4ee29851cd2f8fd8ea4ed516fa97b0`,
  features `baf28d05bbd36e2783ea7ef6f78e793b89ed832d0eaad5dffed23d35c775aa84`

## Compliance verdict (explicit)

**INT8 candidate: NOT SELECTED as default. Whole-pipeline INT8 compliance:
NOT_ESTABLISHED.**

- INT8 classifier strict category: `INT8 weights/activations with float32 IO`.
  It is **not** full-INT8 IO: interpreter input `[1,24,255]` and output `[1,100]`
  are both float32 with scale 0.0.
- INT8 graph: 158 tensors (100 INT8, 54 INT32, 4 FLOAT32), 56 weight-bearing
  tensors (21 INT8 + 35 INT32; no FLOAT16/FLOAT32 weights), 117 tensors with
  non-empty quantization scale/zero-point, 101 operators, 84 with INT8 input,
  86 with INT8 input or output (85.1%).
- Exact float side path (all float-input operators): entry `QUANTIZE[0]`
  tensor 0 -> 57, plus `DEQUANTIZE[62]` (118 -> 119) ->
  `ZEROS_LIKE[63]` (119 -> 120) -> `QUANTIZE[64]` (120 -> 121). Float-output
  operators: `DEQUANTIZE[62]`, `ZEROS_LIKE[63]`, `DEQUANTIZE[100]` (156 -> 157).
  The 4 FLOAT32 tensors are the external input, the two side-path tensors
  (119, 120), and the final output (157). Main 84-operator path is INT8 after
  the entry QUANTIZE; no other float compute is hidden in the main graph.
- FP16 classifier strict category: `FP16` (192 tensors: 38 FLOAT16 weights +
  18 INT32 biases, 117 FLOAT32 activations, 0 quantized tensors, 135 operators,
  0 INT8-touching).
- Deployed detector (`hand_landmarker.task`) is FP16 in both members:
  `hand_detector.tflite` 133 FLOAT16 weights, `hand_landmarks_detector.tflite`
  102 FLOAT16 weights, 0 quantized tensors in either. This alone blocks any
  whole-pipeline INT8 claim.
- Organizer INT8 definition and detector-counting scope remain unresolved.
  Therefore the pinned INT8 classifier remains an opt-in experimental candidate
  only; this evidence does not support replacing the FP16 default, and no
  default was changed here.

## Measured paired accuracy (346 val clips, 345 successful extractions)

| Model | Correct / 346 | Top-1 | Top-5 | Wilson 95% (Top-1) |
|---|---:|---:|---:|---|
| FP16 | 272 | 78.61% | 89.31% (309) | [73.99%, 82.61%] |
| INT8 | 254 | 73.41% | 86.13% (298) | [68.52%, 77.79%] |

- FP16 lead: 18 clips / 5.20 percentage points. No interval or claim beyond
  these counts is made; the gap is not statistically adjudicated here.
- Top-1 agreement on 345 successful extractions: 81.45% (281/345); 64 top-1
  disagreements. Top-5 exact agreement 2.32%, top-5 set agreement 12.17%.
- Paired categories: both correct 242, FP16-only correct 30, INT8-only correct
  12, both wrong 61, frontend failed 1. Of the 64 disagreements: 30
  FP16-correct/INT8-wrong, 12 INT8-correct/FP16-wrong, 22 both-wrong with
  different predictions.
- Frontend failure (counted, not excluded): `5989539775823847-INTUITIVE`
  (P5), no hand in any of 32 sampled frames. 345 unique feature SHA-256 values,
  all-successful-features hash (manifest order)
  `30ee73060bb9c36c73b313a8c89b67177050ca60eff4437f7caf7b715eb9973e`.
- Cross-check vs retained V5 source (`predictions.json` SHA
  `8d980fbe...`, `summary.json` SHA `1ef28344...`): 346 clips checked,
  0 mismatches on status/prediction/top-5/frames-with-hands.
- V6-vs-V7 parity: all 346 clip IDs, feature hashes, and top-1 predictions
  match V6; file bytes differ only in re-measured timings/logit float
  serialization (predictions.json SHA V6 `e1e2c235...`, V7 `8cdef462...`).

Per-signer (successful extractions):

| Signer | n | FP16 | INT8 | FP16-INT8 |
|---|---:|---:|---:|---:|
| P12 | 24 | 13 (54.2%) | 8 (33.3%) | +5 (20.83 pp) |
| P21 | 104 | 79 (76.0%) | 74 (71.2%) | +5 (4.81 pp) |
| P26 | 101 | 82 (81.2%) | 80 (79.2%) | +2 (1.98 pp) |
| P39 | 103 | 92 (89.3%) | 87 (84.5%) | +5 (4.85 pp) |
| P5 | 13 (+1 frontend failure) | 6 (46.2%) | 5 (38.5%) | +1 (7.69 pp) |

Largest FP16-only losses by gloss (2 FP16-only, 0 INT8-only each):
ADVERTISE, BASEBALLCAP, CAPTURE, CHEEK, ELEVATOR1, EMPTY2, FINE1.
`results/summary.json` retains per-signer/per-gloss paired counts, confidence
and margin distributions, true-class probability/rank, hand-band summaries,
category diagnostics, and all 64 disagreements. `results/predictions.json`
retains complete ordered 100-logit vectors, top-10 classes, feature SHA-256,
presence vectors, and timings for every clip.

### Logit / probability drift (345 successful clips)

| Metric | Mean | Median | Min | Max |
|---|---:|---:|---:|---:|
| logit L2 diff | 5.355 | 5.266 | 1.918 | 10.209 |
| logit mean-abs diff | 0.419 | 0.417 | 0.148 | 0.826 |
| logit max-abs diff | 1.742 | 1.604 | 0.563 | 4.543 |
| prob L1 diff | 0.584 | 0.523 | 0.045 | 1.512 |
| prob L2 diff | 0.237 | 0.184 | 0.010 | 0.766 |
| logit correlation | 0.891 | 0.908 | 0.550 | 0.994 |

Drift is largest in the FP16-only-correct category (logit L2 mean 6.211,
prob L1 mean 0.673) vs both-correct (5.256 / 0.604), both-wrong (5.228 /
0.478), and INT8-only-correct (5.849 / 0.509). Worst individual clips by logit
L2: `38571484397842015-MOUTH` (10.209, both correct), `2664208219859696-MYSELF`
(10.148, both correct), `3006731850276776-CHEEK` (10.117, FP16-only correct),
`7227604144782054-PENNY` (10.076, both correct), `46279340594521323-TEACH`
(9.966, both correct). High drift alone does not flip predictions; flip
analysis requires the disagreement table, not a drift threshold.

### Hand coverage

Hand-band summaries are in `summary.json`. The largest band is 12–18 frames
(n=209: FP16 171, INT8 156, +15). No hand-coverage damage is attributed to
INT8 beyond the paired counts; INT8 never sees the detector output differently
(the identical feature tensor feeds both classifiers).

## Runtime, delegate, and footprint

`benchmark.py`: one fixed real feature tensor from clip
`844429155260962-BEE.mp4` (feature SHA
`2ef6a7d3f27f790891f86b1a59b801984eb28820a18c3859dd9b77edb5f48a01`),
300 measured repeats per configuration, 50 warmups, `set_tensor + invoke +
get_tensor` only (excludes decode, tracking, feature building, softmax).
Host: macOS 26.3 arm64, Python 3.11.18, ai-edge-litert 2.3.0, numpy 2.4.6 —
a validation-host measurement, **not** a target-device claim. AUTO logs
XNNPACK delegate creation; `BUILTIN_WITHOUT_DEFAULT_DELEGATES` is a
resolver-only comparison.

| Config (mean / p50 / p95 ms) | FP16 | INT8 |
|---|---:|---:|
| 1 thread AUTO | 0.0566 / 0.0561 / 0.0587 | 0.0357 / 0.0361 / 0.0382 |
| 1 thread no-default-delegates | 0.1585 / 0.1572 / 0.1634 | 0.1104 / 0.1097 / 0.1136 |
| 2 thread AUTO | 0.0856 / 0.0843 / 0.1002 | 0.0528 / 0.0472 / 0.0884 |
| 2 thread no-default-delegates | 0.1254 / 0.1245 / 0.1303 | 0.1012 / 0.1005 / 0.1050 |
| 4 thread AUTO | 0.1358 / 0.1352 / 0.1643 | 0.0619 / 0.0595 / 0.0809 |
| 4 thread no-default-delegates | 0.1002 / 0.0998 / 0.1035 | 0.1021 / 0.1019 / 0.1048 |

Paired-run classifier times over all 345 real tensors: FP16 mean 0.093 ms,
INT8 mean 0.058 ms; frontend mean 241 ms — classifier latency is not the
end-to-end bottleneck. Both models report float32 input/output in every
benchmark configuration, consistent with the graph audit.

Footprint:

| Configuration | Bytes |
|---|---:|
| FP16 classifier | 832,888 |
| INT8 classifier | 509,720 |
| Detector task | 7,819,105 |
| FP16 + detector | 8,651,993 |
| INT8 + detector | 8,328,825 |

INT8 classifier saves 323,168 bytes (38.80% of the classifier file) while
scoring 5.20 pp lower on this validation split.

## Evidence-supported Dev handoff

Do not promise that calibration will close the gap. There is no calibration
implementation in this bundle and no held-out re-quantization result.

1. Keep FP16 as the default until organizer INT8/detector scope is resolved
   and a replacement artifact passes the same paired gates.
2. If Dev re-exports, calibrate/inspect activation ranges by channel and
   signer bucket without using test data. The graph already uses per-channel
   scales; quantify whether those scales are appropriate by inspecting extreme
   tensors, not by assuming a cause.
3. Add missing-hand and near-threshold difficult frames to the representative
   set for each signer, including near-zero hand presence. Track the current
   P5 no-hand failure and each band's paired counts.
4. Treat per-channel quantization, QAT, and mixed precision as separate
   hypotheses. The present graph already has per-channel integer weights and
   INT32 biases with per-channel scales. Record which operations remain float
   in any candidate (entry QUANTIZE, the DEQUANTIZE/ZEROS_LIKE/QUANTIZE side
   path, and the final DEQUANTIZE must be reported explicitly).
5. Evaluate any candidate on the same 345 successful val clips plus frontend
   failures, per signer, per gloss, hand band, confidence margin, and identical
   feature hashes. Require INT8 accuracy not to regress from FP16 before
   calling it a no-regression drop-in.
6. Export the checkpoint SHA, representative-set identity, converter command,
   quantization setting, and operator/dtype map for every candidate.

## Reproduction (exact commands)

From `/tmp/echogesture-v7-agents/A06`, using the retained validation clips
and the pre-existing project venv (no installs):

```bash
/Users/abhijeetmiskin/AppData/EchoGesture/.venv/bin/python -I \
  evaluation/device/v7_int8/audit_tflite.py

/Users/abhijeetmiskin/AppData/EchoGesture/.venv/bin/python -I \
  evaluation/device/v7_int8/run_v7_int8.py

/Users/abhijeetmiskin/AppData/EchoGesture/.venv/bin/python -I \
  evaluation/device/v7_int8/benchmark.py --repeats 300 --warmup 50

/Users/abhijeetmiskin/AppData/EchoGesture/.venv/bin/python -I \
  evaluation/device/v7_int8/test_v7_int8.py -v
```

Expected: audit prints FP16=`FP16`, INT8=`INT8 weights/activations with
float32 IO`, detector=`FP16`, `whole_pipeline_int8=false`; paired run prints
FP16 272 / INT8 254 / FP16 lead 5.2023 pp; benchmark writes 12 configs × 300
repeats; 12 tests pass (note: the V6 test's 3-logit fake interpreter violated
the real `[1,100]` contract and errored in V6 — fixed in V7's test by using a
100-logit fake; V6 code was not modified).

## Files

- `evaluation/device/v7_int8/audit_tflite.py` — strict operator/tensor/scale audit
- `evaluation/device/v7_int8/tflite_audit.json` — full audit evidence (4.4 MB)
- `evaluation/device/v7_int8/run_v7_int8.py` — paired 346-clip FP16/INT8 runner
- `evaluation/device/v7_int8/results/summary.json`, `results/predictions.json`,
  `results/frontend_cache.json`
- `evaluation/device/v7_int8/benchmark.py`, `benchmark.json`
- `evaluation/device/v7_int8/test_v7_int8.py` (12 tests)
- `evaluation/device/v7_int8/README.md`, `HANDOFF.md`
- Logs: `audit.log`, `run.log`, `benchmark.log`

## Limitations

- INT8 is not a no-regression replacement on the measured validation split.
- Graph evidence does not establish organizer INT8 scope; the FP16 detector
  blocks a whole-pipeline INT8 claim.
- Operator placement, activation clipping, and calibration provenance are not
  causally proven from this audit.
- Timing is a local macOS validation-host measurement, not an organizer
  target-device benchmark.
- Logit float values are re-serialized on each run; exact byte equality with
  V6 `predictions.json` is not expected (clip/feature/prediction parity is).
