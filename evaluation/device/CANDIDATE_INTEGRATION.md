# Candidate integration — 8 November 2026

Source: `origin/hackathon/dev-model` at `542a5f95c1bf3bfcea3d2a6bcf2b31d29a8e8cbf`.
Dev's feature implementation was imported unchanged. Integration is local and uncommitted.

## Selected runtime

Default `ECHOGESTURE_MODEL=landmark` selects:

- `models/landmark_classifier_fp16.tflite`, SHA-256
  `a012f2b5f7a1110961079f19eb0e28c9abacb95e4d0787352e4157ebca7044a4`.
- `models/landmark_labels.json`, SHA-256
  `e78f5028e97e2eaf86aff65af403af65061eebef010f0fbd41012293e8019a4a`.
- `src/features.py`, SHA-256
  `baf28d05bbd36e2783ea7ef6f78e793b89ed832d0eaad5dffed23d35c775aa84`.
- `models/hand_landmarker.task`, SHA-256
  `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1`.

Input is `[1,24,255]` float32, output is `[1,100]` float32 logits. The loader
validates shapes, vocabulary indices and required assets. No silent model
fallback occurs. `ECHOGESTURE_MODEL=s3d` retains the original pixel baseline.

The classifier is FP16 with float32 IO, **not an INT8 release solution**. Dev's
INT8 alternative is not selected because its documented accuracy/runtime
behavior is worse. Do not claim the FP16 file satisfies an INT8 requirement.

## Real execution

Downloaded one documented ASL Citizen research clip by HTTP range requests,
outside Git. This is not an organizer test clip and not a representative corpus.

`0009701491968876486-PULL CONVINCE.mp4`, SHA-256
`7976e5a1c2115e1f4e970178db053b324a91427c2789f7233d8e957e6826ab82`.

The shared integrated `predict()` returned `PULLCONVINCE`, confidence 0.7507.
A profiled manifest run on macOS 26.3 arm64, Python 3.11.18 and LiteRT 2.3.0
measured **642.01 ms** complete clip-to-JSON: 27.34 ms sampled decode, 612.15 ms
landmark tracking, 0.24 ms features, 0.17 ms classifier. N=1; representative
performance and accuracy remain **NOT MEASURED**. The 50 ms clip-p95 target is
not met by this sample. A one-clip correct result is not the 70% held-out target.
A separate real run measured 485.22 ms, confirming order-of-magnitude latency
variability; neither is a representative p95.

A seeded low-light severity 0.5 pair (seed 42) returned correct clean and
corrupted predictions, N=1. This is a transform/inference smoke check only,
not evidence of <=5 percentage points robustness degradation.

## INT8 paired verification

A paired identical-feature rerun on the 30 labelled validation clips measures:

- FP16 top-1: **22/30 = 73.33%**
- INT8 top-1: **21/30 = 70.00%**
- Difference: **−3.33 percentage points**
- Top-1 agreement: **25/30 = 83.33%**
- Ground truth in top-5: FP16 **25/30**, INT8 **24/30**
- Logit mean absolute drift: **0.4301**; logit RMSE **0.5737**

INT8 has genuine INT8 weights/activations with float32 IO. It remains unselected
because accuracy, runtime and whole-pipeline compliance have not been established;
the FP16 detector also makes whole-pipeline INT8 compliance unestablished.
Coordinates and exact reproduction are in `evaluation/device/v4_int8/`.

## Validation batch

The integrated candidate was run on a 30-clip ASL Citizen **validation**
subset (not locked test data): 5 signers, 26 glosses, all 30 predictions
succeeded. Top-1 was **22/30 = 73.33%** and top-5 **25/30 = 83.33%**. The
95% Wilson interval for top-1 is 55.6%–85.8%; this small, unbalanced batch
cannot verify Dev's 62.61% claim or establish the 70% target. Reproduce with:

```bash
.venv/bin/python -I evaluation/device/evaluate_candidate.py \
  $TMP/echogesture-.../manifest.csv \
  --output-dir $TMP/echogesture-...
```

A separate profiled manifest run measured **235.21 ms p50** and **309.75 ms
p95** over those 30 clips. In the final regenerated evidence run the complete
p95 was **327.63 ms**; these are sequential laptop measurements, not a stable
target-device p95. Both remain far above the 50 ms target.

## Release and demo verification

- `scripts/release/build_bundle.py` built a provisional bundle with the selected
  FP16 model, detector, shared feature/frontend code, export script, Python 3.11
  Dockerfile and candidate-specific results CSV. `verify --clip` extracted it
  and executed the real `PULLCONVINCE` prediction successfully.
- The offline worker and HUD were exercised with real frames and the real
  prediction. A real-clip HUD render shows `PULLCONVINCE`, confidence 0.7507,
  top-5 and actual latency. This is not a live-camera test; macOS denied camera
  access and no Android SDK/device was available.
- All **161** integration, evaluator, robustness, model-contract, release,
  streaming, evidence and demo tests pass on Python 3.11.

## Measured robustness

All five paired corruptions were run on the same 30-clip validation batch,
using severity 0.5 and seed 42 with no failed pairs:

| Corruption | Accuracy drop (percentage points) | Target interpretation |
|---|---:|---|
| low_light | 3.33 | PASS in this subset |
| crop | 6.67 | FAIL |
| occlusion | 13.33 | FAIL |
| motion_blur | 13.33 | FAIL |
| frame_drop | 13.33 | FAIL |

Each clean/corrupted report is under `robustness_lab/runs/<kind>/robustness.json`.
These are validation-only laptop results, not organizer test evidence.

## Streaming and latency optimization

The original IMAGE tracker remains the production path. A01's cached VIDEO
experiment saved about 12.6 ms mean but changed 12/60 top-1 outputs and carries
state across clips; fresh VIDEO was ~500 ms slower, and explicit CPU mode was
~568 ms slower. None is adopted. A02 found current `grab()` decoding already
fastest and pixel-identical to decode-all; random seeking was ~15× slower.
A03 found VIDEO tracking ~9.8% faster per frame but materially changed the
landmark stream and 20% of top-1 predictions, so it also remains opt-in only.

A real streaming run is available in `streaming_lab/`. On the same 103-frame
clip it returned `PULLCONVINCE` (0.6142), frame-processing p50 4.98 ms/p95
7.47 ms, sign-end-to-caption 0.73 ms, and clip-to-JSON 826.61 ms. These are
file-based streaming timings, not a camera test and not clip p95.

## Remaining work

1. Dev: a compliant INT8 candidate if the organizer requires INT8, plus the
   exact feature/version/artifact manifest for the eventual release. Discuss
   reducing 32-frame MediaPipe cost on the intended device; classifier time is
   not the bottleneck. Do not substitute per-frame timing for clip p95.
2. Abhijeet: run a representative permitted validation batch, paired robustness
   across all five corruptions, then physical-device sustained/power tests.
3. Both: resolve official vocabulary, timing boundary, detector size accounting
   and whether the actual evaluator requires INT8.

## Handoff schema

`evaluation/device/model_contract.py` validates an `integration_manifest.json`
containing all handoff fields and the actual TFLite tensor metadata. The shipped
FP16 model passed real tensor/shape validation. This validates compatibility,
not recognition quality.
