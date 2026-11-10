# EchoGesture: Source Code & Execution Guide

Track 2: Sign Language Prediction on Edge Device  
Samsung R&D Institute India, Bangalore (XR Perception Group & PRISM)

---

## 1. Prerequisites

- Python 3.11 (tested on CPython 3.11.x 64-bit and Python 3.12/3.14)
- CPU or mobile edge accelerator

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 2. Model Files

The default runtime is Dev's landmark candidate, selected with
`ECHOGESTURE_MODEL=landmark`:

- `models/landmark_classifier_fp16.tflite` (832,888 bytes, float16 weights and
  float32 IO; **not INT8**).
- `models/landmark_labels.json`, `models/landmark_model_meta.json` and
  `models/hand_landmarker.task` are required local assets.
- `src/features.py` and `src/landmark_frontend.py` implement Dev's shared
  two-hand transform and frontend. Do not change them independently of the
  corresponding model/config handoff.

`ECHOGESTURE_MODEL=s3d` retains the original pixel research baseline and its
INT8 models. There is no silent fallback from a missing selected model.
Classifier and detector files must both be included in final size accounting.
The current default candidate does not establish INT8, accuracy or clip-p95
compliance; see `evaluation/device/CANDIDATE_INTEGRATION.md`.

---

## 3. Running Prediction

The primary evaluation entrypoint is `main.py`.

### Python API

```python
from main import predict

result = predict("path/to/test_clip.mp4")
print(result)
```

Output format (per Slide 11 specification):
```json
{
  "clip_id": "africa_01386",
  "prediction": "AFRICA",
  "top5": ["AFRICA", "BOOK", "COMPUTER", "DRINK", "CHAIR"],
  "confidence": 0.0266,
  "latency_ms": 23.4
}
```

### CLI Execution

```bash
python main.py path/to/test_clip.mp4
```

---

## 4. Manifest evaluation

From the repository root, use Python 3.11. Create a CSV with `clip_id,path`
columns and optional `ground_truth` and `signer_id` columns. Supply ground truth
for every clip or none. Relative clip paths are resolved from the manifest's
directory; exact Unicode `clip_id` values are preserved.

```bash
python evaluation/device/run_manifest.py /absolute/path/to/manifest.csv \
  --output-dir /absolute/path/to/local-benchmark-output
```

The runner imports this file's real `predict()` function, validates every
output against `models/label100.json` by default, and writes `predictions.jsonl`
and `benchmark.json`. For the landmark candidate pass
`--model models/landmark_classifier_fp16.tflite --labels models/landmark_labels.json
--feature-config src/features.py`. The summary includes artifact SHA-256 values,
runtime/device details, wall-clock p50/p90/p95/p99 using nearest-rank percentiles,
and separate cold (first-call) and warm timings. `--profile` records actual
pipeline stages and explicit failures; strict mode aborts on failed clips.
Accuracy is `null` without complete ground truth. A smoke run is marked
nonrepresentative even if its small sample happens to be correct.

Keep manifests containing private paths, videos and generated benchmark output
outside Git. Laptop measurements are not target-device evidence.

## 5. Robustness and release tools

`evaluation/device/robustness.py` evaluates paired clean/corrupted clips for
occlusion, low light, blur, crop and frame drops. Every clip must be labelled;
temporary corrupted clips are deleted. Small samples remain transform checks,
not representative robustness results.

```bash
python evaluation/device/robustness.py /absolute/path/to/manifest.csv \
  --output-dir /tmp/robustness-low-light --corruption low_light \
  --severity 0.5 --seed 42 --model models/landmark_classifier_fp16.tflite \
  --labels models/landmark_labels.json
```

Use `--validation --split validation` only for an actual documented validation
set with at least 20 labelled clips. Never tune on the locked test set.

`evaluation/device/model_contract.py` validates candidate handoff metadata and
actual TFLite tensors. `scripts/release/generate_results.py` generates measured
or explicit `NOT_MEASURED` rows tied to the model SHA.
`scripts/release/build_bundle.py` builds and verifies provisional ZIP bundles,
including a real extracted `predict()` check when `--clip` is supplied. An
official bundle requires real organizer-manifest predictions.

## 6. Key Architectural Highlights

- **Two-hand landmark features** with explicit presence masks, root/scale
  normalization, temporal features and inter-hand geometry.
- **FP16 TFLite classifier with float32 IO** in the current candidate; INT8
  remains a separate unselected artifact.
- **Full-path profiling** separates sampled decode, MediaPipe tracking,
  feature preparation, classifier invocation and output formatting.
- **Zero cloud inference** — all model computation runs locally. Temporary demo
  captures are deleted after processing.
