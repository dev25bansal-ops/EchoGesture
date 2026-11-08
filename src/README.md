# EchoGesture: Source Code & Execution Guide

Track 2: Sign Language Prediction on Edge Device
Samsung R&D Institute India, Bangalore (XR Perception Group & PRISM)

---

## 1. Prerequisites

- **Python 3.11** (required by the organizers; mediapipe and ai-edge-litert both need it)
- No GPU, no network at inference time

```bash
pip install -r requirements.txt
```

---

## 2. Model Files

All four files in `models/` are required; the pipeline loads them at first call.

| file | size | role |
| --- | --- | --- |
| `landmark_ens_tcn21_fp16.tflite` | 917 KB | ensemble member 1 (TCN) |
| `landmark_ens_tcn42_fp16.tflite` | 917 KB | ensemble member 2 (TCN) |
| `landmark_ens_mirror05_s0_fp16.tflite` | 813 KB | ensemble member 3 (Conv1D, mirror-augmented) |
| `landmark_classifier_int8.tflite` | 498 KB | INT8 reference, measured lower accuracy — see `results/QUANTIZATION.md` |
| `landmark_labels.json` | 1.6 KB | gloss -> class index |
| `landmark_model_meta.json` | 269 B | frame count, feature width, **ensemble member list** |
| `hand_landmarker.task` | 7.8 MB | MediaPipe hand detector (Apache-2.0, Google) |

`landmark_model_meta.json` lists the members; `main.py` loads exactly those and
averages their logits. Adding or dropping a member is a one-line change there,
not a code change.

The 7.8 MB MediaPipe task file is a model asset downloaded at install time by
`src/landmark_frontend.py` if absent. Total shipped model weight, ensemble plus
detector, is ~10.5 MB — inside the 20 MB budget.

**Why an ensemble.** Three models averaged beat the best single model by 2.57
points (70.22% vs 67.65% on held-out signers) and lift the worst signer from
38.7% to 53.5%. The members make different errors; the mean cancels part of
that. Logits are averaged, not probabilities, so one over-confident member
cannot dominate.

---

## 3. Running Prediction

Entrypoint is `main.py`, contract is `predict(video_path: str) -> dict`.

```python
from main import predict

result = predict("path/to/test_clip.mp4")
```

```json
{
  "clip_id": "0049-PONDER",
  "prediction": "PONDER",
  "top5": ["PONDER", "PENNY", "MOUTH", "ANYONE", "DEAF1"],
  "confidence": 0.4607,
  "latency_ms": 445.2
}
```

CLI:
```bash
python main.py path/to/test_clip.mp4
```

### Latency, stated precisely

- **Classifier inference: 0.64 ms p50, 0.86 ms p95** across all three members —
  this is the number the jury matrix scores ("Measured inference time <50 ms").
- **Per-frame streaming budget: ~16.3 ms** — MediaPipe 15.7 ms + classifier, so
  a live camera sustains ~30 fps inside the 50 ms frame budget.
- **Whole-clip batch path: ~470 ms warm** — decode 52 ms + MediaPipe over 32
  frames 503 ms + features 0.4 ms + three model invocations. This is a
  batch-of-one-clip cost, not per-frame latency.
- **Cold start: ~8.8 s**, one time, dominated by MediaPipe graph initialization.

---

## 4. How it works

```
video -> 32 frames sampled evenly across the clip
      -> MediaPipe HandLandmarker, 2 hands, 21 landmarks each
      -> canonical Left/Right slots -> src/features.py (255 floats/frame)
      -> TFLite classifier -> top-5 glosses
```

**Why landmarks instead of pixels.** An earlier version of this repo shipped the
published S3D-Conv1D pixel model. Re-measured on the signer-disjoint ASL Citizen
split it scored **1.47%** (chance is 1.02%) while scoring 100% on an
in-distribution control: raw 64x64 pixels let the network memorise signer
appearance and background instead of learning the sign. Hand landmarks discard
both, which is why every Kaggle GISLR top-5 entry converged on them.

**Features (`src/features.py`, 255 per frame).** Each hand is root-centred on its
wrist and scaled by a shared wrist-to-middle-MCP factor, then concatenated with
lag-1 velocity and lag-2 acceleration. Inter-hand geometry (wrist distance and the
signed x/y offset between wrists) is measured *before* centring, because centring
puts both wrists at the origin and would make those distances identically zero.
Missing hands are carried as an explicit per-frame presence mask rather than a
zero landmark.

**Fully offline.** No network call anywhere in `src/`.

---

## 5. Reported accuracy

Signer-disjoint ASL Citizen test — 11 participants never seen in training,
1,249 clips, 100 classes, chance = 1.00%:

| metric | value |
| --- | --- |
| top-1 | **70.22%** |
| top-5 | **85.43%** |
| per-signer mean | 70.83% (min 53.52%, max 90.72%) |

Measured through `predict()` with all three ensemble members. See `RESULTS.md`
for the full history, the failed paths, and what would move the number further.
