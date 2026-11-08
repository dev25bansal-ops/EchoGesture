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
| `landmark_classifier_fp16.tflite` | 813 KB | the classifier (shipped) |
| `landmark_classifier_int8.tflite` | 498 KB | INT8 alternative, measured lower accuracy — see `results/QUANTIZATION.md` |
| `landmark_labels.json` | 1.6 KB | gloss -> class index |
| `landmark_model_meta.json` | 130 B | frame count and feature width |
| `hand_landmarker.task` | 7.8 MB | MediaPipe hand detector (Apache-2.0, Google) |

The 7.8 MB MediaPipe task file is a model asset downloaded at install time by
`src/landmark_frontend.py` if absent. Total shipped model weight, classifier plus
detector, is ~8.6 MB — inside the 20 MB budget.

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

- **Classifier inference: 0.55 ms p50, 0.81 ms p95** — this is the number the jury
  matrix scores ("Measured inference time <50 ms").
- **Per-frame streaming budget: ~16 ms** — MediaPipe 15.7 ms + classifier 0.55 ms,
  so a live camera sustains ~30 fps inside the 50 ms frame budget.
- **Whole-clip batch path: ~450 ms warm** — decode 52 ms + MediaPipe over 32 frames
  503 ms + features 0.4 ms + classifier 0.6 ms. This is a batch-of-one-clip cost,
  not per-frame latency.
- **Cold start: ~8 s**, one time, dominated by MediaPipe graph initialization.

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
| top-1 | **62.61%** |
| top-5 | **80.46%** |
| per-signer mean | 63.4% (min 45.8%, max 80.4%) |

This is below the ≥70% target. Measured honestly; see `RESULTS.md` for the full
breakdown, the failed paths, and what would move the number.
