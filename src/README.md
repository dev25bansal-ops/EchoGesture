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

The pipeline loads a **nine-member FP16 ensemble** plus the MediaPipe hand detector.
`models/landmark_model_meta.json` lists the members; `main.py` loads exactly those and
averages their logits. Adding or dropping a member is a one-line change there, not a
code change.

| file | size | role |
| --- | --: | --- |
| `landmark_ens_tcn21_fp16.tflite` | 920 KB | member 1 (TCN, 255-wide x,y) |
| `landmark_ens_tcnm61_fp16.tflite` | 816 KB | member 2 (Conv1D, mirror-augmented, 255-wide) |
| `landmark_ens_tcnm62_fp16.tflite` | 816 KB | member 3 (Conv1D, mirror-augmented, 255-wide) |
| `landmark_ens_rob71_fp16.tflite` | 920 KB | member 4 (TCN, stronger augmentation, 255-wide) |
| `landmark_ens_mirror_handdrop_tcn_z_s0_fp16.tflite` | 967 KB | member 5 (TCN, 381-wide x,y,z) |
| `landmark_ens_mirror_handdrop_tcn_z_s1_fp16.tflite` | 967 KB | member 6 (381-wide) |
| `landmark_ens_mirror_handdrop_tcn_z_s2_fp16.tflite` | 967 KB | member 7 (381-wide) |
| `landmark_ens_mirror_handdrop_tcn_z_s7_fp16.tflite` | 967 KB | member 8 (381-wide) |
| `landmark_ens_mirror_handdrop_tcn_z_s13_fp16.tflite` | 967 KB | member 9 (381-wide) |
| `landmark_classifier_int8.tflite` | 498 KB | INT8 reference, measured lower accuracy — see `results/QUANTIZATION.md` |
| `landmark_labels.json` | 1.6 KB | gloss → class index (ordered, 100 entries) |
| `landmark_model_meta.json` | 1.6 KB | frame count, feature width, **ensemble member list** |
| `hand_landmarker.task` | 7.8 MB | MediaPipe hand detector (Apache-2.0, Google) |

**Total shipped model weight: 8.51 MB** ensemble + 7.8 MB detector ≈ **16.3 MB**,
inside the 20 MB budget.

Five of the nine members take the **381-wide x,y,z** feature vector; the four legacy
members take **255-wide x,y**. Which is which is recorded per member in
`members_detail` in the meta sidecar — the width is not inferable from the filename.

**Why an ensemble.** Nine models averaged beat the best single model by 3.92 points
(85.45% vs 81.53% on held-out signers). The members make different errors; the mean
cancels part of that. Logits are averaged, not probabilities, so one over-confident
member cannot dominate.

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

All figures below are **desktop x86 Python**, not on-device.

- **Classifier inference: ~3.1 ms p50 / 3.7 ms p95** for all nine members on the
  native-cache scoring loop. This is the number the jury matrix scores as inference
  time and it is well inside budget.
- **Per-frame MediaPipe: p50 133 ms, p95 199 ms, floor 69 ms** (`scripts/ml/latency_per_frame.py`,
  n=192 frames, IMAGE mode). The floor alone is above the 50 ms budget, so the
  per-frame criterion **fails on this desktop**. VIDEO mode with tracker reuse measured
  142 ms p50 and does not rescue it here. (The repo previously quoted 15.7 ms/frame;
  that figure had no artifact behind it and was retracted 16 Nov.)
- **Whole-clip batch path: ~4,650 ms warm** — the detector dominates (~4.3–4.4 s over
  32 frames); decode is 126–312 ms, features 6–8 ms, classifier ~7 ms.
- **Cold start: ~159 s**, one time, dominated by MediaPipe building its graph lazily on
  the first `detect()`. `main.py` now pays this inside `_init_model()`, so a timed
  `predict()` reports real inference only.

The Android app uses `detectForVideo` at native 640×480 with tracker reuse — a
different detector path from any measured here. **On-device latency, power and thermal
have never been measured** and are reported as `NOT_MEASURED`, not estimated.

---

## 4. How it works

```
video -> 32 frames sampled evenly across the clip
      -> MediaPipe HandLandmarker, 2 hands, 21 landmarks each
      -> canonical Left/Right slots -> src/features.py (255 or 381 floats/frame)
      -> 9-member FP16 TFLite ensemble, mean logits, one softmax -> top-5 glosses
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
1,251 clips, 100 classes, chance = 1.00%. Measured on the **FP16 `.tflite` files
that ship**, on the landmark corpus re-extracted at native 640×480 (the geometry
the Android app actually feeds MediaPipe):

| metric | value |
| --- | --- |
| top-1 | **85.45%** |
| top-5 | **96.64%** |
| per-signer mean | 86.02% (min 72.54%, max 99.01%) |

Val (5 unseen signers, 346 clips): **89.31%** top-1, 97.69% top-5.

Confidence is calibrated: above 0.8 the model is right 99.7% of the time (594 clips);
74% of all errors sit below 0.4 confidence. Answering only clips above 0.5 confidence
gives 95.93% accuracy at 72.6% coverage.

See `RESULTS.md` for the full history, the failed paths, and what would move the
number further.
