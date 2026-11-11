# EchoGesture — system architecture

On-device American Sign Language recognition for a 100-gloss vocabulary, with no
network dependency at inference time.

---

## 1. Pipeline

```
camera / MP4
   │
   ▼
MediaPipe HandLandmarker          models/hand_landmarker.task   7.82 MB
   │  21 landmarks × 2 hands per frame
   ▼
feature extractor                 src/main.py
   │  24 frames × 255 features
   ▼
┌─────────────────────────────────────────┐
│  4 × FP16 TFLite, run independently     │
│    ens_tcn21    → logits[100]           │
│    ens_tcnm61   → logits[100]           │
│    ens_tcnm62   → logits[100]           │
│    ens_rob71    → logits[100]           │
└─────────────────────────────────────────┘
   │  element-wise mean of the four logit vectors
   ▼
single softmax over the mean
   │
   ▼
top-5 glosses + confidence  →  predict() dict
```

The ensemble averages **logits**, not probabilities, and applies **one** softmax
afterwards. Averaging probabilities would change the result; the shipped
behaviour is mean-then-softmax and is what every reported metric measures.

---

## 2. The 255-feature vector

Each of 24 frames carries 255 float32 values. They are not 255 landmark
coordinates — they mix geometry with motion and validity.

| Group | Purpose |
| --- | --- |
| Hand landmark positions | 21 points × 2 hands, normalised to the wrist |
| Velocity per landmark | first difference across frames |
| Acceleration per landmark | second difference across frames |
| Inter-hand geometry | relative offset and distance between the two hands |
| Presence masks | whether each hand was detected in the frame |
| Canonical `[Left, Right]` slots | hands are ordered so a mirrored frame is not a new sign |

Canonical slot ordering matters: without it, the same sign performed with the
camera image-flipped would land in a different part of the feature space.

`STRICT_RESAMPLE=False` — frames are sampled to 24 with the non-strict
resampler, matching training. Changing this invalidates every reported metric.

---

## 3. Per-member tensor indices

The four members were exported by two different architectures, so their input
and output tensor indices are not interchangeable:

| Architecture | members | `in_idx` | `out_idx` |
| --- | --- | --: | --: |
| TCN | `ens_tcn21`, `ens_tcnm61`, `ens_tcnm62` | 0 | 224 |
| Conv1D | `ens_rob71` | 0 | 191 |

The runtime reads these from `models/landmark_model_meta.json` per member rather
than assuming a shared layout. A hardcoded index is the most likely way this
pipeline would silently return wrong logits, so the loader validates member
existence, filename shape, and tensor shapes before the first inference.

A missing member raises `FileNotFoundError` naming it. It never falls back to
running fewer members — a 3-member ensemble would produce plausible, wrong
confidence values.

---

## 4. Android deployment

```
app/android_final/
   MainActivity.kt          camera, tap-to-capture, caption HUD
   LandmarkClassifier.kt    MediaPipe + 255-feature transform
   ModelContract.kt         validates members at startup
   EnsembleAggregation.kt   mean-logit merge, single softmax
```

Four TFLite members and the detector are embedded in the APK as assets. The
Kotlin aggregation path is verified against a frozen logits fixture
(`parity_ensemble_logits.txt`) so the on-device merge cannot drift from the
Python reference.

Built with JDK 17, Android SDK 34, Gradle 8.10.2, AGP 8.5.2, Kotlin 2.0.20.
Debug APK: 75,072,782 B.

---

## 5. Where the time goes

Measured on a macOS arm64 laptop (not target hardware), full clip-to-JSON:

| Stage | Cost |
| --- | --- |
| Hand tracking (MediaPipe) | ~217 ms of ~240 ms — **90%** |
| Feature transform | small |
| 4 × ensemble inference | **0.36–0.41 ms** |

The ensemble is effectively free. The cost is per-frame hand detection, which
is why parallelism helps so much and why a faster model would not:

| Configuration | p50 | p95 |
| --- | --: | --: |
| Singleton | 231.8 ms | 274.2 ms |
| Parallel-2 | 132.0 ms | 154.8 ms |
| Parallel-4 | 79.3 ms | **99.8 ms** |

All three configurations return identical predictions on 12/12 clips.

The 50 ms target is **not met**. Meeting it requires reducing tracked frames or
per-frame detector cost, both of which change the feature semantics and would
require the model to be retrained. A GPU delegate was tested and rejected: it
was slower here and not bit-identical.

---

## 6. Privacy properties

- No network calls on the inference path. The app runs with the radio off.
- No frames, landmarks or captions are persisted or transmitted.
- Weights are local assets; nothing is fetched at runtime.
- The desktop evaluator reads a local file path and writes a local dict.

These are the properties the submission claims. They follow from the code, and
are what the offline test verifies.