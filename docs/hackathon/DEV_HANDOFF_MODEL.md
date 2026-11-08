# Dev → Abhijeet · Model bundle handoff

Follows the field table in [HANDOFF.md](HANDOFF.md#dev--abhijeet--model-bundle).

**Status: ready.** Abhijeet can run the candidate from the documented artifacts and
verify feature and label parity. The accuracy number is below the ≥70% target and is
reported as measured.

---

## Candidate

| | |
| --- | --- |
| Branch | `hackathon/dev-model` |
| Commit | `603065a` + this handoff commit (see `git log`) |
| Shipped artifact | `models/landmark_classifier_fp16.tflite` |
| SHA-256 | `a012f2b5f7a1110961079f19eb0e28c9abacb95e4d0787352e4157ebca7044a4` |
| Bytes | 832,888 |
| Alternative (not shipped) | `models/landmark_classifier_int8.tflite`, SHA-256 `9d24640a23244a3cd00c9fa62257733819f86ed1da5a10194c9b1cb64323700b`, 509,720 bytes |

## Artifacts

| File | SHA-256 | Role |
| --- | --- | --- |
| `models/landmark_classifier_fp16.tflite` | `a012f2b5…7044a4` | classifier, float16 weights / float32 IO |
| `models/landmark_labels.json` | `1af78527…9d5aa27e` | gloss → class index, **ordered** |
| `models/landmark_model_meta.json` | `3ad4ecb2…c9cfe7f14` | frames=24, feature_dim=255, classes=100 |
| `models/hand_landmarker.task` | `fbc2a300…3607cde1` | MediaPipe hand detector, Apache-2.0 |

Feature config is versioned in `src/features.py` via the `FEATURE_DIM` constant and
the module docstring; there is no separate config file. A change to the feature
layout changes `FEATURE_DIM` and invalidates the artifact.

## Input contract

| | |
| --- | --- |
| Tensor shape | `[1, 24, 255]` float32 |
| dtype | float32 (the graph quantises internally; **not** int8 IO) |
| Frames | 24, produced by `build_features(..., expected_frames=24)` |
| Sampling | 32 frames sampled evenly across the whole clip by `read_frames` (not the first 32), resized to 256×256 RGB, then linearly resampled to 24 |
| Coordinates | MediaPipe normalised image coords, x/y in [0,1]; **z is dropped** |
| Normalisation | per-hand root-centre on the wrist, shared wrist-to-MiddleMCP scale across both hands, per-clip |
| Masks | `present` is a `(T,2)` flag array; absent hands are explicit zeros + flag, never inferred from geometry |
| Hand slots | canonical `[Left, Right]` from MediaPipe handedness, never swapped (ASL Citizen is not mirrored, live demo is not mirrored) |

## Output contract

| | |
| --- | --- |
| Shape | `[1, 100]` float32 |
| Values | **logits**, not probabilities — apply softmax before reporting confidence |
| Class axis | index into `landmark_labels.json` |
| Dequantisation | none needed (float32 IO) |

`src/main.py` already does softmax + top-5 + the required dict. Use it rather than
reimplementing.

## Reproduction

```bash
# 1. landmarks (resumable, parallel, ~35 min for 3390 clips)
scratch/s3d_repro/venv311/Scripts/python.exe scripts/ml/extract_parallel.py --workers 8

# 2. train (needs torch, not the mediapipe venv)
python scripts/ml/train.py --epochs 140 --frames 24 --dim 192

# 3. export TFLite (parity-gated; refuses to write models/ on mismatch)
scratch/s3d_repro/venv311/Scripts/python.exe scripts/ml/export_tflite_tf.py

# 4. evaluate on the signer-disjoint test split
scratch/s3d_repro/venv311/Scripts/python.exe scripts/ml/evaluate.py --split test
```

**Smoke fixture.** Any clip under `.scratch/aslcit_vocab_videos/`:

```python
from main import predict
predict("0009701491968876486-PULL CONVINCE.mp4")
# -> {'clip_id': '...PULL CONVINCE', 'prediction': 'PULLCONVINCE',
#     'top5': [...5 glosses...], 'confidence': 0.65, 'latency_ms': ~450}
```

Feature shape check: `build_features(np.zeros((32,2,21,3),np.float32),
np.ones((32,2),np.float32), expected_frames=24)` → shape `(24, 255)`.

## Evidence

| | |
| --- | --- |
| Data | ASL Citizen, 100-gloss subset, official participant-disjoint split |
| Clips | train 1,789 (32 signers) · val 346 (5) · test 1,255 (11) |
| Signer overlap | **zero** across all three splits (verified) |
| Test accuracy (signer-disjoint) | **62.61% top-1, 80.46% top-5**, 1,249 scored clips |
| Chance | 1.00% |
| Float vs FP16 | 75.36% vs 75.36% on val — quantization loses **0.00** |
| TF↔PyTorch parity | max abs logit diff **3.34e-06**, 100% argmax agreement |
| Classifier latency | 0.55 ms p50 / 0.76 ms p95 |
| Per-frame streaming | ~16 ms (MediaPipe 15.7 + classifier 0.55) |
| Cold start | ~7.7 s, one time, MediaPipe graph init |

### Known failure cases

- **6 of 1,255 test clips produce no hand detections in any frame** and are skipped,
  not predicted. `predict()` raises `ValueError` rather than guessing a gloss.
- Accuracy is uneven across signers: **45.8%** (P42, P9) to **80.4%** (P22). Small
  training data per class (17–24 clips) is the binding constraint.
- One-handed signs dominate the vocabulary (14.3% of frames have two hands); the
  inter-hand features are therefore sparse but not dead.

## Data scope

ASL Citizen clips and the extracted landmark cache stay **outside Git**
(`.scratch/` is gitignored). Only the vocabulary, feature code, training/export
scripts and the model artifacts are committed. ASL Citizen is research-use; the
release ships derived model weights, not the videos.

## Open questions for Abhijeet

1. Does the organizer manifest use clip IDs matching the **video stem**? We key
   `clip_id` on `Path(video_path).stem`.
2. Is **62.61%** the number to report, or does the device evaluation re-measure on a
   different clip set? If the latter, the model may need the robustness work in
   `RESULTS.md` before the deadline.
3. Should `predict()` **raise** or return a low-confidence sentinel when no hand is
   detected? Currently raises.
