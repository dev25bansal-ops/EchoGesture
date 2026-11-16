# Dev → Abhijeet · Model bundle handoff

Follows the field table in [HANDOFF.md](HANDOFF.md#dev--abhijeet--model-bundle).

**Status: ready.** Abhijeet can run the candidate from the documented artifacts and
verify feature and label parity. **Test top-1 is 85.45%, which clears the ≥70% target.**

> **Corrected 16 Nov.** Earlier revisions of this document listed four member hashes
> from the pre-mask export (`08417e72…`, `112279bc…`, `01a261d8…`, `27058a85…`) and
> quoted 77.02%. Those binaries no longer exist in `models/` — the mask fix re-exported
> every member. The table below is the current nine-member set, hash-verified against
> `models/` on 16 Nov.

---

## Candidate

| | |
| --- | --- |
| Branch | `hackathon/dev-model` |
| Shipped artifact | **9-model FP16 TFLite ensemble** (see Artifacts) |
| Combined SHA-256 (members, meta order) | `995cc8d0b6f6ac25abb9f7ddca01c3ce71edf8bc2d34e34f4813a2b99dd3b4bb` |
| Total size | 8.51 MB across nine members |
| Alternative (not shipped) | `models/landmark_classifier_int8.tflite`, SHA-256 `9d24640a…4323700b`, 498 KB |

## Breaking change: masked models and per-member feature width (13 Nov)

**Any caller that loads `models/*.tflite` directly must be updated.** Two
things that used to be true no longer are.

### 1. Each model now takes TWO inputs

```
inputs[0] name contains "mask"      ->  (1, 24)   float32  hand-presence
inputs[1] name contains "features"  ->  (1, 24, F) float32
```

**Do not index inputs by position.** The converter reports the mask first
(verified: `['serving_default_mask:0', 'serving_default_features:0']`), so
positional code will feed the mask as features and the model will fail as a
shape mismatch several layers later. Match on the name substring.

`src/main.py` already does this; if you copied the loading code elsewhere,
copy the name-matching with it.

Why: the model was **always trained** with this mask applied to the attention
(`train.TemporalModel` takes `key_padding_mask`), but the export never had a
mask input, so every artifact before today was a different model than the one
trained. Supplying it is worth **+1.84 points on test, +2.90 on val** for a
single member.

### 2. F is not always 255

Five members take **381-wide** x,y,z features; four take the original
**255-wide** x,y. Which is which is recorded per member in the sidecar:

```json
"members_detail": [
  {"file": "landmark_ens_tcn21_fp16.tflite", "use_z": false},
  {"file": "landmark_ens_mirror_handdrop_tcn_z_s0_fp16.tflite", "use_z": true}
]
```

A member with no entry, or `use_z: false`, takes 255. `main.py` builds each
width once from a single landmark extraction and feeds each member its own.

### Verified numbers after the change

| | val | test |
| --- | --: | --: |
| before (4 members, no mask) | 83.77% | 72.78% |
| after (9 members, masked) | **89.31%** | **85.45%** |

The first "after" measurement was 87.25/77.02 on the 256-squash landmark cache. That
cache was later found wrong about presence on 29% of test clips; re-extracting at native
640×480 (the app's geometry) lifted the same nine models to 89.31/85.45. See
`CACHE_REBUILT_2026-11-15.md`.

All 9 members at 100% argmax parity with PyTorch (max|d| 3.8e-06 to 6.1e-05,
gate 1e-2). Re-verified by reloading every `.tflite` from `models/`.


## Artifacts

| File | SHA-256 | Bytes | Role |
| --- | --- | --: | --- |
| `models/landmark_ens_tcn21_fp16.tflite` | `af031d898c98c97abcf66ae7538166316faac458b7b938fe9e3390fce31e5d4c` | 942160 | member 1, TCN, 255-wide |
| `models/landmark_ens_tcnm61_fp16.tflite` | `b0a97aa45c6797b4b85e2f6a1b03945881a94dad5519cb71be47e474fdea54f0` | 835896 | member 2, Conv1D mirror-aug, 255-wide |
| `models/landmark_ens_tcnm62_fp16.tflite` | `1d247397d57e982196e68f3f230adeeabbfcdf673ba01ed541eadd54d874ea8d` | 835896 | member 3, Conv1D mirror-aug, 255-wide |
| `models/landmark_ens_rob71_fp16.tflite` | `105218a89fde1d107441422254805dfb38cc7df25a216ff986801bc15dccb89a` | 942160 | member 4, TCN, 255-wide |
| `models/landmark_ens_mirror_handdrop_tcn_z_s0_fp16.tflite` | `387749f81673719bd4a24813f33c6880c6117ed7be9d6bd50853250f8082901e` | 990544 | member 5, TCN, 381-wide z |
| `models/landmark_ens_mirror_handdrop_tcn_z_s1_fp16.tflite` | `02a2dc70fbd769f793c8f8915c8a30cc568d901bf2e35bffe1752aa762000a35` | 990544 | member 6, 381-wide z |
| `models/landmark_ens_mirror_handdrop_tcn_z_s2_fp16.tflite` | `3b97a2d416faec002bb84a6d2e47c131bb97941f074feed248a2c5a99c1c3b3d` | 990544 | member 7, 381-wide z |
| `models/landmark_ens_mirror_handdrop_tcn_z_s7_fp16.tflite` | `2f13d0eba1f34e68f3e3ac1b2c645d9f75d1953918d69be6d58ca00a1f2b36b2` | 990544 | member 8, 381-wide z |
| `models/landmark_ens_mirror_handdrop_tcn_z_s13_fp16.tflite` | `3d3a61e08d6a4d2a0a8064ae3b63692a55b55a2004ba37a428308a22b6336fba` | 990544 | member 9, 381-wide z |
| `models/landmark_labels.json` | `1af78527…9d5aa27e` | 1593 | gloss → class index, **ordered** |
| `models/landmark_model_meta.json` | `4b522e5d…a237fd1c` | 1653 | frames=24, feature_dim=255, **member list + per-member use_z** |
| `models/hand_landmarker.task` | `fbc2a300…3607cde1` | 7819105 | MediaPipe hand detector, Apache-2.0 |

Combined ensemble SHA-256 (all nine members concatenated in meta order):
`995cc8d0b6f6ac25abb9f7ddca01c3ce71edf8bc2d34e34f4813a2b99dd3b4bb`.

**The member list lives in `landmark_model_meta.json`.** `main.py` loads exactly
the files named there and averages their logits — adding or dropping a member
needs no code change. `predict()` falls back to `landmark_classifier_fp16.tflite`
if the meta file is absent, but that file is no longer shipped; keep the meta.

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

The runtime venv is **not** committed (`.scratch/` is gitignored). Recreate it —
any Python 3.11 works; this was verified on 3.11.9.

```bash
# 0. runtime environment (Python 3.11; verified 3.11.9)
py -V:3.11 -m venv .scratch/venv311
.scratch/venv311/Scripts/python.exe -m pip install numpy opencv-python-headless \
    mediapipe==1.2.2 ai-edge-litert

# 1. landmarks (resumable, parallel, ~35 min for 3390 clips)
.scratch/venv311/Scripts/python.exe scripts/ml/extract_parallel.py --workers 8

# 2. train (needs torch, separate from the runtime venv)
python scripts/ml/train.py --epochs 140 --frames 24 --dim 192

# 3. export TFLite (parity-gated; refuses to write models/ on mismatch)
.scratch/venv311/Scripts/python.exe scripts/ml/export_tflite_tf.py

# 4. evaluate on the signer-disjoint test split
.scratch/venv311/Scripts/python.exe scripts/ml/evaluate.py --split test
```

**Verified on a clean Python 3.11.9 venv** with only the four runtime packages
above: `predict()` returns the exact contract keys on three real clips
(PULLCONVINCE 0.631, SAIL2 0.566, PONDER 0.460), cold start 0.33 s, warm clips
520–580 ms. `torch` is needed only for training and the parity gate, not for
inference.

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
| Test accuracy (signer-disjoint) | **85.45% top-1**, 1,251 scored clips (val 89.31%, 346 clips) |
| Chance | 1.00% |
| Test top-5 | **96.64%** |
| Per-signer | mean 86.02%, **min 72.54%**, max 99.01% |
| Best single member | 81.53% — the ensemble adds +3.92 |
| Float vs FP16 | 75.36% vs 75.36% on val — quantization loses **0.00** (four-member era) |
| TF↔PyTorch parity, all 9 members | 100% argmax agreement; FP16 max&#124;d&#124; 3.8e-06–6.1e-05 (gate: 1e-2) |
| Classifier latency | **3.1 ms p50 / 3.7 ms p95** for the nine members (desktop x86, native-cache loop) |
| Per-frame MediaPipe | **p50 133 ms / p95 199 ms / floor 69 ms** desktop x86 — FAILS the 50 ms budget |
| Cold start | **~159 s**, one time, MediaPipe graph build on first `detect()` (85% of it) |

### Known failure cases

- **4 of 1,255 test clips produce no hand detections in any frame** and are skipped,
  not predicted. `predict()` raises `ValueError` rather than guessing a gloss.
- Accuracy is uneven across signers: **72.54%** (worst) to **99.01%** (best). Small
  training data per class (17–24 clips) is the binding constraint.
- **A hand leaving frame for 25% of a clip costs 5.76 points** (on the native cache) —
  the one robustness case outside the ≤5-point budget. This gap is **open**: a
  corrected feature interpolation was implemented and fully retrained, and it
  made both clean and hand-dropped accuracy worse (64.61% / 47.64% vs 72.78% /
  58.21%), so it was not shipped. `RESULTS.md`, `DROP_HAND_ANALYSIS.md` and
  `EXPERIMENTS.md` have the measurements.
- Pixel-level degradation is harsher than landmark-level: motion blur and
  downscaling cost 13–35 points because they corrupt MediaPipe's input before
  the classifier sees anything.
- One-handed signs dominate the vocabulary (14.3% of frames have two hands); the
  inter-hand features are therefore sparse but not dead.

## Data scope

ASL Citizen clips and the extracted landmark cache stay **outside Git**
(`.scratch/` is gitignored). Only the vocabulary, feature code, training/export
scripts and the model artifacts are committed. ASL Citizen is research-use; the
release ships derived model weights, not the videos.

**Full working tree mirrored to a private Hugging Face dataset** —
[`Dev2506/echogesture-full`](https://huggingface.co/datasets/Dev2506/echogesture-full),
16.9 GB across 181,083 files. That includes the 3,390 ASL Citizen clips, the
landmark caches, both reproduction venvs, training checkpoints and the built
APK, so the environment can be rebuilt without a network install. See
[HF_DATASET.md](HF_DATASET.md) for what is in it, download commands, and the
shipping subset.

**The dataset is private and must stay private.** It contains licensed research
video; publishing it would be redistribution. The models are distributed
through the release bundle instead.

## Open questions for Abhijeet

1. Does the organizer manifest use clip IDs matching the **video stem**? We key
   `clip_id` on `Path(video_path).stem`.
2. Is **85.45%** the number to report, or does the device evaluation re-measure on a
   different clip set? Note that 85.45% is the app's native-640×480 geometry; the
   Python reference `predict()` still reads 256-squash by default (~82.5%) unless
   called with `size="native"`. Both are real, different operating points.
3. Should `predict()` **raise** or return a low-confidence sentinel when no hand is
   detected? Currently raises.
