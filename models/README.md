# Model artifacts and local setup

The trained classifier weights are **not committed to this repository**. They are
derived from ASL Citizen (Microsoft Research) and WLASL (C-UDA) data and have **no
confirmed redistribution permission** — see [`../NOTICE`](../NOTICE). Treat them as
local/private until written authorization is recorded there.

The repository publishes the **model interface** so the runtime is fully specified
even though the bytes are supplied separately.

## Shipped model: 4-member FP16 ensemble

The default landmark runtime loads an ensemble of four independently trained FP16
classifiers and averages their logits before a single softmax. The member list lives
in `landmark_model_meta.json` under `members`; adding or removing a member needs no
code change.

| File | Committed? | Role |
| --- | --- | --- |
| `landmark_ens_tcn21_fp16.tflite` | **no** (restricted) | member 1, TCN |
| `landmark_ens_tcnm61_fp16.tflite` | **no** (restricted) | member 2, Conv1D, mirror-augmented |
| `landmark_ens_tcnm62_fp16.tflite` | **no** (restricted) | member 3, Conv1D, mirror-augmented |
| `landmark_ens_rob71_fp16.tflite` | **no** (restricted) | member 4, TCN, stronger augmentation |
| `landmark_classifier_fp16.tflite` | **no** (restricted) | previous single model; used only when the meta has no `members` key |
| `hand_landmarker.task` | **no** (binary, fetchable) | MediaPipe hand detector, Apache-2.0 |
| `landmark_labels.json` | **yes** | ordered gloss → class index (100 entries) |
| `landmark_model_meta.json` | **yes** | frames / feature dim / classes / `members` list |

Total deployed model bytes: **11.37 MB** (3,543,080 B ensemble + 7,819,105 B detector).

## Interface contract

| | |
| --- | --- |
| Input tensor | `[1, 24, 255]` float32, identical for every member |
| Frames | 24, produced by `build_features(..., expected_frames=24)` |
| Coordinates | MediaPipe normalized x/y in [0,1]; z dropped; per-clip root-centre + wrist→MiddleMCP scale; canonical `[Left, Right]` slots |
| Presence | explicit `(T, 2)` mask; absent hands are zeros + flag |
| Output tensor | `[1, 100]` float32 **logits** per member |
| Aggregation | `logits = mean(member_logits, axis=0)`, then **one** softmax |
| Class axis | index into `landmark_labels.json` |
| Quantization | float16 weights, float32 IO (no dequantization needed) |

**Per-member tensor indices differ.** The TCN members expose output index 224 and the
Conv1D members 191 (input index 0 for all). A loader must resolve each interpreter's
own indices; reusing one member's index on another produces wrong results.

## Artifact fingerprints

Committed interface files:

```text
landmark_labels.json      e78f5028e97e2eaf86aff65af403af65061eebef010f0fbd41012293e8019a4a
landmark_model_meta.json  06f655bbec9c75181e89218d7ba284523a2ee02f5ac30f133e48cde48072ddbc
```

Reference fingerprints for the uncommitted ensemble (verify after supplying locally):

```text
landmark_ens_tcn21_fp16.tflite   112279bc86090470b82b20b510746a946220322c25ec84cb3706a538c73dc22a  (939,152 B)
landmark_ens_tcnm61_fp16.tflite  01a261d8eb6c2c4d8b501f0ce1a36db338365d17fddabc63528a7941b1390f63  (832,888 B)
landmark_ens_tcnm62_fp16.tflite  27058a85e032bee3af444fb1538ff0bf32fd38fa2b01ce01a2f25c3a428d20b3  (832,888 B)
landmark_ens_rob71_fp16.tflite   08417e72da59150ac9adf749e6324c30fb001d4a401394e32f125d3276d1ab21  (939,152 B)
hand_landmarker.task             fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1  (7,819,105 B)
```

## Local setup

```bash
# 1. detector (Apache-2.0, public URL)
curl -L -o models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

# 2. ensemble weights — obtain from the model owner (Dev) under the data terms,
#    place all four in models/, then verify the SHA-256 values above.

# 3. smoke test
.venv/bin/python -I src/main.py "path/to/clip.mp4"
```

Without step 2 the runtime imports and the label/meta contract validates, but
`predict()` cannot run — it raises `FileNotFoundError` naming the missing member.
This is intentional: source is public, weights are not.