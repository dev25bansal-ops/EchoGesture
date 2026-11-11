# Model artifacts and local setup

The trained classifier weights are **not committed to this repository**. They are
derived from ASL Citizen (Microsoft Research) and WLASL (C-UDA) data and have **no
confirmed redistribution permission** — see [`../NOTICE`](../NOTICE). Treat them as
local/private until written authorization is recorded there.

The repository publishes the **model interface** so the runtime is fully specified
even though the bytes are supplied separately.

## Required files for `predict(video_path)`

| File | Committed? | Role |
| --- | --- | --- |
| `landmark_classifier_fp16.tflite` | **no** (restricted) | default classifier, FP16 weights / FP32 IO |
| `landmark_classifier_int8.tflite` | **no** (restricted) | experimental INT8 alternative, not selected |
| `hand_landmarker.task` | **no** (binary, see note) | MediaPipe hand detector, Apache-2.0 |
| `landmark_labels.json` | **yes** | ordered gloss → class index (100 entries) |
| `landmark_model_meta.json` | **yes** | frames / feature dim / classes / quantization |

> `hand_landmarker.task` is Apache-2.0 MediaPipe and is safe to fetch from its
> published URL rather than vendoring the 7.5 MB binary.

## Interface contract

| | |
| --- | --- |
| Input tensor | `[1, 24, 255]` float32 |
| Frames | 24, produced by `build_features(..., expected_frames=24)` |
| Coordinates | MediaPipe normalized x/y in [0,1]; z dropped; per-clip root-centre + wrist→MiddleMCP scale; canonical `[Left, Right]` slots |
| Presence | explicit `(T, 2)` mask; absent hands are zeros + flag |
| Output tensor | `[1, 100]` float32 **logits** (softmax applied by `src/main.py`) |
| Class axis | index into `landmark_labels.json` |
| Quantization | float16 weights, float32 IO (no dequantization needed) |

## Artifact fingerprints (interface files, committed)

```text
landmark_labels.json      e78f5028e97e2eaf86aff65af403af65061eebef010f0fbd41012293e8019a4a
landmark_model_meta.json  7f189a16f4d175042b5481afb973217fc21410832ebd14c5bacf2f516473947e
```

Reference fingerprints for the uncommitted weights (for a locally supplied copy):

```text
landmark_classifier_fp16.tflite  a012f2b5f7a1110961079f19eb0e28c9abacb95e4d0787352e4157ebca7044a4  (832,888 B)
landmark_classifier_int8.tflite  9d24640a23244a3cd00c9fa62257733819f86ed1da5a10194c9b1cb64323700b  (509,720 B)
hand_landmarker.task             fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1  (7,819,105 B)
```

## Local setup

```bash
# 1. detector (Apache-2.0, public URL)
curl -L -o models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

# 2. classifier weights — obtain from the model owner (Dev) under the data terms,
#    then place them in models/ and verify the SHA-256 above.

# 3. verify
shasum -a 256 models/landmark_classifier_fp16.tflite   # must match a012f2b5…
```

Without step 2 the runtime imports and the label/meta contract validates, but
`predict()` cannot run. This is intentional: source is public, weights are not.