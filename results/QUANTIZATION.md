# Quantization: what we measured and why we ship FP16

**Decision: ship `models/landmark_classifier_fp16.tflite` (813 KB, float16 weights,
float32 IO). Keep `models/landmark_classifier_int8.tflite` (498 KB) in the repo as a
measured alternative, not the default.**

The scored requirement is *"Compilation to TFLite/ONNX <20MB"*. Both variants are
real TFLite FlatBuffers under 1 MB, so the criterion is met either way. The choice
comes down to accuracy, and INT8 costs more than it buys here.

## Measured, on the signer-disjoint ASL Citizen val split (345 clips, 100 classes)

| variant | val top-1 | size | accuracy lost |
|---|---|---|---|
| PyTorch float32 (reference) | 76.52% | 1.6 MB ckpt | — |
| TFLite FP32 | 75.36% | 1,575 KB | 0.00 |
| **TFLite FP16 (shipped)** | **75.36%** | **813 KB** | **0.00** |
| TFLite INT8 | 68.99% | 498 KB | **−6.37** |
| PyTorch dynamic INT8 | 75.36% | n/a | 0.00 |

FP16 is bit-for-bit as accurate as the float32 TFLite file and half the size. INT8
drops 6.4 points.

## Why INT8 lost accuracy, and why we did not just fix it

`onnxruntime.quantize_static` and the TF converter both produced an INT8 graph in
which several ops had no integer kernel. The op census showed:

```
101 ops total:  19 RESHAPE, 13 EXPAND_DIMS, 50 int32 tensors,
                2 QUANTIZE, 2 DEQUANTIZE,
                plus MEAN (SE pooling) and GELU running in float
```

Every `QUANTIZE`/`DEQUANTIZE` pair is an activation leaving int8, running in
float32, and coming back — each round trip loses precision, and with three SE
blocks plus three GELU activations that error compounds. `fully_quantize: 0` from
the converter is the same fact stated by the tool.

Two attempts to remove the fallback ops did **not** help:
- replacing `tf.einsum` with a matmul + static reshape (reduced einsum but Keras
  still emitted the dynamic reshapes)
- replacing `tf.tile(tf.shape(x))` in the attention query with a static broadcast

## The runtime-divergence finding

The same INT8 file gave different answers in two runtimes:

| runtime | accuracy on 50 val clips |
|---|---|
| `ai_edge_litert.Interpreter` | 72.0% |
| `tf.lite.Interpreter` | 4.0% (float reference 74.0%) |

A correct model cannot produce 72% and 4% from the same FlatBuffer. Logits differed
by up to 8.4. The likely cause is the mixed float/int8 fallback path being
interpreted differently by the two interpreters. Whatever the mechanism, **a file
that behaves this way is not shippable** — the organizers re-measure on their own
device and could reproduce either number.

FP16 has no such ambiguity: every op is float, and both runtimes agree.

## What would change the decision

INT8 becomes worth revisiting if the model is re-exported with a quantization-aware
training pass, or if the architecture drops the SE blocks and GELU (both of which
lack clean integer kernels in TFLite). At 813 KB for FP16 against a 20 MB budget,
there is no size pressure forcing that work.

## Reproduction

```bash
# runtime venv (not committed - .scratch/ is gitignored)
py -V:3.11 -m venv .scratch/venv311
.scratch/venv311/Scripts/python.exe -m pip install numpy opencv-python-headless \
    mediapipe==1.2.2 ai-edge-litert

# FP16 (shipped): parity-checked against PyTorch, writes the artifact
.scratch/venv311/Scripts/python.exe scripts/ml/export_tflite_tf.py

# INT8 (measured, not shipped)
.scratch/venv311/Scripts/python.exe scripts/ml/export_int8.py \
    --ckpt .scratch/runs/model.pt
```

`export_tflite_tf.py` refuses to write `models/` unless the TF port reproduces
PyTorch logits to <1e-2 with >99% argmax agreement. Measured on the four shipped
members: **100% argmax agreement, max |d| 3.5–4.5e-03** (FP16 rounding).
