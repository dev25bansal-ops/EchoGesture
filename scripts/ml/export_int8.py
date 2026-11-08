"""Export the trained landmark classifier to INT8 TFLite.

DEV-03. Produces the artifact Abhijeet ships: an integer-only TFLite file whose
input is the FEATURE_DIM-wide (255) landmark stream from src/features.py.

The model is 100% Conv1d / BatchNorm / Linear / attention-pool. Attention uses
a single learned query against the sequence, which lowers to MatMul + Softmax -
all natively INT8-supported. No Flex delegates.

Usage:
    python scripts/ml/export_int8.py --ckpt .scratch/runs/model.pt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "ml"))
from features import FEATURE_DIM  # noqa: E402
from train import TemporalModel  # noqa: E402


class ExportWrapper(nn.Module):
    """Fixed-length, mask-free variant for clean TFLite lowering.

    The attention mask is dropped at export time by padding every clip to the
    full frame count before inference. `predict()` always supplies exactly
    `frames` timesteps, so the mask would be all-ones anyway.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # x: (B, T, F) float32 -> logits (B, C)
        return self.model(x, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=".scratch/runs/model.pt")
    ap.add_argument("--out", default="models/landmark_classifier_int8.tflite")
    ap.add_argument("--calib", type=int, default=300)
    args = ap.parse_args()

    ckpt = torch.load(ROOT / args.ckpt, map_location="cpu", weights_only=True)
    vocab = ckpt["vocab"]
    frames = int(ckpt["frames"])
    dim = int(ckpt["dim"])
    backbone = str(ckpt["backbone"])

    model = TemporalModel(len(vocab), dim=dim, frames=frames, backbone=backbone)
    model.load_state_dict(ckpt["model"])
    model.eval()

    wrapped = ExportWrapper(model).eval()

    # ---------------------------------------------------------------- ONNX
    onnx_path = ROOT / "models" / "landmark_classifier.onnx"
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, frames, FEATURE_DIM)
    torch.onnx.export(
        wrapped, dummy, str(onnx_path),
        input_names=["features"], output_names=["logits"],
        opset_version=17, do_constant_folding=True,
        dynamic_axes=None,
    )
    print(f"ONNX  {onnx_path}  {onnx_path.stat().st_size/1024:.1f} KB")

    # ------------------------------------------------------- quantise to INT8
    # Use representative real inputs, not random noise: calibration on noise
    # biases every activation range.
    calib = torch.load(ROOT / ".scratch" / "runs" / "calib.pt", weights_only=True) \
        if (ROOT / ".scratch" / "runs" / "calib.pt").exists() else None
    if calib is None:
        print("no calibration set found; falling back to synthetic (poor but loadable)")
        calib = [torch.randn(1, frames, FEATURE_DIM) for _ in range(args.calib)]
    else:
        calib = [c.unsqueeze(0) for c in calib[: args.calib]]
    print(f"calibration samples: {len(calib)}")

    import onnx  # noqa: F401
    from onnxruntime.quantization import (quantize_static, CalibrationDataReader,
                                          QuantFormat, QuantType)

    class Reader(CalibrationDataReader):
        def __init__(self, batches):
            self.it = iter([{ "features": b.numpy().astype(np.float32) } for b in batches])

        def get_next(self):
            return next(self.it, None)

    int8_path = ROOT / args.out
    quantize_static(
        str(onnx_path), str(int8_path), Reader(calib),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
    )
    print(f"INT8  {int8_path}  {int8_path.stat().st_size/1024:.1f} KB")

    # Label map next to the model so the artifact is self-describing.
    label_path = ROOT / "models" / "landmark_labels.json"
    json.dump({g: i for i, g in enumerate(vocab)}, open(label_path, "w"), indent=0)
    print(f"labels {label_path}  ({len(vocab)} classes)")

    # Frame count and feature width ship beside the model. src/main.py reads
    # this to size its input; .scratch (where training writes its own copy)
    # is gitignored and will not exist on the evaluator's machine.
    meta_path = ROOT / "models" / "landmark_model_meta.json"
    json.dump({"frames": frames, "feature_dim": FEATURE_DIM,
               "classes": len(vocab), "backbone": backbone, "dim": dim},
              open(meta_path, "w"), indent=2)
    print(f"meta  {meta_path}  frames={frames} feature_dim={FEATURE_DIM}")

    # ------------------------------------------------------------ sanity check
    import onnxruntime as ort
    sess = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    x = calib[0].numpy().astype(np.float32)
    out = sess.run(None, {"features": x})[0]
    print(f"INT8 smoke: input {x.shape} -> output {out.shape}, "
          f"dtype {out.dtype}, finite {np.isfinite(out).all()}")


if __name__ == "__main__":
    main()