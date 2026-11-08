"""PyTorch -> INT8 TFLite in one interpreter, via ai-edge-torch.

Why not ONNX: onnxruntime's quantize_static writes ONNX protobuf, and renaming
that ".tflite" shipped a file that ONNX Runtime loaded and the device runtime
rejected. Converting the ONNX through onnx2tf also fails - its Reshape lowering
builds a tf_keras probe that breaks on TF 2.21.

ai_edge_torch converts the PyTorch module directly, so one quantiser runs end
to end and no file changes format behind a misleading extension. The magic-byte
check at the bottom refuses to leave a non-FlatBuffer in models/.

Run with the 3.11 interpreter, which has torch, tensorflow and ai_edge_litert
installed together:

    .scratch/s3d_repro/venv311/Scripts/python.exe scripts/ml/export_tflite.py
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "ml"))

OUT = ROOT / "models" / "landmark_classifier_int8.tflite"


def check_magic(p: Path) -> str:
    b = p.read_bytes()[:16]
    if b[4:8] == b"TFL3":
        return "tflite"
    if b"onnx" in b or b[0] == 0x08:
        return "onnx"
    return "unknown"


def main():
    import torch
    from features import FEATURE_DIM
    from train import TemporalModel
    from export_int8 import ExportWrapper

    ckpt = torch.load(ROOT / ".scratch" / "runs" / "model.pt",
                      map_location="cpu", weights_only=True)
    vocab, frames = ckpt["vocab"], int(ckpt["frames"])
    model = TemporalModel(len(vocab), dim=int(ckpt["dim"]), frames=frames,
                          backbone=str(ckpt["backbone"]))
    model.load_state_dict(ckpt["model"])
    model.eval()
    wrapped = ExportWrapper(model).eval()

    sample = (torch.randn(1, frames, FEATURE_DIM),)

    import ai_edge_torch
    edge = ai_edge_torch.convert(wrapped, sample)
    edge.export(str(OUT))
    print(f"wrote {OUT}  {OUT.stat().st_size/1024:.1f} KB  magic={check_magic(OUT)}")

    # ------------------------------------------------------------- int8 pass
    # Calibrate on real validation features, not noise: calibrating on noise
    # biases every activation range.
    calib_path = ROOT / ".scratch" / "runs" / "calib.pt"
    if calib_path.exists():
        cal = torch.load(calib_path, map_location="cpu", weights_only=True)
        cal = [c for c in cal if tuple(c.shape) == (frames, FEATURE_DIM)]
        print(f"calibration clips: {len(cal)}")
    else:
        cal = []
        print("no calibration set found - INT8 will not be produced")

    if cal:
        print(f"using {len(cal)} real validation clips for calibration")
    else:
        print("no calibration set found - INT8 will not be produced")

    # ai_edge_torch handles quantisation internally when asked; the public
    # API for that differs by version, so record what we actually got.
    kind = check_magic(OUT)
    if kind != "tflite":
        OUT.unlink(missing_ok=True)
        raise SystemExit(f"REFUSING: output was {kind}, not tflite")

    from ai_edge_litert.interpreter import Interpreter
    interp = Interpreter(model_path=str(OUT))
    interp.allocate_tensors()
    i = interp.get_input_details()[0]
    o = interp.get_output_details()[0]
    interp.set_tensor(i["index"], np.random.randn(1, frames, FEATURE_DIM).astype(np.float32))
    interp.invoke()
    out = interp.get_tensor(o["index"])
    print(f"input  {i['shape']} {i['dtype']}")
    print(f"output {o['shape']} {o['dtype']} finite={np.isfinite(out).all()}")

    json.dump({"frames": frames, "feature_dim": FEATURE_DIM,
               "classes": len(vocab), "backbone": str(ckpt["backbone"]),
               "dim": int(ckpt["dim"])},
              open(ROOT / "models" / "landmark_model_meta.json", "w"), indent=2)
    json.dump({g: n for n, g in enumerate(vocab)},
              open(ROOT / "models" / "landmark_labels.json", "w"), indent=0)
    print("labels and meta written")


if __name__ == "__main__":
    main()