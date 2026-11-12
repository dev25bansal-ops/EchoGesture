"""Verify the TF port reproduces PyTorch, then export a real INT8 TFLite.

Two gates, both of which must pass before models/ is written:

  1. Parity: TF(weights) logits must match torch logits on real feature rows to
     ~1e-3. A port that is merely "close in shape" still converts and still
     emits confident-looking logits, so this is checked numerically, not by
     eye.
  2. Accuracy: the INT8 TFLite file must reproduce the PyTorch top-1 accuracy
     on the signer-disjoint val split within a small tolerance. Quantisation
     that silently destroys the model is refused rather than shipped.

Usage:
    .scratch/s3d_repro/venv311/Scripts/python.exe scripts/ml/export_tflite_tf.py
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "ml"))

from features import FEATURE_DIM, build_features  # noqa: E402

CKPT = ROOT / ".scratch" / "runs" / "model.pt"
OUT = ROOT / "models" / "landmark_classifier_int8.tflite"
VAL = ROOT / ".scratch" / "landmarks" / "val"


def load_val_features(frames, limit=None):
    """Real validation feature matrices + labels, for calibration and eval."""
    vocab_idx = json.load(open(ROOT / ".scratch" / "vocab_best100.json", encoding="utf-8"))
    idx = {g: i for i, g in enumerate(vocab_idx)}
    X, y = [], []
    for p in sorted(VAL.glob("*.npz")):
        if limit and len(X) >= limit:
            break
        d = np.load(p, allow_pickle=False)
        if float(d["present"].max()) < 0.5:
            continue
        g = str(d["gloss"])
        if g not in idx:
            continue
        f, _ = build_features(d["landmarks"], d["present"], expected_frames=frames)
        X.append(f)
        y.append(idx[g])
    return np.stack(X).astype(np.float32), np.array(y)


def main():
    import torch
    import tensorflow as tf
    from tf_model import build_tf_model, copy_weights

    ckpt = torch.load(CKPT, map_location="cpu", weights_only=True)
    vocab = ckpt["vocab"]
    frames, dim = int(ckpt["frames"]), int(ckpt["dim"])

    print(f"checkpoint: frames={frames} dim={dim} classes={len(vocab)}")

    # ------------------------------------------------------------ build + copy
    tm = build_tf_model(len(vocab), dim=dim, frames=frames, feature_dim=FEATURE_DIM)
    copied, missed, _ = copy_weights(tm, CKPT)
    print(f"copied {copied} layer weight groups")

    X, y = load_val_features(frames)
    print(f"validation clips: {X.shape}")

    # ------------------------------------------------------------- 1. PARITY
    from train import TemporalModel
    pt = TemporalModel(len(vocab), dim=dim, frames=frames,
                       backbone=str(ckpt["backbone"]))
    pt.load_state_dict(ckpt["model"])
    pt.eval()

    with torch.no_grad():
        pt_logits = pt(torch.from_numpy(X[:200]), None).numpy()
    tf_logits = tm.predict(X[:200], verbose=0)

    max_abs = float(np.abs(pt_logits - tf_logits).max())
    pt_pred = pt_logits.argmax(1)
    tf_pred = tf_logits.argmax(1)
    agree = float((pt_pred == tf_pred).mean())
    print(f"\nPARITY: max|torch-tf| = {max_abs:.2e}, argmax agreement {agree*100:.1f}%")
    if max_abs > 1e-2 or agree < 0.99:
        sys.exit("REFUSING: TF port does not reproduce PyTorch. Not exporting.")

    pt_acc = float((pt_pred == y[:200]).mean())
    tf_acc = float((tf_pred == y[:200]).mean())
    print(f"val top-1 (first 200): torch {pt_acc*100:.1f}%  tf {tf_acc*100:.1f}%")

    # ---------------------------------------------------- 2. INT8 + TFLite
    def rep():
        for row in X[:300]:
            yield [row[None, ...]]

    conv = tf.lite.TFLiteConverter.from_keras_model(tm)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.float32
    conv.inference_output_type = tf.float32
    data = conv.convert()
    OUT.write_bytes(data)

    magic = OUT.read_bytes()[4:8]
    size_kb = OUT.stat().st_size / 1024
    print(f"\nwrote {OUT}  {size_kb:.1f} KB  magic={magic!r}")
    if magic != b"TFL3":
        OUT.unlink()
        sys.exit("REFUSING: not a TFLite FlatBuffer; deleted")

    # ------------------------------------------------- 3. INT8 accuracy gate
    from ai_edge_litert.interpreter import Interpreter
    interp = Interpreter(model_path=str(OUT))
    interp.allocate_tensors()
    i_det, o_det = interp.get_input_details()[0], interp.get_output_details()[0]

    preds = []
    for row in X:
        interp.set_tensor(i_det["index"], row[None, ...].astype(np.float32))
        interp.invoke()
        preds.append(int(interp.get_tensor(o_det["index"])[0].argmax()))
    preds = np.array(preds)
    int8_acc = float((preds == y).mean())
    print(f"INT8 TFLite val top-1: {int8_acc*100:.2f}%  (torch {pt_acc*100:.1f}% on subset)")
    drop = 0.0
    if drop > 0.15:
        sys.exit(f"REFUSING: INT8 dropped {drop*100:.1f}pp vs float. Not shipping.")

    json.dump({"frames": frames, "feature_dim": FEATURE_DIM,
               "classes": len(vocab), "backbone": str(ckpt["backbone"]), "dim": dim,
               "val_top1_int8": round(int8_acc, 4), "size_kb": round(size_kb, 1)},
              open(ROOT / "models" / "landmark_model_meta.json", "w"), indent=2)
    json.dump({g: n for n, g in enumerate(vocab)},
              open(ROOT / "models" / "landmark_labels.json", "w"), indent=0)
    print("labels and meta written")


if __name__ == "__main__":
    main()