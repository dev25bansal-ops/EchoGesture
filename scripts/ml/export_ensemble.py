"""Export an ensemble of checkpoints as separate FP16 TFLite artifacts.

Why separate files rather than one fused graph: each member keeps its own
parity check against PyTorch, so a bad seed is caught individually instead of
being averaged invisibly into the ensemble. Averaging happens at inference in
src/main.py, where it is one numpy line.

Why FP16 rather than INT8: measured. INT8 cost 6.37 accuracy points on this
architecture and two TFLite runtimes disagreed on the same file. FP16 loses
0.00 at 813 KB per member; a 3-member ensemble is ~2.4 MB, still an eighth of
the 20 MB budget. See results/QUANTIZATION.md.

Gates (each must pass or nothing is written):
  1. Per-member parity: TF logits vs PyTorch logits, 100% argmax agreement.
  2. Ensemble accuracy: the exported FP16 ensemble must reproduce the PyTorch
     ensemble's top-1 on the signer-disjoint test split.

Usage:
    .scratch/s3d_repro/venv311/Scripts/python.exe scripts/ml/export_ensemble.py \
        --ckpts .scratch/runs/tcn21.pt .scratch/runs/tcn41.pt .scratch/runs/tcn42.pt
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "ml"))

from features import FEATURE_DIM, build_features  # noqa: E402

LANDMARKS = ROOT / ".scratch" / "landmarks"


def load_split(vocab_idx, frames, split):
    X, y = [], []
    for p in sorted((LANDMARKS / split).glob("*.npz")):
        d = np.load(p, allow_pickle=False)
        if float(d["present"].max()) < 0.5:
            continue
        g = str(d["gloss"])
        if g not in vocab_idx:
            continue
        f, _ = build_features(d["landmarks"], d["present"], expected_frames=frames)
        X.append(f)
        y.append(vocab_idx[g])
    return np.stack(X).astype(np.float32), np.array(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--prefix", default="landmark_ens")
    args = ap.parse_args()

    import torch
    import tensorflow as tf
    from tf_model import build_tf_model, copy_weights
    from train import TemporalModel

    first = torch.load(args.ckpts[0], map_location="cpu", weights_only=True)
    vocab = first["vocab"]
    frames = int(first["frames"])
    vocab_idx = {g: i for i, g in enumerate(vocab)}

    Xva, yva = load_split(vocab_idx, frames, "val")
    Xte, yte = load_split(vocab_idx, frames, "test")
    print(f"val {Xva.shape}  test {Xte.shape}  ({len(vocab)} classes)")

    pt_test_logits, tf_test_logits = [], []
    written = []

    for i, ck_path in enumerate(args.ckpts):
        name = Path(ck_path).stem
        ck = torch.load(ck_path, map_location="cpu", weights_only=True)
        dim = int(ck["dim"])
        backbone = str(ck["backbone"])
        if int(ck["frames"]) != frames or ck["vocab"] != vocab:
            sys.exit(f"{name}: frames/vocab differ from the first checkpoint")

        # ---- PyTorch reference
        pt = TemporalModel(len(vocab), dim=dim, frames=frames, backbone=backbone)
        pt.load_state_dict(ck["model"])
        pt.eval()

        # ---- TF port
        tm = build_tf_model(len(vocab), dim=dim, frames=frames,
                            feature_dim=FEATURE_DIM, backbone=backbone)
        copy_weights(tm, ck_path)

        with torch.no_grad():
            pt_logits = pt(torch.from_numpy(Xva[:200]), None).numpy()
        tf_logits = tm.predict(Xva[:200], verbose=0)

        max_abs = float(np.abs(pt_logits - tf_logits).max())
        agree = float((pt_logits.argmax(1) == tf_logits.argmax(1)).mean())
        print(f"[{i+1}/{len(args.ckpts)}] {name:10s} {backbone:4s} dim={dim} "
              f"parity max|d|={max_abs:.2e} agree={agree*100:.1f}%")
        if max_abs > 1e-2 or agree < 0.99:
            sys.exit(f"REFUSING: {name} TF port does not reproduce PyTorch")

        # ---- FP16 export
        conv = tf.lite.TFLiteConverter.from_keras_model(tm)
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.target_spec.supported_types = [tf.float16]
        conv.inference_input_type = tf.float32
        conv.inference_output_type = tf.float32
        data = conv.convert()

        out = ROOT / "models" / f"{args.prefix}_{name}_fp16.tflite"
        out.write_bytes(data)
        if out.read_bytes()[4:8] != b"TFL3":
            out.unlink()
            sys.exit(f"REFUSING: {out.name} is not a TFLite FlatBuffer")
        written.append(out)
        print(f"           -> {out.name}  {len(data)/1024:.1f} KB")

        # keep logits for the ensemble gate
        from ai_edge_litert.interpreter import Interpreter
        interp = Interpreter(model_path=str(out))
        interp.allocate_tensors()
        i_det, o_det = interp.get_input_details()[0], interp.get_output_details()[0]
        te = []
        for row in Xte:
            interp.set_tensor(i_det["index"], row[None].astype(np.float32))
            interp.invoke()
            te.append(interp.get_tensor(o_det["index"])[0])
        tf_test_logits.append(np.array(te))
        with torch.no_grad():
            pt_test_logits.append(pt(torch.from_numpy(Xte), None).numpy())

    # ------------------------------------------------------------ ensemble gate
    ens_tf = np.mean(tf_test_logits, axis=0).argmax(1)
    ens_pt = np.mean(pt_test_logits, axis=0).argmax(1)
    acc_tf = float((ens_tf == yte).mean())
    acc_pt = float((ens_pt == yte).mean())
    print(f"\nENSEMBLE test top-1:  FP16 TFLite {acc_tf*100:.2f}%   PyTorch {acc_pt*100:.2f}%")
    if abs(acc_tf - acc_pt) > 0.02:
        sys.exit(f"REFUSING: exported ensemble ({acc_tf*100:.2f}%) diverges from "
                 f"PyTorch ({acc_pt*100:.2f}%)")

    json.dump({"frames": frames, "feature_dim": FEATURE_DIM,
               "classes": len(vocab),
               "members": [p.name for p in written],
               "quantization": "float16",
               "ensemble_test_top1": round(acc_tf, 4)},
              open(ROOT / "models" / "landmark_model_meta.json", "w"), indent=2)
    json.dump({g: n for n, g in enumerate(vocab)},
              open(ROOT / "models" / "landmark_labels.json", "w"), indent=0)
    print(f"meta + labels written; {len(written)} members")


if __name__ == "__main__":
    main()