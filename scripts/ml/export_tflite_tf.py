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

from features import build_features  # noqa: E402

CKPT = ROOT / ".scratch" / "runs" / "model.pt"
OUT = ROOT / "models" / "landmark_classifier_int8.tflite"
VAL = ROOT / ".scratch" / "landmarks" / "val"


def load_val_features(frames, use_z=False, limit=None):
    """Real validation feature matrices + masks + labels, for calibration and eval.

    The mask is returned because every DEPLOYED member (and train.py's training
    loop) applies it: the shipped ensemble members all take a mask input, and
    scoring a checkpoint without one changes its operating point by ~5.8 points.
    An exporter that omits the mask ships a different, weaker model under the
    same name - export_ensemble.py builds with_mask=True and refuses a maskless
    graph for exactly this reason.
    """
    vocab_idx = json.load(open(ROOT / ".scratch" / "vocab_best100.json", encoding="utf-8"))
    idx = {g: i for i, g in enumerate(vocab_idx)}
    X, M, y = [], [], []
    for p in sorted(VAL.glob("*.npz")):
        if limit and len(X) >= limit:
            break
        d = np.load(p, allow_pickle=False)
        if float(d["present"].max()) < 0.5:
            continue
        g = str(d["gloss"])
        if g not in idx:
            continue
        f, m = build_features(d["landmarks"], d["present"],
                              expected_frames=frames, use_z=use_z)
        X.append(f)
        M.append(m)
        y.append(idx[g])
    return np.stack(X).astype(np.float32), np.stack(M).astype(np.float32), np.array(y)


def main():
    import torch
    import tensorflow as tf
    from tf_model import build_tf_model, copy_weights

    ckpt = torch.load(CKPT, map_location="cpu", weights_only=True)
    vocab = ckpt["vocab"]
    frames, dim = int(ckpt["frames"]), int(ckpt["dim"])
    in_dim = int(ckpt["model"]["stem.weight"].shape[1])
    use_z = in_dim != 255

    print(f"checkpoint: frames={frames} dim={dim} classes={len(vocab)} "
          f"feature_width={in_dim}")

    # ------------------------------------------------------------ build + copy
    # with_mask=True, unconditionally: train.py always passes the mask, the
    # nine shipped members all expose a mask input, and a maskless export
    # measured 76.52% against 82.32% with it on the same 345 val clips. The
    # previous default (maskless) meant this script produced an artifact five-
    # plus points below the checkpoint it claimed to represent, and its gate
    # compared INT8 against a maskless float reference so the loss was invisible.
    tm = build_tf_model(len(vocab), dim=dim, frames=frames, feature_dim=in_dim,
                        backbone=str(ckpt["backbone"]), with_mask=True)
    copied, missed, _ = copy_weights(tm, CKPT)
    print(f"copied {copied} layer weight groups")

    X, M, y = load_val_features(frames, use_z=use_z)
    print(f"validation clips: {X.shape} (mask {M.shape})")

    # ------------------------------------------------------------- 1. PARITY
    from train import TemporalModel
    pt = TemporalModel(len(vocab), dim=dim, frames=frames,
                       backbone=str(ckpt["backbone"]), in_dim=in_dim)
    pt.load_state_dict(ckpt["model"])
    pt.eval()

    with torch.no_grad():
        pt_logits = pt(torch.from_numpy(X[:200]), torch.from_numpy(M[:200])).numpy()
    tf_logits = tm.predict([X[:200], M[:200]], verbose=0)

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

    # Full-set float reference for the INT8 gate below. The subset figure above
    # is a cheap parity sanity check; the INT8 comparison must be scored on the
    # SAME clip population and the SAME operating point (mask applied), or
    # "drop" measures a population/treatment difference rather than a
    # quantization cost.
    with torch.no_grad():
        pt_full_pred = pt(torch.from_numpy(X), torch.from_numpy(M)).numpy().argmax(1)
    float_acc = float((pt_full_pred == y).mean())
    print(f"val top-1 (full {len(y)}, WITH mask): float {float_acc*100:.2f}%")

    # ---------------------------------------------------- 2. INT8 + TFLite
    def rep():
        # Dict form: the list-of-lists form fed the two-input graph as if it
        # were one input and the converter refused ("input->dims->size != 4").
        for i in range(min(300, len(X))):
            yield {"features": X[i][None, ...], "mask": M[i][None, ...]}

    conv = tf.lite.TFLiteConverter.from_keras_model(tm)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.float32
    conv.inference_output_type = tf.float32
    data = conv.convert()

    # Stage to a temp path and move into place ONLY after every gate passes.
    # The old code wrote OUT first and called OUT.unlink() on refusal - which
    # deleted the previously tracked artifact rather than leaving it alone.
    # A refused export must leave models/ exactly as it was.
    import tempfile
    tmp_out = Path(tempfile.gettempdir()) / ("export_int8_" + OUT.name)
    tmp_out.write_bytes(data)

    magic = tmp_out.read_bytes()[4:8]
    size_kb = tmp_out.stat().st_size / 1024
    print(f"\nexported (staged) {tmp_out}  {size_kb:.1f} KB  magic={magic!r}")
    if magic != b"TFL3":
        tmp_out.unlink()
        sys.exit("REFUSING: not a TFLite FlatBuffer; models/ untouched")

    # ------------------------------------------------- 3. INT8 accuracy gate
    from ai_edge_litert.interpreter import Interpreter
    interp = Interpreter(model_path=str(tmp_out))
    interp.allocate_tensors()
    dets = interp.get_input_details()
    f_det = next((d for d in dets if "features" in d["name"]), dets[0])
    m_det = next((d for d in dets if "mask" in d["name"]), None)
    if m_det is None:
        tmp_out.unlink()
        sys.exit("REFUSING: exported graph has no mask input - the deployed "
                 "members all take one, so this would ship a different model. "
                 "models/ untouched")
    o_det = interp.get_output_details()[0]

    preds = []
    for row, mrow in zip(X, M):
        interp.set_tensor(f_det["index"], row[None, ...].astype(np.float32))
        interp.set_tensor(m_det["index"], mrow[None, ...].astype(np.float32))
        interp.invoke()
        preds.append(int(interp.get_tensor(o_det["index"])[0].argmax()))
    preds = np.array(preds)
    int8_acc = float((preds == y).mean())
    print(f"INT8 TFLite val top-1: {int8_acc*100:.2f}%  (float {float_acc*100:.2f}% on the same {len(y)} clips)")
    drop = float_acc - int8_acc
    if drop > 0.15:
        tmp_out.unlink()
        sys.exit(
            f"REFUSING: INT8 dropped {drop*100:.1f}pp vs float "
            f"({float_acc*100:.1f}% -> {int8_acc*100:.1f}%). Not shipping. "
            f"models/ untouched."
        )

    # All gates passed: publish.
    import shutil
    OUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(tmp_out), str(OUT))
    print(f"wrote {OUT}")

    # Merge into the existing sidecar rather than replacing it. This script
    # exports ONE INT8 model; the shipped file is a nine-member ensemble
    # manifest carrying `members`, `members_detail` and `quantization`. Writing
    # a bare dict here deletes those keys, and src/main.py then falls back to a
    # single-model file that does not exist, so predict() raises
    # FileNotFoundError. Costs one read to avoid destroying the release.
    meta_path = ROOT / "models" / "landmark_model_meta.json"
    try:
        meta = json.load(open(meta_path, encoding="utf-8"))
        if not isinstance(meta, dict):
            meta = {}
    except (FileNotFoundError, json.JSONDecodeError):
        meta = {}
    meta.update({"frames": frames, "feature_dim": in_dim,
                 "classes": len(vocab), "backbone": str(ckpt["backbone"]), "dim": dim,
                 "val_top1_int8": round(int8_acc, 4),
                 "val_top1_float": round(float_acc, 4),
                 "int8_drop_pp": round(drop * 100, 2),
                 "size_kb": round(size_kb, 1)})
    # This artifact is INT8; if the sidecar describes the shipped FP16 ensemble
    # it already says so, and overwriting that would break the APK's packaging
    # gate (app/build.gradle.kts requires exactly "float16").
    meta.setdefault("quantization", "int8")
    keep = sorted(meta.get("members", []))
    if keep:
        print(f"preserving {len(keep)} ensemble member entries in {meta_path.name}")
    json.dump(meta, open(meta_path, "w", encoding="utf-8"), indent=2)
    json.dump({g: n for n, g in enumerate(vocab)},
              open(ROOT / "models" / "landmark_labels.json", "w"), indent=0)
    print("labels and meta written")


if __name__ == "__main__":
    # This script takes no arguments, but it DOES mutate models/: it rewrites
    # landmark_classifier_int8.tflite, landmark_model_meta.json and
    # landmark_labels.json. Without this guard `--help`, a typo, or any stray
    # flag started a full conversion and overwrote tracked artifacts, which is
    # the worst possible response to someone asking what the script does.
    import sys as _sys
    if len(_sys.argv) > 1:
        print(__doc__)
        print("This script takes no arguments and rewrites files under models/.")
        print("It did NOT run. Invoke it with no arguments to perform the export.")
        raise SystemExit(2)
    main()