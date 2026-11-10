#!/usr/bin/env python3
"""FP16 vs INT8 logit/agreement comparison on synthetic feature rows.

This measures quantization-induced divergence, NOT accuracy: synthetic rows
have no labels. Do not cite agreement as accuracy or as compliance evidence.
For a real metric, supply labelled feature rows (.npz with 'X','y') via --npz;
then top-1 for both models is computed on the same rows.

Usage:
  .venv/bin/python -I evaluation/device/int8_lab/compare_fp16_int8.py
  .venv/bin/python -I evaluation/device/int8_lab/compare_fp16_int8.py --npz /path/rows.npz --json out.json
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
from ai_edge_litert.interpreter import Interpreter

ROOT = Path(__file__).resolve().parents[3]
FP = ROOT / "models/landmark_classifier_fp16.tflite"
I8 = ROOT / "models/landmark_classifier_int8.tflite"

def load(p):
    i = Interpreter(model_path=str(p)); i.allocate_tensors(); return i

def run(interp, x):
    inp = interp.get_input_details()[0]; out = interp.get_output_details()[0]
    interp.set_tensor(inp["index"], x[None].astype(np.float32))
    interp.invoke()
    return interp.get_tensor(out["index"])[0].astype(np.float32)

def softmax(z):
    e = np.exp(z - z.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)

def bench(interp, x, n=200):
    for _ in range(20): run(interp, x)
    t0 = time.perf_counter()
    for _ in range(n): run(interp, x)
    return (time.perf_counter() - t0) / n * 1000.0

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", type=Path, default=None, help=".npz with X (N,24,255) float32 and y (N,) int labels")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    labels = json.loads((ROOT / "models/landmark_labels.json").read_text())
    inv = {v: k for k, v in labels.items()}
    fp, i8 = load(FP), load(I8)
    result = {"models": {"fp16": str(FP), "int8": str(I8)}, "input_kind": "synthetic (no labels; agreement only)"}
    if args.npz:
        d = np.load(args.npz, allow_pickle=False)
        X, y = d["X"].astype(np.float32), d["y"].astype(int)
        result["input_kind"] = f"labelled rows from {args.npz} N={len(X)}"
    else:
        rng = np.random.default_rng(args.seed)
        X = np.concatenate([rng.standard_normal((500, 24, 255)) * 0.3,
                            rng.standard_normal((200, 24, 255))]).astype(np.float32)
        y = None
    fl = np.stack([run(fp, r) for r in X]); il = np.stack([run(i8, r) for r in X])
    fp_pred, i8_pred = fl.argmax(1), il.argmax(1)
    pf, pi = softmax(fl), softmax(il)
    result.update({
        "n": int(len(X)),
        "argmax_agreement": float(np.mean(fp_pred == i8_pred)),
        "logit_maxabs_mean": float(np.abs(fl - il).mean()),
        "logit_maxabs_max": float(np.abs(fl - il).max()),
        "logit_corr": float(np.corrcoef(fl.ravel(), il.ravel())[0, 1]),
        "prob_l1_mean": float(np.abs(pf - pi).sum(1).mean()),
        "fp16_top1_conf_mean": float(pf.max(1).mean()),
        "int8_top1_conf_mean": float(pi.max(1).mean()),
        "invoke_ms_fp16_mean": bench(fp, X[0]),
        "invoke_ms_int8_mean": bench(i8, X[0]),
    })
    if y is not None:
        result["fp16_top1"] = float(np.mean(fp_pred == y))
        result["int8_top1"] = float(np.mean(i8_pred == y))
        result["drop_pp"] = (result["fp16_top1"] - result["int8_top1"]) * 100.0
    print(json.dumps(result, indent=2))
    if y is None:
        print("NOTE: synthetic inputs only. This is not accuracy and not a compliance claim.")
    if args.json:
        args.json.write_text(json.dumps(result, indent=2))
    return 0

if __name__ == "__main__":
    sys.exit(main())
