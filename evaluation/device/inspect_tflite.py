#!/usr/bin/env python3
"""Real TFLite tensor/operator inspection for EchoGesture classifier artifacts.

Usage:
  .venv/bin/python -I evaluation/device/int8_lab/inspect_tflite.py
  .venv/bin/python -I evaluation/device/int8_lab/inspect_tflite.py models/landmark_classifier_fp16.tflite
  ... --json out.json

Inspects the FlatBuffer directly (schema_py_generated) and via the LiteRT
Interpreter IO details. Classification is derived from tensor dtypes, never
from filenames. Exit 0 always on success; use --strict for nonzero exit when
any artifact does not meet the INT8 criterion defined below.
"""
import argparse, collections, hashlib, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]

def _reverse(enum_cls):
    rev = {}
    for k in dir(enum_cls):
        if k.startswith("_"):
            continue
        v = getattr(enum_cls, k)
        if isinstance(v, int):
            rev[v] = k
    return rev

def inspect_flatbuffer(path):
    from ai_edge_litert import schema_py_generated as schema
    data = Path(path).read_bytes()
    rev_dt = _reverse(schema.TensorType)
    rev_op = _reverse(schema.BuiltinOperator)
    model = schema.Model.GetRootAsModel(bytearray(data), 0)
    opcodes = []
    for i in range(model.OperatorCodesLength()):
        oc = model.OperatorCodes(i)
        opcodes.append(rev_op.get(oc.BuiltinCode(), str(oc.BuiltinCode())))
    g = model.Subgraphs(0)
    tensors = []
    for i in range(g.TensorsLength()):
        t = g.Tensors(i)
        q = t.Quantization()
        scale, zp = [], []
        if q is not None:
            scale = [float(x) for x in q.ScaleAsNumpy()] if q.ScaleLength() else []
            zp = [int(x) for x in q.ZeroPointAsNumpy()] if q.ZeroPointLength() else []
        buf_len = 0
        bi = t.Buffer()
        if 0 <= bi < model.BuffersLength():
            b = model.Buffers(bi)
            buf_len = int(b.DataLength()) if b.DataLength() else 0
        tensors.append({
            "idx": i,
            "name": t.Name().decode() if t.Name() else "",
            "dtype": rev_dt.get(t.Type(), str(t.Type())),
            "shape": [int(x) for x in t.ShapeAsNumpy()] if t.ShapeLength() else [],
            "scale": scale, "zp": zp, "buffer_bytes": buf_len,
        })
    ops = []
    for i in range(g.OperatorsLength()):
        o = g.Operators(i)
        ops.append({
            "op": opcodes[o.OpcodeIndex()] if o.OpcodeIndex() < len(opcodes) else "?",
            "inputs": [int(x) for x in o.InputsAsNumpy()] if o.InputsLength() else [],
            "outputs": [int(x) for x in o.OutputsAsNumpy()] if o.OutputsLength() else [],
        })
    return {"magic": data[4:8].decode(errors="replace"), "opcodes": opcodes,
            "tensors": tensors, "ops": ops, "buffers": int(model.BuffersLength())}

def inspect_interpreter(path):
    from ai_edge_litert.interpreter import Interpreter
    interp = Interpreter(model_path=str(path))
    interp.allocate_tensors()
    def fmt(d):
        s, z = d["quantization"]
        return {"shape": [int(x) for x in d["shape"]], "dtype": d["dtype"].__name__,
                "scale": float(s), "zero_point": int(z)}
    return {"inputs": [fmt(d) for d in interp.get_input_details()],
            "outputs": [fmt(d) for d in interp.get_output_details()]}

def classify(fb, io):
    """INT8 criterion: weight-bearing (buffer) tensors are INT8 (or INT32 bias
    constants only), a majority of ops touch INT8 tensors, >=1 quantized
    tensor with non-empty scale, and no FLOAT16/FLOAT32 weight tensors.
    A float16 file has FLOAT16 weights + DEQUANTIZE and 0 quantized tensors;
    it fails this criterion and is classified FP16/non-INT8."""
    tensors = fb["tensors"]
    weight = [t for t in tensors if t["buffer_bytes"] > 0]
    w_dtypes = collections.Counter(t["dtype"] for t in weight)
    qt = [t for t in tensors if t["scale"]]
    int8_ops = [o for o in fb["ops"]
                if any(0 <= i < len(tensors) and tensors[i]["dtype"] in ("INT8", "UINT8")
                       for i in o["inputs"])]
    has_f16_w = w_dtypes.get("FLOAT16", 0) > 0
    has_f32_w = w_dtypes.get("FLOAT32", 0) > 0
    all_weight_int = all(t["dtype"] in ("INT8", "INT32") for t in weight)
    is_int8 = (all_weight_int and not has_f16_w and not has_f32_w and len(qt) > 0
               and len(int8_ops) > 0)
    if has_f16_w and not qt:
        label = "FP16 (non-INT8: FLOAT16 weights, no per-tensor INT8 quantization)"
    elif has_f32_w and not qt:
        label = "FP32 (non-INT8)"
    elif is_int8:
        label = "INT8 (quantized weights/activations; float32 IO via internal QUANTIZE/DEQUANTIZE)"
    else:
        label = "MIXED/UNKNOWN (see dtype histogram)"
    return {"weight_dtype_hist": dict(w_dtypes), "tensor_dtype_hist": dict(
        collections.Counter(t["dtype"] for t in tensors)),
        "quantized_tensors": len(qt), "total_tensors": len(tensors),
        "ops_touching_int8": len(int8_ops), "total_ops": len(fb["ops"]),
        "opcode_hist": dict(collections.Counter(fb["opcodes"])),
        "classification": label, "is_int8_by_tensor_test": bool(is_int8),
        "input_dtype": io["inputs"][0]["dtype"] if io else None,
        "output_dtype": io["outputs"][0]["dtype"] if io else None}

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="*", help="tflite files (default: all models/*.tflite)")
    ap.add_argument("--json", type=Path, default=None, help="write full JSON report here")
    ap.add_argument("--strict", action="store_true", help="exit 1 if any file is not INT8 by tensor test")
    args = ap.parse_args(argv)
    paths = [Path(p) for p in args.paths] or sorted((ROOT / "models").glob("*.tflite"))
    report = {}
    failed = False
    for p in paths:
        data = p.read_bytes()
        fb = inspect_flatbuffer(p)
        io = inspect_interpreter(p)
        cls = classify(fb, io)
        report[p.name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                          "magic": fb["magic"], "interpreter_io": io,
                          "flatbuffer": {"buffers": fb["buffers"], "ops": fb["ops"],
                                         "tensors": [ {**t, "scale": t["scale"][:4], "zp": t["zp"][:4]} for t in fb["tensors"]]},
                          **cls}
        print(f"{p.name}: {len(data)} B  sha={report[p.name]['sha256'][:16]}...")
        print(f"  IO: in {[ (d['dtype'],d['shape'],d['scale']) for d in io['inputs']]} out {[ (d['dtype'],d['shape'],d['scale']) for d in io['outputs']]}")
        print(f"  weights {cls['weight_dtype_hist']} tensors {cls['tensor_dtype_hist']} "
              f"quantized {cls['quantized_tensors']}/{cls['total_tensors']} ops@int8 {cls['ops_touching_int8']}/{cls['total_ops']}")
        print(f"  => {cls['classification']}")
        if not cls["is_int8_by_tensor_test"]:
            failed = True
    if args.json:
        args.json.write_text(json.dumps(report, indent=2))
        print(f"JSON written to {args.json}")
    return 1 if (args.strict and failed) else 0

if __name__ == "__main__":
    sys.exit(main())
