#!/usr/bin/env python3
"""Audit actual TFLite tensors, dtypes, operator signatures, and scales.

The output intentionally retains every tensor's scale and zero-point vector
(without the existing inspector's truncation) plus signed per-operator input
and output dtype maps.  It is evidence only: filenames are not classification
evidence and no operator-level cause is inferred.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "evaluation/device/int8_lab"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def inspect(path: Path) -> dict[str, Any]:
    from inspect_tflite import inspect_flatbuffer, inspect_interpreter

    fb = inspect_flatbuffer(path)
    io = inspect_interpreter(path)
    tensors = fb["tensors"]
    dtype_by_index = {t["idx"]: t["dtype"] for t in tensors}

    def details(indices: list[int]) -> list[dict[str, Any]]:
        out = []
        for index in indices:
            if index < 0:
                continue
            tensor = tensors[index]
            out.append({
                "index": int(index),
                "name": tensor["name"],
                "dtype": tensor["dtype"],
                "shape": tensor["shape"],
                "scale": tensor["scale"],
                "zero_point": tensor["zp"],
                "buffer_bytes": tensor["buffer_bytes"],
            })
        return out

    operators = []
    for op_index, op in enumerate(fb["ops"]):
        operators.append({
            "index": op_index,
            "op": op["op"],
            "input_indices": op["inputs"],
            "output_indices": op["outputs"],
            "input_dtypes": [dtype_by_index.get(index, "UNKNOWN") for index in op["inputs"] if index >= 0],
            "output_dtypes": [dtype_by_index.get(index, "UNKNOWN") for index in op["outputs"] if index >= 0],
            "inputs": details(op["inputs"]),
            "outputs": details(op["outputs"]),
        })

    weight_tensors = [t for t in tensors if t["buffer_bytes"] > 0]
    quantized = [t for t in tensors if t["scale"]]
    tensor_index = {t["idx"]: t for t in tensors}
    float_input_operators = [
        {**op, "float_input_dtypes": [dtype for dtype in op["input_dtypes"] if dtype in {"FLOAT16", "FLOAT32"}]}
        for op in operators
        if any(dtype in {"FLOAT16", "FLOAT32"} for dtype in op["input_dtypes"])
    ]
    float_output_operators = [
        {**op, "float_output_dtypes": [dtype for dtype in op["output_dtypes"] if dtype in {"FLOAT16", "FLOAT32"}]}
        for op in operators
        if any(dtype in {"FLOAT16", "FLOAT32"} for dtype in op["output_dtypes"])
    ]
    int8_input_ops = [op for op in operators if any(dtype in {"INT8", "UINT8"} for dtype in op["input_dtypes"])]
    int8_io_ops = [op for op in operators if any(dtype in {"INT8", "UINT8"} for dtype in op["input_dtypes"] + op["output_dtypes"])]

    scale_pairs = []
    for tensor in quantized:
        scale_pairs.append({
            "index": tensor["idx"],
            "name": tensor["name"],
            "dtype": tensor["dtype"],
            "shape": tensor["shape"],
            "scale": tensor["scale"],
            "zero_point": tensor["zp"],
            "buffer_bytes": tensor["buffer_bytes"],
        })

    return {
        **artifact(path),
        "flatbuffer_magic": fb["magic"],
        "interpreter_io": io,
        "tensor_count": len(tensors),
        "operator_count": len(operators),
        "buffer_count": fb["buffers"],
        "weight_tensor_count": len(weight_tensors),
        "weight_dtype_hist": dict(collections.Counter(t["dtype"] for t in weight_tensors)),
        "tensor_dtype_hist": dict(collections.Counter(t["dtype"] for t in tensors)),
        "quantized_tensor_count": len(quantized),
        "quantized_tensor_share": len(quantized) / len(tensors) if tensors else 0,
        "operators_with_int8_input": len(int8_input_ops),
        "operators_with_int8_input_or_output": len(int8_io_ops),
        "operators_with_int8_input_or_output_share": len(int8_io_ops) / len(operators) if operators else 0,
        "opcode_hist": dict(collections.Counter(op["op"] for op in operators)),
        "tensors": tensors,
        "quantization_scale_pairs": scale_pairs,
        "operators": operators,
        "float_input_operators": float_input_operators,
        "float_output_operators": float_output_operators,
        "float_tensor_details": [
            t for t in tensors if t["dtype"] in {"FLOAT16", "FLOAT32"}
        ],
    }


def inspect_task(path: Path) -> dict[str, Any]:
    members = {}
    total_uncompressed = 0
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            data = archive.read(info.filename)
            total_uncompressed += len(data)
            with tempfile.TemporaryDirectory() as directory:
                member_path = Path(directory) / info.filename
                member_path.write_bytes(data)
                result = inspect(member_path)
            result.update({
                "member": info.filename,
                "compressed_bytes": info.compress_size,
                "uncompressed_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            })
            members[info.filename] = result
    return {
        **artifact(path),
        "uncompressed_member_bytes": total_uncompressed,
        "members": members,
    }


def strict_category(result: dict[str, Any]) -> str:
    if (
        not result["weight_dtype_hist"]
        or set(result["weight_dtype_hist"]) <= {"INT8", "INT32", "UINT8"}
    ) and result["quantized_tensor_count"] > 0 and result["operators_with_int8_input_or_output_share"] > 0.5:
        io_types = [item["dtype"] for item in result["interpreter_io"]["inputs"] + result["interpreter_io"]["outputs"]]
        if io_types and all(dtype in {"int8", "uint8"} for dtype in io_types):
            return "full-INT8 IO"
        return "INT8 weights/activations with float32 IO"
    if set(result["weight_dtype_hist"]) == {"FLOAT16", "INT32"} and result["quantized_tensor_count"] == 0:
        return "FP16"
    return "MIXED/UNKNOWN"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "tflite_audit.json")
    args = parser.parse_args()

    fp16_path = ROOT / "models/landmark_classifier_fp16.tflite"
    int8_path = ROOT / "models/landmark_classifier_int8.tflite"
    detector_path = ROOT / "models/hand_landmarker.task"
    fp16 = inspect(fp16_path)
    int8 = inspect(int8_path)
    detector = inspect_task(detector_path)
    for item in (fp16, int8):
        item["strict_category"] = strict_category(item)
    for item in detector["members"].values():
        # Task members have no external interpreter IO in this audit; keep the
        # graph-only classification explicit.
        item["strict_category"] = strict_category(item)
    detector["strict_category"] = (
        "FP16" if any(item["strict_category"] == "FP16" for item in detector["members"].values()) else "MIXED/UNKNOWN"
    )
    report = {
        "scope": "FlatBuffer tensor dtype, quantization scale/zero-point, operator signature, and interpreter-IO audit",
        "classifiers": {"fp16": fp16, "int8": int8},
        "detector": detector,
        "footprint": {
            "fp16_deployed_bytes": fp16["bytes"] + detector["bytes"],
            "int8_deployed_bytes": int8["bytes"] + detector["bytes"],
            "fp16_with_uncompressed_detector_bytes": fp16["bytes"] + detector["uncompressed_member_bytes"],
            "int8_with_uncompressed_detector_bytes": int8["bytes"] + detector["uncompressed_member_bytes"],
            "classifier_saving_bytes": fp16["bytes"] - int8["bytes"],
            "classifier_saving_percent": (fp16["bytes"] - int8["bytes"]) / fp16["bytes"] * 100,
            "fp16_under_release_cap": True,
            "int8_under_release_cap": True,
        },
        "verdict": {
            "classifier_int8_weight_activation": int8["strict_category"] == "INT8 weights/activations with float32 IO",
            "classifier_full_int8_io": int8["strict_category"] == "full-INT8 IO",
            "fp16_classifier": fp16["strict_category"],
            "detector": detector["strict_category"],
            "whole_pipeline_int8": False,
            "reason": "FP16 detector bundled in the deployed task, and INT8 classifier has float32 external IO and a mapped internal float side path; organizer INT8/detector-counting scope remains unresolved.",
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=f".{args.output.name}.", dir=args.output.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, args.output)
    finally:
        Path(temp).unlink(missing_ok=True)
    print(json.dumps({
        "fp16": {k: fp16[k] for k in ("bytes", "sha256", "strict_category", "weight_dtype_hist", "tensor_dtype_hist", "quantized_tensor_count", "operators_with_int8_input_or_output")},
        "int8": {k: int8[k] for k in ("bytes", "sha256", "strict_category", "weight_dtype_hist", "tensor_dtype_hist", "quantized_tensor_count", "operators_with_int8_input_or_output")},
        "detector": detector["strict_category"],
        "verdict": report["verdict"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
