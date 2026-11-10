#!/usr/bin/env python3
"""Inspect EchoGesture quantization and run deterministic real-feature inference.

No training, conversion, downloading, or benchmark loops. All external data are
read-only. This tool classifies graphs from tensor/operator evidence, never names.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.metadata
import json
import platform
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FIXTURES = HERE / "fixtures"
PINS = json.loads((FIXTURES / "provenance.json").read_text(encoding="utf-8"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def artifact(path: Path, expected: dict | None = None) -> dict:
    data = path.read_bytes()
    result = {"path": str(path), "sha256": sha256_bytes(data), "bytes": len(data)}
    if expected:
        for field in ("sha256", "bytes"):
            if field in expected and result[field] != expected[field]:
                raise ValueError(f"{path.name} {field} mismatch: {result[field]} != {expected[field]}")
    return result


def reverse_enum(enum_type) -> dict[int, str]:
    return {value: key for key in dir(enum_type)
            if not key.startswith("_") and isinstance((value := getattr(enum_type, key)), int)}


def inspect_graph(data: bytes, label: str) -> dict:
    from ai_edge_litert import schema_py_generated as schema

    model = schema.Model.GetRootAsModel(bytearray(data), 0)
    dtypes = reverse_enum(schema.TensorType)
    op_names = reverse_enum(schema.BuiltinOperator)
    opcodes = []
    for index in range(model.OperatorCodesLength()):
        opcode = model.OperatorCodes(index)
        opcodes.append(op_names.get(opcode.BuiltinCode(), str(opcode.BuiltinCode())))
    graph = model.Subgraphs(0)
    tensors = []
    for index in range(graph.TensorsLength()):
        tensor = graph.Tensors(index)
        quant = tensor.Quantization()
        buffer_index = tensor.Buffer()
        buffer = model.Buffers(buffer_index) if 0 <= buffer_index < model.BuffersLength() else None
        tensors.append({
            "index": index,
            "name": tensor.Name().decode() if tensor.Name() else "",
            "dtype": dtypes.get(tensor.Type(), str(tensor.Type())),
            "shape": tensor.ShapeAsNumpy().astype(int).tolist() if tensor.ShapeLength() else [],
            "buffer_bytes": int(buffer.DataLength()) if buffer is not None else 0,
            "scales": quant.ScaleAsNumpy().astype(float).tolist() if quant is not None and quant.ScaleLength() else [],
            "zero_points": quant.ZeroPointAsNumpy().astype(int).tolist() if quant is not None and quant.ZeroPointLength() else [],
            "quantized_dimension": int(quant.QuantizedDimension()) if quant is not None else None,
        })
    operators = []
    for index in range(graph.OperatorsLength()):
        operator = graph.Operators(index)
        inputs = operator.InputsAsNumpy().astype(int).tolist() if operator.InputsLength() else []
        outputs = operator.OutputsAsNumpy().astype(int).tolist() if operator.OutputsLength() else []
        operators.append({
            "index": index,
            "op": opcodes[operator.OpcodeIndex()],
            "inputs": inputs,
            "outputs": outputs,
            "input_dtypes": [tensors[i]["dtype"] for i in inputs if i >= 0],
            "output_dtypes": [tensors[i]["dtype"] for i in outputs if i >= 0],
        })
    weights = [tensor for tensor in tensors if tensor["buffer_bytes"]]
    quantized = [tensor for tensor in tensors if tensor["scales"]]
    int8_ops = [op for op in operators if set(op["input_dtypes"] + op["output_dtypes"]) & {"INT8", "UINT8"}]
    float_ops = [op for op in operators if set(op["input_dtypes"] + op["output_dtypes"]) & {"FLOAT16", "FLOAT32"}]
    per_channel = [tensor for tensor in quantized if len(tensor["scales"]) > 1]
    weight_dtypes = dict(collections.Counter(tensor["dtype"] for tensor in weights))
    category = "MIXED/UNKNOWN"
    if set(weight_dtypes) == {"FLOAT16", "INT32"} and not quantized:
        category = "FP16"
    elif set(weight_dtypes) <= {"INT8", "INT32", "UINT8"} and quantized and len(int8_ops) > len(operators) / 2:
        category = "INT8 weights/activations"
    return {
        "label": label,
        "sha256": sha256_bytes(data),
        "bytes": len(data),
        "flatbuffer_magic": data[4:8].decode(errors="replace"),
        "category": category,
        "tensor_count": len(tensors),
        "operator_count": len(operators),
        "tensor_dtype_histogram": dict(collections.Counter(t["dtype"] for t in tensors)),
        "weight_dtype_histogram": weight_dtypes,
        "weight_buffer_bytes": sum(t["buffer_bytes"] for t in weights),
        "quantized_tensor_count": len(quantized),
        "int8_operator_count": len(int8_ops),
        "opcode_histogram": dict(collections.Counter(op["op"] for op in operators)),
        "per_channel_tensor_count": len(per_channel),
        "per_channel_examples": [{key: tensor[key] for key in
                                  ("index", "name", "shape", "quantized_dimension")}
                                 | {"scale_count": len(tensor["scales"]),
                                    "scale_min": min(tensor["scales"]),
                                    "scale_max": max(tensor["scales"]),
                                    "zero_points_unique": sorted(set(tensor["zero_points"]))}
                                 for tensor in per_channel],
        "scale_zero_point_checks": {
            "all_scales_finite_positive": all(np.isfinite(s) and s > 0 for t in quantized for s in t["scales"]),
            "int8_zero_points_in_range": all(-128 <= z <= 127 for t in quantized if t["dtype"] == "INT8" for z in t["zero_points"]),
            "mismatched_scale_zero_point_lengths": [t["index"] for t in quantized if len(t["scales"]) != len(t["zero_points"])],
            "per_tensor_zero_point_nonzero": [t["index"] for t in quantized if len(t["scales"]) == 1 and t["zero_points"] != [0]],
            "per_channel_zero_point_nonzero": [t["index"] for t in per_channel if set(t["zero_points"]) != {0}],
        },
        "float_operators": float_ops,
        "tensors": tensors,
        "operators": operators,
    }


def interpreter_io(path: Path) -> dict:
    from ai_edge_litert.interpreter import Interpreter
    interpreter = Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    def clean(detail):
        scale, zero = detail["quantization"]
        return {"shape": detail["shape"].astype(int).tolist(), "dtype": detail["dtype"].__name__,
                "scale": float(scale), "zero_point": int(zero)}
    return {"inputs": [clean(d) for d in interpreter.get_input_details()],
            "outputs": [clean(d) for d in interpreter.get_output_details()]}


def invoke(path: Path, features: np.ndarray) -> np.ndarray:
    from ai_edge_litert.interpreter import Interpreter
    interpreter = Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    detail_in = interpreter.get_input_details()[0]
    detail_out = interpreter.get_output_details()[0]
    value = np.asarray(features, dtype=detail_in["dtype"])
    if value.shape != tuple(detail_in["shape"]):
        value = value.reshape(detail_in["shape"])
    scale, zero = detail_in["quantization"]
    if detail_in["dtype"] == np.int8:
        value = np.clip(np.rint(value / scale + zero), -128, 127).astype(np.int8)
    interpreter.set_tensor(detail_in["index"], value)
    interpreter.invoke()
    output = interpreter.get_tensor(detail_out["index"]).copy()
    scale, zero = detail_out["quantization"]
    if scale:
        output = (output.astype(np.float32) - zero) * scale
    return np.asarray(output, dtype=np.float32)


def ordered_labels() -> list[str]:
    mapping = json.loads((ROOT / "models/landmark_labels.json").read_text(encoding="utf-8"))
    labels = [""] * len(mapping)
    for label, index in mapping.items():
        labels[index] = label.upper()
    if sorted(mapping.values()) != list(range(len(mapping))) or len(set(labels)) != len(labels) or any(not x for x in labels):
        raise ValueError("Label order is not unique and contiguous")
    return labels


def paired_metrics(reference: np.ndarray, candidate: np.ndarray, y: np.ndarray | None = None) -> dict:
    if reference.shape != candidate.shape or reference.ndim != 2 or reference.shape[1] != 100:
        raise ValueError("Expected matching N x 100 logit matrices")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("Non-finite logits")
    rp, cp = reference.argmax(axis=1), candidate.argmax(axis=1)
    def softmax(z):
        exp = np.exp(z - z.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)
    result = {"n": len(reference), "argmax_agreement": float(np.mean(rp == cp)),
              "logit_mean_absolute_error": float(np.abs(reference - candidate).mean()),
              "logit_l2_per_clip_mean": float(np.linalg.norm(reference - candidate, axis=1).mean()),
              "probability_l1_per_clip_mean": float(np.abs(softmax(reference) - softmax(candidate)).sum(axis=1).mean())}
    if y is not None:
        result.update({"reference_top1": float(np.mean(rp == y)), "candidate_top1": float(np.mean(cp == y)),
                       "reference_minus_candidate_pp": float((np.mean(rp == y) - np.mean(cp == y)) * 100)})
    return result


def feature_contract() -> dict:
    sys.path.insert(0, str(ROOT / "src"))
    from features import build_features
    fixtures = []
    for name in ("parity_chew2.npz", "parity_synthetic.npz"):
        path = ROOT / "evaluation/device/v7_android" / name
        with np.load(path, allow_pickle=False) as data:
            landmarks, present = data["landmarks"], data["present"]
            actual, mask = build_features(landmarks, present, expected_frames=24)
            expected, expected_mask = data["features"], data["mask"]
            fixtures.append({"name": name, "sha256": sha256_bytes(path.read_bytes()),
                             "shape": list(actual.shape), "dtype": str(actual.dtype),
                             "max_abs_feature_error": float(np.max(np.abs(actual - expected))),
                             "mask_exact": bool(np.array_equal(mask, expected_mask)),
                             "nonfinite_values": int((~np.isfinite(actual)).sum()),
                             "missing_left_frames": int((present[:, 0] <= .5).sum()),
                             "missing_right_frames": int((present[:, 1] <= .5).sum()),
                             "present_range": [float(present.min()), float(present.max())]})
    # Exercise the real transform's missing-hand and scale behavior, not a model mock.
    rng = np.random.default_rng(20261110)
    landmarks = rng.normal(0.5, .08, (32, 2, 21, 3)).astype(np.float32)
    present = np.ones((32, 2), np.float32)
    present[4:12, 0] = 0
    present[16:20, :] = 0
    landmarks[4:12, 0] = 0
    landmarks[16:20] = 0
    features, mask = build_features(landmarks, present, expected_frames=24)
    missing_rows = np.abs(features[mask == 0]).max() if (mask == 0).any() else 0.0
    missing_probe = {"shape": list(features.shape), "finite": bool(np.isfinite(features).all()),
                     "zero_mask_frames": int((mask == 0).sum()),
                     "missing_frame_rows_are_all_zero": bool(missing_rows == 0.0),
                     "missing_frame_rows_max_abs": float(missing_rows),
                     "note": "velocity/acceleration are lagged differences, so a frame whose hands are absent still carries "
                             "delta from the previous present frame; the model input discards the mask at inference, so these "
                             "rows are not zeroed before the classifier",
                     "partial_presence_interpolates": bool(np.any((build_features(landmarks, present, expected_frames=8)[0] != 0)))}
    return {"pinned_sha256": PINS["assets"]["features"]["sha256"], "fixtures": fixtures,
            "missing_hand_probe": missing_probe,
            "frozen_contract": PINS["frozen_contract"]}


def verify_existing_pair() -> dict:
    base = ROOT / "evaluation/device/v5_accuracy"
    summary_path = base / "results/val_346/summary.json"
    predictions_path = base / "results/val_346/predictions.json"
    manifest_path = base / "data/val_346/clips/fetched_manifest.csv"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    identities = {name: artifact(path) for name, path in
                  (("summary", summary_path), ("predictions", predictions_path), ("manifest", manifest_path))}
    expected_sha = PINS["paired_evidence"]
    for name, value in identities.items():
        if value["sha256"] != expected_sha[name]["sha256"]:
            raise ValueError(f"Paired evidence {name} SHA changed")
    checks = {"n_clips": len(predictions) == summary["n_clips"] == 346,
              "successful": sum(r["status"] == "ok" for r in predictions) == 345,
              "fp16_correct": sum(r["fp16_top1_correct"] for r in predictions) == summary["fp16"]["top1_correct"],
              "int8_correct": sum(r["int8_top1_correct"] for r in predictions) == summary["int8"]["top1_correct"],
              "manifest_pinned": summary["manifest"]["sha256"] == identities["manifest"]["sha256"],
              "feature_implementation_pinned": summary["features_sha256"] == PINS["assets"]["features"]["sha256"],
              "labels_pinned": summary["labels_sha256"] == PINS["assets"]["labels"]["sha256"]}
    if not all(checks.values()):
        raise ValueError(f"Paired evidence failed checks: {checks}")
    successful = [r for r in predictions if r["status"] == "ok"]
    return {"identities": identities, "checks": checks, "n": len(predictions),
            "fp16_top1": summary["fp16"], "int8_top1": summary["int8"],
            "paired": summary["paired"], "per_signer": summary["per_signer"],
            "feature_hash_unique": len({r["features_sha"] for r in successful}),
            "feature_hash_combined_sha256": sha256_bytes(bytes.fromhex("".join(r["features_sha"] for r in successful))),
            "missing_hand_clip": next(r["clip_id"] for r in predictions if r["status"] != "ok"),
            "note": "This verifies stored paired records, not a fresh 346-clip execution."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "results/audit.json")
    parser.add_argument("--skip-real-clips", action="store_true", help="Use retained real landmark NPZ fixtures only")
    args = parser.parse_args()
    assets = {name: artifact(ROOT / item["path"], item) for name, item in PINS["assets"].items()}
    fp16_path = ROOT / PINS["assets"]["fp16"]["path"]
    int8_path = ROOT / PINS["assets"]["int8"]["path"]
    fp16 = inspect_graph(fp16_path.read_bytes(), "fp16") | {"interpreter_io": interpreter_io(fp16_path)}
    int8 = inspect_graph(int8_path.read_bytes(), "int8") | {"interpreter_io": interpreter_io(int8_path)}
    detector_members = []
    with zipfile.ZipFile(ROOT / PINS["assets"]["detector"]["path"]) as archive:
        for info in archive.infolist():
            detector_members.append(inspect_graph(archive.read(info), info.filename)
                                    | {"compressed_bytes": info.compress_size})
    detector = {"members": detector_members,
                "category": "FP16" if all(m["category"] == "FP16" for m in detector_members) else "MIXED/UNKNOWN",
                "uncompressed_bytes": sum(m["bytes"] for m in detector_members)}
    contract = feature_contract()
    fixture_rows, logits = [], {"fp16": [], "int8": []}
    for name in ("parity_chew2.npz", "parity_synthetic.npz"):
        path = ROOT / "evaluation/device/v7_android" / name
        with np.load(path, allow_pickle=False) as data:
            features = data["features"]
        row = {"name": name}
        for model_name, model_path in (("fp16", fp16_path), ("int8", int8_path)):
            output = invoke(model_path, features)
            logits[model_name].append(output[0])
            row[model_name] = {"finite": bool(np.isfinite(output).all()), "shape": list(output.shape),
                               "dtype": str(output.dtype), "top1": ordered_labels()[int(output.argmax())],
                               "min": float(output.min()), "max": float(output.max()),
                               "sum": float(output.sum()), "logits": output[0].tolist()}
        row["parity"] = paired_metrics(np.array([logits["fp16"][-1]]), np.array([logits["int8"][-1]]))
        fixture_rows.append(row)
    real_clips = []
    if not args.skip_real_clips:
        import csv
        sys.path.insert(0, str(ROOT / "src"))
        from features import build_features
        from landmark_frontend import detect_hands, read_frames
        ids = PINS["real_clip_ids"]
        manifest = ROOT / PINS["paired_evidence"]["manifest"]["path"]
        with manifest.open(newline="", encoding="utf-8-sig") as stream:
            rows = {r["clip_id"]: r for r in csv.DictReader(stream)}
        for clip_id in ids:
            row = rows[clip_id]
            path = manifest.parent / row["path"]
            item = {"clip_id": clip_id, "sha256": sha256_bytes(path.read_bytes()), "ground_truth": row["ground_truth"]}
            frames = read_frames(path, max_frames=32)
            landmarks, present = detect_hands(frames, mode="image")
            item.update({"decoded_frames": len(frames), "frames_with_hands": int((present.max(axis=1) > .5).sum()),
                         "presence": present.tolist()})
            if not frames or present.max() < .5:
                item["status"] = "no_hand_detected"
            else:
                features, mask = build_features(landmarks, present, expected_frames=24)
                item.update({"status": "ok", "feature_sha256": sha256_bytes(features.astype("<f4").tobytes()),
                             "feature_range": [float(features.min()), float(features.max())],
                             "feature_std": float(features.std())})
                for model_name, model_path in (("fp16", fp16_path), ("int8", int8_path)):
                    output = invoke(model_path, features)
                    item[model_name] = {"prediction": ordered_labels()[int(output.argmax())],
                                        "finite": bool(np.isfinite(output).all()), "logits": output[0].tolist()}
            real_clips.append(item)
    report = {"scope": "quantization inspection and real inference; no training/download/conversion/timing benchmark",
              "repo_head": PINS["repo_head"], "runtime": {"python": platform.python_version(),
              "platform": platform.platform(), "ai_edge_litert": importlib.metadata.version("ai-edge-litert"),
              "numpy": np.__version__}, "assets": assets, "labels": {"ordered": ordered_labels(),
              "count": len(ordered_labels())}, "feature_contract": contract,
              "classifiers": {"fp16": fp16, "int8": int8}, "detector": detector,
              "footprint": {"fp16_plus_detector_bytes": assets["fp16"]["bytes"] + assets["detector"]["bytes"],
              "int8_plus_detector_bytes": assets["int8"]["bytes"] + assets["detector"]["bytes"],
              "classifier_saving_bytes": assets["fp16"]["bytes"] - assets["int8"]["bytes"],
              "classifier_saving_percent": (assets["fp16"]["bytes"] - assets["int8"]["bytes"]) / assets["fp16"]["bytes"] * 100,
              "cap_interpretations": {"20_MiB": 20 * 1024 * 1024, "20_MB": 20_000_000}},
              "real_feature_inference": fixture_rows, "real_clip_inference": real_clips,
              "existing_paired_346": verify_existing_pair(),
              "calibration_code_audit": PINS["calibration_code_audit"],
              "verdict": {"selected_classifier": "FP16", "candidate_classifier": int8["category"] + " with float32 IO",
                          "candidate_is_improvement": False, "whole_pipeline_int8": False,
                          "official_interpretation": "UNRESOLVED", "default_model_changed": False}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=args.output.parent,
                                     prefix=args.output.name + ".", delete=False) as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), "verdict": report["verdict"],
                      "footprint": report["footprint"], "real_clip_inference": [
                          {k: v for k, v in row.items() if k not in ("fp16", "int8", "presence")}
                          | {name: row[name]["prediction"] for name in ("fp16", "int8") if name in row}
                          for row in real_clips]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
