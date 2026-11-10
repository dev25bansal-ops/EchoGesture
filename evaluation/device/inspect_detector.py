#!/usr/bin/env python3
"""Inspect hand_landmarker.task (zip of 2 TFLite) and compute combined footprint.

The detector bundles hand_detector.tflite + hand_landmarks_detector.tflite,
both FP16 weights with no INT8 quantization (verified by tensor inspection).
Usage: .venv/bin/python -I evaluation/device/int8_lab/inspect_detector.py [--json out.json]
"""
import argparse, collections, hashlib, json, sys, zipfile
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from inspect_tflite import inspect_flatbuffer, classify

ROOT = Path(__file__).resolve().parents[3]
TASK = ROOT / "models/hand_landmarker.task"
FP, I8 = ROOT / "models/landmark_classifier_fp16.tflite", ROOT / "models/landmark_classifier_int8.tflite"

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)
    report = {"task_file_bytes": TASK.stat().st_size,
              "task_file_sha256": hashlib.sha256(TASK.read_bytes()).hexdigest(), "members": {}}
    total = 0
    with zipfile.ZipFile(TASK) as z:
        for info in z.infolist():
            data = z.read(info.filename)
            total += len(data)
            tmp = Path("/tmp") / f"_det_{info.filename}"
            tmp.write_bytes(data)
            fb = inspect_flatbuffer(tmp)
            io_cls = classify(fb, None)
            tmp.unlink(missing_ok=True)
            report["members"][info.filename] = {"bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(), **io_cls}
            print(f"{info.filename}: {len(data)} B weights {io_cls['weight_dtype_hist']} quantized {io_cls['quantized_tensors']}/{io_cls['total_tensors']} => {io_cls['classification']}")
    report["members_uncompressed_bytes"] = total
    fp_b, i8_b = FP.stat().st_size, I8.stat().st_size
    report["footprint"] = {
        "fp16_classifier_bytes": fp_b, "int8_classifier_bytes": i8_b,
        "detector_task_file_bytes": report["task_file_bytes"],
        "fp16_plus_detector_file_bytes": fp_b + report["task_file_bytes"],
        "int8_plus_detector_file_bytes": i8_b + report["task_file_bytes"],
        "int8_plus_detector_uncompressed_bytes": i8_b + total,
        "limit_20MiB_bytes": 20 * 1024 * 1024, "limit_20MB_bytes": 20_000_000,
        "fp16_total_under_20MiB_by": 20 * 1024 * 1024 - (fp_b + report["task_file_bytes"]),
        "int8_total_under_20MiB_by": 20 * 1024 * 1024 - (i8_b + report["task_file_bytes"]),
        "detector_counted_note": "Detector itself is FP16 (not INT8). Whether the organizer counts detector bytes toward the 20 MB cap is UNCONFIRMED; report both totals.",
    }
    print(json.dumps(report["footprint"], indent=2))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2))
    return 0

if __name__ == "__main__":
    sys.exit(main())
