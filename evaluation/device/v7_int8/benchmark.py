#!/usr/bin/env python3
"""Measured classifier runtime/delegate comparison on a fixed real tensor.

This uses the retained V5 paired runner's feature contract for one real clip
from the official validation cache.  The tensor is computed once and reused
for every timing configuration; no detector or feature timing is mixed into
classifier timing.  Delegate labels refer to the LiteRT resolver setting and
are reported separately from the XNNPACK log line emitted by the runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE_ROOT = Path("/tmp/echogesture-v7-agents/A06")
sys.path.insert(0, str(ROOT / "src"))

FP16 = ROOT / "models/landmark_classifier_fp16.tflite"
INT8 = ROOT / "models/landmark_classifier_int8.tflite"
DETECTOR = ROOT / "models/hand_landmarker.task"
FP16_SHA = "a012f2b5f7a1110961079f19eb0e28c9abacb95e4d0787352e4157ebca7044a4"
INT8_SHA = "9d24640a23244a3cd00c9fa62257733819f86ed1da5a10194c9b1cb64323700b"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pct(values: list[float], f: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, int(f * len(ordered)) - 1))]


def summarize(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean_ms": statistics.fmean(values),
        "p50_ms": statistics.median(values),
        "p90_ms": pct(values, 0.90),
        "p95_ms": pct(values, 0.95),
        "p99_ms": pct(values, 0.99),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=HERE / "benchmark.json")
    ap.add_argument("--clip", type=Path, default=SOURCE_ROOT / "evaluation/device/v5_accuracy/data/val_346/clips/844429155260962-BEE.mp4")
    ap.add_argument("--repeats", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=50)
    args = ap.parse_args()
    if args.repeats < 1 or args.warmup < 0:
        raise ValueError("repeats must be positive and warmup nonnegative")
    if not args.clip.is_file():
        raise FileNotFoundError(args.clip)

    from ai_edge_litert.interpreter import Interpreter, OpResolverType
    from landmark_frontend import detect_hands, read_frames
    from features import build_features

    frames = read_frames(str(args.clip), max_frames=32)
    landmarks, present = detect_hands(frames, mode="image")
    features, _ = build_features(landmarks, present, expected_frames=24)
    tensor = features[None, ...].astype(np.float32)
    if tensor.shape != (1, 24, 255):
        raise ValueError("unexpected feature tensor shape")
    fixed_sha = hashlib.sha256(tensor.astype("<f4").tobytes()).hexdigest()
    in_dtypes = {}
    out_dtypes = {}
    configs = {}
    log_capture = []
    # Record the actual runtime environment and hardware, but never turn this
    # into a target-device claim.
    try:
        import importlib.metadata as metadata
        versions = {name: metadata.version(name) for name in ("numpy", "ai-edge-litert", "opencv-python", "mediapipe")}
    except Exception:
        versions = {"python": sys.version.split()[0]}

    for threads in (1, 2, 4):
        for key, path, expected_sha in (("fp16", FP16, FP16_SHA), ("int8", INT8, INT8_SHA)):
            actual_sha = sha256(path)
            if actual_sha != expected_sha:
                raise ValueError(f"SHA mismatch for {path}: {actual_sha}")
            for resolver_name, resolver in (
                ("AUTO", OpResolverType.AUTO),
                ("BUILTIN_WITHOUT_DEFAULT_DELEGATES", OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES),
            ):
                interpreter = Interpreter(
                    model_path=str(path),
                    num_threads=threads,
                    experimental_op_resolver_type=resolver,
                )
                interpreter.allocate_tensors()
                ind = interpreter.get_input_details()[0]
                outd = interpreter.get_output_details()[0]
                in_dtypes[key] = str(ind["dtype"])
                out_dtypes[key] = str(outd["dtype"])
                for _ in range(args.warmup):
                    interpreter.set_tensor(ind["index"], tensor)
                    interpreter.invoke()
                times = []
                for _ in range(args.repeats):
                    started = time.perf_counter_ns()
                    interpreter.set_tensor(ind["index"], tensor)
                    interpreter.invoke()
                    raw = interpreter.get_tensor(outd["index"])
                    times.append((time.perf_counter_ns() - started) / 1_000_000)
                logits = np.asarray(raw[0], dtype=np.float32)
                name = f"{key}_threads{threads}_{resolver_name}"
                configs[name] = {
                    "model": key,
                    "num_threads": threads,
                    "op_resolver": resolver_name,
                    "input": {"shape": [int(x) for x in ind["shape"]], "dtype": str(ind["dtype"]), "scale": float(ind["quantization"][0]), "zero_point": int(ind["quantization"][1])},
                    "output": {"shape": [int(x) for x in outd["shape"]], "dtype": str(outd["dtype"]), "scale": float(outd["quantization"][0]), "zero_point": int(outd["quantization"][1])},
                    "times_set_invoke_get_tensor": summarize(times),
                    "output_logits_sha256": hashlib.sha256(logits.astype("<f4").tobytes()).hexdigest(),
                    "output_top1_argmax": int(np.argmax(logits)),
                    "output_logits_abs_max": float(np.max(np.abs(logits))),
                }

    report = {
        "status": "completed",
        "scope": "classifier-only measured repeated fixed-input benchmark; no target-device extrapolation",
        "fixed_input": {
            "clip": str(args.clip),
            "clip_sha256": sha256(args.clip),
            "feature_shape": [int(x) for x in tensor.shape],
            "feature_dtype": str(tensor.dtype),
            "feature_sha256": fixed_sha,
            "frontend": "existing landmark_frontend.py + features.py; identical tensor reused for every config",
        },
        "models": {
            "fp16": {"path": str(FP16), "sha256": FP16_SHA, "bytes": FP16.stat().st_size, "input_dtype": in_dtypes.get("fp16"), "output_dtype": out_dtypes.get("fp16")},
            "int8": {"path": str(INT8), "sha256": INT8_SHA, "bytes": INT8.stat().st_size, "input_dtype": in_dtypes.get("int8"), "output_dtype": out_dtypes.get("int8")},
            "detector": {"path": str(DETECTOR), "sha256": sha256(DETECTOR), "bytes": DETECTOR.stat().st_size},
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "versions": versions,
            "host_note": "measured on the current validation host; no target-device claim",
        },
        "configuration": {
            "repeats": args.repeats,
            "warmup": args.warmup,
            "serialized": True,
            "fixed_input_reused": True,
            "timing_boundary": "set_tensor + invoke + get_tensor only; excludes decode, tracking, feature preparation, and softmax",
            "delegate_note": "AUTO is the normal LiteRT path and emits XNNPACK delegate logs; BUILTIN_WITHOUT_DEFAULT_DELEGATES is a resolver-only comparison and may still have runtime-specific behavior.",
        },
        "configs": configs,
        "footprint": {
            "classifier_saving_bytes": FP16.stat().st_size - INT8.stat().st_size,
            "classifier_saving_percent": (FP16.stat().st_size - INT8.stat().st_size) / FP16.stat().st_size * 100,
            "fp16_plus_detector_bytes": FP16.stat().st_size + DETECTOR.stat().st_size,
            "int8_plus_detector_bytes": INT8.stat().st_size + DETECTOR.stat().st_size,
        },
        "recorded_at": datetime.now(timezone.utc).isoformat(),
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
    print(json.dumps({"output": str(args.output), "configs": list(configs), "fp16_plus_detector": report["footprint"]["fp16_plus_detector_bytes"], "int8_plus_detector": report["footprint"]["int8_plus_detector_bytes"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
