#!/usr/bin/env python3
"""Verify the real predict() loader boundary without modifying it.

The selected default is read from the live runtime. The INT8 candidate is then
loaded through an in-process override of `_model_paths` only, proving whether
its tensor contract is accepted or rejected by the unmodified loader.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))

import main  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def base_state() -> dict:
    backend, model, labels = main._model_paths()
    return {"backend": backend, "model": str(model), "labels": str(labels),
            "model_sha256": sha256(model)}


def reset() -> None:
    main._INTERPRETER = None
    main._ACTIVE_MODEL = None
    main._INPUT_DETAILS = main._OUTPUT_DETAILS = None


def snapshot() -> dict:
    return {
        "backend": main._MODEL_BACKEND,
        "frames": int(main._FRAMES),
        "input": {"shape": main._INPUT_DETAILS["shape"].astype(int).tolist(),
                  "dtype": main._INPUT_DETAILS["dtype"].__name__,
                  "scale": float(main._INPUT_DETAILS["quantization"][0]),
                  "zero_point": int(main._INPUT_DETAILS["quantization"][1])},
        "output": {"shape": main._OUTPUT_DETAILS["shape"].astype(int).tolist(),
                   "dtype": main._OUTPUT_DETAILS["dtype"].__name__,
                   "scale": float(main._OUTPUT_DETAILS["quantization"][0]),
                   "zero_point": int(main._OUTPUT_DETAILS["quantization"][1])},
        "labels": len(main._IDX_TO_LABEL),
    }


def probe() -> dict:
    rng = np.random.default_rng(20261110)
    features = rng.normal(0, .3, (1, 24, 255)).astype(np.float32)
    main._INTERPRETER.set_tensor(main._INPUT_DETAILS["index"], features)
    main._INTERPRETER.invoke()
    raw = main._INTERPRETER.get_tensor(main._OUTPUT_DETAILS["index"])[0].astype(np.float32)
    if main._OUTPUT_SCALE > 0:
        raw = (raw - main._OUTPUT_ZP) * main._OUTPUT_SCALE
    return {"finite": bool(np.isfinite(raw).all()), "shape": list(raw.shape),
            "min": float(raw.min()), "max": float(raw.max()), "sum": float(raw.sum())}


def main_entry(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "results/runtime_boundary.json")
    args = parser.parse_args(argv)
    selected = base_state()
    reset()
    main._init_model()
    default_result = snapshot() | {"probe": probe()}
    int8 = ROOT / "models/landmark_classifier_int8.tflite"
    original = main._model_paths
    try:
        main._model_paths = lambda: ("landmark", int8, ROOT / "models/landmark_labels.json")
        reset()
        main._init_model()
        int8_result = snapshot() | {"probe": probe()}
        int8_error = None
    except Exception as exc:
        int8_result = None
        int8_error = f"{type(exc).__name__}: {exc}"
    finally:
        main._model_paths = original
        reset()
        main._init_model()
    report = {"selected": selected, "default_loader": default_result,
              "int8_loaded_without_changing_loader": int8_result,
              "int8_rejection": int8_error,
              "int8_io_is_drop_in": bool(int8_result and int8_result["input"]["dtype"] == "float32"
                                         and int8_result["output"]["dtype"] == "float32"),
              "default_restored": snapshot(),
              "runtime_sha256": sha256(ROOT / "src/main.py")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=args.output.parent,
                                     prefix=args.output.name + ".", delete=False) as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(args.output)
    print(json.dumps({key: value for key, value in report.items()
                      if key in ("selected", "default_loader", "int8_loaded_without_changing_loader",
                                 "int8_rejection", "int8_io_is_drop_in")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main_entry())