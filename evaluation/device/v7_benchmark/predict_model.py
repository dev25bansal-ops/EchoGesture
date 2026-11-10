"""Benchmark-only predictor wrapper: selects FP16 or INT8 without changing production defaults.

It imports the real src/main.py from a chosen checkout, pins the backend to
"landmark", and overrides only `_model_paths` to point at the requested
classifier. `predict()` and `get_last_profile()` are re-exported unchanged so
`run_manifest.evaluate_profiled` / benchmark_lab `matrix.run_cell` keep working.

Usage from the paired runner (or directly):
    python predict_model.py /tmp/echogesture-validation-30/<clip>.mp4 \
        --variant int8 --source-root /path/to/checkout

No production model default is changed: the wrapper only affects this process.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

VARIANT_FILES = {
    "fp16": "landmark_classifier_fp16.tflite",
    "int8": "landmark_classifier_int8.tflite",
}

# Candidate pins for the V7 benchmark (validated by the runner, not trusted here).
VARIANT_SHA256 = {
    "fp16": "a012f2b5f7a1110961079f19eb0e28c9abacb95e4d0787352e4157ebca7044a4",
    "int8": "9d24640a23244a3cd00c9fa62257733819f86ed1da5a10194c9b1cb64323700b",
}


def load_predictor(source_root: Path, variant: str):
    """Load src/main.py from source_root and pin the classifier variant.

    Returns the predict callable; its __module__ exposes get_last_profile().
    """
    if variant not in VARIANT_FILES:
        raise ValueError(f"Unknown variant {variant!r}; expected one of {sorted(VARIANT_FILES)}")
    main_path = source_root.resolve() / "src" / "main.py"
    if not main_path.is_file():
        raise FileNotFoundError(f"Predictor not found: {main_path}")
    os.environ["ECHOGESTURE_MODEL"] = "landmark"
    spec = importlib.util.spec_from_file_location(f"echogesture_v7_bench_{variant}", main_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not callable(getattr(module, "predict", None)):
        raise RuntimeError(f"{main_path} does not expose predict()")

    def _model_paths():
        base_dir = main_path.resolve().parent.parent / "models"
        return "landmark", base_dir / VARIANT_FILES[variant], base_dir / "landmark_labels.json"

    # Benchmark-only override of model selection. Production src/main.py is untouched.
    module._model_paths = _model_paths
    return module.predict


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--variant", choices=sorted(VARIANT_FILES), required=True)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args(argv)
    predict = load_predictor(args.source_root, args.variant)
    result = predict(str(args.video), profile=args.profile)
    print(json.dumps(result, indent=2))
    if args.profile:
        getter = getattr(sys.modules[predict.__module__], "get_last_profile", None)
        if callable(getter):
            print(json.dumps(getter(), indent=2))


if __name__ == "__main__":
    main()
