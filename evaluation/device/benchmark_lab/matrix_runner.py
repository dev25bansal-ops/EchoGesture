"""Matrix runner CLI — one heavy process at a time, no install.

Usage:
  python matrix_runner.py /path/to/manifest.csv --output-dir /path/out \
      --predictor src/main.py --labels models/landmark_labels.json \
      --model models/landmark_classifier_fp16.tflite \
      --feature-config src/features.py \
      --repeats 1 --warmup 1 [--no-profile] \
      --same-clip-id ID     # optional identical-clip stability cell

When --same-clip-id is omitted, one fresh-manifest cell is run.
Adds a second identical-clip cell when provided (warmup cells are discarded).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matrix  # noqa: E402
import run_manifest  # noqa: E402


def main(argv=None):
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--predictor", type=Path, default=root / "src/main.py")
    parser.add_argument("--labels", type=Path, default=root / "models/landmark_labels.json")
    parser.add_argument("--model", type=Path, default=root / "models/landmark_classifier_fp16.tflite")
    parser.add_argument("--feature-config", type=Path, default=root / "src/features.py")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--no-profile", action="store_true")
    parser.add_argument("--same-clip-id", default=None)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--dataset", default="ASL Citizen (30-clip validation batch)")
    parser.add_argument("--hardware-note",
                        default="Mac laptop (Apple M5 Pro, macOS 26.3 arm64) — NOT target device")
    args = parser.parse_args(argv)

    try:
        items = run_manifest.read_manifest(args.manifest.resolve())
        labels = run_manifest.load_labels(args.labels)
        unknown = sorted({t for _, _, t, *_ in items if t} - labels)
        if unknown:
            raise run_manifest.EvaluationError(f"Ground truth outside vocabulary: {unknown}")
        for art in (args.predictor, args.labels, args.model):
            if not art.is_file():
                raise run_manifest.EvaluationError(f"Artifact not found: {art}")
        predict = run_manifest.load_predictor(args.predictor)
        mod_name = getattr(predict, "__module__", None)

        cells = []
        cells.append(matrix.run_cell(
            items, predict, labels,
            matrix.MatrixCellConfig(
                name="fresh_manifest", profile=not args.no_profile,
                repeats=args.repeats, warmup=args.warmup,
            ),
            predict_module_name=mod_name,
        ))
        if args.same_clip_id:
            cells.append(matrix.run_cell(
                items, predict, labels,
                matrix.MatrixCellConfig(
                    name="identical_clip_stability", profile=not args.no_profile,
                    repeats=max(5, args.repeats), warmup=args.warmup,
                    same_clip_id=args.same_clip_id,
                    note="Identical-clip repeats for median/IQR stability only; not a target-device p95.",
                ),
                predict_module_name=mod_name,
            ))

        artifact_paths = [("predictor", args.predictor), ("labels", args.labels), ("model", args.model)]
        if args.feature_config:
            artifact_paths.append(("feature_config", args.feature_config))
        artifacts = {name: {"path": str(p.resolve()), "sha256": run_manifest.sha256(p),
                            "bytes": p.stat().st_size} for name, p in artifact_paths}

        summary = matrix.build_summary(
            cells, manifest_path=args.manifest.resolve(), artifact_paths=artifacts,
            fixture_type="real", split=args.split, dataset=args.dataset,
            hardware_note=args.hardware_note,
            code_commit=matrix._git_sha(root),
        )
        # write without predictions.jsonl duplication: matrix cells already hold wall times;
        # also write predictions list for audit
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "benchmark_matrix.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
    except (run_manifest.EvaluationError, OSError, ValueError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
