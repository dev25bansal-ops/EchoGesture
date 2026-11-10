"""Paired FP16/INT8 benchmark runner for EchoGesture V7 (laptop harness + target-device protocol).

Compares FP16 vs INT8 full-clip p50/p95 ONLY on factual, identical inputs:

- Same manifest, same clips, same order, same predictor (src/main.py) — the only
  difference between variants is which classifier file the benchmark-only
  wrapper points `_model_paths()` at (see predict_model.py).
- Both candidate files are hash-pinned before any run; a mismatch aborts the run.
- Timing comes from `matrix.run_cell` (nearest-rank percentiles, external wall
  time via perf_counter_ns, warmup discarded, cold/warm split, failures reported
  separately).  No p95, power or thermal numbers are fabricated.
- INT8 frontend compatibility is checked up-front (see README.md): the INT8
  file's input/output tensors are float32 with the identical [1,24,255]->[1,100]
  signature and the same landmark_labels.json vocabulary, so the production
  `src/main.py` predictor drives both variants with no frontend change.

Usage (isolated-process protocol: one heavy process at a time):
    # Orchestrated paired run (runs FP16 then INT8 as separate subprocesses)
    python paired_runner.py --manifest /tmp/echogesture-validation-30/manifest.csv \
        --output-dir results/laptop_run --source-root /path/to/checkout

    # Single-variant cell executed by the orchestrator (normally not called directly)
    python paired_runner.py --manifest /tmp/.../manifest.csv --output-dir results/laptop_run \
        --variant fp16 --cell-out results/laptop_run/fp16_cell.json \
        --source-root /path/to/checkout

Each variant cell JSON is written atomically (tempfile + os.replace).  The
merged `paired.json` in --output-dir joins the two cells by clip_id and reports
per-variant latency, stage aggregates, and paired agreement / delta metrics.
`hardware_is_target_device` is False unless --hardware-is-target is explicitly
passed together with a --device-model string; laptop numbers are therefore
never presentable as target-device evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "benchmark_lab"))

import predict_model  # noqa: E402
import run_manifest  # noqa: E402
import matrix  # noqa: E402

CANDIDATE_SHA256 = dict(predict_model.VARIANT_SHA256)
CANDIDATE_FILES = dict(predict_model.VARIANT_FILES)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_artifact_sha256(path: Path, expected: str, label: str) -> str:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(
            f"{label} hash mismatch: expected {expected}, got {actual} ({path}). Refusing to benchmark."
        )
    return actual


def estimate_run_seconds(clips: int, per_clip_ms: float = 230.0) -> float:
    """Rough harness time hint for protocol output (not a measurement)."""
    return clips * per_clip_ms / 1000.0


def _accepts_clip_id(predict) -> bool:
    try:
        param = inspect.signature(predict).parameters.get("clip_id")
        return param is not None and param.kind is inspect.Parameter.KEYWORD_ONLY
    except (TypeError, ValueError):
        return False


def _load_predictor(source_root: Path, variant: str):
    model_dir = source_root / "models"
    for var, filename in CANDIDATE_FILES.items():
        path = model_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Candidate {var} file missing: {path}")
    model_path = model_dir / CANDIDATE_FILES[variant]
    check_artifact_sha256(model_path, CANDIDATE_SHA256[variant], f"{variant} classifier")
    return predict_model.load_predictor(source_root, variant)


def run_variant_cell(items, predict, labels, variant: str, *, warmup: int = 1):
    """Run one fresh-manifest cell (each clip exactly once, in manifest order).

    Mirrors matrix.run_cell's wall-time / profile / failure bookkeeping but also
    keeps per-clip predictions so FP16/INT8 can be paired by clip_id.  Warmup
    calls are discarded; identical-clip repeats are not used here.
    """
    if not items:
        raise ValueError("Manifest contains no clips")
    accepts_clip = _accepts_clip_id(predict)
    module_name = getattr(predict, "__module__", None)

    def _call(cid, cpath):
        kwargs = {"profile": True}
        if accepts_clip:
            kwargs["clip_id"] = cid
        return predict(str(cpath), **kwargs)

    for i in range(warmup):
        cid, cpath, _, *_ = items[i % len(items)]
        try:
            _call(cid, cpath)
        except Exception:
            pass  # warmup failures ignored by design (discarded cell)

    wall_ms, profiles, failures, per_clip = [], [], [], []
    correct = total_labelled = 0
    for cid, cpath, truth, *_ in items:
        started = time.perf_counter_ns()
        try:
            value = _call(cid, cpath)
            validated = run_manifest.validate_prediction(value, cid, labels)
        except Exception as exc:  # keep parity with matrix.run_cell: report, never drop silently
            failures.append({"clip_id": cid, "error_type": type(exc).__name__, "error": str(exc)})
            continue
        elapsed = (time.perf_counter_ns() - started) / 1e6
        wall_ms.append(elapsed)
        per_clip.append({
            "clip_id": cid,
            "wall_ms": elapsed,
            "prediction": validated["prediction"],
            "confidence": validated["confidence"],
            "top5": validated.get("top5", []),
            "ground_truth": truth or None,
        })
        getter = None
        if module_name:
            getter = getattr(sys.modules.get(module_name), "get_last_profile", None)
        if callable(getter):
            sp = getter()
            if isinstance(sp, dict) and "stages" in sp:
                profiles.append({"clip_id": cid, **sp})
        if truth:
            total_labelled += 1
            correct += validated["prediction"] == truth

    cold, warm = matrix._split_cold_warm(wall_ms)
    stage_agg, stage_extras = matrix._aggregate_stage_profiles(profiles)
    rep = len(per_clip) >= 20 and not failures and total_labelled > 0
    return {
        "cell": f"fresh_manifest_{variant}",
        "variant": variant,
        "config": {
            "repeats": 1,
            "warmup": warmup,
            "same_clip_id": None,
            "profile": True,
            "repeat_mode": "fresh_manifest",
            "note": "Each clip exactly once, manifest order; identical-clip repeats not used for p95.",
        },
        "clip_count": len(items),
        "successful_predictions": len(per_clip),
        "failures": failures,
        "failed_predictions": failures,
        "accuracy": {
            "clips_with_ground_truth": sum(1 for _, _, t, *_ in items if t),
            "correct": correct if total_labelled else 0,
            "top1": (correct / total_labelled) if total_labelled and not failures else None,
            "representative_accuracy_measured": rep,
            "qualification": (
                "Representative accuracy requires a documented validation split; "
                "flag is only completeness/size check"
            ),
            "note": f"Accuracy computed over {total_labelled} labelled successful calls",
        },
        "latency_all_calls": matrix._timing_summary(wall_ms),
        "cold_call": matrix._timing_summary(cold),
        "warm_calls": matrix._timing_summary(warm),
        "stage_aggregates": stage_agg,
        "stage_extras": stage_extras,
        "profiles_count": len(profiles),
        "all_wall_ms": wall_ms,
        "per_clip": per_clip,
    }


def _write_json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            Path(tmp).unlink(missing_ok=True)
        except OSError:
            pass
        raise


def compute_paired(cell_a: dict, cell_b: dict) -> dict:
    """Pair two fresh-manifest cells by clip_id; compute deltas and agreement.

    Raises if clip sets differ — a FP16/INT8 comparison on non-identical inputs
    would be meaningless, so it is refused rather than reported.
    """
    a = {r["clip_id"]: r for r in cell_a["per_clip"]}
    b = {r["clip_id"]: r for r in cell_b["per_clip"]}
    if set(a) != set(b):
        only_a = sorted(set(a) - set(b))
        only_b = sorted(set(b) - set(a))
        raise ValueError(f"Clip sets differ (only in first={only_a}, only in second={only_b}); refusing comparison")
    ids = sorted(a)
    n = len(ids)
    if n == 0:
        # No successes on either side: report empty-instead-of-fabricated.
        return {
            "n": 0,
            "clip_ids": [],
            "paired_wall_ms": [],
            "prediction_agreement": None,
            "top5_overlap_mean": None,
            "wall_delta_ms": None,
            "wall_delta_pct_of_fp16": None,
        }

    def _pct(vals):
        if not vals:
            return None
        s = sorted(vals)
        return s[max(0, math.ceil(0.5 * len(s)) - 1)]

    def _p95(vals):
        if not vals:
            return None
        s = sorted(vals)
        return s[max(0, math.ceil(0.95 * len(s)) - 1)]

    agree = sum(1 for cid in ids if a[cid]["prediction"] == b[cid]["prediction"])
    overlaps = []
    deltas = []
    ratios = []
    for cid in ids:
        ta, tb = a[cid].get("top5") or [], b[cid].get("top5") or []
        if ta and tb:
            overlaps.append(len(set(ta) & set(tb)) / min(len(ta), len(tb)))
        d = b[cid]["wall_ms"] - a[cid]["wall_ms"]
        deltas.append(d)
        if a[cid]["wall_ms"]:
            ratios.append(100.0 * d / a[cid]["wall_ms"])

    return {
        "n": n,
        "clip_ids": ids,
        "paired_wall_ms": [{"clip_id": cid, "first_ms": a[cid]["wall_ms"], "second_ms": b[cid]["wall_ms"]} for cid in ids],
        "prediction_agreement": agree / n if n else None,
        "top5_overlap_mean": statistics.mean(overlaps) if overlaps else None,
        "wall_delta_ms": {
            "p50": _pct(deltas),
            "p95": _p95(deltas),
            "mean": statistics.mean(deltas),
        },
        "wall_delta_pct_of_first": {
            "p50": _pct(ratios),
            "p95": _p95(ratios),
            "mean": statistics.mean(ratios) if ratios else None,
        },
        "per_variant_latency_all_calls": {
            "first": cell_a["latency_all_calls"],
            "second": cell_b["latency_all_calls"],
        },
        "stage_aggregate_delta_ms": {
            stage: {
                "first_mean": (cell_a["stage_aggregates"].get(stage) or {}).get("mean_ms"),
                "second_mean": (cell_b["stage_aggregates"].get(stage) or {}).get("mean_ms"),
            }
            for stage in sorted(set(cell_a["stage_aggregates"]) | set(cell_b["stage_aggregates"]))
        },
    }


def build_paired_summary(cells: dict, *, manifest_path: Path, artifacts: dict,
                         hardware_note: str, hardware_is_target: bool,
                         device_model: str | None, fixture_type: str = "real",
                         split: str = "validation",
                         dataset: str = "ASL Citizen (30-clip validation batch)"):
    """Assemble the paired FP16/INT8 top-level summary (mirrors matrix.build_summary fields)."""
    first = cells["fp16"]
    second = cells["int8"]
    paired = compute_paired(first, second)
    manifest_sha = sha256(Path(manifest_path))
    all_ok = not first["failures"] and not second["failures"]
    return {
        "status": "passed_schema" if all_ok else "completed_with_failures",
        "representative_performance_measured": bool(
            first["latency_all_calls"]["count"] >= 20
            and second["latency_all_calls"]["count"] >= 20
            and not first["failures"] and not second["failures"]
        ),
        "performance_qualification": (
            "N>=20 successful real clips per variant with zero failures on identical inputs; "
            "still NOT target-device evidence unless hardware_is_target_device is true and "
            "the target-device conditions in TARGET_DEVICE.md are recorded."
        ),
        "fixture_type": fixture_type,
        "split": split,
        "dataset": dataset,
        "hardware_is_target_device": bool(hardware_is_target),
        "target_device_status": "measured" if hardware_is_target else "blocked_not_target_hardware",
        "device_model": device_model,
        "hardware_note": hardware_note,
        "cells": [first, second],
        "paired": paired,
        "artifacts": artifacts,
        "manifest": {"path": str(Path(manifest_path).resolve()), "sha256": manifest_sha},
        "power": "NOT_MEASURED; requires a documented external meter or qualified platform method — do not infer from CPU",
        "thermal": "NOT_MEASURED unless a sustained CSV (>=600s) analyzed by thermal_report.py is attached for this run",
        "code_commit": matrix._git_sha(Path(__file__).resolve().parents[3]),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "device_model": platform.node(),
            "inference_runtime": run_manifest._runtime_version(),
            "delegate": "XNNPACK CPU (LiteRT)",
            "thread_count": "runtime default; not explicitly configured",
        },
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Runtime/interface measurement on the recorded hardware only; not licensing/accuracy compliance evidence",
        "comparison_ri_note": "FP16/INT8 deltas are paired by clip_id on identical inputs; they are latency/accuracy-delta measurements, not a requalification of either model.",
    }


def _resolve_paths(source_root: Path, labels_name: str = "landmark_labels.json"):
    models = source_root / "models"
    return models / "landmark_labels.json"


def _run_single_variant(args) -> int:
    source_root = args.source_root.resolve()
    manifest_path = args.manifest.resolve()
    items = run_manifest.read_manifest(manifest_path)
    labels_path = _resolve_paths(source_root)
    if not labels_path.is_file():
        raise run_manifest.EvaluationError(f"Labels not found: {labels_path}")
    labels = run_manifest.load_labels(labels_path)
    unknown = sorted({t for _, _, t, *_ in items if t} - labels)
    if unknown:
        raise run_manifest.EvaluationError(f"Ground truth outside vocabulary: {unknown}")
    predict = _load_predictor(source_root, args.variant)
    cell = run_variant_cell(items, predict, labels, args.variant, warmup=args.warmup)
    _write_json_atomic(Path(args.cell_out), cell)
    print(json.dumps({
        "variant": args.variant,
        "count": cell["latency_all_calls"]["count"],
        "p50_ms": cell["latency_all_calls"]["p50_ms"],
        "p95_ms": cell["latency_all_calls"]["p95_ms"],
        "failures": len(cell["failures"]),
    }, indent=2))
    return 0


def _run_paired(args) -> int:
    source_root = args.source_root.resolve()
    manifest_path = args.manifest.resolve()
    output_dir = args.output_dir
    # Artifact hashes (models, labels, predictor, matrix, run_manifest, predict_model).
    models = source_root / "models"
    artifacts = {}
    for var in ("fp16", "int8"):
        p = models / CANDIDATE_FILES[var]
        check_artifact_sha256(p, CANDIDATE_SHA256[var], f"{var} classifier")
        artifacts[f"model_{var}"] = {"path": str(p.resolve()), "sha256": sha256(p), "bytes": p.stat().st_size}
    labels_path = models / "landmark_labels.json"
    if not labels_path.is_file():
        raise FileNotFoundError(f"Labels not found: {labels_path}")
    artifacts["labels"] = {"path": str(labels_path.resolve()), "sha256": sha256(labels_path), "bytes": labels_path.stat().st_size}
    for name, rel in (("predictor", "src/main.py"),
                      ("run_manifest", "evaluation/device/run_manifest.py"),
                      ("matrix", "evaluation/device/benchmark_lab/matrix.py"),
                      ("predict_model", "evaluation/device/v7_benchmark/predict_model.py")):
        p = source_root / rel
        if not p.is_file():
            # Source-root may be the original checkout; fall back to local copies.
            p = HERE / Path(rel).name
        if p.is_file():
            artifacts[name] = {"path": str(p.resolve()), "sha256": sha256(p), "bytes": p.stat().st_size}

    # Run each variant in its own subprocess (isolated model load, one heavy process at a time).
    cells = {}
    for variant in ("fp16", "int8"):
        cell_out = output_dir / f"{variant}_cell.json"
        cmd = [
            sys.executable, "-I", str(Path(__file__).resolve()),
            str(manifest_path),
            "--output-dir", str(output_dir),
            "--source-root", str(source_root),
            "--variant", variant,
            "--cell-out", str(cell_out),
            "--warmup", str(args.warmup),
        ]
        print(f"[paired_runner] running {variant} ...", flush=True)
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout, end="")
        if result.returncode != 0:
            print("STDOUT:", result.stdout, file=sys.stderr)
            print("STDERR:", result.stderr, file=sys.stderr)
            raise run_manifest.EvaluationError(f"{variant} cell failed with exit {result.returncode}")
        cells[variant] = json.loads(cell_out.read_text(encoding="utf-8"))

    if args.hardware_is_target and not args.device_model:
        raise run_manifest.EvaluationError("--hardware-is-target requires --device-model")
    if not args.hardware_is_target:
        needle = args.hardware_note.lower()
        if "not target device" not in needle and "not_target" not in needle:
            raise run_manifest.EvaluationError(
                "Non-target hardware runs must record hardware_note containing 'NOT target device'"
            )

    summary = build_paired_summary(
        cells,
        manifest_path=manifest_path,
        artifacts=artifacts,
        hardware_note=args.hardware_note,
        hardware_is_target=args.hardware_is_target,
        device_model=args.device_model if args.hardware_is_target else None,
        fixture_type=args.fixture_type,
        split=args.split,
        dataset=args.dataset,
    )
    _write_json_atomic(output_dir / "paired.json", summary)
    print(json.dumps({
        "status": summary["status"],
        "fp16_p50_ms": summary["paired"]["per_variant_latency_all_calls"]["first"]["p50_ms"],
        "fp16_p95_ms": summary["paired"]["per_variant_latency_all_calls"]["first"]["p95_ms"],
        "int8_p50_ms": summary["paired"]["per_variant_latency_all_calls"]["second"]["p50_ms"],
        "int8_p95_ms": summary["paired"]["per_variant_latency_all_calls"]["second"]["p95_ms"],
        "target_device_status": summary["target_device_status"],
    }, indent=2))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[3],
                        help="Checkout root containing src/main.py and models/")
    parser.add_argument("--variant", choices=("fp16", "int8"), default=None,
                        help="Single-variant mode (used by the orchestrator subprocess)")
    parser.add_argument("--cell-out", type=Path, default=None, help="Cell JSON path (single-variant mode)")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--hardware-note",
                        default="Mac laptop (Apple M5 Pro, macOS 26.3 arm64) — NOT target device")
    parser.add_argument("--hardware-is-target", action="store_true",
                        help="Set only when the run hardware IS the actual target device")
    parser.add_argument("--device-model", default=None,
                        help="Exact target device model; required with --hardware-is-target")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--dataset", default="ASL Citizen (30-clip validation batch)")
    parser.add_argument("--fixture-type", choices=("real", "synthetic"), default="real")
    args = parser.parse_args(argv)

    try:
        if args.variant:
            if args.cell_out is None:
                raise run_manifest.EvaluationError("--variant requires --cell-out")
            return _run_single_variant(args)
        return _run_paired(args)
    except (run_manifest.EvaluationError, OSError, ValueError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
