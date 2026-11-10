"""Run the submission predictor for every clip in a CSV manifest.

The manifest requires clip_id,path columns and may include ground_truth.
Relative clip paths are resolved from the manifest's directory and MUST stay inside an allowed input root (manifest directory by default; see --input-root / ECHOGESTURE_INPUT_ROOT for judge layouts). Paths escaping every allowed root are rejected.
"""

import argparse
import csv
import hashlib
import importlib.util
import inspect
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

REQUIRED_FIELDS = ("clip_id", "path")
OUTPUT_FIELDS = ("clip_id", "prediction", "confidence", "latency_ms")

_ENV_INPUT_ROOT = "ECHOGESTURE_INPUT_ROOT"
_GENERIC_DEVICE_MODEL = "generic-device"


def _is_within(child, parent):
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _parse_allowed_roots(allowed_roots):
    raw = []
    if allowed_roots is None:
        pass
    elif isinstance(allowed_roots, (str, os.PathLike)):
        raw.append(allowed_roots)
    else:
        raw.extend(allowed_roots)
    env_value = os.environ.get(_ENV_INPUT_ROOT, "") or ""
    if env_value.strip():
        raw.extend(p for p in (p.strip() for p in env_value.split(os.pathsep)) if p)
    seen = []
    out = []
    for entry in raw:
        candidate = Path(entry).expanduser().resolve()
        if candidate not in seen:
            seen.append(candidate)
            out.append(candidate)
    return out


def _effective_roots(manifest_path, allowed_roots=None):
    manifest_dir = Path(manifest_path).expanduser().resolve().parent
    roots = _parse_allowed_roots(allowed_roots)
    if manifest_dir not in roots:
        roots = [manifest_dir, *roots]
    return roots


def _resolve_clip(relpath, manifest_path, allowed_roots, line):
    if "\x00" in relpath:
        raise EvaluationError(f"Manifest line {line}: clip path contains NUL byte")
    expanded = Path(relpath).expanduser()
    manifest_dir = Path(manifest_path).expanduser().resolve().parent
    candidate = expanded if expanded.is_absolute() else manifest_dir / expanded
    resolved = candidate.resolve()
    roots = _effective_roots(manifest_path, allowed_roots)
    for root in roots:
        if _is_within(resolved, root) and resolved != root:
            return resolved
    joined = ", ".join(str(r) for r in roots)
    raise EvaluationError(
        f"Manifest line {line}: clip path escapes allowed input root: {relpath!r} "
        f"(resolved={resolved}; allowed roots: {joined})"
    )


def _sanitized_relative(path, root):
    # Repo-relative only when the artifact genuinely lives under the
    # repo; otherwise basename. _ABS_LEAK_RE-safe: no /private// /tmp/
    # /Users/ segments, no drive prefix, no hostname.
    try:
        rel = Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    except (ValueError, OSError):
        return Path(str(path).replace(chr(92), "/")).name
    first = rel.split("/", 1)[0] if rel else ""
    if first in ("private", "tmp", "var", "Users", "home") or Path(rel).is_absolute():
        return Path(str(path).replace(chr(92), "/")).name
    return rel


class EvaluationError(Exception):
    pass


def load_predictor(path):
    path = path.resolve()
    if not path.is_file():
        raise EvaluationError(f"Predictor not found: {path}")
    spec = importlib.util.spec_from_file_location("echogesture_predictor", path)
    if spec is None or spec.loader is None:
        raise EvaluationError(f"Cannot import predictor: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(spec.name, None)
        raise EvaluationError(f"Cannot import {path}: {exc}") from exc
    if not callable(getattr(module, "predict", None)):
        raise EvaluationError(f"Missing callable predict(video_path): {path}")
    return module.predict


def load_labels(path):
    try:
        labels = {label.upper() for label in json.loads(path.read_text(encoding="utf-8"))}
    except (OSError, ValueError, AttributeError, TypeError) as exc:
        raise EvaluationError(f"Cannot read vocabulary {path}: {exc}") from exc
    if not labels:
        raise EvaluationError(f"Empty vocabulary: {path}")
    return labels


def read_manifest(path, allowed_roots=None):
    try:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            missing = [field for field in REQUIRED_FIELDS if field not in (reader.fieldnames or [])]
            if missing:
                raise EvaluationError(f"Manifest missing columns: {', '.join(missing)}")
            rows = list(reader)
    except (OSError, csv.Error) as exc:
        raise EvaluationError(f"Cannot read manifest {path}: {exc}") from exc
    if not rows:
        raise EvaluationError("Manifest contains no clips")
    seen = set()
    items = []
    for line, row in enumerate(rows, start=2):
        clip_id = row.get("clip_id") or ""
        relative_path = (row.get("path") or "").strip()
        if not clip_id.strip() or not relative_path:
            raise EvaluationError(f"Manifest line {line}: clip_id and path are required")
        if clip_id in seen:
            raise EvaluationError(f"Manifest line {line}: duplicate clip_id {clip_id!r}")
        seen.add(clip_id)
        try:
            clip_path = _resolve_clip(relative_path, path, allowed_roots, line)
        except EvaluationError:
            raise
        if not clip_path.is_file():
            raise EvaluationError(f"Manifest line {line}: clip not found: {clip_path}")
        signer = (row.get("signer_id") or "").strip()
        truth = (row.get("ground_truth") or "").strip().upper()
        items.append((clip_id, clip_path, truth, signer))
    return items


def validate_prediction(value, clip_id, labels):
    if not isinstance(value, dict):
        raise EvaluationError(f"{clip_id}: predict() must return a JSON object")
    missing = [field for field in OUTPUT_FIELDS if field not in value]
    if missing:
        raise EvaluationError(f"{clip_id}: missing output fields: {', '.join(missing)}")
    if value["clip_id"] != clip_id:
        raise EvaluationError(f"{clip_id}: returned clip_id {value['clip_id']!r}")
    for field in ("prediction", "confidence", "latency_ms"):
        if isinstance(value[field], bool):
            raise EvaluationError(f"{clip_id}: invalid {field}")
    prediction = value["prediction"]
    if not isinstance(prediction, str) or prediction != prediction.upper() or prediction not in labels:
        raise EvaluationError(f"{clip_id}: prediction is not an uppercase vocabulary label")
    confidence = value["confidence"]
    latency = value["latency_ms"]
    if not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise EvaluationError(f"{clip_id}: confidence must be finite and in [0, 1]")
    if not isinstance(latency, (int, float)) or not math.isfinite(latency) or latency < 0:
        raise EvaluationError(f"{clip_id}: latency_ms must be finite and nonnegative")
    if "top5" in value:
        top5 = value["top5"]
        if (not isinstance(top5, list) or not 1 <= len(top5) <= 5
                or any(not isinstance(label, str) or label != label.upper() or label not in labels
                       for label in top5)
                or len(set(top5)) != len(top5) or top5[0] != prediction):
            raise EvaluationError(f"{clip_id}: invalid top5 vocabulary, ranking or length")
    return value


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    rank = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[rank]


def timings(values):
    return {
        "count": len(values),
        "p50_ms": percentile(values, 0.50),
        "p90_ms": percentile(values, 0.90),
        "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99),
        "min_ms": min(values) if values else None,
        "max_ms": max(values) if values else None,
    }


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_sha(root):
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def write_pair_atomically(directory, predictions, summary):
    directory.mkdir(parents=True, exist_ok=True)
    temp_paths = []
    try:
        for name, text in (
            ("predictions.jsonl", "".join(json.dumps(value, allow_nan=False, ensure_ascii=False) + "\n" for value in predictions)),
            ("benchmark.json", json.dumps(summary, indent=2, ensure_ascii=False) + "\n"),
        ):
            fd, temp_name = tempfile.mkstemp(prefix=f".{name}.", dir=directory)
            temp_paths.append((Path(temp_name), directory / name))
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
        for source, destination in temp_paths:
            os.replace(source, destination)
    finally:
        for source, _ in temp_paths:
            source.unlink(missing_ok=True)


def _runtime_version():
    try:
        import ai_edge_litert
        return f"ai-edge-litert {ai_edge_litert.__version__}"
    except (ImportError, AttributeError):
        return "unknown"


def _accepts_clip_id(predict):
    try:
        parameter = inspect.signature(predict).parameters.get("clip_id")
        return parameter is not None and parameter.kind is inspect.Parameter.KEYWORD_ONLY
    except (TypeError, ValueError):
        return False


def evaluate(items, predict, labels):
    labelled = {clip_id: truth for clip_id, _, truth, *_ in items if truth}
    if labelled and len(labelled) != len(items):
        raise EvaluationError("Supply ground_truth for every clip or none; partial accuracy is not reported")
    accepts_clip_id = _accepts_clip_id(predict)
    predictions = []
    wall_times = []
    correct = 0
    for clip_id, clip_path, truth, *_ in items:
        started_ns = time.perf_counter_ns()
        try:
            value = predict(str(clip_path), clip_id=clip_id) if accepts_clip_id else predict(str(clip_path))
        except Exception as exc:
            raise EvaluationError(f"{clip_id}: predict() failed: {type(exc).__name__}: {exc}") from exc
        wall_times.append((time.perf_counter_ns() - started_ns) / 1e6)
        predictions.append(validate_prediction(value, clip_id, labels))
        if truth:
            correct += predictions[-1]["prediction"] == truth
    return predictions, wall_times, {
        "clips_with_ground_truth": len(labelled),
        "correct": correct,
        "top1": correct / len(labelled) if labelled else None,
        "representative_accuracy_measured": len(items) >= 20 and bool(labelled),
    }


def evaluate_profiled(items, predict, labels):
    labelled = {clip_id: truth for clip_id, _, truth, *_ in items if truth}
    if labelled and len(labelled) != len(items):
        raise EvaluationError("Supply ground_truth for every clip or none; partial accuracy is not reported")
    if "profile" not in inspect.signature(predict).parameters:
        raise EvaluationError("Predictor does not support profile=True")
    accepts_clip_id = _accepts_clip_id(predict)
    predictions, wall_times, profiles, failures = [], [], [], []
    correct = 0
    for clip_id, clip_path, truth, *_ in items:
        started_ns = time.perf_counter_ns()
        kwargs = {"profile": True}
        if accepts_clip_id:
            kwargs["clip_id"] = clip_id
        try:
            value = predict(str(clip_path), **kwargs)
            predictions.append(validate_prediction(value, clip_id, labels))
        except Exception as exc:
            failures.append({"clip_id": clip_id, "error_type": type(exc).__name__, "error": str(exc)})
            continue
        wall_times.append((time.perf_counter_ns() - started_ns) / 1e6)
        getter = getattr(sys.modules[predict.__module__], "get_last_profile", None)
        if not callable(getter):
            raise EvaluationError("Predictor does not expose get_last_profile()")
        stage_profile = getter()
        if not isinstance(stage_profile, dict) or "stages" not in stage_profile:
            raise EvaluationError(f"{clip_id}: missing valid stage profile")
        profiles.append({"clip_id": clip_id, **stage_profile})
        if truth:
            correct += predictions[-1]["prediction"] == truth
    accuracy = {
        "clips_with_ground_truth": len(labelled),
        "correct": correct,
        "top1": correct / len(labelled) if labelled and not failures else None,
        "representative_accuracy_measured": len(items) >= 20 and not failures and bool(labelled),
        "qualification": "Representative accuracy requires a documented validation split; this flag is only a completeness/size check",
    }
    return predictions, wall_times, accuracy, {"failures": failures, "profiles": profiles}


def main(argv=None):
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--predictor", type=Path, default=root / "src/main.py")
    parser.add_argument("--labels", type=Path, default=root / "models/label100.json")
    parser.add_argument("--model", type=Path, default=root / "models/s3d_conv1d_qat_int8.tflite")
    parser.add_argument("--feature-config", type=Path)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--input-root", action="append", default=[],
                        help="Additional allowed input root for clip paths (repeatable); manifest directory is always allowed. Also via ECHOGESTURE_INPUT_ROOT.")
    parser.add_argument("--fixture-type", choices=("real", "synthetic"), default="real")
    parser.add_argument("--split", default="unspecified")
    parser.add_argument("--dataset", default="unspecified")
    args = parser.parse_args(argv)
    try:
        items = read_manifest(args.manifest.resolve(), allowed_roots=(args.input_root or None))
        labels = load_labels(args.labels)
        unknown = sorted({truth for _, _, truth, *_ in items if truth} - labels)
        if unknown:
            raise EvaluationError(f"Ground truth outside vocabulary: {unknown}")
        for artifact in (args.predictor, args.labels, args.model):
            if not artifact.is_file():
                raise EvaluationError(f"Artifact not found: {artifact}")
        predict = load_predictor(args.predictor)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        evaluator = evaluate_profiled if args.profile else evaluate
        if args.profile:
            predictions, wall_times, accuracy, details = evaluator(items, predict, labels)
        else:
            predictions, wall_times, accuracy = evaluator(items, predict, labels)
            details = {"failures": [], "profiles": []}
        artifact_paths = [("predictor", args.predictor), ("labels", args.labels), ("model", args.model)]
        if args.feature_config:
            artifact_paths.append(("feature_config", args.feature_config))
        for _, artifact in artifact_paths:
            if not artifact.is_file():
                raise EvaluationError(f"Artifact not found: {artifact}")
        status = "passed_schema" if not details["failures"] else "completed_with_failures"
        representative = args.fixture_type == "real" and not details["failures"] and len(items) >= 20
        summary = {
            "status": status,
            "representative_performance_measured": representative,
            "performance_qualification": (
                "Real clips supplied; sample size still depends on coverage of the evaluation corpus."
                if representative else
                "Representative performance NOT MEASURED: requires at least 20 successfully measured real clips and no failures."
            ),
            "clip_count": len(items),
            "successful_predictions": len(predictions),
            "failed_predictions": details["failures"],
            "accuracy": accuracy,
            "latency": {
                "all_calls": timings(wall_times),
                "cold_call": timings(wall_times[:1]),
                "warm_calls": timings(wall_times[1:]),
                "method": "External wall time around each predict() call, perf_counter_ns; nearest-rank percentiles",
                "included": "Video input through returned JSON; excludes manifest I/O and process startup",
                "excluded_failures": len(details["failures"]),
            },
            "stage_profiles": details["profiles"],
            "artifacts": {
                name: {"path": _sanitized_relative(str(path), root), "sha256": sha256(path), "bytes": path.stat().st_size}
                for name, path in artifact_paths
            },
            "manifest": {"path": _sanitized_relative(str(args.manifest), root), "sha256": sha256(args.manifest),
                         "dataset": args.dataset, "split": args.split},
            "code_commit": git_sha(root),
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "machine": platform.machine(),
                "device_model": _GENERIC_DEVICE_MODEL,
                "inference_runtime": _runtime_version(),
                "delegate": "XNNPACK CPU (observed LiteRT log; current baseline)",
                "thread_count": "runtime default; not explicitly configured",
            },
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "fixture_type": args.fixture_type,
            "scope": "Runtime/interface measurement only unless complete ground_truth is supplied; not target-device evidence",
        }
        write_pair_atomically(args.output_dir, predictions, summary)
    except (EvaluationError, OSError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
