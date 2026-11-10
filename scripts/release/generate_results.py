"""Generate explicit result rows from a candidate's evidence files only.

Hardened (A12): every row traces to candidate SHA, feature config, code
revision, dataset/split, clips/signers, device/runtime, timestamp and method.
Accuracy stays NOT_MEASURED unless a representative labelled validation run
exists; latency stays NOT_MEASURED if the run is nonrepresentative; model
size reports total artifact accounting (classifier + detector + auxiliaries),
not classifier bytes alone. Small smoke runs can never yield PASS.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evaluation/device"))
from evidence_lab import (
    ProvenanceError,
    artifact_inventory,
    describe_output_semantics,
    extract_provenance,
    per_signer_rows,
)

# v8_privacy hardening: sanitise evidence before use; refuse to emit
# CSV rows that still embed a hostname or private absolute path, and
# refuse to emit raw participant IDs (per-signer rows use pseudonyms
# only when ECHOGESTURE_SIGNER_SALT is set, otherwise aggregate-only).
_V8_DEVICE_DIR = str(Path(__file__).resolve().parents[2] / "evaluation" / "device" / "v8_privacy")
if _V8_DEVICE_DIR not in sys.path:
    sys.path.insert(0, _V8_DEVICE_DIR)
try:
    from sanitize_evidence import assert_sanitized, sanitize_benchmark_dict  # noqa: E402
except ImportError:  # v8_privacy tree not present (older checkout): no-op
    assert_sanitized = None  # type: ignore
    sanitize_benchmark_dict = None  # type: ignore

COLUMNS = ["metric", "target", "measured", "unit", "device", "runtime", "pass", "notes"]

# Representative gates (mirrors run_manifest / robustness): a PASS requires
# both the threshold and these sample conditions.
MIN_REPRESENTATIVE_CLIPS = 20


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _accuracy_is_representative(benchmark):
    accuracy = benchmark.get("accuracy") or {}
    if not isinstance(accuracy, dict):
        return False
    if "representative_accuracy_measured" in accuracy:
        return bool(accuracy["representative_accuracy_measured"])
    # Fallback used by older benchmark JSON: stage profiles imply a real run.
    return len(benchmark.get("stage_profiles", [])) >= MIN_REPRESENTATIVE_CLIPS


def _load_predictions(predictions_path):
    if predictions_path is None:
        return None
    try:
        text = Path(predictions_path).read_text(encoding="utf-8")
    except OSError:
        return None
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows or None


def generate(benchmark_path, robustness_paths, output_path, model_sha,
             power_path=None, thermal_path=None, predictions_path=None,
             extra_artifacts=(), output_semantics=None, heldout_path=None):
    benchmark = load(benchmark_path)
    if sanitize_benchmark_dict is not None:
        benchmark = sanitize_benchmark_dict(benchmark, repo_root=Path(__file__).resolve().parents[2])
    if assert_sanitized is not None:
        leaks = assert_sanitized(benchmark)
        if leaks:
            raise ValueError(f"Benchmark still embeds private path/hostname after sanitisation: {leaks[0][:120]}")
    required = ("artifacts", "environment", "accuracy", "latency", "manifest")
    missing = [key for key in required if key not in benchmark]
    if missing:
        raise KeyError(f"Benchmark missing required fields: {', '.join(missing)}")
    prov = extract_provenance(benchmark, Path(benchmark_path).name)
    # Stale evidence rejection: benchmark must be bound to this candidate.
    try:
        prov.require_bound_to(model_sha, stale_label="benchmark")
    except ProvenanceError as exc:
        raise ValueError(str(exc)) from exc
    device = prov.device
    runtime = prov.runtime
    trace = prov.trace_string()
    missing = prov.missing_provenance()
    trace_note = trace + ("; missing provenance: " + ",".join(missing) if missing else "")

    accuracy = benchmark.get("accuracy") or {}
    top1 = accuracy.get("top1") if isinstance(accuracy, dict) else None
    accuracy_measured = top1 is not None and _accuracy_is_representative(benchmark)
    clip_count = benchmark.get("clip_count")
    measured_accuracy = 100 * top1 if top1 is not None else "NOT_MEASURED"
    rows = [
        row("top1_accuracy", ">=70", measured_accuracy, "%", device, runtime,
            status(accuracy_measured, accuracy_measured and top1 >= 0.70),
            trace_note + f"; clips={clip_count}; representative={accuracy_measured}; "
            "PASS requires >=20 labelled clips, complete ground truth, documented validation split"),
    ]

    latency = (benchmark.get("latency") or {}).get("all_calls") or {}
    p95 = latency.get("p95_ms")
    perf_measured = bool(benchmark.get("representative_performance_measured"))
    latency_measured = perf_measured and p95 is not None
    rows.append(row(
        "latency_p95", "<50", p95 if p95 is not None else "NOT_MEASURED", "ms", device, runtime,
        "NOT_MEASURED" if not latency_measured else status(True, p95 < 50),
        trace_note + f"; complete clip path, cold and warm included; clips={clip_count}; "
        f"representative={perf_measured}; smoke runs cannot PASS"))

    # --- Total artifact size accounting (not classifier-only) ---
    inventory = artifact_inventory(benchmark.get("artifacts"), extra_artifacts,
                                   root=Path(benchmark_path).parent)
    breakdown = inventory.as_dict()
    total_bytes = breakdown["total_bytes"]
    classifier_bytes = breakdown["classifier_bytes"]
    if total_bytes is None:
        size_measured, size_value = "NOT_MEASURED", "NOT_MEASURED"
        size_pass = "NOT_MEASURED"
        size_note = "Total artifact inventory incomplete: an artifact file is missing; " \
            "classifier-only bytes are NOT a total and cannot PASS"
    else:
        size_measured = total_bytes / 1e6
        size_value = size_measured
        size_pass = "PASS" if total_bytes < 20e6 else "FAIL"
        size_note = (
            f"Total shipped artifacts {total_bytes} bytes ({size_measured:.3f} MB); "
            f"classifier-only {classifier_bytes} bytes; "
            "includes detector/auxiliary assets listed in notes")
    item_list = "; ".join(
        f"{i['name']}={i['bytes']}B sha256={ (i['sha256'] or 'missing')[:12]}"
        for i in breakdown["items"])
    rows.append(row("model_size", "<20", size_value, "MB", device, runtime,
                    size_pass, size_note + f"; artifacts: {item_list}"))

    # --- Explicit confidence semantics (logits -> softmax, uncalibrated) ---
    semantics = output_semantics or describe_output_semantics()
    rows.append(row(
        "confidence_semantics", "logits+softmax", "softmax_top1", "n/a", device, runtime,
        "NOT_MEASURED",
        f"Model output semantics: {semantics['output_semantics']} "
        f"({semantics['basis']}); confidence={semantics['confidence_definition']}; "
        f"calibrated={semantics['calibrated']}; {semantics['calibration_note']}"))

    # --- Per-signer accuracy, only if the benchmark supplies signer data ---
    predictions = _load_predictions(predictions_path)
    signer_rows = per_signer_rows(benchmark, predictions)
    if signer_rows:
        for s in signer_rows:
            rows.append(row(
                s["metric"], ">=70", s["measured"], "%", device, runtime,
                "NOT_MEASURED",
                f"{trace_note}; per-signer breakdown from supplied signer data only; "
                f"correct={s['correct']}/{s['total']}; no PASS claimed per signer"))
    else:
        rows.append(row(
            "top1_accuracy_per_signer", ">=70", "NOT_SIGNER_DATA_SUPPLIED", "%", device, runtime,
            "NOT_MEASURED",
            trace_note + "; benchmark/predictions carry no signer breakdown; "
            "per-signer accuracy not derivable and not estimated"))

    for path in robustness_paths:
        data = load(path)
        rprov = extract_provenance(data, Path(path).name)
        try:
            rprov.require_bound_to(model_sha, stale_label=f"robustness evidence: {path}")
        except ProvenanceError as exc:
            raise ValueError(str(exc)) from exc
        drop = data.get("accuracy_drop_percentage_points")
        rep = bool(data.get("representative_robustness_measured"))
        measured = rep and drop is not None
        rows.append(row(
            f"accuracy_drop_{data.get('corruption_type', 'unknown')}", "<=5",
            drop if drop is not None else "NOT_MEASURED", "pt", device, runtime,
            "NOT_MEASURED" if not measured else status(True, drop <= 5),
            rprov.trace_string() + f"; N={data.get('sample_count')}; paired clean/corrupt; "
            f"representative={rep}; corruption={data.get('corruption_type')}/"
            f"severity={data.get('corruption_severity')}/seed={data.get('seed')}"))

    for metric, target, unit, path in (
        ("avg_power", "<=500", "mW", power_path),
        ("thermal_drift_10min", "<=20", "%", thermal_path),
    ):
        if path is None:
            rows.append(row(metric, target, "NOT_MEASURED", unit, "unavailable", "unavailable",
                            "NOT_MEASURED", "No physical-device measurement supplied"))
            continue
        data = load(path)
        pprov = extract_provenance(data, Path(path).name)
        try:
            pprov.require_bound_to(model_sha, stale_label=f"{metric} evidence: {path}")
        except ProvenanceError as exc:
            raise ValueError(str(exc)) from exc
        value = data.get("value")
        limit = float(target[2:])
        measured_flag = value is not None
        rows.append(row(metric, target,
                        value if value is not None else "NOT_MEASURED", unit,
                        data.get("device", device), runtime,
                        "NOT_MEASURED" if not measured_flag else
                        status(True, value <= limit),
                        pprov.trace_string() + f"; method={data.get('method', 'unspecified')}"))

    if assert_sanitized is not None:
        leaks = assert_sanitized(rows)
        if leaks:
            raise ValueError(f"Refusing to write results still embedding private path/hostname: {leaks[0][:120]}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def row(metric, target, measured, unit, device, runtime, passed, notes):
    return {"metric": metric, "target": target, "measured": measured, "unit": unit,
            "device": device, "runtime": runtime, "pass": passed, "notes": notes}


def status(measured, passes):
    return "PASS" if passes else "FAIL" if measured else "NOT_MEASURED"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--robustness", type=Path, nargs="*", default=[])
    parser.add_argument("--model-sha", required=True)
    parser.add_argument("--power", type=Path)
    parser.add_argument("--thermal", type=Path)
    parser.add_argument("--predictions", type=Path,
                        help="predictions.jsonl from the same run, if available")
    parser.add_argument("--artifact", action="append", default=[],
                        help="extra shipped artifact file (name=path or path); "
                             "repeatable; included in total size accounting")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    extra = []
    for item in args.artifact:
        if "=" in item:
            name, path = item.split("=", 1)
            extra.append((name, path))
        else:
            extra.append(item)
    try:
        rows = generate(args.benchmark, args.robustness, args.output, args.model_sha,
                        args.power, args.thermal, args.predictions, extra)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
