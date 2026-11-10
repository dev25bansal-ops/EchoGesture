"""Report performance drift from a sustained-workload CSV.

Input columns: timestamp_seconds,latency_ms; optional temperature_c,
thermal_status,device. No power estimate is inferred from CPU or temperature.
"""

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_manifest import percentile


def analyze(rows, window_seconds=30, model_sha=None):
    samples = [(float(row["timestamp_seconds"]), float(row["latency_ms"]), row) for row in rows]
    if len(samples) < 2 or any(second <= 0 or latency < 0 for second, latency, _ in samples):
        raise ValueError("Need at least two valid timestamped latency samples")
    samples.sort()
    elapsed = samples[-1][0] - samples[0][0]
    if elapsed < 600:
        raise ValueError(f"Thermal drift requires at least 600 seconds; got {elapsed:.1f}")
    first_end = samples[0][0] + window_seconds
    last_start = samples[-1][0] - window_seconds
    early = [latency for second, latency, _ in samples if second <= first_end]
    late = [latency for second, latency, _ in samples if second >= last_start]
    if not early or not late:
        raise ValueError("No samples in early/late measurement window")
    early_p95, late_p95 = percentile(early, 0.95), percentile(late, 0.95)
    if early_p95 == 0:
        raise ValueError("Early p95 must be nonzero")
    return {
        "metric": "thermal_drift_10min", "value": 100 * (late_p95 - early_p95) / early_p95,
        "early_p95_ms": early_p95, "late_p95_ms": late_p95,
        "elapsed_seconds": elapsed, "window_seconds": window_seconds,
        "early_count": len(early), "late_count": len(late),
        "device": samples[0][2].get("device", "unknown"),
        "model_sha256": model_sha,
        "method": "100 * (last-window p95 - first-window p95) / first-window p95; performance drift, not temperature change",
        "power": "NOT_MEASURED; requires a documented external meter or qualified platform method",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--model-sha", required=True)
    parser.add_argument("--window", type=int, default=30)
    args = parser.parse_args(argv)
    try:
        with args.csv.open(newline="", encoding="utf-8") as stream:
            result = analyze(list(csv.DictReader(stream)), args.window, args.model_sha)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
