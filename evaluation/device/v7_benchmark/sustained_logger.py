#!/usr/bin/env python3
"""Sustained-workload CSV logger for the 10-minute thermal protocol.

Pipe each row to CSV: timestamp_seconds,latency_ms,clip_id,device,temperature_c,thermal_status
Loops the manifest continuously until --duration seconds have elapsed. Each clip
is called through the benchmark predictor (FP16, or INT8 with --variant int8).
temperature_c/thermal_status are left blank unless the platform exposes them
without sudo — they are never fabricated. Power is NOT inferred.

Usage:
    python sustained_logger.py /tmp/echogesture-validation-30/manifest.csv \
        --source-root /path/to/checkout --duration 600 --output sustained.csv \
        [--device-model "Exact target device model" --variant fp16|--variant int8]
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "benchmark_lab"))

import paired_runner  # noqa: E402
import run_manifest  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--variant", choices=("fp16", "int8"), default="fp16")
    parser.add_argument("--duration", type=float, default=600.0)
    parser.add_argument("--output", type=Path, default=Path("sustained.csv"))
    parser.add_argument("--device-model", default="unknown")
    parser.add_argument("--ambient-c", default="")
    parser.add_argument("--charger", default="")
    parser.add_argument("--battery-pct", default="")
    args = parser.parse_args(argv)
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        items = run_manifest.read_manifest(args.manifest.resolve())
    except (run_manifest.EvaluationError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    try:
        predict = paired_runner._load_predictor(args.source_root.resolve(), args.variant)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    accepts_clip = paired_runner._accepts_clip_id(predict)
    start = time.monotonic()
    deadline = start + args.duration
    rows = []
    while time.monotonic() < deadline:
        for cid, cpath, *_ in items:
            if time.monotonic() >= deadline:
                break
            t0 = time.monotonic()
            kwargs = {"profile": True}
            if accepts_clip:
                kwargs["clip_id"] = cid
            try:
                predict(str(cpath), **kwargs)
                status = "ok"
            except Exception as exc:  # keep collecting; exhaustion of deadline still applies
                status = f"failure:{type(exc).__name__}"
            latency_ms = (time.monotonic() - t0) * 1000.0
            rows.append({
                "timestamp_seconds": f"{time.monotonic() - start:.3f}",
                "latency_ms": f"{latency_ms:.3f}",
                "clip_id": cid,
                "device": args.device_model,
                "temperature_c": "",
                "thermal_status": status,
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "timestamp_seconds", "latency_ms", "clip_id", "device", "temperature_c", "thermal_status"])
        writer.writeheader()
        writer.writerows(rows)
    elapsed = time.monotonic() - start
    print(f"wrote {len(rows)} calls over {elapsed:.1f}s to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
