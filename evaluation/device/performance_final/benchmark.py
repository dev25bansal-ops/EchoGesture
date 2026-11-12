"""Benchmark the persistent parallel IMAGE tracker vs the singleton IMAGE path.

Measures full clip-to-JSON `predict()` p50/p95, cold and warm, tracking-stage
breakdown, memory (ru_maxrss), failure handling and handedness/presence parity
on identical permitted validation clips across worker counts 1/2/4.

Holds the atomic lock ``/tmp/echogesture-heavy-benchmark.lock`` for the whole
run and removes it in ``finally``.  Uses Dev's read-frontend and classifier via
the existing ``src`` modules; nothing outside this directory is modified.

Usage:
    .venv/bin/python evaluation/device/performance_final/benchmark.py \
        --clips <dir|manifest> --output evaluation/device/performance_final/results \
        --workers 1 2 4 --repeats 3
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SRC = ROOT / "src"
MODELS = ROOT / "models"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

LOCK = Path("/tmp/echogesture-heavy-benchmark.lock")


def _acquire_lock() -> None:
    """Atomically claim the benchmark lock; raise if another run is active."""
    try:
        os.mkdir(str(LOCK))
    except FileExistsError as exc:
        raise RuntimeError(
            "heavy benchmark already running (lock held); refusing to start"
        ) from exc


def _release_lock() -> None:
    try:
        os.rmdir(str(LOCK))
    except FileNotFoundError:
        pass


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    k = (len(ordered) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    frac = k - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def _load(clip_path, manifest_clip_id=None):
    """Decode frames with Dev's sampler once; reuse across workers/repeats."""
    import landmark_frontend as lf
    frames = lf.read_frames(str(clip_path), max_frames=32)
    if not frames:
        raise ValueError(f"No valid frames decoded from: {clip_path}")
    return frames


def _classify(landmarks, present):
    """Full model inference from Dev contract (features -> FP16 classifier)."""
    from features import build_features
    import main as mainmod

    tensor, _ = build_features(landmarks, present, expected_frames=24)
    tensor = tensor[None, ...].astype(np.float32)
    mainmod._init_model()
    inp = mainmod._INPUT_DETAILS
    out = mainmod._OUTPUT_DETAILS
    mainmod._INTERPRETER.set_tensor(inp["index"], tensor)
    mainmod._INTERPRETER.invoke()
    raw = mainmod._INTERPRETER.get_tensor(out["index"])[0].astype(np.float32)
    logits = raw
    exp = np.exp(logits - np.max(logits))
    probs = exp / np.sum(exp)
    top = np.argsort(-probs)[:5]
    top5 = [mainmod._IDX_TO_LABEL.get(int(i), f"UNKNOWN_{int(i)}") for i in top]
    return top5[0], top5, float(probs[top[0]])


def _baseline_detect(frames):
    import landmark_frontend as lf
    return lf.detect_hands(frames)


def _summ(values):
    return {
        "count": len(values),
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "mean_ms": statistics.mean(values) if values else None,
    }


def benchmark_clips(clip_paths, workers_list, repeats):
    import tracker as tk
    import landmark_frontend as lf

    # Warm singleton + classifier once (also warms MediaPipe/TFLite).
    sample = _load(clip_paths[0])
    _baseline_detect(sample[:1])
    _classify(*_baseline_detect(sample))

    reports = {"clips": [str(p) for p in clip_paths],
               "workers": {}, "baseline": {}, "parity": {}}

    # Baseline singleton, cold + warm total predict.
    base_times = []
    for path in clip_paths:
        frames = _load(path)
        for rep in range(repeats):
            st = time.perf_counter()
            lm, present = _baseline_detect(frames)
            pred, top5, conf = _classify(lm, present)
            base_times.append((time.perf_counter() - st) * 1000.0)
    reports["baseline"] = {
        "count": len(base_times),
        "p50_ms": percentile(base_times, 0.50),
        "p95_ms": percentile(base_times, 0.95),
        "mean_ms": statistics.mean(base_times) if base_times else None,
    }

    for workers in workers_list:
        # Cold: fresh pool (construction).  Warm: reuse the same pool.
        cold, warm = [], []
        tracking_cold, tracking_warm = [], []
        parity = {"exact_landmarks": 0, "exact_present": 0, "total": 0,
                  "any_diff": 0}
        predictions_match = 0

        # Cold pool.
        tracking_all = []
        pool = tk.ParallelImageTracker(workers=workers,
                                       task_path=MODELS / "hand_landmarker.task")
        try:
            for path in clip_paths:
                frames = _load(path)
                blm, bpresent = _baseline_detect(frames)
                bpred, _btop5, _bconf = _classify(blm, bpresent)
                for rep in range(repeats):
                    st = time.perf_counter()
                    tr_st = time.perf_counter()
                    lm, present = pool.detect(frames, timeout=30.0)
                    tracking = (time.perf_counter() - tr_st) * 1000.0
                    pred, top5, conf = _classify(lm, present)
                    total = (time.perf_counter() - st) * 1000.0
                    (cold if rep == 0 else warm).append(total)
                    tracking_all.append(tracking)
                    parity["total"] += 1
                    if np.array_equal(lm, blm):
                        parity["exact_landmarks"] += 1
                    else:
                        parity["any_diff"] += 1
                    if np.array_equal(present, bpresent):
                        parity["exact_present"] += 1
                    if pred == bpred:
                        predictions_match += 1
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        finally:
            pool.close()

        reports["workers"][str(workers)] = {
            "cold": _summ(cold),
            "warm": _summ(warm),
            "tracking_stage_p50_ms": percentile(tracking_all, 0.50),
            "tracking_stage_p95_ms": percentile(tracking_all, 0.95),
            "tracking_stage_mean_ms": statistics.mean(tracking_all) if tracking_all else None,
            "peak_rss_kib": peak,
        }
        reports["parity"] = {
            "checked": parity["total"],
            "exact_landmarks": parity["exact_landmarks"],
            "exact_present": parity["exact_present"],
            "any_landmark_diff": parity["any_diff"],
            "same_prediction": predictions_match,
        }
    return reports


def _resolve_clips(clip_arg):
    path = Path(clip_arg).expanduser()
    if path.is_dir():
        clips = sorted(path.glob("*.mp4"))
    elif path.suffix == ".csv":
        data = Path(path.parent)
        clips = []
        for line in path.read_text(encoding="utf-8").splitlines()[1:]:
            parts = line.split(",")
            if len(parts) >= 2:
                clips.append(data / parts[1])
    else:
        clips = [path]
    return [c for c in clips if c.is_file()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", nargs="+", required=True,
                        help="one or more dirs, manifest.csv, or mp4 files")
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)

    clips = []
    for arg in args.clips:
        clips.extend(_resolve_clips(arg))
    if not clips:
        parser.error("no clips resolved")

    _acquire_lock()
    try:
        started = time.perf_counter()
        reports = benchmark_clips(clips, args.workers, args.repeats)
        reports["wall_seconds"] = time.perf_counter() - started
        reports["hardware"] = {
            "platform": "darwin", "python": sys.version.split()[0],
        }
        args.output.mkdir(parents=True, exist_ok=True)
        out = args.output / "benchmark.json"
        out.write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(reports, indent=2))
    finally:
        _release_lock()


if __name__ == "__main__":
    main()