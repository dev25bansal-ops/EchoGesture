"""Simulate production `predict()` from a one-shot decode, under the start lock.

Production `predict()` does:

    read_frames()   decode all 32 sampled frames, sequentially
    tracker.detect()  all 32 frames, 4 workers
    build_features()
    ensemble

so per clip the wall time is decode + detect.  A process that serves clips
concurrently loses that serialisation whenever decode of clip n+1 overlaps
detection of clip n.  This harness measures that overlap directly, with no
change to ``src/``:

    mode=serial  N clip workers, each running decode-then-detect-and-classify
    mode=split   one decode thread per clip (production read_frames) + one
                 persistent detect pool shared across clips, so decode overlaps
                 detection

Both modes do the identical maths; only the scheduling differs.  Prediction,
top-5, feature-tensor hash and landmark hash are recorded per clip so parity
between modes (and against the production baseline) is checkable.

Holds ``/tmp/echogesture-heavy-benchmark.lock`` for the whole run and removes it
in ``finally``.

Usage:
    .venv/bin/python evaluation/device/performance_final/concurrency_a02.py \
        [--clips DIR] [--max-clips 12] [--samples 3] [--workers 4] \
        [--output results_candidates/concurrency.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

LOCK = Path("/tmp/echogesture-heavy-benchmark.lock")
DEFAULT_CLIPS = ROOT / "evaluation/device/v5_accuracy/data/val_30"
DEFAULT_OUTPUT = HERE / "results_candidates/concurrency.json"


def _sha(arr) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]


def _percentile(values, q):
    if not values:
        return float("nan")
    ordered = sorted(values)
    k = (len(ordered) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] * (1 - k + lo) + ordered[hi] * (k - lo)


def _summ(values):
    return {"p50_ms": _percentile(values, 0.50), "p95_ms": _percentile(values, 0.95),
            "mean_ms": statistics.mean(values), "min_ms": min(values),
            "max_ms": max(values), "count": len(values)}


def _load_main():
    import importlib.util
    spec = importlib.util.spec_from_file_location("eg_main_conc", SRC / "main.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _classify(mainmod, landmarks, present):
    from features import build_features
    tensor, _ = build_features(landmarks, present, expected_frames=mainmod._FRAMES)
    tensor = tensor[None, ...].astype(np.float32)
    logits, _ = mainmod._run_ensemble(tensor)
    exp = np.exp(logits - np.max(logits))
    probs = exp / np.sum(exp)
    top = np.argsort(-probs)[:5]
    top5 = [mainmod._IDX_TO_LABEL.get(int(i), "?") for i in top]
    return {"prediction": top5[0], "top5": top5,
            "tensor_sha": _sha(tensor), "landmarks_sha": _sha(landmarks)}


def run_mode(mode: str, clips, workers: int, samples: int, clip_workers: int):
    """Return (per-clip latency samples by clip index, prediction records)."""
    import landmark_frontend as lf
    import tracker as tk

    mainmod = _load_main()
    mainmod._init_model()
    lf.read_frames(str(clips[0]), max_frames=32)          # warm decoder

    latencies: dict[str, list[float]] = {}
    records: dict[str, dict] = {}
    pool = tk.ParallelImageTracker(workers=workers)
    decode_executor = None
    try:
        pool.detect(lf.read_frames(str(clips[0]), max_frames=32))  # warm landmarkers
        if mode == "split":
            decode_executor = ThreadPoolExecutor(
                max_workers=clip_workers, thread_name_prefix="a02-decode")

        def serial(job):
            clip = clips[job]
            t = time.perf_counter()
            frames = lf.read_frames(str(clip), max_frames=32)
            landmarks, present = pool.detect(frames)
            result = _classify(mainmod, landmarks, present)
            return job, (time.perf_counter() - t) * 1000.0, result

        def split(job):
            clip = clips[job]
            t = time.perf_counter()
            future = decode_executor.submit(
                lf.read_frames, str(clip), 32)
            frames = future.result()
            landmarks, present = pool.detect(frames)
            result = _classify(mainmod, landmarks, present)
            return job, (time.perf_counter() - t) * 1000.0, result

        work = serial if mode == "serial" else split
        for sample in range(samples):
            with ThreadPoolExecutor(max_workers=clip_workers,
                                    thread_name_prefix="a02-clip") as clip_pool:
                for job, ms, result in clip_pool.map(work, range(len(clips))):
                    key = clips[job].name
                    latencies.setdefault(key, []).append(ms)
                    if key not in records:
                        records[key] = result
    finally:
        if decode_executor is not None:
            decode_executor.shutdown(wait=True)
        pool.close()
    return latencies, records


def _resolve(clip_dir, limit):
    clips = sorted(clip_dir.glob("*.mp4"))
    if not clips:
        raise SystemExit(f"no .mp4 clips in {clip_dir}")
    return clips[:limit]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", type=Path, default=DEFAULT_CLIPS)
    parser.add_argument("--max-clips", type=int, default=12)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--clip-workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    clips = _resolve(args.clips, args.max_clips)
    try:
        os.mkdir(str(LOCK))
    except FileExistsError:
        print("refusing to start: heavy benchmark lock held", file=sys.stderr)
        return 2
    try:
        report = {"workers": args.workers, "clip_workers": args.clip_workers,
                  "samples": args.samples, "clips": [c.name for c in clips],
                  "modes": {}}
        for mode in ("serial", "split"):
            lat, recs = run_mode(mode, clips, args.workers, args.samples,
                                 args.clip_workers)
            flat = [v for vals in lat.values() for v in vals]
            report["modes"][mode] = {
                "per_clip_p50": {k: statistics.median(v) for k, v in lat.items()},
                "all_samples": _summ(flat),
                "records": recs,
            }
            print(f"[{mode}] clip-p50={statistics.median(flat):7.2f} "
                  f"clip-p95={_percentile(flat, 0.95):7.2f} n={len(flat)}")

        serial = report["modes"]["serial"]["records"]
        split = report["modes"]["split"]["records"]
        diffs = [k for k in serial if serial[k] != split[k]]
        report["parity_serial_vs_split"] = {
            "clips": len(serial), "mismatches": diffs}
        print(f"parity serial vs split: {len(serial) - len(diffs)}/{len(serial)} identical")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
        return 0
    finally:
        try:
            os.rmdir(str(LOCK))
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    sys.exit(main())