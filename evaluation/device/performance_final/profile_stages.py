"""Stage profiler for the shipped landmark pipeline (A02).

Measures, on the 12 permitted validation clips, where full clip-to-JSON time
actually goes when `predict()` runs with the parallel tracker:

    decode (read)         cap.read() of each sampled frame
    grab     (skip)       cap.grab() over non-sampled frames
    resize   (per frame)  cv2.resize -> 256x256
    cvtColor (per frame)  cv2.cvtColor BGR->RGB
    detect   (per frame)  MediaPipe HandLandmarker IMAGE, 2 hands
    build_features
    ensemble inference

All timings are manual ``perf_counter_ns`` sums: cProfile cannot see inside
OpenCV / MediaPipe C calls, and instrumenting them per call distorts the
Python-level split anyway.

The script never edits ``src/``; it imports Dev's read-only front end, the
feature transform and the A01 tracker, and holds
``/tmp/echogesture-heavy-benchmark.lock`` for the whole run.

Usage:
    .venv/bin/python evaluation/device/performance_final/profile_stages.py \
        [--clips DIR] [--max-clips 12] [--workers 4] [--repeats 2] \
        [--output results_profile/stages.json]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(HERE))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

LOCK = Path("/tmp/echogesture-heavy-benchmark.lock")
DEFAULT_CLIPS = ROOT / "evaluation/device/v5_accuracy/data/val_30"
DEFAULT_OUTPUT = HERE / "results_profile/stages.json"
FRAME_SIZE = 256


def _acquire_lock():
    try:
        os.mkdir(str(LOCK))
    except FileExistsError as exc:
        raise RuntimeError(
            "heavy benchmark already running (lock held); refusing to start") from exc


def _release_lock():
    try:
        os.rmdir(str(LOCK))
    except FileNotFoundError:
        pass


def _resolve_clips(clip_dir: Path, limit: int):
    clips = sorted(clip_dir.glob("*.mp4"))
    if not clips:
        raise SystemExit(f"no .mp4 clips in {clip_dir}")
    return clips[:limit]


def _sample_targets(total: int, max_frames: int):
    return np.unique(np.linspace(0, total - 1, max_frames).astype(int))


def profile_read(clip: Path, max_frames: int = 32) -> dict:
    """One instrumented pass of Dev's read_frames, split into its primitives.

    Mirrors src/landmark_frontend.read_frames exactly (same targets, same
    grab/read loop, same resize/cvtColor) so the totals match production.
    """
    t = {"decode_ms": 0.0, "grab_ms": 0.0, "resize_ms": 0.0, "cvt_ms": 0.0}
    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {clip}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames = []
    if total > max_frames:
        targets = _sample_targets(total, max_frames)
        cur, decoded = 0, 0
        for target in targets:
            s = time.perf_counter_ns()
            while cur < target:
                if not cap.grab():
                    break
                cur += 1
            t["grab_ms"] += (time.perf_counter_ns() - s) / 1e6
            s = time.perf_counter_ns()
            ok, f = cap.read()
            t["decode_ms"] += (time.perf_counter_ns() - s) / 1e6
            if not ok:
                break
            cur += 1
            decoded += 1
            s = time.perf_counter_ns()
            r = cv2.resize(f, (FRAME_SIZE, FRAME_SIZE))
            t["resize_ms"] += (time.perf_counter_ns() - s) / 1e6
            s = time.perf_counter_ns()
            rgb = cv2.cvtColor(r, cv2.COLOR_BGR2RGB)
            t["cvt_ms"] += (time.perf_counter_ns() - s) / 1e6
            frames.append(rgb)
    else:
        while len(frames) < max_frames:
            s = time.perf_counter_ns()
            ok, f = cap.read()
            t["decode_ms"] += (time.perf_counter_ns() - s) / 1e6
            if not ok:
                break
            s = time.perf_counter_ns()
            r = cv2.resize(f, (FRAME_SIZE, FRAME_SIZE))
            t["resize_ms"] += (time.perf_counter_ns() - s) / 1e6
            s = time.perf_counter_ns()
            rgb = cv2.cvtColor(r, cv2.COLOR_BGR2RGB)
            t["cvt_ms"] += (time.perf_counter_ns() - s) / 1e6
            frames.append(rgb)
    cap.release()
    t["frames"] = len(frames)
    t["total_ms"] = sum(t[k] for k in ("decode_ms", "grab_ms", "resize_ms", "cvt_ms"))
    return t, frames


def profile_detect(frames, landmarker) -> dict:
    """Per-frame MediaPipe IMAGE detection cost using Dev's detect_hands."""
    import landmark_frontend as lf
    s = time.perf_counter_ns()
    landmarks, present = lf.detect_hands(frames, landmarker=landmarker, mode="image")
    total = (time.perf_counter_ns() - s) / 1e6
    return {"total_ms": total, "per_frame_ms": total / max(1, len(frames))}, landmarks, present


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", type=Path, default=DEFAULT_CLIPS)
    parser.add_argument("--max-clips", type=int, default=12)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    clips = _resolve_clips(args.clips, args.max_clips)
    _acquire_lock()
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("eg_main_profile", SRC / "main.py")
        mainmod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mainmod
        spec.loader.exec_module(mainmod)

        import landmark_frontend as lf
        from features import build_features
        import tracker as tk

        mainmod._init_model()
        # Warm everything (first MediaPipe/TFLite call includes graph setup).
        _, warm_frames = profile_read(clips[0])
        warm_lm = lf.get_landmarker("image")
        lf.detect_hands(warm_frames[:2], landmarker=warm_lm, mode="image")
        per_clip = []
        with tk.ParallelImageTracker(workers=args.workers) as pool:
            pool.detect(warm_frames)
            for clip in clips:
                row = {"clip": clip.name, "reads": [], "detects": []}
                frames = None
                for _ in range(args.repeats):
                    read_t, frames = profile_read(clip)
                    row["reads"].append(read_t)

                    det_t, landmarks, present = profile_detect(frames, warm_lm)
                    row["detects"].append(det_t)

                    s = time.perf_counter_ns()
                    feat_t, _ = build_features(
                        landmarks, present, expected_frames=mainmod._FRAMES)
                    row.setdefault("features_ms", []).append(
                        (time.perf_counter_ns() - s) / 1e6)

                    s = time.perf_counter_ns()
                    tensor = feat_t[None, ...].astype(np.float32)
                    mainmod._run_ensemble(tensor)
                    row.setdefault("ensemble_ms", []).append(
                        (time.perf_counter_ns() - s) / 1e6)

                    s = time.perf_counter_ns()
                    pool.detect(frames)
                    row.setdefault("parallel_detect_ms", []).append(
                        (time.perf_counter_ns() - s) / 1e6)
                row["frames"] = len(frames)
                per_clip.append(row)
                print(f"{clip.name[:40]:40s} "
                      f"read={statistics.mean(r['total_ms'] for r in row['reads']):6.2f} "
                      f"detect={statistics.mean(d['total_ms'] for d in row['detects']):6.2f} "
                      f"feat={statistics.mean(row['features_ms']):5.2f} "
                      f"ens={statistics.mean(row['ensemble_ms']):5.2f} "
                      f"par{args.workers}={statistics.mean(row['parallel_detect_ms']):6.2f}")

        def mean_of(key, sub=None):
            vals = []
            for row in per_clip:
                if sub is None:
                    vals.append(statistics.mean(row[key]))
                else:
                    vals.append(statistics.mean(r[sub] for r in row[key]))
            return statistics.mean(vals)

        summary = {
            "clips": len(per_clip),
            "workers": args.workers,
            "read_mean_ms": mean_of("reads", "total_ms"),
            "read_decode_ms": mean_of("reads", "decode_ms"),
            "read_grab_ms": mean_of("reads", "grab_ms"),
            "read_resize_ms": mean_of("reads", "resize_ms"),
            "read_cvt_ms": mean_of("reads", "cvt_ms"),
            "singleton_detect_ms": mean_of("detects", "total_ms"),
            "singleton_detect_per_frame_ms": mean_of("detects", "per_frame_ms"),
            "parallel_detect_ms": mean_of("parallel_detect_ms"),
            "features_ms": mean_of("features_ms"),
            "ensemble_ms": mean_of("ensemble_ms"),
        }
        summary["serial_wall_ms"] = (
            summary["read_mean_ms"] + summary["singleton_detect_ms"]
            + summary["features_ms"] + summary["ensemble_ms"])
        summary["parallel4_wall_ms"] = (
            summary["read_mean_ms"] + summary["parallel_detect_ms"]
            + summary["features_ms"] + summary["ensemble_ms"])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(
            {"summary": summary, "per_clip": per_clip}, indent=2) + "\n",
            encoding="utf-8")
        print(json.dumps(summary, indent=2))
        print(f"wrote {args.output}")
        return 0
    finally:
        _release_lock()


if __name__ == "__main__":
    sys.exit(main())