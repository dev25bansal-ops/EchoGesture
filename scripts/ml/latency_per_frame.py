"""Measure per-frame MediaPipe cost, which is the latency criterion.

The repo has been quoting 15.7 ms/frame with no retained artifact behind it.
This script produces one, so the number is reproducible rather than inherited.

Reports both IMAGE mode (one frame at a time, what the batch path does) and
VIDEO mode (tracker reused, what the Android app does).

Usage:
    python scripts/ml/latency_per_frame.py --frames 30
"""
import argparse
import glob
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

VIDEOS = ROOT / ".scratch" / "aslcit_vocab_videos"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, default=5)
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--size", default="256", help="'256' or 'native'")
    ap.add_argument("--warmup", type=int, default=3,
                    help="detector calls discarded before timing (graph init)")
    ap.add_argument("--out", default="results/latency_per_frame.json")
    args = ap.parse_args()

    import numpy as np
    import landmark_frontend as lf

    size = None if str(args.size).lower() == "native" else int(args.size)
    clips = sorted(glob.glob(str(VIDEOS / "*.mp4")))[: args.clips]

    out = {"size": args.size, "frames_per_clip": args.frames, "clips": len(clips),
           "warmup_calls": args.warmup, "modes": {}}

    for mode in ("image", "video"):
        # Fresh landmarker per mode so neither inherits the other's state.
        lf._LANDMARKER = None
        per_frame = []
        for c in clips:
            frames = lf.read_frames(c, max_frames=args.frames, size=size)
            if not frames:
                continue
            for _ in range(args.warmup):
                lf.detect_hands([frames[0]], mode=mode)
            for f in frames:
                t0 = time.perf_counter()
                lf.detect_hands([f], mode=mode)
                per_frame.append((time.perf_counter() - t0) * 1000.0)
        per_frame.sort()
        n = len(per_frame)
        rec = {
            "n": n,
            "p50_ms": round(statistics.median(per_frame), 2),
            "p95_ms": round(per_frame[min(n - 1, int(round(0.95 * (n - 1))))], 2),
            "min_ms": round(per_frame[0], 2),
            "mean_ms": round(statistics.fmean(per_frame), 2),
        }
        out["modes"][mode] = rec
        print(f"  {mode:5s} mode: n={n:4d}  p50 {rec['p50_ms']:7.2f} ms  "
              f"p95 {rec['p95_ms']:7.2f}  min {rec['min_ms']:7.2f}")

    print("\n  NOTE: desktop x86. This is NOT an on-device figure.")
    Path(ROOT / args.out).write_text(json.dumps(out, indent=2))
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
