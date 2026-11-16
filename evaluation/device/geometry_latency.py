"""Latency: native-resolution frames vs the shipped 256 squash.

Isolates the resize cost by running the same production predict() path twice,
patching only the frame geometry. Warm only — cold start is measured separately
by the caller.
"""

from __future__ import annotations

import glob
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import landmark_frontend as LF  # noqa: E402
import main as M  # noqa: E402


def bench(clips, size, n):
    orig = LF.read_frames

    def patched(video_path, max_frames=32, size_=256):
        return orig(video_path, max_frames=max_frames, size=size)

    LF.read_frames = patched
    try:
        ts = []
        for c in clips[:n]:
            t = time.perf_counter()
            M.predict(c)
            ts.append((time.perf_counter() - t) * 1000)
    finally:
        LF.read_frames = orig
    return ts


def main():
    clips = sorted(glob.glob(
        "/Users/abhijeetmiskin/AppData/EchoGesture/"
        "evaluation/device/v5_accuracy/data/val_30/*.mp4"))
    M._init_model()
    M.predict(clips[0])  # warm the graph

    out = {}
    for label, size in (("squash256", 256), ("native", None)):
        ts = bench(clips, size, 12)
        out[label] = ts
        print(f"{label:10s} p50={statistics.median(ts):7.1f} ms  "
              f"min={min(ts):7.1f}  max={max(ts):7.1f}  n={len(ts)}")

    a, b = out["squash256"], out["native"]
    print(f"\ndelta p50: {statistics.median(b) - statistics.median(a):+.1f} ms")
    print("note: laptop CPU, full clip-to-JSON incl. MediaPipe; not a phone")


if __name__ == "__main__":
    main()