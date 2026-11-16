"""Measure per-frame MediaPipe cost under a persistent detector pool.

`latency_per_frame.py` times one landmarker serially. MediaPipe releases the GIL
inside detect(), and there are 28 logical cores here, so a pool of K landmarkers
processing consecutive frames concurrently should drop the p95 if the serial call
is not already saturating the machine.

This measures the SAME number results.csv scores (per-frame detector cost against
the <=50 ms budget), just with a pool in front of it. It is a scheduling change
plus a measurement, not a model change: the detector, the frames and the features
are identical to the serial path.

Latency is reported per FRAME, not per clip, so it is comparable to the 133 ms
baseline and to the 50 ms budget. The wall-clock to drain a whole clip is also
printed, because that is what a user feels.

Usage:
    python scripts/ml/latency_pool.py --workers 4 --frames 32 --clips 5
    python scripts/ml/latency_pool.py --workers 1 --sweep
"""
import argparse
import glob
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

VIDEOS = ROOT / ".scratch" / "aslcit_vocab_videos"


def percentile(sorted_vals, q):
    if not sorted_vals:
        return None
    i = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, default=5)
    ap.add_argument("--frames", type=int, default=32, help="frames sampled per clip")
    ap.add_argument("--workers", type=int, default=4, help="size of the detector pool")
    ap.add_argument("--size", default="256", help="'256' or 'native'")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--sweep", action="store_true",
                    help="measure workers = 1,2,4,8 in one run")
    ap.add_argument("--out", default="results/latency_pool.json")
    args = ap.parse_args()

    import landmark_frontend as lf

    size = None if str(args.size).lower() == "native" else int(args.size)
    clips = sorted(glob.glob(str(VIDEOS / "*.mp4")))[: args.clips]
    if not clips:
        sys.exit(f"no clips under {VIDEOS}")

    # Decode every frame ONCE up front so decode time never lands in the
    # per-frame detector measurement (the serial script decodes inside the loop).
    all_frames = []
    for c in clips:
        fr = lf.read_frames(c, max_frames=args.frames, size=size)
        all_frames.extend(fr)
    if not all_frames:
        sys.exit("no frames decoded")

    def make_pool(k):
        lf._LANDMARKER = None  # each worker gets its own landmarker instance
        pool = [lf.get_landmarker("image") for _ in range(k)]
        # warm every instance so graph build never lands in a timed call
        for lm in pool:
            for _ in range(args.warmup):
                lf.detect_hands([all_frames[0]], landmarker=lm, mode="image")
        return pool

    def run(k):
        pool = make_pool(k)
        # One detector per frame, round-robin. detect_hands is single-frame here,
        # so per-frame cost is measured directly; concurrency is the only change.
        def one(i_frame):
            lm = pool[i_frame[0] % k]
            t0 = time.perf_counter()
            lf.detect_hands([i_frame[1]], landmarker=lm, mode="image")
            return (time.perf_counter() - t0) * 1000.0

        # Wall clock to drain the whole frame set, and per-frame service time.
        t_wall = time.perf_counter()
        with ThreadPoolExecutor(max_workers=k) as ex:
            per_frame = list(ex.map(one, list(enumerate(all_frames))))
        wall_ms = (time.perf_counter() - t_wall) * 1000.0
        per_frame.sort()
        n = len(per_frame)
        return {
            "workers": k,
            "n_frames": n,
            "per_frame_p50": round(statistics.median(per_frame), 2),
            "per_frame_p95": round(percentile(per_frame, 0.95), 2),
            "per_frame_min": round(per_frame[0], 2),
            "per_frame_max": round(per_frame[-1], 2),
            "frames_per_second": round(n / (wall_ms / 1000.0), 2),
            "clip_equivalent_ms": round(wall_ms / max(1, len(clips)), 2),
        }

    workers = [1, 2, 4, 8] if args.sweep else [args.workers]
    results = [run(k) for k in workers]

    print(f"\nsize={args.size}  frames/clip={args.frames}  clips={len(clips)}  "
          f"total frames={len(all_frames)}\n")
    hdr = f"{'workers':>7} {'p50':>8} {'p95':>8} {'min':>8} {'fps':>7} {'clip_ms':>9}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(f"{r['workers']:>7} {r['per_frame_p50']:>8.1f} {r['per_frame_p95']:>8.1f} "
              f"{r['per_frame_min']:>8.1f} {r['frames_per_second']:>7.1f} "
              f"{r['clip_equivalent_ms']:>9.1f}")

    out = {"size": args.size, "frames_per_clip": args.frames, "clips": len(clips),
           "results": results}
    (ROOT / args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    best = min(results, key=lambda r: r["per_frame_p95"])
    print(f"best p95: {best['per_frame_p95']:.1f} ms at {best['workers']} workers "
          f"(serial baseline ~199 ms p95)")


if __name__ == "__main__":
    main()
