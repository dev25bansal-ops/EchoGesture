"""CLI: run one sequential file-mode streaming execution.

This is not a live-camera claim. Reuse the installed project environment; do
not launch a second concurrent model process.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from streaming_runtime import run_landmark_file, run_video_file  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", nargs="?", help="mp4 path (or omit with --landmarks)")
    parser.add_argument("--landmarks", type=Path, help="timestamped .npz instead of a video")
    parser.add_argument("--out", type=Path, help="write JSON here (default: stdout)")
    parser.add_argument("--capacity", type=int, default=96)
    parser.add_argument("--max-capture-gap-ms", type=float, default=350.0)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()
    if args.landmarks:
        result, _session = run_landmark_file(
            args.landmarks, capacity=args.capacity, do_warmup=not args.no_warmup
        )
    elif args.video:
        result, _session = run_video_file(
            args.video,
            capacity=args.capacity,
            do_warmup=not args.no_warmup,
            max_frames=args.max_frames,
            max_capture_gap_ms=args.max_capture_gap_ms,
        )
    else:
        parser.error("provide a video path or --landmarks")
    text = json.dumps(result, indent=2)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    print(text)


if __name__ == "__main__":
    main()
