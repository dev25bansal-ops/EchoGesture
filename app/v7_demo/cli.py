"""CLI for the A10 V7 offline demo wrappers.

    python -m app.v7_demo --video <clip.mp4> [--render-output hud.png]
    python -m app.v7_demo --camera 0 --headless [--captures 1]

File mode reads the clip in place, runs the real ``src/main.py::predict``
once, renders the width-fitted HUD, and exits 0 (CAPTION/RETRY) or 1
(error). Camera mode never claims success without a real frame probe;
on TCC denial it prints the macOS guidance and exits 2.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.v7_demo.reliable import (  # noqa: E402
    PROVISIONAL_THRESHOLD,
    run_file,
    run_session,
)
from app.offline_demo import CameraUnavailableError  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--threshold", type=float, default=PROVISIONAL_THRESHOLD)
    parser.add_argument("--predictor", type=Path, default=ROOT / "src/main.py")
    parser.add_argument("--video", type=Path, default=None,
                        help="Camera-less fallback: real predict() on one clip")
    parser.add_argument("--render-output", type=Path, default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--captures", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--result-timeout", type=float, default=30.0)
    args = parser.parse_args(argv)
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be in [0, 1]")
    if args.captures < 1:
        parser.error("--captures must be >= 1")
    try:
        if args.video:
            report = run_file(args.video, args.predictor, args.threshold,
                              args.render_output)
        else:
            report = run_session(
                args.camera, args.seconds, args.predictor, args.threshold,
                headless=args.headless or True,
                max_captures=args.captures, max_frames=args.max_frames,
                result_timeout_s=args.result_timeout)
    except CameraUnavailableError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report.to_dict(), indent=2))
    return report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
