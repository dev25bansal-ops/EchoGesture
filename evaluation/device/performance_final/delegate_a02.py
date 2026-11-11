"""Mesure le coût par image du détecteur MediaPipe selon le delegate.

Question posée : le delegate GPU (Metal) réduit-il le coût MediaPipe par image
par rapport au CPU sur cette machine ? Le gain est plafonné par le temps de
décodage (~14 ms par clip), donc seul un coût par image nettement inférieur
change la conclusion.

Ce laboratoire est autonome : il ne modifie ni ``src/`` ni le tracker. Il
mesure uniquement ``detect_hands`` (même options de landmarker que la
production : IMAGE, 2 mains, seuils 0.3) sur des images réelles lues avec le
lecteur de production.

Aucune prediction n'est produite ici ; la parité est vérifiée sur les
landmarks (np.array_equal) et non sur le modèle.
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

import numpy as np  # noqa: E402

LOCK = Path("/tmp/echogesture-heavy-benchmark.lock")
DEFAULT_CLIPS = ROOT / "evaluation/device/v5_accuracy/data/val_30"
DEFAULT_OUTPUT = HERE / "results_candidates/delegate.json"


def _make_landmarker(delegate_name: str):
    import mediapipe as mp  # noqa: F401
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions

    delegate = BaseOptions.Delegate.GPU if delegate_name == "gpu" else BaseOptions.Delegate.CPU
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(ROOT / "models/hand_landmarker.task"),
            delegate=delegate),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    return vision.HandLandmarker.create_from_options(options)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", type=Path, default=DEFAULT_CLIPS)
    parser.add_argument("--max-clips", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    import landmark_frontend as lf

    clips = sorted(args.clips.glob("*.mp4"))[:args.max_clips]
    try:
        os.mkdir(str(LOCK))
    except FileExistsError:
        print("refusing to start: heavy benchmark lock held", file=sys.stderr)
        return 2
    try:
        report = {"clips": [c.name for c in clips], "delegates": {}}
        landmarks_by_delegate = {}
        for name in ("cpu", "gpu"):
            try:
                landmarker = _make_landmarker(name)
            except Exception as exc:  # report, never hide
                report["delegates"][name] = {"error": f"{type(exc).__name__}: {exc}"}
                print(f"[{name}] unavailable: {type(exc).__name__}: {exc}")
                continue
            per_clip, per_frame = [], []
            landmarks_by_delegate[name] = {}
            try:
                for clip in clips:
                    frames = lf.read_frames(str(clip), max_frames=32)
                    landmarks_by_delegate[name][clip.name] = []
                    for _ in range(args.repeats):
                        started = time.perf_counter_ns()
                        lm, pr = lf.detect_hands(frames, landmarker=landmarker,
                                                 mode="image")
                        ms = (time.perf_counter_ns() - started) / 1e6
                        per_clip.append(ms)
                        per_frame.append(ms / max(1, len(frames)))
                        landmarks_by_delegate[name][clip.name].append(
                            (lm, pr))
            finally:
                landmarker.close()
            report["delegates"][name] = {
                "clip_ms_mean": statistics.mean(per_clip),
                "clip_ms_p50": statistics.median(per_clip),
                "per_frame_ms": statistics.mean(per_frame),
                "runs": len(per_clip),
            }
            print(f"[{name}] per-frame={statistics.mean(per_frame):.2f} ms "
                  f"clip={statistics.mean(per_clip):.1f} ms")

        if len(landmarks_by_delegate) == 2:
            identical = []
            diff = []
            for clip in clips:
                cpu = landmarks_by_delegate["cpu"][clip.name][0]
                gpu = landmarks_by_delegate["gpu"][clip.name][0]
                same = (np.array_equal(cpu[0], gpu[0])
                        and np.array_equal(cpu[1], gpu[1]))
                (identical if same else diff).append(clip.name)
            report["landmark_parity"] = {
                "identical": len(identical), "different": diff,
                "bit_identical_all": not diff,
            }
            print(f"landmark parity: {len(identical)}/{len(clips)} bit-identical, "
                  f"{len(diff)} differ")

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