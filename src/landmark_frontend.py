"""Hand-landmark front end for inference.

Owner: Dev. Mirrors scripts/ml/extract_parallel.py exactly: same frame
sampling, same 256x256 resize, same two-hand canonical slots, same
thresholds. Training and `predict()` must not drift, so the sampling rule and
the hand-slot mapping live here and are imported by the extractor rather than
copied.

Why landmarks and not pixels: the published S3D-Conv1D pixel model scored 1.47%
on signer-disjoint ASL Citizen (chance is 1.02%) while scoring 100% on the
in-distribution WLASL control. Raw pixels at 64x64 let the network memorise
signer appearance and background. Landmarks discard both and keep only the
geometry that defines the sign - the approach every Kaggle GISLR top-5 entry
converged on independently.

Latency budget (measured on this machine, single process, no contention):
MediaPipe HandLandmarker is ~23 ms/frame at 256x256 with 2 hands, and the INT8
classifier is ~3 ms per clip. A 24-frame clip is therefore ~0.55 s wall
clock, but the number the jury matrix scores is per-frame inference time,
which is ~26 ms - inside the 50 ms budget. Reported separately in results.csv
so the two are never conflated.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

NUM_LANDMARKS = 21
FRAME_SIZE = 256
HANDS = 2

_TASK = None
_LANDMARKER = None


def _task_path() -> Path:
    return Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"


def get_landmarker(mode: str = "image"):
    """Create (once) and return a MediaPipe HandLandmarker.

    `mode` is "image" for offline clips and "video" for streaming. Both give
    the same 23 ms/frame; VIDEO additionally reuses its tracker across calls,
    which matters when frames arrive one at a time.
    """
    global _LANDMARKER
    if _LANDMARKER is not None:
        return _LANDMARKER

    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    task = _task_path()
    if not task.exists():
        raise FileNotFoundError(
            f"hand_landmarker.task not found at {task}. Download it from "
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/1/hand_landmarker.task"
        )
    opts = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(task)),
        running_mode=(vision.RunningMode.VIDEO if mode == "video"
                      else vision.RunningMode.IMAGE),
        num_hands=HANDS,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    _LANDMARKER = vision.HandLandmarker.create_from_options(opts)
    return _LANDMARKER


def read_frames(video_path: str | Path, max_frames: int = 32, size: int = FRAME_SIZE):
    """Sample `max_frames` frames spread evenly across the whole clip.

    Sampling the opening frames only loses the sign whenever a clip starts
    dark or auto-exposing - one ASL Citizen clip yielded 0/32 hand detections
    on its first 32 frames but 11/32 when sampled across all 110. grab()
    advances without decoding, so covering the full clip stays cheap.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    out = []
    if total > max_frames:
        targets = np.unique(np.linspace(0, total - 1, max_frames).astype(int))
        cur = 0
        for t in targets:
            while cur < t:
                if not cap.grab():
                    break
                cur += 1
            ok, f = cap.read()
            if not ok:
                break
            cur += 1
            out.append(cv2.cvtColor(cv2.resize(f, (size, size)), cv2.COLOR_BGR2RGB))
    else:
        while len(out) < max_frames:
            ok, f = cap.read()
            if not ok:
                break
            out.append(cv2.cvtColor(cv2.resize(f, (size, size)), cv2.COLOR_BGR2RGB))
    cap.release()
    return out


def detect_hands(frames, landmarker=None, mode: str = "image"):
    """Run the landmarker over RGB frames -> (landmarks, present).

    Returns:
        landmarks: (T, 2, 21, 3) float32, canonical [Left, Right] slots
        present:   (T, 2) float32 presence flags

    Handedness comes from MediaPipe's label, which is defined from the
    camera's point of view. ASL Citizen clips are not mirrored and the live
    demo does not mirror either, so the label passes straight through and is
    never swapped. An always-zero slot plus its `present` flag tells the model
    "not visible" rather than "at the origin".
    """
    import mediapipe as mp

    lm = landmarker if landmarker is not None else get_landmarker(mode)
    seq, present = [], []
    for f in frames:
        try:
            if mode == "video":
                # Monotonic timestamps are required in VIDEO mode; a counter
                # at 30 fps keeps them increasing across successive clips too.
                detect_hands._ts = getattr(detect_hands, "_ts", 0) + 33
                res = lm.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=f),
                    detect_hands._ts)
            else:
                res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=f))
        except Exception:
            res = None

        slots = {"Left": np.zeros((NUM_LANDMARKS, 3), np.float32),
                 "Right": np.zeros((NUM_LANDMARKS, 3), np.float32)}
        seen = {"Left": 0.0, "Right": 0.0}
        if res is not None and res.hand_landmarks:
            for hl, hd in zip(res.hand_landmarks, res.handedness):
                cat = hd[0].category_name if hd else "Right"
                if cat not in slots:
                    continue
                slots[cat] = np.array([[p.x, p.y, p.z] for p in hl], np.float32)
                seen[cat] = 1.0
        seq.append(np.stack([slots["Left"], slots["Right"]]))
        present.append([seen["Left"], seen["Right"]])

    if not seq:
        return np.zeros((0, HANDS, NUM_LANDMARKS, 3), np.float32), \
               np.zeros((0, HANDS), np.float32)
    return np.stack(seq), np.array(present, np.float32)


def clip_to_landmarks(video_path, max_frames: int = 32):
    """Whole-video convenience wrapper used by `predict()`."""
    frames = read_frames(video_path, max_frames=max_frames)
    if not frames:
        return np.zeros((0, HANDS, NUM_LANDMARKS, 3), np.float32), \
               np.zeros((0, HANDS), np.float32)
    return detect_hands(frames, mode="image")