"""Extract hand landmarks from ASL Citizen clips using MediaPipe Tasks.

DEV-02 handoff artifact. One .npz per clip: raw 21x3 landmarks (normalised
image coords), a per-frame presence flag, the gloss and the participant.

Why the Tasks API and not the raw .tflite models: the palm-detection .tflite
files in `_altbench/mp` expose a 2016-row anchor tensor that it does not
document, and decoding it by hand produced detections that were not
believable (score 1.58, centre outside the image). MediaPipe's own
HandLandmarker task does detect-then-track internally and is the supported
path. Pin mediapipe==1.2.2.

Writes to `.scratch/landmarks/` which is gitignored - these are caches, not
artifacts. Only the vocabulary, the feature config and the model are shipped.

Usage (must use a 3.11 interpreter; ai_edge_litert and mediapipe both need it):
    .scratch/s3d_repro/venv311/Scripts/python.exe scripts/ml/extract_landmarks.py --split train
"""

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

ROOT = Path(__file__).resolve().parents[2]
ASLCIT = ROOT / ".scratch" / "eval_research" / "aslcit"
VIDEOS = ROOT / ".scratch" / "aslcit_vocab_videos"
OUT = ROOT / ".scratch" / "landmarks"
VOCAB = ROOT / ".scratch" / "vocab_best100.json"
TASK = ROOT / "models" / "hand_landmarker.task"


def ensure_task_model():
    """Download the HandLandmarker task bundle once."""
    if TASK.exists():
        return TASK
    import urllib.request
    TASK.parent.mkdir(parents=True, exist_ok=True)
    url = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
           "hand_landmarker/float16/1/hand_landmarker.task")
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, TASK)
    print(f"saved {TASK.stat().st_size/1e6:.1f} MB -> {TASK}")
    return TASK


def read_frames(path, size=256, max_frames=32):
    """Read `max_frames` frames spread evenly across the whole clip.

    Sampling only the opening frames loses signs whenever a clip starts on a
    dark or auto-exposing frame - one ASL Citizen clip opens almost black and
    yields zero hand detections across its first 32 frames. Striding with
    grab() (which advances without decoding) keeps the cost of covering the
    full clip close to reading 32 frames straight through.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames = []
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
            frames.append(cv2.cvtColor(cv2.resize(f, (size, size)), cv2.COLOR_BGR2RGB))
    else:
        while len(frames) < max_frames:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(cv2.resize(f, (size, size)), cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def make_landmarker(mode=vision.RunningMode.VIDEO):
    opts = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(ensure_task_model())),
        running_mode=mode,
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    return vision.HandLandmarker.create_from_options(opts)


def make_image_landmarker():
    """IMAGE mode avoids the monotonic-timestamp constraint across clips."""
    return make_landmarker(vision.RunningMode.IMAGE)


RECYCLE_EVERY = 200   # clips before rebuilding the landmarker


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "val", "test", "all"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--max-clips", type=int, default=0,
                    help="exit after processing this many clips (0 = no cap). "
                         "Bounding the run keeps MediaPipe's native state from "
                         "accumulating into an abort.")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    vocab = set(json.load(open(VOCAB, encoding="utf-8")))
    landmarker = make_image_landmarker()

    # val/test first when running all: they are small and were starved when
    # the process aborted partway through train.
    splits = ["val", "test", "train"] if args.split == "all" else [args.split]
    grand = dict(ok=0, miss=0, dead=0)
    processed = 0

    for split in splits:
        if args.max_clips and processed >= args.max_clips:
            break
        rows = list(csv.DictReader(open(ASLCIT / "splits" / f"{split}.csv", encoding="utf-8")))
        rows = [r for r in rows if r["Gloss"] in vocab]
        if args.limit:
            rows = rows[: args.limit]
        out_dir = OUT / split
        out_dir.mkdir(parents=True, exist_ok=True)

        ok = miss = dead = 0
        t0 = time.perf_counter()
        for n, r in enumerate(rows):
            if args.max_clips and processed >= args.max_clips:
                break
            src = VIDEOS / r["Video file"]
            dst = out_dir / (src.stem + ".npz")
            if dst.exists():
                ok += 1
                continue
            if not src.exists():
                dead += 1
                continue

            if processed and processed % RECYCLE_EVERY == 0:
                landmarker.close()
                landmarker = make_image_landmarker()

            try:
                frames = read_frames(src, max_frames=args.frames)
            except Exception:
                dead += 1
                continue
            if not frames:
                dead += 1
                continue

            # TWO hands per frame, in canonical left/right slots. Most ASL
            # signs are two-handed, so keeping only the dominant hand would
            # discard the sign. MediaPipe returns handedness labels, which are
            # from the camera's point of view; ASL Citizen clips are not
            # mirrored, so we map label "Left"/"Right" straight through and
            # never swap them.
            seq, present = [], []
            for f in frames:
                try:
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=f)
                    res = landmarker.detect(mp_img)
                except Exception:
                    res = None
                slots = {"Left": np.zeros((21, 3), np.float32),
                         "Right": np.zeros((21, 3), np.float32)}
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

            processed += 1

            # Write even when no hand was found. The presence flags carry that
            # fact to training, and recording it stops the resumable loop from
            # retrying the same undetectable clip on every pass.
            np.savez_compressed(dst,
                                landmarks=np.stack(seq),      # (T, 2, 21, 3)
                                present=np.array(present, np.float32),  # (T, 2)
                                gloss=r["Gloss"],
                                participant=r["Participant ID"],
                                split=split)
            if any(v for frame in present for v in frame):
                ok += 1
            else:
                miss += 1
            if processed % 50 == 0:
                el = time.perf_counter() - t0
                print(f"  [{split}] {n+1}/{len(rows)} ok={ok} miss={miss} dead={dead} "
                      f"{el:.0f}s ({processed/max(el,1e-9):.2f}/s)", flush=True)

        el = time.perf_counter() - t0
        print(f"{split}: ok={ok} no_hand={miss} unreadable={dead} of {len(rows)} "
              f"in {el:.0f}s", flush=True)
        grand["ok"] += ok; grand["miss"] += miss; grand["dead"] += dead

    landmarker.close()
    print(f"TOTAL ok={grand['ok']} no_hand={grand['miss']} unreadable={grand['dead']}")
    rate = grand["ok"] / max(grand["ok"] + grand["miss"], 1)
    print(f"hand-detection rate: {rate*100:.1f}%  "
          f"(the plan's fallback trigger is <80%)")
    print(f"processed {processed} clips this run")


if __name__ == "__main__":
    main()