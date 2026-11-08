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
    cap = cv2.VideoCapture(str(path))
    frames = []
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "val", "test", "all"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--frames", type=int, default=32)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    vocab = set(json.load(open(VOCAB, encoding="utf-8")))
    landmarker = make_image_landmarker()

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    grand = dict(ok=0, miss=0, dead=0)

    for split in splits:
        rows = list(csv.DictReader(open(ASLCIT / "splits" / f"{split}.csv", encoding="utf-8")))
        rows = [r for r in rows if r["Gloss"] in vocab]
        if args.limit:
            rows = rows[: args.limit]
        out_dir = OUT / split
        out_dir.mkdir(parents=True, exist_ok=True)

        ok = miss = dead = 0
        t0 = time.perf_counter()
        for n, r in enumerate(rows):
            src = VIDEOS / r["Video file"]
            dst = out_dir / (src.stem + ".npz")
            if dst.exists():
                ok += 1
                continue
            if not src.exists():
                dead += 1
                continue

            frames = read_frames(src, max_frames=args.frames)
            if not frames:
                dead += 1
                continue

            # 21x3 for the dominant hand per frame, plus presence
            seq, present = [], []
            for f in frames:
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=f)
                res = landmarker.detect(mp_img)
                if res.hand_landmarks:
                    # pick the hand with the largest bounding box (closest)
                    best, best_area = None, -1.0
                    for hl in res.hand_landmarks:
                        xs = [p.x for p in hl]; ys = [p.y for p in hl]
                        area = (max(xs) - min(xs)) * (max(ys) - min(ys))
                        if area > best_area:
                            best, best_area = hl, area
                    seq.append(np.array([[p.x, p.y, p.z] for p in best], np.float32))
                    present.append(1.0)
                else:
                    seq.append(np.zeros((21, 3), np.float32))
                    present.append(0.0)

            if not any(present):
                miss += 1
                continue

            np.savez_compressed(dst,
                                landmarks=np.stack(seq),
                                present=np.array(present, np.float32),
                                gloss=r["Gloss"],
                                participant=r["Participant ID"],
                                split=split)
            ok += 1
            if (n + 1) % 100 == 0:
                el = time.perf_counter() - t0
                print(f"  [{split}] {n+1}/{len(rows)} ok={ok} miss={miss} dead={dead} "
                      f"{el:.0f}s ({ok/max(el,1e-9):.2f}/s)", flush=True)

        el = time.perf_counter() - t0
        print(f"{split}: ok={ok} no_hand={miss} unreadable={dead} of {len(rows)} "
              f"in {el:.0f}s ({ok/max(el,1e-9):.2f}/s)", flush=True)
        grand["ok"] += ok; grand["miss"] += miss; grand["dead"] += dead

    print(f"TOTAL ok={grand['ok']} no_hand={grand['miss']} unreadable={grand['dead']}")
    rate = grand["ok"] / max(grand["ok"] + grand["miss"], 1)
    print(f"hand-detection rate: {rate*100:.1f}%  "
          f"(the plan's fallback trigger is <80%)")


if __name__ == "__main__":
    main()