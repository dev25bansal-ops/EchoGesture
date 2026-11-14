"""Parallel, crash-resilient hand-landmark extraction.

Why a process pool rather than one long loop: MediaPipe aborts the whole
process natively (SIGABRT, not a Python exception) after enough clips, so no
try/except can save it. Isolating the landmarker in worker processes bounds
the damage to one in-flight clip, and a replacement worker picks up the rest.
It also uses the machine: MediaPipe runs ~76 ms/frame single-threaded, so a
serial pass over the corpus takes over an hour.

Each worker owns its own HandLandmarker and recycles it periodically. Results
are written to a temp file and renamed, so a worker killed mid-write cannot
leave a truncated .npz that the resumable loop would then skip.

Usage:
    .scratch/s3d_repro/venv311/Scripts/python.exe scripts/ml/extract_parallel.py --workers 10
"""

import argparse
import csv
import json
import os
import sys
import time
import traceback
from multiprocessing import Process, Queue
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ASLCIT = ROOT / ".scratch" / "eval_research" / "aslcit"
VIDEOS = ROOT / ".scratch" / "aslcit_vocab_videos"
OUT = ROOT / ".scratch" / "landmarks"
VOCAB = ROOT / ".scratch" / "vocab_best100.json"
TASK = ROOT / "models" / "hand_landmarker.task"
RECYCLE_EVERY = 150


def extract_one(landmarker, mp, mp_image, src, frames_n):
    """Read and detect one clip. Returns (landmarks, present) or None."""
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    imgs = []
    if total > frames_n:
        # Even stride across the clip. Sampling the opening frames only loses
        # the sign whenever a clip starts dark or auto-exposing - one clip
        # scored 0/32 hands on its first 32 frames but 11/32 when sampled
        # across its full 110.
        targets = np.unique(np.linspace(0, total - 1, frames_n).astype(int))
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
            imgs.append(cv2.cvtColor(cv2.resize(f, (256, 256)), cv2.COLOR_BGR2RGB))
    else:
        while len(imgs) < frames_n:
            ok, f = cap.read()
            if not ok:
                break
            imgs.append(cv2.cvtColor(cv2.resize(f, (256, 256)), cv2.COLOR_BGR2RGB))
    cap.release()
    if not imgs:
        return None

    seq, present = [], []
    for f in imgs:
        try:
            res = landmarker.detect(mp_image(image_format=mp.ImageFormat.SRGB, data=f))
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

    return np.stack(seq), np.array(present, np.float32)


def worker(wid, tasks, results, frames_n):
    """Pull clip paths off the shared queue until it is empty."""
    import numpy as np
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    landmarker = None
    done = 0
    try:
        while True:
            try:
                item = tasks.get_nowait()
            except Exception:
                break
            if item is None:
                break
            dst, src, gloss, participant, split = item
            try:
                if landmarker is None or done % RECYCLE_EVERY == 0:
                    if landmarker is not None:
                        try:
                            landmarker.close()
                        except Exception:
                            pass
                    opts = vision.HandLandmarkerOptions(
                        base_options=mp_python.BaseOptions(model_asset_path=str(TASK)),
                        running_mode=vision.RunningMode.IMAGE,
                        num_hands=2,
                        min_hand_detection_confidence=0.3,
                        min_hand_presence_confidence=0.3,
                        min_tracking_confidence=0.3,
                    )
                    landmarker = vision.HandLandmarker.create_from_options(opts)

                out = extract_one(landmarker, mp, mp.Image, Path(src), frames_n)
                if out is None:
                    results.put(("dead", dst, 0))
                    done += 1
                    continue
                lm, pres = out
                # np.savez_compressed appends ".npz" unless the path already
                # ends with it, so hand it an open file handle to keep the
                # temp name exact and the rename atomic.
                tmp = str(dst) + f".tmp{wid}"
                with open(tmp, "wb") as fh:
                    np.savez_compressed(fh, landmarks=lm, present=pres,
                                        gloss=gloss, participant=participant, split=split)
                os.replace(tmp, dst)
                results.put(("ok" if pres.max() > 0.5 else "miss", dst, len(lm)))
                done += 1
            except Exception:
                results.put(("err", dst, traceback.format_exc()[-300:]))
                done += 1
    finally:
        if landmarker is not None:
            try:
                landmarker.close()
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 3))
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--splits", default="val,test,train")
    ap.add_argument("--limit", type=int, default=0, help="cap clips this run (0 = all)")
    args = ap.parse_args()

    vocab = set(json.load(open(VOCAB, encoding="utf-8")))
    queue_items = []
    for split in args.splits.split(","):
        split = split.strip()
        if not split:
            continue
        out_dir = OUT / split
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = list(csv.DictReader(open(ASLCIT / "splits" / f"{split}.csv", encoding="utf-8")))
        rows = [r for r in rows if r["Gloss"] in vocab]
        for r in rows:
            src = VIDEOS / r["Video file"]
            if not src.exists():
                continue
            dst = out_dir / (src.stem + ".npz")
            if dst.exists():
                continue
            queue_items.append((str(dst), str(src), r["Gloss"],
                                r["Participant ID"], split))

    if args.limit:
        queue_items = queue_items[: args.limit]
    if not queue_items:
        print("nothing left to extract")
        return 0

    print(f"{len(queue_items)} clips across {args.workers} workers "
          f"({args.frames} frames each)", flush=True)

    tasks, results = Queue(), Queue()
    for item in queue_items:
        tasks.put(item)

    procs = [Process(target=worker, args=(i, tasks, results, args.frames), daemon=True)
             for i in range(args.workers)]
    t0 = time.perf_counter()
    for p in procs:
        p.start()

    tally = {"ok": 0, "miss": 0, "dead": 0, "err": 0}
    finished = 0
    last_report = 0.0
    while finished < len(queue_items):
        try:
            kind, dst, extra = results.get(timeout=30)
        except Exception:
            alive = [p for p in procs if p.is_alive()]
            if not alive:
                print("all workers exited early", flush=True)
                break
            continue
        tally[kind] += 1
        finished += 1
        if kind == "err":
            print(f"  ERR {dst}\n{extra}", flush=True)

        now = time.perf_counter()
        if now - last_report > 20 or finished == len(queue_items):
            last_report = now
            el = now - t0
            rate = finished / max(el, 1e-9)
            eta = (len(queue_items) - finished) / max(rate, 1e-9)
            print(f"  {finished}/{len(queue_items)} {kind}:{tally[kind]} "
                  f"{rate:.2f}/s eta {eta/60:.1f}m  "
                  f"[ok {tally['ok']} miss {tally['miss']} dead {tally['dead']} err {tally['err']}]",
                  flush=True)

    for p in procs:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()

    el = time.perf_counter() - t0
    print(f"\ndone {finished}/{len(queue_items)} in {el/60:.1f} min  "
          f"ok={tally['ok']} no_hand={tally['miss']} unreadable={tally['dead']} errors={tally['err']}")
    for s in ("train", "val", "test"):
        n = len(list((OUT / s).glob("*.npz"))) if (OUT / s).exists() else 0
        print(f"  {s}: {n} cached")
    return 0


if __name__ == "__main__":
    sys.exit(main())