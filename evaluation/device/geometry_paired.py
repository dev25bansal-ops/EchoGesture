"""Paired geometry comparison through the production evaluator.

Runs the FROZEN src/main.py predict() path twice per clip — 256-squash (shipped
default) and native-resolution — using the same nine models, mask, features and
aggregator. Nothing about the model changes; only the frame resize does.

The native path here is a *local harness*: it does not modify src/, it patches
only the resize call so the rest of the production pipeline is exercised
unchanged.

Usage:
    python evaluation/device/geometry_paired.py --clips DIR [--limit N] --out FILE
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import landmark_frontend as LF  # noqa: E402
import main as M  # noqa: E402


def truth_of(path: str) -> str:
    base = Path(path).stem
    return base.split("-", 1)[1].replace(" ", "").upper() if "-" in base else base.upper()


def norm(pred: str, truth: str) -> bool:
    return pred == truth or pred.rstrip("1234567890") == truth or pred == truth + "1"


def run_with(geometry: str, clips: list[str]) -> list[dict]:
    """Run the production predict() with the resize patched for `geometry`."""
    original = LF.read_frames

    def read_frames_native(video_path, max_frames=32, size=256):
        frames = _decode(video_path, max_frames)
        return frames  # no resize at all

    def read_frames_letterbox(video_path, max_frames=32, size=256):
        frames = _decode(video_path, max_frames)
        out = []
        for f in frames:
            h, w = f.shape[:2]
            s = max(h, w)
            canvas = cv2.copyMakeBorder(
                f, 0, s - h, 0, s - w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            out.append(cv2.resize(canvas, (size, size)))
        return out

    if geometry == "native":
        LF.read_frames = read_frames_native
    elif geometry == "letterbox512":
        LF.read_frames = read_frames_letterbox
    else:
        LF.read_frames = original

    results = []
    try:
        for c in clips:
            try:
                r = M.predict(c)
                t = truth_of(c)
                results.append({
                    "clip_id": r["clip_id"], "truth": t,
                    "prediction": r["prediction"], "top5": r["top5"],
                    "confidence": r["confidence"], "latency_ms": r["latency_ms"],
                    "correct": norm(r["prediction"], t),
                })
            except Exception as e:  # noqa: BLE001
                results.append({"clip": Path(c).name,
                                "error": f"{type(e).__name__}: {e}"})
    finally:
        LF.read_frames = original
    return results


def _decode(video_path, max_frames):
    """Same sampling logic as read_frames, without the resize."""
    import numpy as np
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
            out.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    else:
        while len(out) < max_frames:
            ok, f = cap.read()
            if not ok:
                break
            out.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--geometries", default="squash256,native")
    args = ap.parse_args()

    clips = sorted(glob.glob(str(Path(args.clips) / "*.mp4")))
    if args.limit:
        clips = clips[: args.limit]
    if not clips:
        print("no clips found", file=sys.stderr)
        return 2

    payload = {"clips": len(clips), "geometries": {}}
    for g in args.geometries.split(","):
        g = g.strip()
        payload["geometries"][g] = run_with(g, clips)
        ok = sum(1 for r in payload["geometries"][g] if r.get("correct"))
        err = sum(1 for r in payload["geometries"][g] if "error" in r)
        print(f"{g:14s} {ok}/{len(clips)} = {100*ok/len(clips):.2f}%  errors={err}")

    # paired comparison on clips that scored in both
    names = list(payload["geometries"])
    if len(names) == 2:
        a, b = (payload["geometries"][n] for n in names)
        da = {r["clip_id"]: r["correct"] for r in a if "clip_id" in r}
        db = {r["clip_id"]: r["correct"] for r in b if "clip_id" in r}
        common = set(da) & set(db)
        only_a = sum(1 for c in common if da[c] and not db[c])
        only_b = sum(1 for c in common if db[c] and not da[c])
        both = sum(1 for c in common if da[c] and db[c])
        # McNemar exact (binomial) two-sided on discordant pairs
        n = only_a + only_b
        if n:
            from math import comb
            k = min(only_a, only_b)
            p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
        else:
            p = 1.0
        payload["paired"] = {
            "n": len(common), "both_correct": both,
            f"{names[0]}_only": only_a, f"{names[1]}_only": only_b,
            "mcnemar_exact_p": round(p, 6),
        }
        print(f"paired n={len(common)}  {names[0]}-only={only_a}  "
              f"{names[1]}-only={only_b}  exact p={p:.6f}")

    Path(args.out).write_text(json.dumps(payload, indent=1))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())