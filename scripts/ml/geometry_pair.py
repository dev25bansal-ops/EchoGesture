"""Paired per-clip comparison of detector input geometry, clean condition.

The 2x2 matrix (results/pixel_matrix/) showed clean top-1 moving from 85.83%
(256 squash) to 90.83% (512 letterbox) on the same 120 clips with the same nine
checkpoints. That is a train/test geometry mismatch the shipped models were never
asked about: they were trained on 256-squash landmarks, and the app itself feeds
MediaPipe native 640x480. This script scores the SAME clips under BOTH geometries
and runs McNemar on the discordant pairs, so the claim rests on a paired test
rather than two independent 120-clip proportions.

Usage:
    python scripts/ml/geometry_pair.py [--limit 120] [--out results/geometry_pair.json]
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from math import erfc, sqrt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "ml"))

from features import build_features  # noqa: E402
from landmark_frontend import read_frames, detect_hands  # noqa: E402
from train import TemporalModel  # noqa: E402

VIDEOS = ROOT / ".scratch" / "aslcit_vocab_videos"
SPLIT = ROOT / ".scratch" / "eval_research" / "aslcit" / "splits" / "test.csv"

CKPTS = [
    ".scratch/runs/tcn21.pt", ".scratch/runs/tcnm61.pt",
    ".scratch/runs/tcnm62.pt", ".scratch/runs/rob71.pt",
    ".scratch/search/mirror_handdrop_tcn_z_s0.pt",
    ".scratch/search/mirror_handdrop_tcn_z_s1.pt",
    ".scratch/search/mirror_handdrop_tcn_z_s2.pt",
    ".scratch/search/mirror_handdrop_tcn_z_s7.pt",
    ".scratch/search/mirror_handdrop_tcn_z_s13.pt",
]


def load_models():
    models, labels = [], None
    for c in CKPTS:
        ck = torch.load(ROOT / c, map_location="cpu", weights_only=True)
        if labels is None:
            labels = ck["vocab"]
        m = TemporalModel(len(ck["vocab"]), dim=int(ck["dim"]),
                          frames=int(ck["frames"]), backbone=str(ck["backbone"]),
                          in_dim=int(ck["model"]["stem.weight"].shape[1]))
        m.load_state_dict(ck["model"])
        m.eval()
        models.append(m)
    return models, labels


def predict(models, lm, pres):
    """Average LOGITS across members, then one softmax.

    This is what src/main.py and LandmarkClassifier.kt both do, so it is the
    number the app actually produces. An earlier version of this script
    averaged per-member softmax probabilities instead, which is a different
    estimator - mean-probability aggregation measured +0.9 on test / -0.9 on
    val against logit averaging and was rejected (results/EXPERIMENTS.md).
    Scoring geometry with the rejected aggregator would have compared the
    shipped models at an operating point we do not ship.
    """
    logits = []
    for m in models:
        uz = m.stem.weight.shape[1] != 255
        f, mk = build_features(lm, pres, expected_frames=24, use_z=uz)
        with torch.no_grad():
            lg = m(torch.from_numpy(f)[None], torch.from_numpy(mk)[None])
        logits.append(lg[0].numpy())
    avg = np.mean(logits, axis=0)
    e = np.exp(avg - avg.max())
    return e / e.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=120, help="0 = every clip in the split")
    ap.add_argument("--out", default="results/geometry_pair.json")
    args = ap.parse_args()

    models, labels = load_models()
    idx = {g: i for i, g in enumerate(labels)}

    rows = list(csv.DictReader(open(SPLIT, encoding="utf-8")))
    rows = [r for r in rows if r["Gloss"] in idx]
    if args.limit:
        rows = rows[: args.limit]

    # squash256 is what the shipped models were TRAINED on. letterbox512 is what
    # the robustness work settled on. app640x480 is what CameraScreen.kt actually
    # feeds MediaPipe (setTargetResolution(640,480), no resize in HandTracker).
    geoms = {"squash256": (256, False), "letterbox512": (512, True),
             "app640x480": (640, True)}
    per_clip = []
    for r in rows:
        p = VIDEOS / (Path(r["Video file"]).stem + ".mp4")
        if not p.exists():
            continue
        rec = {"clip": r["Video file"], "truth": r["Gloss"]}
        for name, (size, lb) in geoms.items():
            fr = read_frames(str(p), max_frames=32, size=size, letterbox=lb)
            if not fr:
                rec[name] = None
                continue
            lm, pres = detect_hands(fr, mode="image")
            if lm.shape[0] == 0 or float(pres.max()) < 0.5:
                rec[name] = None
                continue
            q = predict(models, lm, pres)
            rec[name] = labels[int(q.argmax())]
        per_clip.append(rec)

    n = len(per_clip)

    # Every geometry must be scored on ONE clip set, or the three rows of the
    # table are not comparable. Restrict to clips that every geometry resolved.
    common = [r for r in per_clip
              if all(r.get(k) is not None for k in geoms)]

    def mcnemar(x_name, y_name):
        """Paired test over `common`, so all comparisons share a denominator.

        Two bugs fixed 15 Nov:
          * the denominator was `both + neither + a + b`, counting the
            concordant pairs McNemar never uses, so each comparison's accuracy
            came out over its own set and the same geometry got two values;
          * `p` used `2*(1 - Phi(|z|))` unclamped, which exceeds 1 whenever
            `|a-b| <= 1` (a=b=1 gave p=1.52). `erfc(|z|/sqrt2)` is the
            two-sided p and cannot exceed 1.
        """
        a = b = both = neither = 0
        for rec in common:
            x, y = rec[x_name], rec[y_name]
            cx, cy = x == rec["truth"], y == rec["truth"]
            both += cx and cy
            neither += (not cx) and (not cy)
            a += cx and not cy
            b += cy and not cx
        n_cmp = both + neither + a + b
        if (a + b) == 0:
            z, p_two = 0.0, 1.0
        else:
            z = (abs(a - b) - 1) / sqrt(a + b)
            p_two = min(1.0, erfc(abs(z) / sqrt(2)))
        return {
            "n": n_cmp,
            "both_correct": both, "both_wrong": neither,
            f"{x_name}_only_right": a, f"{y_name}_only_right": b,
            "mcnemar_z": round(z, 3), "mcnemar_p_two_sided": round(p_two, 4),
            f"{x_name}_acc_pct": round(100 * (both + a) / max(1, n_cmp), 2),
            f"{y_name}_acc_pct": round(100 * (both + b) / max(1, n_cmp), 2),
        }

    out = {
        "n_clips": n,
        "geometries": {k: {"size": s, "letterbox": l} for k, (s, l) in geoms.items()},
        "squash256_vs_letterbox512": mcnemar("squash256", "letterbox512"),
        "squash256_vs_app640x480": mcnemar("squash256", "app640x480"),
        "letterbox512_vs_app640x480": mcnemar("letterbox512", "app640x480"),
        "per_clip": per_clip,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))

    print(f"clips scored: {n}\n")
    for key in ("squash256_vs_letterbox512", "squash256_vs_app640x480",
                "letterbox512_vs_app640x480"):
        m = out[key]
        x, y = key.split("_vs_")
        print(f"{key}")
        print(f"  n {m['n']}  both-ok {m['both_correct']}"
              f"  both-bad {m['both_wrong']}  {x}-only {m[f'{x}_only_right']}"
              f"  {y}-only {m[f'{y}_only_right']}")
        print(f"  {x:14s} {m[f'{x}_acc_pct']:6.2f}%   {y:14s} {m[f'{y}_acc_pct']:6.2f}%"
              f"   McNemar z={m['mcnemar_z']:.2f} p={m['mcnemar_p_two_sided']:.4f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
