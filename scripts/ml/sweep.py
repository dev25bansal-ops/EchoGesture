"""Evaluate test-time augmentation and ensemble choices, quickly.

Batches every clip instead of looping per clip, so a full sweep over crops and
member subsets finishes in minutes rather than hours.

TTA tried here:
  temporal crops - evaluate several windows of the clip and average. The sign
                   occupies the middle of an ASL Citizen clip, so a window
                   centred on it should beat a fixed start-at-0 window.
  none           - single forward pass, the current shipped behaviour.

Usage:
    python scripts/ml/sweep.py --ckpts .scratch/runs/tcn21.pt ...
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "ml"))

from features import build_features  # noqa: E402

LANDMARKS = ROOT / ".scratch" / "landmarks"


def load(split, vocab_idx, frames=24):
    """Return (landmarks, present, labels) as object arrays plus flat features."""
    LM, PR, Y = [], [], []
    for p in sorted((LANDMARKS / split).glob("*.npz")):
        d = np.load(p, allow_pickle=False)
        if float(d["present"].max()) < 0.5:
            continue
        g = str(d["gloss"])
        if g not in vocab_idx:
            continue
        LM.append(d["landmarks"].astype(np.float32))
        PR.append(d["present"].astype(np.float32))
        Y.append(vocab_idx[g])
    return LM, PR, np.array(Y)


def build_at(LM, PR, frames, offset_frac):
    """Features for every clip, cropping `frames` starting at offset_frac of the clip."""
    out = []
    for lm, pr in zip(LM, PR):
        T = len(pr)
        if T > frames:
            span = T - frames
            s = int(round(span * offset_frac))
            lm2, pr2 = lm[s:s + frames], pr[s:s + frames]
        else:
            lm2, pr2 = lm, pr
        f, _ = build_features(lm2, pr2, expected_frames=frames)
        out.append(f)
    return np.stack(out).astype(np.float32)


def logits(models, X):
    import torch
    with torch.no_grad():
        return [m(torch.from_numpy(X), None).numpy() for m in models]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    import torch
    from train import TemporalModel

    models, vocab, frames = [], None, None
    for c in args.ckpts:
        ck = torch.load(c, map_location="cpu", weights_only=True)
        vocab = ck["vocab"] if vocab is None else vocab
        frames = int(ck["frames"])
        m = TemporalModel(len(ck["vocab"]), dim=int(ck["dim"]),
                          frames=frames, backbone=str(ck["backbone"]))
        m.load_state_dict(ck["model"])
        m.eval()
        models.append(m)
    vocab_idx = {g: i for i, g in enumerate(vocab)}

    LM, PR, Y = load(args.split, vocab_idx)
    print(f"{len(Y)} clips, {len(models)} model(s)\n")

    # ---- TTA sweep on the full ensemble
    print("temporal TTA (offset of the 24-frame window within the clip):")
    cache = {}
    for name, fracs in (("start", [0.0]),
                        ("centre", [0.5]),
                        ("end", [1.0]),
                        ("3-window", [0.0, 0.5, 1.0]),
                        ("5-window", [0.0, 0.25, 0.5, 0.75, 1.0])):
        acc = []
        for fr in fracs:
            X = cache.setdefault(fr, build_at(LM, PR, frames, fr))
            acc.append(np.mean(logits(models, X), axis=0))
        pred = np.mean(acc, axis=0).argmax(1)
        print(f"  {name:10s} {100*(pred == Y).mean():6.2f}%")

    # ---- member-subset sweep, using the best single window
    print("\nmember subsets (no TTA):")
    X = cache[0.5]
    L = {Path(c).stem: lg for c, lg in zip(args.ckpts, logits(models, X))}
    names = list(L)
    rows = []
    for r in range(1, len(names) + 1):
        for combo in itertools.combinations(names, r):
            acc = 100 * (np.mean([L[k] for k in combo], axis=0).argmax(1) == Y).mean()
            rows.append((acc, combo))
    rows.sort(reverse=True)
    for acc, combo in rows[:8]:
        print(f"  {acc:6.2f}%  {'+'.join(combo)}")


if __name__ == "__main__":
    main()