"""Compare a checkpoint against the shipped 4-model ensemble on the test split.

Used to decide whether a newly-trained model (e.g. one trained on the merged
train+val split) actually beats what ships. Reports both alone and added to
the ensemble, because a model can be weak alone and still help, or strong
alone and redundant.

Usage:
    python scripts/ml/compare.py --ckpt .scratch/runs/pv201.pt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "ml"))

from features import build_features  # noqa: E402

LANDMARKS = ROOT / ".scratch" / "landmarks"
SHIPPED = ["tcn21", "tcnm61", "tcnm62", "rob71"]


def load_test(vocab_idx, frames=24):
    X, y = [], []
    for p in sorted((LANDMARKS / "test").glob("*.npz")):
        d = np.load(p, allow_pickle=False)
        if float(d["present"].max()) < 0.5:
            continue
        g = str(d["gloss"])
        if g not in vocab_idx:
            continue
        f, _ = build_features(d["landmarks"], d["present"], expected_frames=frames)
        X.append(f)
        y.append(vocab_idx[g])
    return np.stack(X).astype(np.float32), np.array(y)


def logits_of(path, X):
    import torch
    from train import TemporalModel
    ck = torch.load(path, map_location="cpu", weights_only=True)
    m = TemporalModel(len(ck["vocab"]), dim=int(ck["dim"]),
                      frames=int(ck["frames"]), backbone=str(ck["backbone"]),
                      dropout=float(ck.get("dropout", 0.2)))
    m.load_state_dict(ck["model"])
    m.eval()
    with torch.no_grad():
        return m(torch.from_numpy(X), None).numpy(), ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--frames", type=int, default=24)
    args = ap.parse_args()

    first = __import__("torch").load(ROOT / args.ckpt, map_location="cpu",
                                     weights_only=True)
    vocab_idx = {g: i for i, g in enumerate(first["vocab"])}
    X, y = load_test(vocab_idx, args.frames)
    print(f"test: {len(y)} clips, {len(vocab_idx)} classes\n")

    new_lg, ck = logits_of(ROOT / args.ckpt, X)
    print(f"  {Path(args.ckpt).stem:16s} alone            "
          f"{100*(new_lg.argmax(1)==y).mean():6.2f}%")

    shipped, names = [], []
    for n in SHIPPED:
        p = ROOT / ".scratch" / "runs" / f"{n}.pt"
        if not p.exists():
            continue
        lg, _ = logits_of(p, X)
        shipped.append(lg)
        names.append(n)
    if shipped:
        base = np.mean(shipped, axis=0)
        print(f"  {'shipped ensemble':16s} ({len(names)} models)  "
              f"{100*(base.argmax(1)==y).mean():6.2f}%")
        both = np.mean(shipped + [new_lg], axis=0)
        print(f"  {'+ new model':16s}                "
              f"{100*(both.argmax(1)==y).mean():6.2f}%")
        delta = 100 * ((both.argmax(1) == y).mean() - (base.argmax(1) == y).mean())
        print(f"\n  effect of adding it: {delta:+.2f} points")

    print(f"\n  checkpoint: backbone={ck['backbone']} dim={ck['dim']} "
          f"frames={ck['frames']} dropout={ck.get('dropout', 0.2)}")


if __name__ == "__main__":
    main()