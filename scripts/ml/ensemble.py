"""Ensemble several trained seeds and measure signer-disjoint accuracy.

Averaging logits across independently-seeded runs is the cheapest reliable gain
in this setting: each model makes different errors on unseen signers, and the
mean cancels part of that variance. It needs no new data, which matters because
the corpus is the binding constraint (17-24 clips per class) and no more clips
exist.

Reports val AND test. Test is the only number that counts; val is included so a
regression shows up in both.

Usage:
    python scripts/ml/ensemble.py --ckpts .scratch/runs/model.pt \
        .scratch/runs/seed7.pt .scratch/runs/seed13.pt
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "ml"))

from features import build_features  # noqa: E402

LANDMARKS = ROOT / ".scratch" / "landmarks"


def load_split(vocab_idx, frames, split):
    """Raw landmarks, presence, labels and signers for one split.

    Features are built per checkpoint, not here: a 255-wide checkpoint needs
    x,y and a 381-wide one x,y,z, so one shared feature matrix would force a
    single width for every member.
    """
    LM, PR, y, signers = [], [], [], []
    for p in sorted((LANDMARKS / split).glob("*.npz")):
        d = np.load(p, allow_pickle=False)
        if float(d["present"].max()) < 0.5:
            continue
        g = str(d["gloss"])
        if g not in vocab_idx:
            continue
        LM.append(d["landmarks"])
        PR.append(d["present"])
        y.append(vocab_idx[g])
        signers.append(str(d["participant"]))
    return LM, PR, np.array(y), signers


def logits_for(ckpt_path, LM, PR):
    """Softmax-free logits from one checkpoint."""
    import torch
    from train import TemporalModel

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    # in_dim from the checkpoint's stem: five of the nine shipped members are
    # 381-wide z models, and the 255 default failed to load them ("size
    # mismatch for stem.weight"), so those members could not be ensembled.
    in_dim = int(ck["model"]["stem.weight"].shape[1])
    frames = int(ck["frames"])
    m = TemporalModel(len(ck["vocab"]), dim=int(ck["dim"]),
                      frames=frames, backbone=str(ck["backbone"]),
                      in_dim=in_dim)
    m.load_state_dict(ck["model"])
    m.eval()
    use_z = in_dim != 255
    X = np.stack([build_features(lm, pr, expected_frames=frames, use_z=use_z)[0]
                  for lm, pr in zip(LM, PR)]).astype(np.float32)
    with torch.no_grad():
        out = m(torch.from_numpy(X), None).numpy()
    return out, ck


def report(name, preds, y, signers):
    acc = float((preds == y).mean())
    per = {}
    for p, ok in zip(signers, (preds == y).tolist()):
        c, t = per.get(p, (0, 0))
        per[p] = (c + int(ok), t + 1)
    accs = [c / t for c, t in per.values()]
    print(f"{name:24s} {acc*100:6.2f}%   per-signer mean {100*np.mean(accs):5.1f}% "
          f"min {100*min(accs):5.1f}%")
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--splits", default="val,test")
    args = ap.parse_args()

    first = __import__("torch").load(args.ckpts[0], map_location="cpu", weights_only=True)
    vocab = first["vocab"]
    frames = int(first["frames"])
    vocab_idx = {g: i for i, g in enumerate(vocab)}

    for split in args.splits.split(","):
        split = split.strip()
        LM, PR, y, signers = load_split(vocab_idx, frames, split)
        print(f"\n=== {split} ({len(y)} clips) ===")

        all_logits = []
        for c in args.ckpts:
            lg, _ = logits_for(c, LM, PR)
            all_logits.append(lg)
            report(Path(c).stem, lg.argmax(1), y, signers)

        if len(all_logits) > 1:
            mean_lg = np.mean(all_logits, axis=0)
            report(f"ENSEMBLE ({len(all_logits)})", mean_lg.argmax(1), y, signers)


if __name__ == "__main__":
    main()