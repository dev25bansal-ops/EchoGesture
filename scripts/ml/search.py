"""Search the augmentation and architecture space that the shipped run did not cover.

The shipped 4-member ensemble is two backbones (conv, tcn) x 2 seeds each, all
trained with the same augmentation. `results/EXPERIMENTS.md` records what was
rejected: more data (+0.24, noise), INT8 (-6.37), wider dim (-15), 32 frames
(-8), speed perturbation (-5.45), strict resample (-8), mirror TTA on a
mirror-trained model (-1.01).

Measured error structure (13 Nov, .scratch/oracle.py):

    both hands visible   76.8% over 1,138 test clips
    one hand <25%        31.5% over   111 test clips
    single-hand errors   76 of 340 = 22% of all errors

So the model is not uniformly weak; it collapses when a hand leaves frame, and
that single bucket holds a fifth of the loss. The winning augmentation should
therefore be the one that trains on that condition - mirror (handedness) and
stronger hand dropout - not more capacity or more epochs.

This script trains candidate models with a configurable recipe and evaluates
every one on val AND test, because a change that only wins on test at n=1249
is selection noise. McNemar against the shipped baseline decides.

Usage:
    python scripts/ml/search.py --list
    python scripts/ml/search.py --config mirror_handdrop --seed 0
    python scripts/ml/search.py --sweep
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "ml"))

from features import build_features, FEATURE_DIM, feature_dim  # noqa: E402
from train import TemporalModel, index_split, LANDMARKS, VOCAB  # noqa: E402

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import Dataset, DataLoader  # noqa: E402

OUT = ROOT / ".scratch" / "search"


# ------------------------------------------------------------------ augment

def mirror(lm, pres):
    """Swap hand slots and mirror x. Left/right handedness, same sign."""
    lm2 = lm.copy()
    lm2[:, 0], lm2[:, 1] = lm[:, 1].copy(), lm[:, 0].copy()
    lm2[..., 0] = 1.0 - lm2[..., 0]
    return lm2, pres[:, ::-1].copy()


def hand_dropout(lm, pres, p=0.5, strength=0.5):
    """Zero one hand slot for a contiguous span.

    The shipped augmentation caps the span at half the clip and fires on 50% of
    clips. Measured: single-hand clips score 31.5% vs 76.8%. 22% of test errors
    sit in that bucket, and only 4% of test clips reach <25% single-hand
    presence, so the model meets the condition rarely and handles it badly.
    Strength >0.5 makes long absences the norm during training.
    """
    if np.random.rand() >= p:
        return lm, pres
    n = np.random.randint(1, max(2, int(len(pres) * strength)))
    start = np.random.randint(0, max(1, len(pres) - n))
    pres = pres.copy()
    which = np.random.randint(0, 2)
    pres[start:start + n, which] = 0.0
    if np.random.rand() < 0.25:
        pres[start:start + n, 1 - which] = 0.0
    return lm, pres


RECIPES = {
    # The shipped recipe, reproduced here so the harness validates against it.
    "base": dict(mirror_p=0.0, handdrop_p=0.5, handdrop_strength=0.5,
                 noise=0.5, jitter=0.1, backbone="conv", dim=192, epochs=60),
    # Mirror augmentation: full handedness invariance.
    "mirror": dict(mirror_p=0.5, handdrop_p=0.5, handdrop_strength=0.5,
                   noise=0.5, jitter=0.1, backbone="conv", dim=192, epochs=60),
    # Stronger hand dropout: meet the failure condition far more often.
    "handdrop": dict(mirror_p=0.0, handdrop_p=0.7, handdrop_strength=0.85,
                     noise=0.5, jitter=0.1, backbone="conv", dim=192, epochs=60),
    # Both, which is the combination the error structure argues for.
    "mirror_handdrop": dict(mirror_p=0.5, handdrop_p=0.7, handdrop_strength=0.85,
                            noise=0.5, jitter=0.1, backbone="conv", dim=192, epochs=60),
    # Same but on the tcn backbone, for ensemble diversity.
    "mirror_handdrop_tcn": dict(mirror_p=0.5, handdrop_p=0.7, handdrop_strength=0.85,
                                noise=0.5, jitter=0.1, backbone="tcn", dim=192, epochs=60),
    # Lower dropout: the model overfits hard (99.4% train), so more regularisation
    # from the augmentations may pay without needing a capacity change.
    "mirror_handdrop_drop3": dict(mirror_p=0.5, handdrop_p=0.7, handdrop_strength=0.85,
                                  noise=0.5, jitter=0.1, backbone="conv", dim=192,
                                  epochs=60, dropout=0.3),
    # The representation ablation. A third-party repo reports dropping z is
    # worth +7.1pp (transformer) to +11.3pp (TCN) on WLASL-100, but that is a
    # different corpus and we have never reproduced it. Our z spans -0.071 to
    # 0.006, about a tenth of the x/y range, so the signal may not transfer.
    "mirror_handdrop_tcn_z": dict(mirror_p=0.5, handdrop_p=0.7, handdrop_strength=0.85,
                                  noise=0.5, jitter=0.1, backbone="tcn", dim=192,
                                  epochs=60, use_z=True),
    "base_z": dict(mirror_p=0.0, handdrop_p=0.5, handdrop_strength=0.5,
                   noise=0.5, jitter=0.1, backbone="conv", dim=192, epochs=60,
                   use_z=True),
}


# ------------------------------------------------------------------ dataset

# Decoding a compressed .npz is a zip inflate, and at 1,789 clips x 60 epochs
# that is 107,000 of them per run — measured at 41 ms/batch, which is Python
# and I/O, not GPU. The whole corpus as float32 is ~220 MB, so it is held in
# RAM once and indexed from there.
_CACHE = {}


def preload(items):
    """Read every clip once into RAM. Keyed by path."""
    for path, _label, _signer in items:
        if path in _CACHE:
            continue
        d = np.load(path, allow_pickle=False)
        _CACHE[path] = (d["landmarks"].astype(np.float32),
                        d["present"].astype(np.float32))
    return len(_CACHE)


def ensure_cached(items):
    """Populate on demand so every caller works regardless of what was preloaded."""
    missing = [it for it in items if it[0] not in _CACHE]
    if missing:
        preload(missing)


class ClipDataset(Dataset):
    def __init__(self, items, frames, recipe, train=False):
        self.items, self.frames, self.r, self.train = items, frames, recipe, train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label, signer = self.items[i]
        lm, pres = _CACHE[path]
        lm, pres = lm.copy(), pres.copy()

        if self.train:
            r = self.r
            if r["mirror_p"] and np.random.rand() < r["mirror_p"]:
                lm, pres = mirror(lm, pres)
            if r["handdrop_p"]:
                lm, pres = hand_dropout(lm, pres, r["handdrop_p"],
                                        r["handdrop_strength"])
            T = len(pres)
            if T > self.frames:
                start = np.random.randint(0, T - self.frames + 1)
                lm, pres = lm[start:start + self.frames], pres[start:start + self.frames]
            if r["jitter"]:
                keep = np.random.rand(len(pres)) > r["jitter"]
                if keep.sum() >= 4:
                    lm, pres = lm[keep], pres[keep]
            if np.random.rand() < r["noise"]:
                # Per-landmark jitter on x,y only. z is a MediaPipe depth
                # estimate, not a measurement, so perturbing it fakes
                # information the sensor does not have.
                noise = np.random.randn(*lm[..., :2].shape).astype(np.float32) * 0.01
                lm = lm.copy()
                lm[..., :2] += noise
                np.clip(lm[..., :2], 0.0, 1.0, out=lm[..., :2])

        feats, mask = build_features(lm, pres, expected_frames=self.frames,
                                     use_z=bool(self.r.get("use_z", False)))
        return torch.from_numpy(feats), torch.from_numpy(mask), label, signer


def load_items(vocab_idx, split):
    return index_split(vocab_idx, split)[0]


# ------------------------------------------------------------------ evaluate

def predict_all(model, items, frames, device, batch=64, use_z=False):
    ensure_cached(items)
    model.eval()
    feats, ys = [], []
    # use_z must match training: it changes the feature WIDTH, not just values.
    # Building the eval set with the default here silently fed a 255-wide input
    # to a 381-wide model.
    ds = ClipDataset(items, frames, {"mirror_p": 0, "handdrop_p": 0, "noise": 0,
                                     "jitter": 0, "handdrop_strength": 0,
                                     "use_z": use_z}, train=False)
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0)
    with torch.no_grad():
        for f, m, y, _s in dl:
            logits = model(f.to(device), m.to(device))
            feats.append(logits.cpu().numpy())
            ys.append(y.numpy())
    return np.concatenate(feats), np.concatenate(ys)


def acc(logits, ys):
    return float((logits.argmax(1) == ys).mean())


def mcnemar(a_ok, b_ok):
    w = int((b_ok & ~a_ok).sum())
    l = int((a_ok & ~b_ok).sum())
    n = w + l
    z = 0.0 if not n else (w - n / 2) / np.sqrt(n / 4)
    return w, l, z


# ------------------------------------------------------------------ train

def train_one(cfg, seed, train_items, val_items, vocab, device, quiet=False):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    frames = cfg.get("frames", 24)
    model = TemporalModel(len(vocab), dim=cfg["dim"], frames=frames,
                          backbone=cfg["backbone"],
                          dropout=cfg.get("dropout", 0.2),
                          in_dim=feature_dim(bool(cfg.get("use_z", False)))
                          ).to(device)
    # One pass over the .npz files instead of one per sample per epoch.
    preload(train_items); preload(val_items)
    tr = DataLoader(ClipDataset(train_items, frames, cfg, train=True),
                    batch_size=32, shuffle=True, num_workers=0)
    va = DataLoader(ClipDataset(val_items, frames, cfg, train=False),
                    batch_size=64, shuffle=False, num_workers=0)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    lossf = nn.CrossEntropyLoss(label_smoothing=0.1)

    best, best_state = 0.0, None
    # Checkpoint selection does not need per-epoch resolution: every eval is a
    # full forward pass over val (346 clips) with its own build_features, and
    # at 60 epochs that was the largest single cost in the run.
    eval_every = cfg.get("eval_every", 3)
    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        for f, m, y, _s in tr:
            opt.zero_grad()
            loss = lossf(model(f.to(device), m.to(device)), y.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        if ep % eval_every and ep != cfg["epochs"]:
            continue
        a, _c, _t, _ps = evaluate_model(model, va, device)
        if a > best:
            best = a
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if not quiet:
                print(f"    ep {ep:3d} val {a*100:5.2f}%  <- best", flush=True)
    model.load_state_dict(best_state)
    return model, best


def evaluate_model(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for f, m, y, _s in loader:
            p = model(f.to(device), m.to(device)).argmax(1).cpu()
            correct += int((p == y).sum()); total += len(y)
    return correct / max(total, 1), correct, total, {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="base")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=0, help="override recipe epochs")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--train-split", default="train")
    args = ap.parse_args()

    if args.list:
        for k, v in RECIPES.items():
            print(f"{k:26s} {v}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device}")
    vocab = json.load(open(VOCAB, encoding="utf-8"))
    vidx = {g: i for i, g in enumerate(vocab)}

    train_items = load_items(vidx, args.train_split)
    val_items = load_items(vidx, "val")
    test_items = load_items(vidx, "test")
    print(f"train {len(train_items)} | val {len(val_items)} | test {len(test_items)}")

    OUT.mkdir(parents=True, exist_ok=True)
    names = list(RECIPES) if args.sweep else [args.config]
    summary = []
    for name in names:
        cfg = dict(RECIPES[name])
        if args.epochs:
            cfg["epochs"] = args.epochs
        t0 = time.time()
        print(f"\n=== {name} seed={args.seed} epochs={cfg['epochs']} "
              f"{cfg['backbone']} ===", flush=True)
        model, bestv = train_one(cfg, args.seed, train_items, val_items,
                                 vocab, device)
        uz = bool(cfg.get("use_z", False))
        lv, yv = predict_all(model, val_items, cfg.get("frames", 24), device, use_z=uz)
        lt, yt = predict_all(model, test_items, cfg.get("frames", 24), device, use_z=uz)
        av, at = acc(lv, yv), acc(lt, yt)
        print(f"  val {av*100:6.2f}%  test {at*100:6.2f}%  "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)
        torch.save({"model": model.state_dict(), "vocab": vocab,
                    "dim": cfg["dim"], "frames": cfg.get("frames", 24),
                    "backbone": cfg["backbone"], "recipe": name,
                    "dropout": cfg.get("dropout", 0.2)},
                   OUT / f"{name}_s{args.seed}.pt")
        np.savez(OUT / f"{name}_s{args.seed}_logits.npz",
                 val_logits=lv, val_y=yv, test_logits=lt, test_y=yt)
        summary.append(dict(name=name, seed=args.seed, val=av, test=at,
                            params=sum(p.numel() for p in model.parameters())))

    print("\n--- summary ---")
    for s in summary:
        print(f"  {s['name']:26s} val {s['val']*100:6.2f}%  test {s['test']*100:6.2f}%")
    json.dump(summary, open(OUT / f"summary_s{args.seed}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
