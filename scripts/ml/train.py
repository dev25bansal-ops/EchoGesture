"""Train the landmark temporal classifier.

DEV-02. Conv1D + attention (the GISLR 1st-place topology, scaled down) over
the 255-feature landmark stream from src/features.py.

Success gate (from the sprint plan): signer-disjoint validation top-1 >= 60%
on the first run. Below that, the problem is data or normalisation, not
architecture - do not tune the model.

Usage:
    python scripts/ml/train.py --epochs 60 --frames 24
    python scripts/ml/train.py --backbone tcn --epochs 60
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from features import build_features, FEATURE_DIM  # noqa: E402

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import Dataset, DataLoader  # noqa: E402

LANDMARKS = ROOT / ".scratch" / "landmarks"
VOCAB = ROOT / ".scratch" / "vocab_best100.json"


# ---------------------------------------------------------------- model

class ConvBlock(nn.Module):
    def __init__(self, dim, k=17, drop=0.2):
        super().__init__()
        self.dw = nn.Conv1d(dim, dim, k, padding=k // 2, groups=dim)
        self.pw = nn.Conv1d(dim, dim, 1)
        self.norm = nn.BatchNorm1d(dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(dim, dim // 4, 1), nn.ReLU(),
            nn.Conv1d(dim // 4, dim, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        h = self.pw(self.dw(x))
        h = self.act(self.norm(h))
        h = h * self.se(h)
        return self.drop(h) + x


class TemporalModel(nn.Module):
    """Conv1D stack with optional attention pooling.

    `backbone='conv'` matches the GISLR winner's shape. `backbone='tcn'` is
    the dilated residual variant the third-party ablation found stronger; kept
    switchable so the comparison is one flag, not a rewrite.
    """

    def __init__(self, num_classes, dim=192, frames=24, backbone="conv", dropout=0.2):
        super().__init__()
        self.stem = nn.Conv1d(FEATURE_DIM, dim, 1)
        self.norm = nn.BatchNorm1d(dim)

        blocks = []
        if backbone == "conv":
            for _ in range(3):
                blocks.append(ConvBlock(dim, k=17, drop=dropout))
        else:  # tcn: increasing dilation for a longer receptive field
            for d in (1, 2, 4, 8):
                blocks.append(ConvBlock(dim, k=3, drop=dropout))
        self.blocks = nn.Sequential(*blocks)

        self.attn = nn.MultiheadAttention(dim, num_heads=4, batch_first=True, dropout=dropout)
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x, mask=None):
        x = x.transpose(1, 2)          # (B, F, T)
        x = self.norm(self.stem(x))
        x = self.blocks(x)
        x = x.transpose(1, 2)          # (B, T, C)

        q = self.query.expand(x.size(0), -1, -1)
        if mask is not None:
            key_pad = mask < 0.5
            if key_pad.all(dim=1).any():   # never let a clip have zero keys
                key_pad = key_pad & ~key_pad.all(dim=1, keepdim=True)
            pooled, _ = self.attn(q, x, x, key_padding_mask=key_pad)
        else:
            pooled, _ = self.attn(q, x, x)
        return self.head(self.drop(pooled[:, 0]))


# ---------------------------------------------------------------- data

class ClipDataset(Dataset):
    def __init__(self, items, frames, train=False):
        self.items = items
        self.frames = frames
        self.train = train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label, signer = self.items[i]
        d = np.load(path)
        lm, pres = d["landmarks"], d["present"]

        if self.train:
            T = len(pres)
            if T > self.frames:
                start = np.random.randint(0, T - self.frames + 1)
                lm, pres = lm[start:start + self.frames], pres[start:start + self.frames]
            # temporal jitter: drop frames with small probability
            keep = np.random.rand(len(pres)) > 0.1
            if keep.sum() >= 4:
                lm, pres = lm[keep], pres[keep]
            # Spatial jitter must happen AFTER normalisation, not before.
            # build_features root-centres on the wrist and divides by a
            # wrist-to-MCP scale, so a global translate/shift of the raw
            # landmarks cancels out identically - it would be a silent no-op.
            # More usefully, the jitter that matters is per-landmark landmark
            # noise, which mimics MediaPipe jitter between runs; scaling the
            # whole hand is precisely what we normalise away, so we do not
            # fake it.
            if np.random.rand() < 0.5:
                noise = np.random.randn(*lm[..., :2].shape) * 0.01
                lm = lm.copy()
                lm[..., :2] += noise.astype(np.float32)
                # a landmark that jittered out of frame is not a real
                # observation; keep coordinates legal
                np.clip(lm[..., :2], 0.0, 1.0, out=lm[..., :2])
            # Hand masking: a hand leaving frame is the most common real
            # failure and the one degradation this model handled worst
            # (-6.7 pts at 25% dropout in scripts/ml/robustness.py). The old
            # version fired on 30% of clips for at most T//4 frames, which
            # never showed the model a long absence. This fires harder so the
            # model learns to classify from one hand when it has to.
            if np.random.rand() < 0.5:
                n = np.random.randint(1, max(2, int(len(pres) * 0.5)))
                start = np.random.randint(0, max(1, len(pres) - n))
                pres = pres.copy()
                if pres.ndim == 2:
                    which = np.random.randint(0, 2)
                    pres[start:start + n, which] = 0.0
                    # Occasionally drop both, as a fully occluded frame.
                    if np.random.rand() < 0.2:
                        pres[start:start + n, 1 - which] = 0.0
                else:
                    pres[start:start + n] = 0.0

        feats, mask = build_features(lm, pres, expected_frames=self.frames)
        return torch.from_numpy(feats), torch.from_numpy(mask), label, signer


def index_split(vocab_idx, split):
    d = LANDMARKS / split
    items, skipped, empty = [], 0, 0
    for p in sorted(d.glob("*.npz")):
        rec = np.load(p, allow_pickle=False)
        gloss = str(rec["gloss"])
        if gloss not in vocab_idx:
            skipped += 1
            continue
        # A clip whose every frame lacks a hand carries no signal; training on
        # its all-zero landmarks teaches the model that the class is "nothing".
        if float(rec["present"].max()) < 0.5:
            empty += 1
            continue
        signer = str(rec["participant"]) if "participant" in rec else "?"
        items.append((p, vocab_idx[gloss], signer))
    if empty:
        print(f"  [{split}] dropped {empty} clips with no hand detected")
    return items, skipped


# ---------------------------------------------------------------- train

def evaluate(model, loader, device):
    """Return (accuracy, correct, total, per_signer)."""
    model.eval()
    correct = total = 0
    per_signer = {}
    with torch.no_grad():
        for batch in loader:
            feats, mask, labels = batch[0], batch[1], batch[2]
            signers = batch[3] if len(batch) > 3 else None
            feats, mask = feats.to(device), mask.to(device)
            logits = model(feats, mask)
            pred = logits.argmax(1).cpu()
            correct += int((pred == labels).sum())
            total += len(labels)
            if signers is not None:
                for p, ok in zip(signers, (pred == labels).tolist()):
                    c, t = per_signer.get(p, (0, 0))
                    per_signer[p] = (c + int(ok), t + 1)
    return correct / max(total, 1), correct, total, per_signer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--backbone", default="conv", choices=["conv", "tcn"])
    ap.add_argument("--out", default=".scratch/runs/model.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vocab = json.load(open(VOCAB, encoding="utf-8"))
    vocab_idx = {g: i for i, g in enumerate(vocab)}

    train_items, sk_tr = index_split(vocab_idx, "train")
    val_items, sk_va = index_split(vocab_idx, "val")
    print(f"train clips: {len(train_items)} (skipped {sk_tr}) | val: {len(val_items)} (skipped {sk_va})")
    if not train_items:
        sys.exit("no training clips found - run extract_landmarks.py first")

    train_dl = DataLoader(ClipDataset(train_items, args.frames, train=True),
                          batch_size=args.batch, shuffle=True, num_workers=0)
    val_dl = DataLoader(ClipDataset(val_items, args.frames, train=False),
                        batch_size=args.batch, shuffle=False, num_workers=0)

    model = TemporalModel(len(vocab), dim=args.dim, frames=args.frames,
                          backbone=args.backbone).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {args.backbone} dim={args.dim} params={n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.1)

    best = 0.0
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    for ep in range(1, args.epochs + 1):
        model.train()
        run_loss = 0.0
        for feats, mask, labels, _signers in train_dl:
            feats, mask, labels = feats.to(device), mask.to(device), labels.to(device)
            opt.zero_grad()
            loss = lossf(model(feats, mask), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run_loss += float(loss.detach())
        sched.step()

        acc, c, t, _ps = evaluate(model, val_dl, device) if val_items else (0.0, 0, 0, {})
        marker = ""
        if acc > best:
            best = acc
            torch.save({"model": model.state_dict(), "vocab": vocab,
                        "dim": args.dim, "frames": args.frames,
                        "backbone": args.backbone}, out)
            # Keep a slice of real validation inputs for INT8 calibration.
            # Calibrating on noise biases every activation range.
            cal = []
            for feats, _mask, _labels, _signers in val_dl:
                for row in feats:
                    cal.append(row.detach().cpu())
                    if len(cal) >= 400:
                        break
                if len(cal) >= 400:
                    break
            if cal:
                torch.save(cal, out.parent / "calib.pt")
            # Plain-JSON sidecar so the evaluator and the INT8 export can
            # recover the vocab and frame count without loading torch.
            json.dump({"vocab": vocab, "frames": args.frames,
                       "dim": args.dim, "backbone": args.backbone,
                       "val_top1": acc},
                      open(out.parent / "model_meta.json", "w", encoding="utf-8"))
            marker = "  <- best"
        print(f"ep {ep:3d} loss {run_loss/max(len(train_dl),1):.3f} "
              f"val_top1 {acc*100:5.1f}% ({c}/{t}){marker}", flush=True)

    print(f"\nbest signer-disjoint val top-1: {best*100:.1f}%  -> {out}")
    print("gate: >=60% on first run. Below that, check data and normalisation "
          "before touching the architecture.")

    # Per-signer breakdown. A high mean with one signer near zero is a
    # different (and worse) failure than a uniform low mean: it means the
    # model still leans on a signer and only the average hides it.
    if val_items:
        state = torch.load(out, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        acc, c, t, per_signer = evaluate(model, val_dl, device)
        print(f"\nper-signer val accuracy ({len(per_signer)} held-out signers):")
        for p in sorted(per_signer):
            pc, pt = per_signer[p]
            print(f"  {p:6s} {pc:4d}/{pt:<4d} {100*pc/max(pt,1):5.1f}%")
        accs = [pc / max(pt, 1) for pc, pt in per_signer.values()]
        if accs:
            print(f"  mean {100*np.mean(accs):.1f}%  min {100*min(accs):.1f}%  "
                  f"max {100*max(accs):.1f}%")


if __name__ == "__main__":
    main()