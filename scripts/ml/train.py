"""Train the landmark temporal classifier.

DEV-02. Conv1D + attention (the GISLR 1st-place topology, scaled down) over
the 126-feature landmark stream from src/features.py.

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
        path, label = self.items[i]
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
            # spatial jitter: small translation and scale
            lm = lm.copy()
            lm[:, :, :2] += np.random.randn() * 0.01
            lm[:, :, :2] *= 1.0 + np.random.randn() * 0.05
            # hand masking: simulate occlusion
            if np.random.rand() < 0.3:
                n = np.random.randint(1, max(2, len(pres) // 4))
                start = np.random.randint(0, max(1, len(pres) - n))
                pres = pres.copy()
                pres[start:start + n] = 0.0

        feats, mask = build_features(lm, pres, expected_frames=self.frames)
        return torch.from_numpy(feats), torch.from_numpy(mask), label


def index_split(vocab_idx, split):
    d = LANDMARKS / split
    items, skipped = [], 0
    for p in sorted(d.glob("*.npz")):
        rec = np.load(p, allow_pickle=False)
        gloss = str(rec["gloss"])
        if gloss not in vocab_idx:
            skipped += 1
            continue
        items.append((p, vocab_idx[gloss]))
    return items, skipped


# ---------------------------------------------------------------- train

def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    per_signer = {}
    with torch.no_grad():
        for feats, mask, labels in loader:
            feats, mask = feats.to(device), mask.to(device)
            logits = model(feats, mask)
            pred = logits.argmax(1).cpu()
            correct += int((pred == labels).sum())
            total += len(labels)
    return correct / max(total, 1), correct, total


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
        for feats, mask, labels in train_dl:
            feats, mask, labels = feats.to(device), mask.to(device), labels.to(device)
            opt.zero_grad()
            loss = lossf(model(feats, mask), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run_loss += float(loss)
        sched.step()

        acc, c, t = evaluate(model, val_dl, device) if val_items else (0.0, 0, 0)
        marker = ""
        if acc > best:
            best = acc
            torch.save({"model": model.state_dict(), "vocab": vocab,
                        "dim": args.dim, "frames": args.frames,
                        "backbone": args.backbone}, out)
            # Keep a slice of real validation inputs for INT8 calibration.
            # Calibrating on noise biases every activation range.
            cal = []
            for feats, _mask, _labels in val_dl:
                for row in feats:
                    cal.append(row.detach().cpu())
                    if len(cal) >= 400:
                        break
                if len(cal) >= 400:
                    break
            if cal:
                torch.save(cal, out.parent / "calib.pt")
            marker = "  <- best"
        print(f"ep {ep:3d} loss {run_loss/max(len(train_dl),1):.3f} "
              f"val_top1 {acc*100:5.1f}% ({c}/{t}){marker}", flush=True)

    print(f"\nbest signer-disjoint val top-1: {best*100:.1f}%  -> {out}")
    print("gate: >=60% on first run. Below that, check data and normalisation "
          "before touching the architecture.")


if __name__ == "__main__":
    main()