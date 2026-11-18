"""Compare a checkpoint against the shipped nine-member ensemble on test.

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
# The shipped members, derived from models/landmark_model_meta.json rather than
# hardcoded. The old list ["tcn21","tcnm61","tcnm62","rob71"] was the FOUR-member
# generation and printed 72.78% as "shipped ensemble" while the manifest ships
# nine members at 85.45% - so every "does this checkpoint beat what ships"
# decision was judged against a baseline two generations stale.
def shipped_member_names():
    meta = json.load(open(ROOT / "models" / "landmark_model_meta.json", encoding="utf-8"))
    names = []
    for f in meta["members"]:
        n = f
        for pre in ("landmark_ens_", "landmark_classifier_"):
            if n.startswith(pre):
                n = n[len(pre):]
        for suf in ("_fp16.tflite", ".tflite"):
            if n.endswith(suf):
                n = n[: -len(suf)]
        names.append(n)
    return names


SHIPPED = shipped_member_names()


def checkpoint_path(name):
    """Resolve a member name to its .pt checkpoint on disk, or None."""
    for d in (ROOT / ".scratch" / "runs", ROOT / ".scratch" / "search"):
        p = d / f"{name}.pt"
        if p.exists():
            return p
    return None


def load_test(vocab_idx):
    """Raw landmarks + presence per clip, so each checkpoint can build features
    at its own width (255-wide x,y or 381-wide x,y,z) from the same clips."""
    LM, PR, y = [], [], []
    for p in sorted((LANDMARKS / "test").glob("*.npz")):
        d = np.load(p, allow_pickle=False)
        if float(d["present"].max()) < 0.5:
            continue
        g = str(d["gloss"])
        if g not in vocab_idx:
            continue
        LM.append(d["landmarks"])
        PR.append(d["present"])
        y.append(vocab_idx[g])
    return LM, PR, np.array(y)


def logits_of(path, LM, PR):
    import torch
    from train import TemporalModel
    ck = torch.load(path, map_location="cpu", weights_only=True)
    # in_dim must come from the checkpoint's stem: five of the nine shipped
    # members take 381-wide x,y,z features, and the 255 default made the
    # state_dict load fail with "size mismatch for stem.weight" - so no
    # z-checkpoint could be compared at all.
    in_dim = int(ck["model"]["stem.weight"].shape[1])
    frames = int(ck["frames"])
    m = TemporalModel(len(ck["vocab"]), dim=int(ck["dim"]),
                      frames=frames, backbone=str(ck["backbone"]),
                      dropout=float(ck.get("dropout", 0.2)), in_dim=in_dim)
    m.load_state_dict(ck["model"])
    m.eval()
    use_z = in_dim != 255
    X = np.stack([build_features(lm, pr, expected_frames=frames, use_z=use_z)[0]
                  for lm, pr in zip(LM, PR)]).astype(np.float32)
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
    LM, PR, y = load_test(vocab_idx)
    print(f"test: {len(y)} clips, {len(vocab_idx)} classes\n")

    new_lg, ck = logits_of(ROOT / args.ckpt, LM, PR)
    print(f"  {Path(args.ckpt).stem:16s} alone            "
          f"{100*(new_lg.argmax(1)==y).mean():6.2f}%")

    shipped, names, missing = [], [], []
    for n in SHIPPED:
        p = checkpoint_path(n)
        if p is None:
            missing.append(n)
            continue
        lg, _ = logits_of(p, LM, PR)
        shipped.append(lg)
        names.append(n)
    if missing:
        print(f"\n  NOTE: {len(missing)} of {len(SHIPPED)} shipped members have no "
              f"PyTorch checkpoint on disk ({', '.join(missing)}); the baseline "
              f"below covers {len(names)}/{len(SHIPPED)} members, not the full "
              f"shipped ensemble")
    if shipped:
        base = np.mean(shipped, axis=0)
        print(f"  {'shipped ensemble':16s} ({len(names)}/{len(SHIPPED)} models)  "
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