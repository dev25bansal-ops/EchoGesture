"""Is the robustness loss train/eval mismatch, or information loss?

Training (scripts/ml/search.py hand_dropout) zeroes ONLY the presence flag and
leaves the landmark coordinates intact. Evaluation (scripts/ml/robustness.py
c_drop_hand) zeroes BOTH. So the model trains on "flag says gone, numbers still
mean something" and is scored on "flag says gone, numbers are zeros".

Three corruptions, same seed, same spans, shipped 9-member ensemble:
  coords+flag  -- the evaluation corruption (what robustness.py scores)
  flag_only    -- the TRAINING distribution: flag 0, coordinates untouched
  coords_only  -- coordinates zeroed, flag left at 1 (no mask headroom)

Reading the result:
  If flag_only is much better than coords+flag, the model DOES use the numbers
  behind an absent flag, and the scored failure is a train/eval mismatch that a
  matched retrain could close.
  If all three are close, the information is simply gone and no preprocessing
  or augmentation change recovers it -- the row is information-limited.
"""
import glob
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"E:\samsung")
sys.path.insert(0, str(ROOT / "src"))

CACHE = ROOT / ".scratch" / "landmarks_native" / "test"
DROP_FRAC = 0.25
SEED = 20261122
FRAMES = 24


def drop(lm, present, seed, mode):
    """Same spans for every mode, so the three rows are directly comparable."""
    rng = np.random.default_rng(seed)
    lm = lm.copy()
    present = present.copy()
    T = lm.shape[0]
    n = max(1, int(T * DROP_FRAC))
    for h in range(lm.shape[1]):
        if rng.random() < 0.5:
            s = int(rng.integers(0, max(1, T - n)))
            if mode in ("coords+flag", "coords_only"):
                lm[s:s + n, h] = 0.0
            if mode in ("coords+flag", "flag_only"):
                present[s:s + n, h] = 0.0
    return lm, present


def main():
    import main as M
    M._init_model()
    import json
    labels = json.loads((ROOT / "models" / "landmark_labels.json").read_text())
    from features import build_features
    print(f"ensemble members: {len(M._INTERPRETERS)}", flush=True)

    def score(lm, present):
        widths = {}
        for uz in set(M._MEMBER_USE_Z) or {False}:
            widths[uz] = build_features(lm, present, expected_frames=FRAMES, use_z=uz)
        logits = []
        for interp, (in_idx, out_idx, mask_idx), uz in zip(
                M._INTERPRETERS, M._MEMBER_IO, M._MEMBER_USE_Z):
            feats, mask = widths[uz]
            interp.set_tensor(in_idx, feats[None, ...].astype(np.float32))
            if mask_idx >= 0:
                interp.set_tensor(mask_idx, mask[None, ...].astype(np.float32))
            interp.invoke()
            logits.append(interp.get_tensor(out_idx)[0].astype(np.float32))
        return int(np.argmax(np.mean(logits, axis=0)))

    files = sorted(glob.glob(str(CACHE / "*.npz")))
    data = []
    for p in files:
        d = np.load(p, allow_pickle=True)
        data.append((d["landmarks"], d["present"], str(d["gloss"])))
    g2i = {g: i for i, g in enumerate(labels)}
    print(f"clips: {len(data)}", flush=True)

    modes = ["coords+flag", "flag_only", "coords_only"]
    res = {m: 0 for m in modes}
    clean_ok = n = 0
    for i, (lm, present, gloss) in enumerate(data):
        if gloss not in g2i:
            continue
        y = g2i[gloss]
        n += 1
        if score(lm, present) == y:
            clean_ok += 1
        for m in modes:
            dlm, dpres = drop(lm, present, SEED + i, m)
            if score(dlm, dpres) == y:
                res[m] += 1
        if (i + 1) % 250 == 0:
            print(f"  {i+1}/{len(data)}", flush=True)

    ca = 100.0 * clean_ok / n
    print(f"\nclean       n={n}  {ca:6.2f}%")
    print("=== corruption mode vs the scored number (lower delta is better) ===")
    for m in modes:
        a = 100.0 * res[m] / n
        print(f"{m:12s} {a:6.2f}%  delta {a-ca:+6.2f}")


if __name__ == "__main__":
    main()
