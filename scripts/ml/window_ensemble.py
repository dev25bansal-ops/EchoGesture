"""Test multi-window ensembling as a robustness strategy, on both splits.

The app cannot know where the sign sits in the capture window. Rather than
guessing the best single crop, average logits over several windows so the
prediction does not depend on hitting one exact placement. This is the same
argument that made the 4-member ensemble work, applied to time instead of models.

Reports val and test, and a paired McNemar against the shipped full-clip window.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from features import build_features  # noqa: E402
import ai_edge_litert.interpreter as lit  # noqa: E402

FRAMES = 24
# The windows the app could reasonably hold, as (lo, hi) fractions.
WINDOWS = {
    "full": (0.00, 1.00),
    "last90": (0.10, 1.00),
    "last80": (0.20, 1.00),
    "last70": (0.30, 1.00),
    "mid80": (0.10, 0.90),
    "last65": (0.35, 1.00),
}

# Window sets to compare against the shipped single full-clip window.
STRATEGIES = {
    "shipped (full clip only)": ["full"],
    "mean of full+last90+last80": ["full", "last90", "last80"],
    "mean of full+last90+last80+last70": ["full", "last90", "last80", "last70"],
    "mean of last90+last80+last70": ["last90", "last80", "last70"],
    "mean of all six": list(WINDOWS),
    "best single: last80": ["last80"],
}


def load_models():
    meta = json.load(open(ROOT / "models/landmark_model_meta.json", encoding="utf-8"))
    out = []
    for m in meta["members"]:
        it = lit.Interpreter(model_path=str(ROOT / "models" / m))
        it.allocate_tensors()
        out.append((it, it.get_input_details()[0]["index"],
                    it.get_output_details()[0]["index"]))
    return out


def logits(models, feats):
    lg = []
    for it, ii, oi in models:
        it.set_tensor(ii, feats[None].astype(np.float32))
        it.invoke()
        lg.append(it.get_tensor(oi)[0].astype(np.float32))
    return np.mean(lg, axis=0)


def clip_logits(models, lm, pres, windows):
    """Return {window: logits} for one clip."""
    T = len(pres)
    out = {}
    for name in windows:
        lo, hi = WINDOWS[name]
        a = int(T * lo)
        b = max(int(T * hi), a + 8)
        sub_lm, sub_pres = lm[a:b], pres[a:b]
        if len(sub_pres) < 8:
            sub_lm, sub_pres = lm, pres
        f, _ = build_features(sub_lm, sub_pres, expected_frames=FRAMES)
        out[name] = logits(models, f)
    return out


def run(split, models, idx):
    per_strategy = {k: [] for k in STRATEGIES}
    for p in sorted((ROOT / ".scratch" / "landmarks" / split).glob("*.npz")):
        d = np.load(p, allow_pickle=False)
        if float(d["present"].max()) < 0.5:
            continue
        g = str(d["gloss"])
        if g not in idx:
            continue
        y = idx[g]
        lg = clip_logits(models, d["landmarks"], d["present"], WINDOWS)
        for sname, wins in STRATEGIES.items():
            avg = np.mean([lg[w] for w in wins], axis=0)
            per_strategy[sname].append(int(np.argmax(avg)) == y)
    return {k: np.array(v) for k, v in per_strategy.items()}


def mcnemar(a, b):
    """a, b are boolean correctness arrays for the same clips."""
    wins = int((b & ~a).sum())   # b right, a wrong
    loss = int((a & ~b).sum())
    n = wins + loss
    if not n:
        return wins, loss, 0.0
    z = (wins - n / 2) / np.sqrt(n / 4)
    return wins, loss, z


def main():
    vocab = json.load(open(ROOT / "models/landmark_labels.json", encoding="utf-8"))
    labels = [g for g, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
    idx = {g: i for i, g in enumerate(labels)}
    models = load_models()

    val = run("val", models, idx)
    test = run("test", models, idx)

    base_name = "shipped (full clip only)"
    print(f"{'strategy':36s} {'val':>8s} {'test':>8s} {'test d':>8s}")
    for s in STRATEGIES:
        v = val[s].mean()
        t = test[s].mean()
        d = (t - test[base_name].mean()) * 100
        print(f"{s:36s} {v*100:7.2f}% {t*100:7.2f}% {d:+7.2f}")

    print("\n--- paired McNemar vs shipped, on test ---")
    for s in STRATEGIES:
        if s == base_name:
            continue
        w, l, z = mcnemar(test[base_name], test[s])
        flag = "  <-- significant" if abs(z) > 1.96 else ""
        print(f"  {s:36s} +{w:3d}/-{l:3d}  z={z:5.2f}{flag}")

    print("\n--- paired McNemar vs shipped, on val ---")
    for s in STRATEGIES:
        if s == base_name:
            continue
        w, l, z = mcnemar(val[base_name], val[s])
        flag = "  <-- significant" if abs(z) > 1.96 else ""
        print(f"  {s:36s} +{w:3d}/-{l:3d}  z={z:5.2f}{flag}")


if __name__ == "__main__":
    main()