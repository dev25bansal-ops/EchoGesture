"""Cross-check the window finding on val, and test it inside the ensemble.

The 72.78% baseline was measured on test, and the four members were selected on
val, so a window that wins on test alone could be selection noise. This checks
the same conditions on BOTH splits and reports the standard error, so a 1.4 pt
"win" can be judged against what noise looks like at n=1249.
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
CONDITIONS = {
    "full clip": (0.0, 1.0),
    "last 80%": (0.20, 1.0),
    "last 85%": (0.15, 1.0),
    "last 75%": (0.25, 1.0),
    "trim first 15%": (0.15, 1.0),
    "last 90%": (0.10, 1.0),
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


def run_split(split, models, labels_idx):
    clips = []
    for p in sorted((ROOT / ".scratch" / "landmarks" / split).glob("*.npz")):
        d = np.load(p, allow_pickle=False)
        if float(d["present"].max()) < 0.5:
            continue
        g = str(d["gloss"])
        if g not in labels_idx:
            continue
        clips.append((d["landmarks"], d["present"], labels_idx[g]))

    out = {}
    for name, (lo, hi) in CONDITIONS.items():
        ok = 0
        for lm, pres, y in clips:
            T = len(pres)
            a, b = int(T * lo), max(int(T * hi), int(T * lo) + 8)
            sub_lm, sub_pres = lm[a:b], pres[a:b]
            if len(sub_pres) < 8:
                sub_lm, sub_pres = lm, pres
            f, _ = build_features(sub_lm, sub_pres, expected_frames=FRAMES)
            ok += int(np.argmax(logits(models, f)) == y)
        out[name] = ok / len(clips)
    return out, len(clips)


def main():
    vocab = json.load(open(ROOT / "models/landmark_labels.json", encoding="utf-8"))
    labels = [g for g, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
    idx = {g: i for i, g in enumerate(labels)}
    models = load_models()

    rows = {}
    ns = {}
    for split in ("val", "test"):
        rows[split], ns[split] = run_split(split, models, idx)

    print(f"{'condition':18s} {'val':>8s} {'test':>8s} {'test d':>8s}")
    base_test = rows["test"]["full clip"]
    for name in CONDITIONS:
        v, t = rows["val"][name], rows["test"][name]
        print(f"{name:18s} {v*100:7.2f}% {t*100:7.2f}% {(t-base_test)*100:+7.2f}")

    n = ns["test"]
    se = np.sqrt(base_test * (1 - base_test) / n) * 100
    print(f"\ntest n={n}  standard error ±{se:.2f} pts")
    print("a change must exceed about 2 SE (~%.1f pts) to be believed" % (2 * se))

    # Paired test: same clips, so compare per-clip agreement rather than the
    # marginal accuracies, which is far more sensitive.
    print("\n--- paired comparison, full clip vs last 80% (test) ---")
    diffs = []
    for p in sorted((ROOT / ".scratch" / "landmarks" / "test").glob("*.npz")):
        d = np.load(p, allow_pickle=False)
        if float(d["present"].max()) < 0.5:
            continue
        g = str(d["gloss"])
        if g not in idx:
            continue
        y = idx[g]
        T = len(d["present"])
        corpus = {}
        for name, (lo, hi) in (("full", (0.0, 1.0)), ("last80", (0.20, 1.0))):
            a, b = int(T * lo), max(int(T * hi), int(T * lo) + 8)
            f, _ = build_features(d["landmarks"][a:b], d["present"][a:b],
                                  expected_frames=FRAMES)
            corpus[name] = int(np.argmax(logits(models, f)) == y)
        diffs.append(corpus["last80"] - corpus["full"])

    diffs = np.array(diffs)
    wins, losses = int((diffs > 0).sum()), int((diffs < 0).sum())
    print(f"  last80 fixes {wins} clips, breaks {losses} clips, net {wins-losses:+d}")
    # McNemar: under the null, wins ~ Binomial(wins+losses, 0.5)
    n_d = wins + losses
    if n_d:
        z = (wins - n_d / 2) / np.sqrt(n_d / 4)
        print(f"  McNemar z = {z:.2f}  (|z|>1.96 would be p<0.05)")


if __name__ == "__main__":
    main()