"""Does capture-window placement change accuracy?

The Android app keeps 32 analysed frames and resamples the whole window to the
model's 24. If the user taps mid-sign, the window is misaligned relative to the
sign's extent. This measures how much that costs on the test split, and whether
averaging several candidate windows recovers it.

Windows are simulated by trimming the landmark sequence to a sub-span and then
running the shipped preprocessing, which resamples whatever it is given to 24
frames — exactly what the app does.
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
LANDMARKS = ROOT / ".scratch" / "landmarks" / "test"


def load_models():
    """Every shipped member with its input indices resolved BY NAME.

    Index 0 is the (1,24) mask on every shipped member; the positional
    get_input_details()[0] this file used fed features into the mask and
    crashed with a dimension mismatch before printing any accuracy.
    """
    meta = json.load(open(ROOT / "models/landmark_model_meta.json", encoding="utf-8"))
    detail = {d["file"]: bool(d.get("use_z", False))
              for d in meta.get("members_detail", []) or []}
    out = []
    for m in meta["members"]:
        it = lit.Interpreter(model_path=str(ROOT / "models" / m))
        it.allocate_tensors()
        dets = it.get_input_details()
        f_i = next(d["index"] for d in dets if "features" in d["name"])
        m_i = next((d["index"] for d in dets if "mask" in d["name"]), -1)
        out.append((it, f_i, it.get_output_details()[0]["index"], m_i,
                    detail.get(m, False)))
    return out


def logits(models, lm, pres, frames=FRAMES):
    """Mean-logit over members, each at its own feature width."""
    built = {}
    lg = []
    for it, fi, oi, mi, uz in models:
        if uz not in built:
            built[uz] = build_features(lm, pres, expected_frames=frames, use_z=uz)
        f, mask = built[uz]
        it.set_tensor(fi, f[None].astype(np.float32))
        if mi >= 0:
            it.set_tensor(mi, mask[None].astype(np.float32))
        it.invoke()
        lg.append(it.get_tensor(oi)[0].astype(np.float32))
    return np.mean(lg, axis=0)


def main():
    vocab = json.load(open(ROOT / "models/landmark_labels.json", encoding="utf-8"))
    labels = [g for g, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
    idx = {g: i for i, g in enumerate(labels)}
    models = load_models()

    clips = []
    for p in sorted(LANDMARKS.glob("*.npz")):
        d = np.load(p, allow_pickle=False)
        if float(d["present"].max()) < 0.5:
            continue
        g = str(d["gloss"])
        if g not in idx:
            continue
        clips.append((d["landmarks"], d["present"], idx[g]))

    print(f"{len(clips)} test clips, {len(models)} members\n")

    # Each condition is a (lo, hi) fraction of the clip the window covers.
    # 1.0/1.0 is the shipped behaviour: the whole clip.
    conditions = {
        "full clip (shipped)": (0.0, 1.0),
        "trim first 15%": (0.15, 1.0),
        "trim last 15%": (0.0, 0.85),
        "centre 80%": (0.10, 0.90),
        "centre 70%": (0.15, 0.85),
        "centre 60%": (0.20, 0.80),
        "first 80%": (0.0, 0.80),
        "last 80%": (0.20, 1.0),
    }

    results = {}
    per_clip = {k: [] for k in conditions}

    for lm, pres, y in clips:
        T = len(pres)
        for name, (lo, hi) in conditions.items():
            a, b = int(T * lo), max(int(T * hi), int(T * lo) + 8)
            sub_lm, sub_pres = lm[a:b], pres[a:b]
            if len(sub_pres) < 8:
                sub_lm, sub_pres = lm, pres
            per_clip[name].append((logits(models, sub_lm, sub_pres), y))

    for name in conditions:
        rows = per_clip[name]
        acc = np.mean([np.argmax(lg) == y for lg, y in rows])
        results[name] = acc
        print(f"  {name:24s} {acc*100:6.2f}%")

    print("\n--- recovery strategies ---")

    # Strategy A: vote across the four centre-ish windows.
    vote_keys = ["full clip (shipped)", "centre 80%", "centre 70%", "trim first 15%"]
    correct = 0
    for i in range(len(clips)):
        total = None
        for k in vote_keys:
            lg, y = per_clip[k][i]
            total = lg if total is None else total + lg
        correct += int(np.argmax(total) == y)
    print(f"  mean logits over 4 windows        {100*correct/len(clips):6.2f}%")

    # Strategy B: max-confidence window wins.
    correct = 0
    for i in range(len(clips)):
        best_lg, best_p = None, -1
        for k in vote_keys:
            lg, y = per_clip[k][i]
            p = float(np.max(np.exp(lg - lg.max()) / np.exp(lg - lg.max()).sum()))
            if p > best_p:
                best_p, best_lg = p, lg
        correct += int(np.argmax(best_lg) == y)
    print(f"  most-confident window             {100*correct/len(clips):6.2f}%")

    base = results["full clip (shipped)"]
    best = max(results.items(), key=lambda kv: kv[1])
    print(f"\n  shipped {base*100:.2f}%  |  best single window {best[0]} {best[1]*100:.2f}%")


if __name__ == "__main__":
    main()