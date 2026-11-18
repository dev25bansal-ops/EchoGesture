"""Re-measure the app-gate rows of results/robustness_gated.json.

Abhijeet's objection: the raw row of that file reports drop_hand at 79.94, which
is the coords-kept DIAGNOSTIC, while the harness both-zeroed value is 79.70.
This script measures the gate on the actual harness corruption, so the gated
rows rest on a corruption the app would really see.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "ml"))

import robustness as R  # noqa: E402
from eval_tflite_cache import load_ensemble  # noqa: E402
from features import build_features  # noqa: E402


def score(members, frames, lm, pres):
    built = {}
    for uz in {m["use_z"] for m in members}:
        built[uz] = build_features(lm, pres, expected_frames=frames, use_z=uz)
    lg = []
    for mb in members:
        f, m = built[mb["use_z"]]
        mb["it"].set_tensor(mb["f"], f[None].astype(np.float32))
        if mb["m"] is not None:
            mb["it"].set_tensor(mb["m"], m[None].astype(np.float32))
        mb["it"].invoke()
        lg.append(mb["it"].get_tensor(mb["o"])[0].astype(np.float32))
    logits = np.mean(lg, axis=0)
    e = np.exp(logits - logits.max())
    return e / e.sum()


def acc_at_gate(probs, ys, margin):
    if margin is None:
        return 100.0 * float(np.mean(np.argmax(probs, 1) == ys)), 1.0
    s = np.sort(probs, 1)
    top1, top2 = s[:, -1], s[:, -2]
    keep = (top1 - top2) >= margin
    if keep.sum() == 0:
        # No clip cleared the gate. Report that as None, never NaN: json.dumps
        # writes a bare `NaN` token for float("nan"), which is not valid JSON and
        # which a strict parser or a schema validator rejects outright. A None
        # plus coverage 0.0 says "nothing was measured" unambiguously.
        return None, 0.0
    return 100.0 * float(np.mean(np.argmax(probs[keep], 1) == ys[keep])), float(keep.mean())


def _out_path(value: str):
    """Absolute paths are used as given; relative ones resolve against the repo.

    ROOT / "/abs/path" yields the absolute path, but ROOT / "results/x.json" is
    what the docs promise, and a caller in another cwd otherwise writes outside
    the repo. Resolving here means --out behaves the same from anywhere.
    """
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=".scratch/landmarks_native")
    ap.add_argument("--split", default="test")
    ap.add_argument("--corruption", default="drop_hand_25pct")
    ap.add_argument("--seed", type=int, default=None,
                    help="seed for the corruption; default 1234 (robustness.py line 229)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=".scratch/bughunt2/gate_recheck.json")
    args = ap.parse_args()

    R.CACHE_ROOT = ROOT / args.cache
    meta = json.load(open(ROOT / "models" / "landmark_model_meta.json", encoding="utf-8"))
    vocab = {g: i for i, g in enumerate(json.load(open(ROOT / ".scratch" / "vocab_best100.json",
                                                       encoding="utf-8")))}
    members, frames = load_ensemble()

    paths = sorted((ROOT / args.cache / args.split).glob("*.npz"))
    if args.limit:
        paths = paths[: args.limit]
    fn = R.CORRUPTIONS[args.corruption]
    seed = 1234 if args.seed is None else args.seed
    rng = np.random.default_rng(seed)

    print(f"cache={args.cache} split={args.split} corruption={args.corruption} "
          f"seed={seed} members={len(members)} frames={frames}")

    clean_p, drop_p, ys = [], [], []
    for p in paths:
        d = np.load(p, allow_pickle=False)
        if float(d["present"].max()) < 0.5:
            continue
        g = str(d["gloss"])
        if g not in vocab:
            continue
        lm, pres = d["landmarks"], d["present"]
        clean_p.append(score(members, frames, lm, pres))
        cl, cp = fn(lm, pres, rng)
        drop_p.append(score(members, frames, cl, cp))
        ys.append(vocab[g])

    clean_p = np.array(clean_p)
    drop_p = np.array(drop_p)
    ys = np.array(ys)
    print(f"scored {len(ys)} clips\n")

    rows = []
    for name, margin in [("raw (every clip)", None), ("app gate margin>=0.15", 0.15),
                         ("stricter gate margin>=0.30", 0.30)]:
        c, cc = acc_at_gate(clean_p, ys, margin)
        dr, dc = acc_at_gate(drop_p, ys, margin)
        if c is None or dr is None:
            # Nothing cleared this gate, so there is no delta to report and no
            # verdict to give. NOT_MEASURED is the honest third state; calling it
            # PASS or FAIL would both be inventions.
            rows.append({"mode": name, "gate_margin": margin,
                         "clean_pct": None, "drop_pct": None, "delta_pp": None,
                         "clean_coverage": round(cc, 3), "drop_coverage": round(dc, 3),
                         "verdict": "NOT_MEASURED"})
            print(f"{name:26s} no clip cleared this gate "
                  f"(clean cov {cc*100:.0f}%, drop cov {dc*100:.0f}%)  NOT_MEASURED")
            continue
        delta = c - dr
        rows.append({"mode": name, "gate_margin": margin, "clean_pct": round(c, 2),
                     "drop_pct": round(dr, 2), "delta_pp": round(delta, 2),
                     "clean_coverage": round(cc, 3), "drop_coverage": round(dc, 3),
                     "verdict": "PASS" if delta <= 5.0 else "FAIL"})
        print(f"{name:26s} clean {c:6.2f} ({cc*100:4.0f}% cov)  "
              f"drop {dr:6.2f} ({dc*100:4.0f}% cov)  delta {delta:+5.2f}  "
              f"{'PASS' if delta <= 5.0 else 'FAIL'}")

    json.dump({"corruption": args.corruption, "cache": args.cache, "split": args.split,
               "n_clips": len(ys), "seed": seed, "rows": rows},
              open(_out_path(args.out), "w"), indent=2, allow_nan=False)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
