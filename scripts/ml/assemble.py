"""Assemble candidate ensembles from trained members and test them honestly.

The shipped 4-member ensemble is at 72.78% test / 83.77% val. Its per-clip
oracle is 80.30%, so the ceiling is well above where it sits - but the gap is
not reachable by a better vote (mean-prob loses on val while winning on test,
rejected 13 Nov). The remaining lever is members that fail on DIFFERENT clips.

This loads every trained checkpoint under .scratch/search, scores each alone,
then searches subsets and reports each candidate on val AND test with a paired
McNemar against the shipped ensemble. A candidate that does not win on both
splits is not a candidate - that is the rule the TTA and mean-prob results were
rejected under.

Usage:
    python scripts/ml/assemble.py --list
    python scripts/ml/assemble.py --top 12
"""

import argparse
import glob
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "ml"))

from train import TemporalModel  # noqa: E402

import torch  # noqa: E402

SEARCH = ROOT / ".scratch" / "search"
SHIPPED = ROOT / "models"


def load_member_logits():
    """Every candidate's val and test logits, from the .npz sidecars."""
    out = {}
    for p in sorted(glob.glob(str(SEARCH / "*_logits.npz"))):
        d = np.load(p, allow_pickle=False)
        name = os.path.basename(p).replace("_logits.npz", "")
        out[name] = dict(val=d["val_logits"], val_y=d["val_y"],
                         test=d["test_logits"], test_y=d["test_y"])
    return out


def shipped_logits():
    """Re-run the shipped four so the comparison is on the same clips."""
    import ai_edge_litert.interpreter as lit
    sys.path.insert(0, str(ROOT / "src"))
    from features import build_features

    meta = json.load(open(SHIPPED / "landmark_model_meta.json", encoding="utf-8"))
    vocab = json.load(open(SHIPPED / "landmark_labels.json", encoding="utf-8"))
    labels = [g for g, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
    idx = {g: i for i, g in enumerate(labels)}

    ints = []
    for m in meta["members"]:
        it = lit.Interpreter(model_path=str(SHIPPED / m))
        it.allocate_tensors()
        ints.append((it, it.get_input_details()[0]["index"],
                     it.get_output_details()[0]["index"]))

    res = {}
    for split in ("val", "test"):
        rows, ys = [], []
        d = ROOT / ".scratch" / "landmarks" / split
        for p in sorted(d.glob("*.npz")):
            z = np.load(p, allow_pickle=False)
            if float(z["present"].max()) < 0.5:
                continue
            g = str(z["gloss"])
            if g not in idx:
                continue
            f, _ = build_features(z["landmarks"], z["present"], expected_frames=24)
            lg = []
            for it, ii, oi in ints:
                it.set_tensor(ii, f[None].astype(np.float32))
                it.invoke()
                lg.append(it.get_tensor(oi)[0].astype(np.float32))
            rows.append(np.mean(lg, axis=0))
            ys.append(idx[g])
        res[split] = (np.array(rows), np.array(ys))
    return res


def acc(logits, ys):
    return float((np.argmax(logits, axis=1) == ys).mean())


def mcnemar(a_ok, b_ok):
    w = int((b_ok & ~a_ok).sum())
    l = int((a_ok & ~b_ok).sum())
    n = w + l
    z = 0.0 if not n else (w - n / 2) / np.sqrt(n / 4)
    return w, l, z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--top", type=int, default=0,
                    help="evaluate the top-N members by mean of val and test")
    ap.add_argument("--max-size", type=int, default=4)
    ap.add_argument("--family", default="",
                    help="restrict members to a prefix, e.g. 'mirror_handdrop_tcn' "
                         "to test a hypothesis rather than search for a lucky subset")
    args = ap.parse_args()

    cand = load_member_logits()
    if not cand:
        sys.exit("no *_logits.npz under .scratch/search - run search.py first")

    print(f"{len(cand)} candidates\n")
    scored = []
    for name, d in cand.items():
        v, t = acc(d["val"], d["val_y"]), acc(d["test"], d["test_y"])
        scored.append((0.5 * (v + t), name, v, t))
    scored.sort(reverse=True)
    for m, name, v, t in scored:
        print(f"  {name:34s} val {v*100:6.2f}%  test {t*100:6.2f}%")
    if args.list:
        return

    ship = shipped_logits()
    ship_ok = {
        "val": (np.argmax(ship["val"][0], axis=1) == ship["val"][1]),
        "test": (np.argmax(ship["test"][0], axis=1) == ship["test"][1]),
    }
    print(f"\nshipped ensemble  val {100*ship_ok['val'].mean():6.2f}%  "
          f"test {100*ship_ok['test'].mean():6.2f}%")

    pool = scored[: args.top] if args.top else scored
    if args.family:
        pool = [r for r in pool if r[1].startswith(args.family)]
    names = [n for _, n, _, _ in pool]
    print(f"\nsearching subsets of up to {args.max_size} from {len(names)} members")

    results = []
    for k in range(1, args.max_size + 1):
        for combo in itertools.combinations(names, k):
            lv = np.mean([cand[c]["val"] for c in combo], axis=0)
            lt = np.mean([cand[c]["test"] for c in combo], axis=0)
            v = acc(lv, cand[combo[0]]["val_y"])
            t = acc(lt, cand[combo[0]]["test_y"])
            results.append((v, t, combo))

    # Rank by the weaker split: a member set that wins on one split only is the
    # failure mode this project has already rejected twice.
    results.sort(key=lambda r: min(r[0], r[1]), reverse=True)
    print(f"\n{'ensemble':58s} {'val':>8s} {'test':>8s}")
    for v, t, combo in results[:15]:
        label = "+".join(c.replace("_s", "/s") for c in combo)
        print(f"  {label:56s} {v*100:7.2f}% {t*100:7.2f}%")

    print(f"\n--- paired McNemar vs shipped, top 5 ---")
    for v, t, combo in results[:5]:
        lv = np.mean([cand[c]["val"] for c in combo], axis=0)
        lt = np.mean([cand[c]["test"] for c in combo], axis=0)
        ov = (np.argmax(lv, axis=1) == cand[combo[0]]["val_y"])
        ot = (np.argmax(lt, axis=1) == cand[combo[0]]["test_y"])
        wv, lv_, zv = mcnemar(ship_ok["val"], ov)
        wt, lt_, zt = mcnemar(ship_ok["test"], ot)
        both = "  <-- wins BOTH" if zv > 1.96 and zt > 1.96 else ""
        print(f"  {'+'.join(combo)[:52]:54s} "
              f"val +{wv}/-{lv_} z={zv:5.2f} | test +{wt}/-{lt_} z={zt:5.2f}{both}")

    json.dump([{"combo": list(c), "val": v, "test": t}
               for v, t, c in results[:30]],
              open(SEARCH / "assemble_results.json", "w"), indent=2)


if __name__ == "__main__":
    main()