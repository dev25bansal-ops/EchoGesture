"""Score the SHIPPED TFLite ensemble on a chosen landmark cache.

This is the number that corresponds to what the app runs: the FP16 .tflite
members, not the PyTorch checkpoints. Every other accuracy figure in the repo
comes from reloading .pt files; the app ships .tflite, so a claim about the app
has to be measured on these.

Inputs are resolved BY NAME (the converter reports the mask as input 0), width
per member from members_detail, and aggregation is mean-logit then one softmax,
matching src/main.py and LandmarkClassifier.kt.

Usage:
    python scripts/ml/eval_tflite_cache.py --cache .scratch/landmarks_native
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def load_ensemble():
    import ai_edge_litert.interpreter as lit

    meta = json.load(open(ROOT / "models" / "landmark_model_meta.json", encoding="utf-8"))
    detail = {d["file"]: bool(d.get("use_z", False)) for d in meta.get("members_detail", [])}
    members = []
    for name in meta["members"]:
        it = lit.Interpreter(model_path=str(ROOT / "models" / name))
        it.allocate_tensors()
        dets = {d["name"]: d for d in it.get_input_details()}
        f_det = next(d for n, d in dets.items() if "features" in n)
        m_det = next((d for n, d in dets.items() if "mask" in n), None)
        members.append({
            "name": name, "it": it,
            "f": f_det["index"], "m": m_det["index"] if m_det else None,
            "o": it.get_output_details()[0]["index"],
            "width": int(f_det["shape"][-1]),
            "use_z": detail.get(name, False),
        })
    return members, int(meta["frames"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=".scratch/landmarks_native")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from features import build_features

    members, frames = load_ensemble()
    labels = json.load(open(ROOT / "models" / "landmark_labels.json", encoding="utf-8"))
    idx = {g: i for g, i in labels.items()}

    # Sanity: the per-member width must match what features.py can emit.
    widths = sorted({m["width"] for m in members})
    print(f"{len(members)} tflite members, widths {widths}, frames {frames}")

    files = sorted((ROOT / args.cache / args.split).glob("*.npz"))
    if not (ROOT / args.cache / args.split).is_dir():
        sys.exit(f"REFUSING: {ROOT/args.cache/args.split} is not a directory - "
                 f"check --cache/--split (a typo'd path used to score 0 clips, "
                 f"write a 0.00% artifact and exit 0)")
    if args.limit:
        files = files[: args.limit]

    t0 = time.perf_counter()
    per_member_ok = {m["name"]: 0 for m in members}
    ens_ok = n = nodet = 0
    per_clip_ms = []
    for p in files:
        d = np.load(p, allow_pickle=False)
        g = str(d["gloss"])
        if g not in idx:
            continue
        if float(d["present"].max()) < 0.5:
            nodet += 1
            continue
        truth = idx[g]
        lm = d["landmarks"].astype(np.float32)
        pres = d["present"].astype(np.float32)

        t_clip = time.perf_counter()
        built = {}
        for uz in {m["use_z"] for m in members}:
            built[uz] = build_features(lm, pres, expected_frames=frames, use_z=uz)

        logits = []
        for m in members:
            f, mk = built[m["use_z"]]
            m["it"].set_tensor(m["f"], f[None].astype(np.float32))
            if m["m"] is not None:
                m["it"].set_tensor(m["m"], mk[None].astype(np.float32))
            m["it"].invoke()
            lg = m["it"].get_tensor(m["o"])[0].astype(np.float32)
            logits.append(lg)
            if int(lg.argmax()) == truth:
                per_member_ok[m["name"]] += 1
        avg = np.mean(logits, axis=0)
        ens_ok += int(avg.argmax() == truth)
        n += 1
        per_clip_ms.append((time.perf_counter() - t_clip) * 1000.0)

    el = time.perf_counter() - t0
    if n == 0:
        # Same class of defect the repo already fixed in robustness_gated_recheck:
        # a zero-clip run is an error, not a 0.00% measurement. Emitting a
        # well-formed JSON here was indistinguishable from a real run that
        # collapsed, and downstream readers cannot tell the difference.
        sys.exit(f"REFUSING: scored 0 clips from {args.cache}/{args.split} "
                 f"(skipped {nodet} with no hand, {len(files)} files found). "
                 f"No accuracy number exists for this input.")
    lat_sorted = sorted(per_clip_ms)
    p50 = lat_sorted[len(lat_sorted) // 2]
    p95 = lat_sorted[min(len(lat_sorted) - 1, int(round(0.95 * (len(lat_sorted) - 1))))]

    print(f"\nscored {n} clips from {args.cache}/{args.split} "
          f"(skipped {nodet} with no hand) in {el:.0f}s")
    print(f"\n{'member':46s} {'top1':>8s}")
    for m in members:
        print(f"  {m['name']:44s} {100*per_member_ok[m['name']]/n:7.2f}%")
    ens = 100 * ens_ok / n
    print(f"\n  ENSEMBLE (mean-logit): {ens:.2f}%")
    # The field named "ms/clip" is total elapsed / clips: a mean over the whole
    # scoring loop (decode is not in it, but np.load and build_features are).
    # It is NOT a p95 - results.csv used to republish it as latency_model_only_p95,
    # which understated the tail by ~1.3-2x. Both statistics are printed and
    # both are written to the artifact, each labelled for what it is.
    print(f"  classifier time: mean {1000*el/n:.1f} ms/clip, "
          f"per-clip p50 {p50:.2f} / p95 {p95:.2f} ms for {len(members)} members")

    if args.out:
        # Resolve relative paths against the repo root, like every sibling
        # script (make_predictions.py, robustness.py, train.py, ...). The old
        # open(args.out) wrote next to the caller's cwd - silently outside the
        # repo, or FileNotFoundError after all the scoring work.
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "cache": args.cache, "split": args.split, "n": n, "no_detection": nodet,
                "ensemble_top1_pct": round(ens, 4),
                "per_member_pct": {m["name"]: round(100*per_member_ok[m["name"]]/n, 4)
                                   for m in members},
                "classifier_ms_per_clip": round(1000*el/n, 3),
                "latency_mean_ms": round(float(np.mean(per_clip_ms)), 3),
                "latency_p50_ms": round(float(p50), 3),
                "latency_p95_ms": round(float(p95), 3),
            }, f, indent=2, allow_nan=False)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
