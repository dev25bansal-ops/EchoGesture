"""Regenerate the submission's predictions.jsonl on the native cache.

`results/predictions.jsonl` is a required deliverable and the version in the
repo was written 9 Nov: four members, the 256-squash cache, and a `latency_ms`
field measured before the ensemble grew to nine. Every number in it was from a
system we no longer ship.

This writes the same schema from the shipped nine FP16 members scored on
`.scratch/landmarks_native` — the geometry the Android app actually uses — with
mean-logit aggregation, matching `src/main.py` and `LandmarkClassifier.kt`.

Usage:
    python scripts/ml/make_predictions.py --cache .scratch/landmarks_native
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=".scratch/landmarks_native")
    ap.add_argument("--split", default="test")
    # Derived from --split, not hardcoded. The old defaults named the test
    # deliverables unconditionally, so `--split val` wrote val rows into
    # results/predictions.jsonl and a val summary into eval_summary_test.json —
    # two files that say "test" while holding validation data.
    ap.add_argument("--out", default=None,
                    help="default: results/predictions_<split>.jsonl for val, "
                         "results/predictions.jsonl for test")
    ap.add_argument("--summary-out", default=None,
                    help="default: results/eval_summary_<split>.json")
    args = ap.parse_args()
    if args.out is None:
        args.out = ("results/predictions.jsonl" if args.split == "test"
                    else f"results/predictions_{args.split}.jsonl")
    if args.summary_out is None:
        args.summary_out = f"results/eval_summary_{args.split}.json"

    import numpy as np
    import ai_edge_litert.interpreter as lit
    from features import build_features

    meta = json.load(open(ROOT / "models" / "landmark_model_meta.json", encoding="utf-8"))
    detail = {d["file"]: bool(d.get("use_z", False)) for d in meta.get("members_detail", [])}
    labels_map = json.load(open(ROOT / "models" / "landmark_labels.json", encoding="utf-8"))
    idx = {g: i for g, i in labels_map.items()}
    idx_to_label = {i: g for g, i in idx.items()}
    frames = int(meta["frames"])

    members = []
    for name in meta["members"]:
        it = lit.Interpreter(model_path=str(ROOT / "models" / name))
        it.allocate_tensors()
        dets = {d["name"]: d for d in it.get_input_details()}
        f_det = next(d for n, d in dets.items() if "features" in n)
        m_det = next((d for n, d in dets.items() if "mask" in n), None)
        members.append({
            "it": it, "f": f_det["index"],
            "m": m_det["index"] if m_det else None,
            "o": it.get_output_details()[0]["index"],
            "use_z": detail.get(name, False),
        })

    files = sorted((ROOT / args.cache / args.split).glob("*.npz"))
    preds, latencies = [], []
    skipped = 0
    correct = top5_hits = 0
    signer_ok, signer_n = {}, {}

    for p in files:
        d = np.load(p, allow_pickle=False)
        g = str(d["gloss"])
        if g not in idx or float(d["present"].max()) < 0.5:
            skipped += 1
            continue
        lm = d["landmarks"].astype(np.float32)
        pres = d["present"].astype(np.float32)

        t0 = time.perf_counter()
        built = {uz: build_features(lm, pres, expected_frames=frames, use_z=uz)
                 for uz in {m["use_z"] for m in members}}
        logits = []
        for m in members:
            f, mk = built[m["use_z"]]
            m["it"].set_tensor(m["f"], f[None].astype(np.float32))
            if m["m"] is not None:
                m["it"].set_tensor(m["m"], mk[None].astype(np.float32))
            m["it"].invoke()
            logits.append(m["it"].get_tensor(m["o"])[0].astype(np.float32))
        avg = np.mean(logits, axis=0)
        e = np.exp(avg - avg.max())
        probs = e / e.sum()
        latencies.append((time.perf_counter() - t0) * 1000.0)

        order = np.argsort(-probs)[:5]
        preds.append({
            "clip_id": p.stem,
            "prediction": idx_to_label[int(order[0])],
            "top5": [idx_to_label[int(i)] for i in order],
            "confidence": round(float(probs[order[0]]), 4),
            "latency_ms": round(latencies[-1], 3),
        })

        # Accuracy is the reason these summaries exist; the earlier version of
        # this script wrote latency and metadata but no accuracy at all, which
        # is why eval_summary_*.json came back missing top1/top5.
        hit = int(order[0]) == idx[g]
        in5 = idx[g] in [int(i) for i in order]
        correct += int(hit)
        top5_hits += int(in5)
        signer = str(d["participant"]) if "participant" in d.files else "unknown"
        signer_ok[signer] = signer_ok.get(signer, 0) + int(hit)
        signer_n[signer] = signer_n.get(signer, 0) + 1

    # Refuse before touching the output file. `results/predictions.jsonl` is a
    # tracked deliverable; opening it in "w" truncates it even if the very next
    # statement raises, which is how an empty cache silently destroyed it.
    if not preds:
        raise SystemExit(
            f"scored 0 clips from {args.cache}/{args.split} — refusing to overwrite "
            f"{args.out} with an empty deliverable")

    out = ROOT / args.out
    with open(out, "w", encoding="utf-8") as fh:
        for r in preds:
            fh.write(json.dumps(r) + "\n")

    lat = sorted(latencies)
    n = len(preds)
    p50 = statistics.median(lat)
    p95 = lat[min(n - 1, int(round(0.95 * (n - 1))))]
    print(f"wrote {out}  ({n} rows, {skipped} skipped)")
    print(f"  classifier latency p50 {p50:.2f} ms  p95 {p95:.2f} ms  (nine members)")

    if args.summary_out:
        per_signer = {s: round(100.0 * signer_ok[s] / signer_n[s], 2) for s in sorted(signer_n)}
        signer_vals = list(per_signer.values())
        summary = {
            "split": args.split,
            "n": n,
            "skipped": skipped,
            "classes": len(idx),
            "chance_pct": round(100 / len(idx), 2),
            "top1_pct": round(100.0 * correct / n, 2),
            "top5_pct": round(100.0 * top5_hits / n, 2),
            "signers": len(signer_n),
            "per_signer_mean_pct": round(sum(signer_vals) / len(signer_vals), 2) if signer_vals else None,
            "per_signer_min_pct": min(signer_vals) if signer_vals else None,
            "per_signer_max_pct": max(signer_vals) if signer_vals else None,
            "latency_p50_ms": round(p50, 3),
            "latency_p95_ms": round(p95, 3),
            "cache": args.cache,
            "members": len(members),
            "quantization": "FP16",
            "aggregator": "mean-logit (shipped)",
        }
        (ROOT / args.summary_out).write_text(json.dumps(summary, indent=2))
        print(f"  wrote {args.summary_out}  top1 {summary['top1_pct']}%  top5 {summary['top5_pct']}%")


if __name__ == "__main__":
    main()
