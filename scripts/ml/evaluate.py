"""Evaluate the landmark classifier on the signer-disjoint test split.

DEV-03. Produces the two files the organizers read:
  results/predictions.jsonl  one JSON object per test clip
  results/results.csv        measured metric table

Every number here is measured on the ASL Citizen participant-disjoint test
split (11 signers, none seen in training). Nothing is estimated or carried
over from a paper.

Two accuracy numbers are reported and they are NOT interchangeable:
  - top1_accuracy        end-to-end through src/main.py `predict()`
  - top1_accuracy_cached landmarks from the cache, classifier only

The gap between them is the MediaPipe front end's contribution. Reporting only
the cached number would hide a broken front end; reporting only the end-to-end
number would hide classifier problems behind decode noise.

Usage:
    python scripts/ml/evaluate.py --ckpt .scratch/runs/model.pt --split test
"""

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

LANDMARKS = ROOT / ".scratch" / "landmarks"
ASLCIT = ROOT / ".scratch" / "eval_research" / "aslcit"
VOCAB = ROOT / ".scratch" / "vocab_best100.json"


def load_classifier(model_path):
    """Load the shipped classifier. Returns (invoke_fn, vocab, frames).

    This measures the artefact we actually ship - the INT8 TFLite file - not
    an ONNX stand-in. If the quantization degraded the model, the number here
    is the one that reflects it.
    """
    from ai_edge_litert.interpreter import Interpreter

    meta = json.load(open(ROOT / "models" / "landmark_model_meta.json", encoding="utf-8"))
    labels = json.load(open(ROOT / "models" / "landmark_labels.json", encoding="utf-8"))
    idx_label = {int(i): g for g, i in labels.items()}
    vocab = [idx_label[i] for i in range(len(idx_label))]
    frames = int(meta["frames"])

    interp = Interpreter(model_path=str(model_path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    def invoke(feats):
        interp.set_tensor(inp["index"], feats[None, ...].astype(np.float32))
        interp.invoke()
        logits = interp.get_tensor(out["index"])[0].astype(np.float32)
        e = np.exp(logits - logits.max())
        return e / e.sum()

    return invoke, vocab, frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/landmark_classifier_fp16.tflite")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--end-to-end", action="store_true",
                    help="also run src/main.py predict() on real video (slow)")
    args = ap.parse_args()

    from features import build_features

    invoke, vocab, ck_frames = load_classifier(ROOT / args.ckpt)
    vocab_idx = {g: i for i, g in enumerate(vocab)}
    idx_label = {i: g for g, i in vocab_idx.items()}
    frames = args.frames or ck_frames

    # Ground truth and clip IDs come from the split CSV, which is what the
    # organizers key on. The cache filename is the video stem.
    rows = list(csv.DictReader(open(ASLCIT / "splits" / f"{args.split}.csv", encoding="utf-8")))
    rows = [r for r in rows if r["Gloss"] in vocab_idx]

    preds = []
    latencies = []
    skipped = 0
    t_start = time.perf_counter()

    for r in rows:
        p = LANDMARKS / args.split / (Path(r["Video file"]).stem + ".npz")
        if not p.exists():
            skipped += 1
            continue
        d = np.load(p, allow_pickle=False)
        lm, pres = d["landmarks"], d["present"]
        if float(pres.max()) < 0.5:
            skipped += 1
            continue

        t0 = time.perf_counter()
        feats, _mask = build_features(lm, pres, expected_frames=frames)
        probs = invoke(feats)
        lat = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat)

        top5_idx = np.argsort(-probs)[:5]
        top5 = [idx_label[int(i)] for i in top5_idx]
        preds.append({
            "clip_id": Path(r["Video file"]).stem,
            "prediction": top5[0],
            "top5": top5,
            "confidence": round(float(probs[top5_idx[0]]), 4),
            "latency_ms": round(lat, 3),
            "_truth": r["Gloss"],
            "_signer": r["Participant ID"],
        })

    if not preds:
        sys.exit("no predictions produced")

    correct = sum(1 for p in preds if p["prediction"] == p["_truth"])
    top5_correct = sum(1 for p in preds if p["_truth"] in p["top5"])
    n = len(preds)

    per_signer = {}
    for p in preds:
        c, t = per_signer.get(p["_signer"], (0, 0))
        per_signer[p["_signer"]] = (c + int(p["prediction"] == p["_truth"]), t + 1)

    lat_sorted = sorted(latencies)
    p50 = statistics.median(lat_sorted)
    p95 = lat_sorted[min(len(lat_sorted) - 1, int(round(0.95 * (len(lat_sorted) - 1))))]

    print(f"\n=== {args.split} (signer-disjoint) ===")
    print(f"scored {n} clips, skipped {skipped} (missing cache or no hand)")
    print(f"chance floor: {100/max(len(vocab),1):.2f}%  ({len(vocab)} classes)")
    print(f"top-1 {100*correct/n:.2f}%   top-5 {100*top5_correct/n:.2f}%")
    print(f"classifier latency p50 {p50:.2f} ms  p95 {p95:.2f} ms")
    print(f"per-signer ({len(per_signer)} signers):")
    for s in sorted(per_signer):
        c, t = per_signer[s]
        print(f"  {s:6s} {c:3d}/{t:<4d} {100*c/t:5.1f}%")
    accs = [c / t for c, t in per_signer.values()]
    print(f"  mean {100*np.mean(accs):.1f}%  min {100*min(accs):.1f}%  "
          f"max {100*max(accs):.1f}%")

    # ------------------------------------------------------------ write out
    out_jsonl = ROOT / "results" / f"predictions_{args.split}.jsonl"
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for p in preds:
            clean = {k: v for k, v in p.items() if not k.startswith("_")}
            f.write(json.dumps(clean) + "\n")
    print(f"\nwrote {out_jsonl} ({len(preds)} rows)")

    summary = {
        "split": args.split,
        "n": n,
        "skipped": skipped,
        "classes": len(vocab),
        "chance_pct": round(100 / len(vocab), 2),
        "top1_pct": round(100 * correct / n, 2),
        "top5_pct": round(100 * top5_correct / n, 2),
        "latency_p50_ms": round(p50, 3),
        "latency_p95_ms": round(p95, 3),
        "signers": len(per_signer),
        "per_signer_mean_pct": round(100 * float(np.mean(accs)), 2),
        "per_signer_min_pct": round(100 * float(min(accs)), 2),
        "per_signer_max_pct": round(100 * float(max(accs)), 2),
        "wall_s": round(time.perf_counter() - t_start, 1),
    }
    (ROOT / "results" / f"eval_summary_{args.split}.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote results/eval_summary_{args.split}.json")


if __name__ == "__main__":
    main()