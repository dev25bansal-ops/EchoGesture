"""Evaluate the landmark classifier on the signer-disjoint test split.

DEV-03. Produces the two files the organizers read:
  results/predictions.jsonl  one JSON object per test clip
  results/results.csv        measured metric table

Every number here is measured on the ASL Citizen participant-disjoint test
split (11 signers, none seen in training). Nothing is estimated or carried
over from a paper.

Two accuracy numbers are reported and they are NOT interchangeable:
  - top1_accuracy_end_to_end  through src/main.py `predict()` (--end-to-end)
  - top1_accuracy_cached      landmarks from the cache, classifier only (default)

The gap between them is the MediaPipe front end's contribution. Reporting only
the cached number would hide a broken front end; reporting only the end-to-end
number would hide classifier problems behind decode noise.

The default scores the SHIPPED nine-member ensemble from
models/landmark_model_meta.json (mean-logit, mask by name), not the retired
single INT8 file - the previous default, landmark_classifier_fp16.tflite, was
deleted in the ensemble commit and made this script crash on a bare invocation.

Usage:
    python scripts/ml/evaluate.py --split test
    python scripts/ml/evaluate.py --split test --end-to-end
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
    # Match on the NAME, not the position. The TFLite converter does not
    # preserve the order the Keras model declared its inputs in, and it
    # reports the mask first - indexing [0] as "features" feeds a (1,24) mask
    # where a (1,24,F) feature matrix belongs.
    dets = interp.get_input_details()
    f_det = next((d for d in dets if "features" in d["name"]), dets[0])
    m_det = next((d for d in dets if "mask" in d["name"]), None)
    out = interp.get_output_details()[0]

    def invoke(feats, mask=None):
        interp.set_tensor(f_det["index"], feats[None, ...].astype(np.float32))
        if m_det is not None:
            if mask is None:
                raise ValueError(
                    f"{Path(model_path).name} takes a mask input; scoring it "
                    f"without one measures a different model than was trained")
            interp.set_tensor(m_det["index"], mask[None, ...].astype(np.float32))
        interp.invoke()
        logits = interp.get_tensor(out["index"])[0].astype(np.float32)
        e = np.exp(logits - logits.max())
        return e / e.sum()

    # Width is read off the graph, not from the sidecar: a model file is the
    # one thing that cannot disagree with the features it needs. The feature
    # axis is last, so a 255-wide member takes x,y and a 381-wide one x,y,z.
    width = int(f_det["shape"][-1])
    use_z = width != 255
    if width not in (255, 381):
        raise ValueError(f"{Path(model_path).name}: unexpected feature width {width}")
    return invoke, vocab, frames, use_z


def load_ensemble_classifier():
    """Score the SHIPPED nine-member ensemble, mean-logit, mask by name.

    This is the path src/main.py and the app run. Returns
    (invoke_from_landmarks, vocab, frames); invoke takes the raw per-clip
    landmarks and presence flags and returns probabilities, because the nine
    members do not share one feature width (four are 255-wide, five 381-wide).
    """
    from ai_edge_litert.interpreter import Interpreter

    meta = json.load(open(ROOT / "models" / "landmark_model_meta.json", encoding="utf-8"))
    labels = json.load(open(ROOT / "models" / "landmark_labels.json", encoding="utf-8"))
    idx_label = {int(i): g for g, i in labels.items()}
    vocab = [idx_label[i] for i in range(len(idx_label))]
    frames = int(meta["frames"])
    detail = {d["file"]: bool(d.get("use_z", False))
              for d in meta.get("members_detail", []) or []}

    members = []
    for name in meta["members"]:
        p = ROOT / "models" / name
        if not p.exists():
            sys.exit(f"Ensemble member missing: {p}")
        interp = Interpreter(model_path=str(p))
        interp.allocate_tensors()
        dets = interp.get_input_details()
        f_det = next((d for d in dets if "features" in d["name"]), dets[0])
        m_det = next((d for d in dets if "mask" in d["name"]), None)
        members.append({"it": interp, "f": f_det["index"],
                        "m": m_det["index"] if m_det else None,
                        "o": interp.get_output_details()[0]["index"],
                        "use_z": detail.get(name, False)})

    def invoke_from_landmarks(lm, pres):
        from features import build_features
        feats = {}
        for uz in {m["use_z"] for m in members}:
            feats[uz] = build_features(lm, pres, expected_frames=frames, use_z=uz)
        logits = []
        for m in members:
            f, mk = feats[m["use_z"]]
            m["it"].set_tensor(m["f"], f[None].astype(np.float32))
            if m["m"] is not None:
                m["it"].set_tensor(m["m"], mk[None].astype(np.float32))
            m["it"].invoke()
            logits.append(m["it"].get_tensor(m["o"])[0].astype(np.float32))
        lg = np.mean(logits, axis=0)
        e = np.exp(lg - lg.max())
        return e / e.sum()

    return invoke_from_landmarks, vocab, frames


def main():
    ap = argparse.ArgumentParser()
    # The default used to be landmark_classifier_fp16.tflite, deleted in
    # cdafe12 when the ensemble replaced it - so the script could not run at all
    # as shipped. Default to the ensemble manifest and score the mean-logit
    # ensemble (the shipped path); a single --ckpt still works for ablations.
    ap.add_argument("--ckpt", default=None,
                    help="single .tflite classifier to score; default: the nine-member "
                         "ensemble from models/landmark_model_meta.json (shipped path)")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--end-to-end", action="store_true",
                    help="also run src/main.py predict() on real video (slow)")
    args = ap.parse_args()

    from features import build_features

    if args.ckpt:
        ckpt_path = ROOT / args.ckpt
        if not ckpt_path.exists():
            sys.exit(f"--ckpt {ckpt_path} does not exist. The former default "
                     "models/landmark_classifier_fp16.tflite was removed when the "
                     "ensemble became the shipped artifact; omit --ckpt to score "
                     "the ensemble, or point at an existing .tflite file.")
        invoke, vocab, ck_frames, use_z = load_classifier(ckpt_path)
    else:
        invoke, vocab, ck_frames = load_ensemble_classifier()
        use_z = None
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

    # --end-to-end used to be parsed and advertised in the help text but never
    # referenced, so passing it silently measured nothing extra while the module
    # docstring promised a second, non-interchangeable accuracy number. Run the
    # real MediaPipe front end + src/main.py predict() on the clips that have a
    # video on disk, and report BOTH numbers, labelled.
    e2e = None
    if args.end_to_end:
        from main import predict as e2e_predict
        videos_dir = ROOT / ".scratch" / "aslcit_vocab_videos"
        videos = {p.stem: p for p in videos_dir.glob("*.mp4")}
        ok = tot = 0
        e2e_skipped = 0
        print(f"end-to-end: running src/main.py predict() on "
              f"{len(rows)} clips (slow)")
        for r in rows:
            stem = Path(r["Video file"]).stem
            vp = videos.get(stem)
            if vp is None:
                e2e_skipped += 1
                continue
            try:
                res = e2e_predict(str(vp))
            except ValueError:
                e2e_skipped += 1   # no hand / no coverage - same skip rule
                continue
            tot += 1
            ok += int(res["prediction"] == r["Gloss"])
        e2e = (ok, tot, e2e_skipped)

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
        if args.ckpt:
            # Single .tflite: width comes from the graph (255 -> x,y; 381 -> x,y,z).
            feats, mask = build_features(lm, pres, expected_frames=frames, use_z=use_z)
            probs = invoke(feats, mask)
        else:
            # Ensemble: invoke() takes the raw landmarks and builds each width.
            probs = invoke(lm, pres)
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
    e2e_pct = None
    if e2e is not None:
        e2e_ok, e2e_n, e2e_skipped = e2e
        if e2e_n:
            e2e_pct = round(100 * e2e_ok / e2e_n, 2)
            print(f"end-to-end (src/main.py predict()): {e2e_pct:.2f}% "
                  f"on {e2e_n} clips ({e2e_skipped} skipped: no video or no hand)")
            print(f"classifier-only (cached landmarks): "
                  f"{100*correct/n:.2f}% on {n} clips")
            print(f"the {100*correct/n - e2e_pct:+.2f} pt gap is the MediaPipe "
                  f"front end's contribution")
        else:
            print("end-to-end: no clips had a readable video; nothing measured")

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
    if e2e_pct is not None:
        summary["top1_accuracy_end_to_end_pct"] = e2e_pct
        summary["top1_accuracy_cached_pct"] = round(100 * correct / n, 2)
    (ROOT / "results" / f"eval_summary_{args.split}.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote results/eval_summary_{args.split}.json")


if __name__ == "__main__":
    main()