"""A08 V7: paired clean/control/corrupt evaluation (detector run once per arm).

Takes the recovered V6 collect_frames / classifier harness and adds V7
wrappers. All detector state is shared between baseline and candidate
within one arm, so a candidate's landmarks differ from baseline only by
its wrapper. No prints beyond progress lines.

Usage:
  python evaluate_v7.py <manifest> --output-dir <dir> --corruption {low_light,crop,occlusion,motion_blur,frame_drop} --wrapper {baseline,repair_slot_flicker,absent_slot_zero,smooth_present_frames,combo_repair_smooth_zero} [--severity 0.5 --seed 42 --limit N --transport reencode]
Control: --corruption clean_reencode runs the codec-only arm.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
DEVICE = ROOT / "evaluation" / "device"
V6 = DEVICE / "v6_robustness"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(DEVICE))
sys.path.insert(0, str(V6))
sys.path.insert(0, str(HERE))

from diagnose import (  # noqa: E402
    FP16_SHA, INT8_SHA, get_interpreters, setup_interp, aggregate,
    load_manifest, read_labels, reliability, run_classifier, probs,
    resize_like_frontend, resample_rgb, encode_decode, sha256,
)
from evaluate_candidate_v6 import collect_frames  # noqa: E402
from features import build_features  # noqa: E402
from landmark_frontend import detect_hands  # noqa: E402
from robustness_lab import wilson_interval  # noqa: E402
from wrapper_v7 import apply_v7  # noqa: E402


def predict_both(clip_frames, classifier_state, labels, wrapper, wparams):
    lm, present = detect_hands(clip_frames, mode="image")
    lm = np.asarray(lm, dtype=np.float32)
    present = np.asarray(present, dtype=np.float32)
    if lm.shape[0] == 0 or float(present.max()) < .5:
        raise ValueError("No hand detected in any frame")
    rel_base_pre = reliability(lm, present)
    lm_c, pr_c = apply_v7(lm, present, wrapper, **wparams)
    rel_cand = reliability(lm_c, pr_c)
    tensor_base, _ = build_features(lm, present, expected_frames=24)
    tensor_cand, _ = build_features(lm_c, pr_c, expected_frames=24)
    tb = tensor_base[None, ...].astype(np.float32)
    tc = tensor_cand[None, ...].astype(np.float32)
    models = {}
    for name, interp, ind, outd in classifier_state:
        lb = run_classifier(interp, ind, outd, tb)
        pb = probs(lb)
        topb = np.argsort(-pb)[:5]
        lc = run_classifier(interp, ind, outd, tc)
        pc = probs(lc)
        topc = np.argsort(-pc)[:5]
        models[name] = {
            "baseline": {"prediction": labels[int(topb[0])], "confidence": float(pb[topb[0]]),
                         "top5": [labels[int(i)] for i in topb], "logits": lb.tolist(),
                         "logits_sha256": hashlib.sha256(np.asarray(lb, np.float32).tobytes()).hexdigest()},
            "candidate": {"prediction": labels[int(topc[0])], "confidence": float(pc[topc[0]]),
                          "top5": [labels[int(i)] for i in topc], "logits": lc.tolist(),
                          "logits_sha256": hashlib.sha256(np.asarray(lc, np.float32).tobytes()).hexdigest()},
            "agreement": bool(labels[int(topb[0])] == labels[int(topc[0])]),
        }
    return {"reliability_base": rel_base_pre, "reliability_candidate": rel_cand,
            "tensor_base_sha256": hashlib.sha256(tb.tobytes()).hexdigest(),
            "tensor_candidate_sha256": hashlib.sha256(tc.tobytes()).hexdigest(),
            "landmarks_sha256": hashlib.sha256(lm.tobytes()).hexdigest(),
            "present_sha256": hashlib.sha256(present.tobytes()).hexdigest(),
            "landmarks_candidate_sha256": hashlib.sha256(lm_c.tobytes()).hexdigest(),
            "models": models}


def aggregate_records(records):
    out = {"n": len(records), "failed_pairs": sum(bool(r.get("error")) for r in records)}
    for kind in ("baseline", "candidate"):
        for model in ("fp16", "int8"):
            for arm in ("clean", "corrupted"):
                vals = [r for r in records if not r.get("error") and arm in r]
                c = sum(r[arm]["models"][model][kind]["prediction"] == r["ground_truth"] for r in vals)
                conf = [r[arm]["models"][model][kind]["confidence"] for r in vals]
                out.setdefault(kind, {}).setdefault(model, {})[arm] = {
                    "correct": c, "n": len(vals), "top1": c / len(vals) if vals else None,
                    "wilson95": wilson_interval(c, len(vals)) if vals else [None, None],
                    "mean_confidence": float(np.mean(conf)) if conf else None}
    for kind in ("baseline", "candidate"):
        for model in ("fp16", "int8"):
            vals = [r for r in records if not r.get("error") and "clean" in r and "corrupted" in r]
            c2w = sum(r["clean"]["models"][model][kind]["prediction"] == r["ground_truth"]
                      and r["corrupted"]["models"][model][kind]["prediction"] != r["ground_truth"] for r in vals)
            w2c = sum(r["clean"]["models"][model][kind]["prediction"] != r["ground_truth"]
                      and r["corrupted"]["models"][model][kind]["prediction"] == r["ground_truth"] for r in vals)
            same = sum(r["clean"]["models"][model][kind]["prediction"] == r["corrupted"]["models"][model][kind]["prediction"] for r in vals)
            out.setdefault("pairs", {})[f"{kind}_{model}"] = {
                "clean_correct_corrupted_wrong": c2w, "clean_wrong_corrupted_correct": w2c,
                "agreement": same / len(vals) if vals else None, "n": len(vals)}
    # per-corruption, per-clip flip detail for the Dev report
    out["flips"] = []
    for r in records:
        if r.get("error") or "clean" not in r or "corrupted" not in r:
            continue
        out["flips"].append({"clip_id": r["clip_id"], "ground_truth": r["ground_truth"],
                             "fp16_clean_base": r["clean"]["models"]["fp16"]["baseline"]["prediction"],
                             "fp16_corrupt_base": r["corrupted"]["models"]["fp16"]["baseline"]["prediction"],
                             "fp16_clean_cand": r["clean"]["models"]["fp16"]["candidate"]["prediction"],
                             "fp16_corrupt_cand": r["corrupted"]["models"]["fp16"]["candidate"]["prediction"],
                             "int8_clean_base": r["clean"]["models"]["int8"]["baseline"]["prediction"],
                             "int8_corrupt_base": r["corrupted"]["models"]["int8"]["baseline"]["prediction"],
                             "int8_clean_cand": r["clean"]["models"]["int8"]["candidate"]["prediction"],
                             "int8_corrupt_cand": r["corrupted"]["models"]["int8"]["candidate"]["prediction"],
                             "clean_detection_cand": r["clean"]["reliability_candidate"]["hand_detection_rate"],
                             "corrupt_detection_cand": r["corrupted"]["reliability_candidate"]["hand_detection_rate"],
                             "clean_detection_base": r["clean"]["reliability_base"]["hand_detection_rate"],
                             "corrupt_detection_base": r["corrupted"]["reliability_base"]["hand_detection_rate"]})
    return out


def comparable_records(records):
    converted = []
    for r in records:
        if r.get("error"):
            converted.append(r)
            continue
        q = dict(r)
        for arm in ("clean", "corrupted"):
            q[arm] = {"reliability": r[arm]["reliability_candidate"],
                      "models": {m: {kind: r[arm]["models"][m][kind] for kind in ("baseline", "candidate")}
                                 for m in ("fp16", "int8")}}
        converted.append(q)
    return converted


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--corruption", default="low_light")
    ap.add_argument("--severity", type=float, default=.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--wrapper", default="baseline")
    ap.add_argument("--wrapper-params", default="{}")
    ap.add_argument("--transport", choices=["reencode", "direct"], default="reencode")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    wparams = json.loads(args.wrapper_params)
    outdir = args.output_dir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    workdir = outdir / "work"
    workdir.mkdir(exist_ok=True)
    manifest = args.manifest.resolve()
    items = load_manifest(manifest, args.limit, args.only)
    cstate = []
    for name, (path, interp) in zip(("fp16", "int8"), get_interpreters()):
        ind, outd = setup_interp(interp)
        cstate.append((name, interp, ind, outd))
    labels = read_labels()
    records = []
    for i, item in enumerate(items):
        started = time.time()
        try:
            kind = "low_light" if args.corruption == "clean_reencode" else args.corruption
            sev = 0.0 if args.corruption == "clean_reencode" else args.severity
            clean_rgb, corr_rgb, clip_seed, nsrc, fps = collect_frames(
                item, i, kind, sev, args.seed, args.transport, workdir,
                skip_corruption=(args.corruption == "clean_reencode"))
            if args.corruption == "clean_reencode":
                corr_rgb, _ = encode_decode(clean_rgb, fps, workdir, f"codec-{i:05d}")
            clean = predict_both(clean_rgb, cstate, labels, args.wrapper, wparams)
            corr = predict_both(corr_rgb, cstate, labels, args.wrapper, wparams)
            records.append({"clip_id": item["clip_id"], "path": item["path"], "signer_id": item["signer"],
                            "ground_truth": item["truth"], "corruption": args.corruption,
                            "severity": args.severity, "seed": args.seed, "corruption_seed": clip_seed,
                            "transport": args.transport, "candidate": args.wrapper,
                            "wrapper_parameters": wparams, "source_frame_count": nsrc, "source_fps": fps,
                            "clean": clean, "corrupted": corr, "feature_l2": None,
                            "elapsed_seconds": time.time() - started})
        except Exception as exc:
            records.append({"clip_id": item["clip_id"], "path": item["path"], "signer_id": item["signer"],
                            "ground_truth": item["truth"], "corruption": args.corruption,
                            "severity": args.severity, "seed": args.seed, "transport": args.transport,
                            "candidate": args.wrapper, "wrapper_parameters": wparams,
                            "error": f"{type(exc).__name__}: {exc}", "elapsed_seconds": time.time() - started})
        print(f"[{i+1}/{len(items)}] {item['clip_id']} {records[-1].get('error','ok')}", flush=True)
    summary = aggregate_records(records)
    summary.update({"status": "complete", "recorded_at": datetime.now(timezone.utc).isoformat(),
                    "manifest": str(manifest), "manifest_sha256": sha256(manifest),
                    "corruption": args.corruption, "severity": args.severity, "seed": args.seed,
                    "transport": args.transport, "candidate": args.wrapper, "wrapper_parameters": wparams,
                    "n_requested": len(items),
                    "models": {"fp16": {"sha256": FP16_SHA}, "int8": {"sha256": INT8_SHA}},
                    "source_sha256": {"features": sha256(SRC / "features.py"), "frontend": sha256(SRC / "landmark_frontend.py"),
                                      "wrapper": sha256(HERE / "wrapper_v7.py"), "stock_main": sha256(SRC / "main.py")},
                    "scope": "Validation-only; official test untouched. Baseline/candidate share detector output per arm. Failures retained.",
                    "records": records})
    (outdir / "comparison.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n")
    with (outdir / "comparison.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    with (outdir / "comparison.csv").open("w", newline="") as f:
        fields = ["clip_id", "signer_id", "ground_truth", "error", "baseline_fp16_clean", "baseline_fp16_corrupt", "candidate_fp16_clean", "candidate_fp16_corrupt",
                  "baseline_int8_clean", "baseline_int8_corrupt", "candidate_int8_clean", "candidate_int8_corrupt",
                  "clean_detection", "corrupt_detection"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            def g(arm, model, kind):
                try:
                    return r[arm]["models"][model][kind]["prediction"]
                except Exception:
                    return ""
            w.writerow({"clip_id": r.get("clip_id"), "signer_id": r.get("signer_id"), "ground_truth": r.get("ground_truth"), "error": r.get("error"),
                        "baseline_fp16_clean": g("clean", "fp16", "baseline"), "baseline_fp16_corrupt": g("corrupted", "fp16", "baseline"),
                        "candidate_fp16_clean": g("clean", "fp16", "candidate"), "candidate_fp16_corrupt": g("corrupted", "fp16", "candidate"),
                        "baseline_int8_clean": g("clean", "int8", "baseline"), "baseline_int8_corrupt": g("corrupted", "int8", "baseline"),
                        "candidate_int8_clean": g("clean", "int8", "candidate"), "candidate_int8_corrupt": g("corrupted", "int8", "candidate"),
                        "clean_detection": r.get("clean", {}).get("reliability_candidate", {}).get("hand_detection_rate"),
                        "corrupt_detection": r.get("corrupted", {}).get("reliability_candidate", {}).get("hand_detection_rate")})
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
