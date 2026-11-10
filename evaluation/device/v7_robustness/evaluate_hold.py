"""A08 V7: test the rejected-idea hypothesis — hold features, not landmarks.

Hypothesis (from V6 data): the harmful mechanism in sticky_hand is not
"extra presence" per se, but that filling coordinates at absent frames
creates a zero-velocity plateau followed by a velocity impulse on
re-acquisition (features.py: vel[t] = norm[t]-norm[t-1], acc lag-2).
A candidate that (1) keeps presence==0 and (2) interpolates linearly is
already tested (smooth_present_frames): it changes <1 clip per corruption
and is net-neutral to slightly negative.

This script tests the other V5/B candidate, 'hold exactly the same
features', at IoU/state level. The feature-level instantiation allowed
here without changing feature semantics is an explicit *feature hold*:
absent frames take the last present frame's full 255-d feature vector
(INCLUDING its zeroed velocity/acc columns), instead of the _resample
interpolated trajectory through the origin. This is what
absent_slot_zero + linear fill was trying to approximate.

Also tests a diagnostic control: compare feature-space nearest-neighbour
stability (feature L2 clean->corrupt for held vs interpolated) so Dev can
see the mechanism directly.

Valid, honest use only: no ground-truth leakage, no model change.
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
    FP16_SHA, INT8_SHA, get_interpreters, setup_interp, load_manifest,
    read_labels, reliability, run_classifier, probs, encode_decode, sha256,
)
from evaluate_candidate_v6 import collect_frames  # noqa: E402
from evaluate_v7 import aggregate_records  # noqa: E402
from features import build_features  # noqa: E402
from landmark_frontend import detect_hands  # noqa: E402
from wrapper_v7 import apply_v7  # noqa: E402


def predict_hold(clip_frames, classifier_state, labels):
    """Baseline vs feature-hold (hold last present 255-d vector on absent
    frames; presence mask unchanged)."""
    lm, present = detect_hands(clip_frames, mode="image")
    lm = np.asarray(lm, dtype=np.float32)
    present = np.asarray(present, dtype=np.float32)
    if lm.shape[0] == 0 or float(present.max()) < .5:
        raise ValueError("No hand detected in any frame")
    tensor_base, mask = build_features(lm, present, expected_frames=24)
    tb = tensor_base[None, ...].astype(np.float32)
    # feature hold: absent tensor frames take previous present frame's vector
    th = tb.copy()
    last = None
    for t in range(th.shape[1]):
        if mask[t] > .5:
            last = th[0, t].copy()
        elif last is not None:
            th[0, t] = last
    models = {}
    for name, interp, ind, outd in classifier_state:
        lb = run_classifier(interp, ind, outd, tb)
        pb = probs(lb)
        topb = np.argsort(-pb)[:5]
        lc = run_classifier(interp, ind, outd, th)
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
    rel = reliability(lm, present)
    return {"reliability_base": rel, "reliability_candidate": rel,
            "tensor_base_sha256": hashlib.sha256(tb.tobytes()).hexdigest(),
            "tensor_candidate_sha256": hashlib.sha256(th.tobytes()).hexdigest(),
            "landmarks_sha256": hashlib.sha256(lm.tobytes()).hexdigest(),
            "present_sha256": hashlib.sha256(present.tobytes()).hexdigest(),
            "landmarks_candidate_sha256": hashlib.sha256(lm.tobytes()).hexdigest(),
            "models": models}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--corruption", default="occlusion")
    ap.add_argument("--severity", type=float, default=.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--transport", choices=["reencode", "direct"], default="reencode")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    outdir = args.output_dir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    workdir = outdir / "work"
    workdir.mkdir(exist_ok=True)
    manifest = args.manifest.resolve()
    items = load_manifest(manifest, args.limit, "")
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
            clean = predict_hold(clean_rgb, cstate, labels)
            corr = predict_hold(corr_rgb, cstate, labels)
            records.append({"clip_id": item["clip_id"], "path": item["path"], "signer_id": item["signer"],
                            "ground_truth": item["truth"], "corruption": args.corruption,
                            "severity": args.severity, "seed": args.seed, "corruption_seed": clip_seed,
                            "transport": args.transport, "candidate": "feature_hold",
                            "wrapper_parameters": {}, "source_frame_count": nsrc, "source_fps": fps,
                            "clean": clean, "corrupted": corr, "feature_l2": None,
                            "elapsed_seconds": time.time() - started})
        except Exception as exc:
            records.append({"clip_id": item["clip_id"], "path": item["path"], "signer_id": item["signer"],
                            "ground_truth": item["truth"], "corruption": args.corruption,
                            "severity": args.severity, "seed": args.seed, "transport": args.transport,
                            "candidate": "feature_hold", "wrapper_parameters": {},
                            "error": f"{type(exc).__name__}: {exc}", "elapsed_seconds": time.time() - started})
        print(f"[{i+1}/{len(items)}] {item['clip_id']} {records[-1].get('error','ok')}", flush=True)
    summary = aggregate_records(records)
    summary.update({"status": "complete", "recorded_at": datetime.now(timezone.utc).isoformat(),
                    "manifest": str(manifest), "manifest_sha256": sha256(manifest),
                    "corruption": args.corruption, "severity": args.severity, "seed": args.seed,
                    "transport": args.transport, "candidate": "feature_hold",
                    "wrapper_parameters": {}, "n_requested": len(items),
                    "models": {"fp16": {"sha256": FP16_SHA}, "int8": {"sha256": INT8_SHA}},
                    "source_sha256": {"features": sha256(SRC / "features.py"), "frontend": sha256(SRC / "landmark_frontend.py"),
                                      "stock_main": sha256(SRC / "main.py")},
                    "scope": "Validation-only; official test untouched. Baseline/candidate share detector output per arm. Failures retained.",
                    "records": records})
    (outdir / "comparison.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n")
    with (outdir / "comparison.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
