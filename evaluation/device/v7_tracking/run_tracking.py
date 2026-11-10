#!/usr/bin/env python3
"""Run pinned IMAGE-mode Dev tracking on a labelled manifest, once per clip.

Detection runs once per clip; baseline and all opt-in landmark candidates are
evaluated from that identical raw detection output. The production frontend,
feature builder, and models are imported unchanged. Every successful clip
stores raw landmarks/presence so a later comparison never redetects.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path("/tmp/echogesture-v7-agents/A03")
SRC = ROOT / "src"
DEVICE = ROOT / "evaluation/device"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(DEVICE))
sys.path.insert(0, str(HERE))

from candidates import apply_candidate  # noqa: E402
from v6_robustness.diagnose import (  # noqa: E402
    FP16_SHA, INT8_SHA, get_interpreters, load_manifest, read_labels,
    reliability, run_classifier, setup_interp, probs, sha256,
)
from features import build_features  # noqa: E402
from landmark_frontend import read_frames, detect_hands  # noqa: E402

OPTIONS = {}


def model_prediction(interp, ind, outd, tensor):
    logits = run_classifier(interp, ind, outd, tensor)
    p = probs(logits)
    top = np.argsort(-p)[:5]
    return p, {
        "prediction": OPTIONS[int(top[0])],
        "prediction_index": int(top[0]),
        "confidence": float(p[top[0]]),
        "top5": [OPTIONS[int(i)] for i in top],
        "top5_probabilities": [float(p[int(i)]) for i in top],
        "logits_sha256": hashlib.sha256(np.asarray(logits, np.float32).tobytes()).hexdigest(),
        "logits": logits.tolist(),
    }


def classifier_outputs(classifier_state, tensor):
    out = {}
    for name, interp, ind, outd in classifier_state:
        p, record = model_prediction(interp, ind, outd, tensor)
        out[name] = record
    return out


def main() -> int:
    global OPTIONS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--arm", choices=["clean", "corrupted"], required=True)
    parser.add_argument("--corruption", default="")
    parser.add_argument("--severity", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-gap", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--only", default="")
    args = parser.parse_args()

    for modules, name in ((SRC / "features.py", "features"), (SRC / "landmark_frontend.py", "frontend")):
        if not modules.exists():
            raise SystemExit(f"Missing {name} source: {modules}")
    outdir = args.output_dir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    workdir = outdir / "work"
    workdir.mkdir(exist_ok=True)
    manifest = args.manifest.resolve()
    items = load_manifest(manifest, args.limit, args.only)
    cstate = []
    for name, (_path, interp) in zip(("fp16", "int8"), get_interpreters()):
        ind, outd = setup_interp(interp)
        cstate.append((name, interp, ind, outd))
    labels = read_labels()
    OPTIONS = {int(k): v for k, v in labels.items()}

    records = []
    raw_landmarks = []
    raw_present = []
    started = time.time()
    try:
        from corruptions import CorruptionSpec, corrupt_frames, decode_video, write_video  # noqa: E402
        from robustness_lab import derive_clip_seed  # noqa: E402

        for i, item in enumerate(items):
            rec = {"clip_id": item["clip_id"], "path": item["path"],
                   "signer_id": item["signer"], "ground_truth": item["truth"],
                   "arm": args.arm}
            try:
                if args.arm == "clean":
                    frames = read_frames(item["path"], max_frames=32)
                    if not frames:
                        raise ValueError("No valid frames decoded")
                    transport = "stock_read_frames"
                    corruption_seed = None
                else:
                    if not args.corruption:
                        raise ValueError("Corruption is required for corrupted arm")
                    corruption_seed = derive_clip_seed(args.seed, i, item["clip_id"])
                    source_frames, source_fps = decode_video(item["path"])
                    corrupt = corrupt_frames(
                        source_frames, CorruptionSpec(args.corruption, args.severity, corruption_seed))
                    p = workdir / f"corrupt-{i:05d}.mp4"
                    write_video(p, corrupt, source_fps)
                    frames = read_frames(p, max_frames=32)
                    p.unlink(missing_ok=True)
                    if not frames:
                        raise ValueError("No valid corrupted frames decoded")
                    transport = "decode_corrupt_write_reencode_read_frames"
                landmarks, present = detect_hands(frames, mode="image")
                landmarks = np.asarray(landmarks, dtype=np.float32)
                present = np.asarray(present, dtype=np.float32)
                if landmarks.shape[0] == 0 or float(present.max()) < 0.5:
                    raise ValueError("No hand detected in any frame")
                raw_landmarks.append(landmarks)
                raw_present.append(present)
                rec.update({
                    "frames": len(frames),
                    "landmarks_sha256": hashlib.sha256(landmarks.tobytes()).hexdigest(),
                    "present_sha256": hashlib.sha256(present.tobytes()).hexdigest(),
                    "raw_reliability": reliability(landmarks, present),
                    "candidates": {},
                })
                for candidate in ("baseline", "slot", "gap", "slot_gap"):
                    trial_lm, trial_present, info = apply_candidate(
                        landmarks, present, candidate, max_gap=args.max_gap)
                    tensor, mask = build_features(trial_lm, trial_present, expected_frames=24)
                    models = classifier_outputs(cstate, tensor[None, ...].astype(np.float32))
                    rec["candidates"][candidate] = {
                        "candidate_contract": info,
                        "tensor_sha256": hashlib.sha256(tensor.tobytes()).hexdigest(),
                        "feature_l2": float(np.linalg.norm(tensor)),
                        "landmarks_sha256": hashlib.sha256(trial_lm.tobytes()).hexdigest(),
                        "present_sha256": hashlib.sha256(trial_present.tobytes()).hexdigest(),
                        "models": models,
                        "mask": mask.tolist(),
                    }
                rec["corruption_seed"] = corruption_seed
                rec["transport"] = transport
            except Exception as exc:
                rec["error"] = f"{type(exc).__name__}: {exc}"
            records.append(rec)
            print(f"[{i + 1}/{len(items)}] {item['clip_id']} {rec.get('error', 'ok')}", flush=True)
    finally:
        # Temporary corrupted clips are removed by removing the work directory.
        import shutil
        shutil.rmtree(workdir, ignore_errors=True)

    if raw_landmarks:
        frame_lengths = sorted(set(len(x) for x in raw_landmarks))
        if len(frame_lengths) == 1:
            np.save(outdir / "raw_landmarks.npy", np.stack(raw_landmarks, axis=0))
            np.save(outdir / "raw_present.npy", np.stack(raw_present, axis=0))
        else:
            # Frame-drop clips can decode fewer than 32 sampled frames. Retain
            # them as a pickle-free JSON list instead of refusing to save evidence.
            (outdir / "raw_tracks.json").write_text(json.dumps({
                "frame_lengths": [len(x) for x in raw_landmarks],
                "landmarks": [x.tolist() for x in raw_landmarks],
                "present": [x.tolist() for x in raw_present],
            }, ensure_ascii=False) + "\n")

    def summarize(candidates_records):
        summary = {}
        for candidate in ("baseline", "slot", "gap", "slot_gap"):
            vals = [r for r in candidates_records if candidate in r.get("candidates", {})]
            summary[candidate] = {}
            for model in ("fp16", "int8"):
                correct = sum(
                    r["candidates"][candidate]["models"][model]["prediction"] == r["ground_truth"]
                    for r in vals)
                summary[candidate][model] = {"correct": int(correct), "n": len(vals),
                                              "top1": correct / len(vals) if vals else None}
        return summary

    output = {
        "status": "complete",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "arm": args.arm,
        "corruption": args.corruption or None,
        "severity": args.severity if args.arm == "corrupted" else None,
        "seed": args.seed,
        "max_gap": args.max_gap,
        "n_requested": len(items),
        "n_records": len(records),
        "n_success": sum("error" not in r for r in records),
        "failed": sum("error" in r for r in records),
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "models": {"fp16": {"sha256": FP16_SHA}, "int8": {"sha256": INT8_SHA}},
        "source_sha256": {"features": sha256(SRC / "features.py"),
                          "frontend": sha256(SRC / "landmark_frontend.py")},
        "summary": summarize(records),
        "records": records,
        "runtime": {"python": sys.version, "platform": platform.platform(),
                    "numpy": np.__version__},
        "elapsed_seconds": time.time() - started,
        "scope": "One detection per clip; all candidates share identical raw detector output.",
    }
    (outdir / "run.json").write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    with (outdir / "records.jsonl").open("w") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(output["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
