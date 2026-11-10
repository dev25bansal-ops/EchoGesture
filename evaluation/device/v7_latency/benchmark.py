"""V7 latency lab: paired identical-clip full clip-to-JSON benchmark.

Same structure as v5_latency/benchmark.py, adapted for v7_candidates.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import v7_candidates as cand  # noqa: E402


def sha256(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            d.update(b)
    return d.hexdigest()


def summary(v):
    if not v:
        return {"count": 0}
    o = sorted(v)
    def pct(f):
        import math
        return o[max(0, math.ceil(f * len(o)) - 1)]
    return {"count": len(v), "mean_ms": statistics.fmean(v), "p50_ms": pct(0.50),
            "p90_ms": pct(0.90), "p95_ms": pct(0.95), "p99_ms": pct(0.99),
            "min_ms": min(v), "max_ms": max(v)}


def git_sha():
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip() or None
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest", type=Path)
    p.add_argument("--output-dir", type=Path, default=HERE / "results")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--strategies", default="baseline")
    p.add_argument("--limit", type=int, default=30)
    args = p.parse_args()

    for f, v in [("--repeats", args.repeats), ("--warmup", args.warmup), ("--limit", args.limit)]:
        if v < (1 if f != "--warmup" else 0):
            p.error(f"{f} out of range")

    strategies = [s.strip().lower() for s in args.strategies.split(",") if s.strip()]
    if not strategies or any(s not in cand.STRATEGIES for s in strategies):
        p.error("--strategies must be a comma-separated subset of " + ",".join(cand.STRATEGIES))

    rows = list(csv.DictReader(args.manifest.open(newline="", encoding="utf-8-sig")))
    if len(rows) != 30:
        raise RuntimeError(f"expected 30 validation clips, got {len(rows)}")
    rows = rows[:args.limit]

    try:
        import ai_edge_litert
        litert = ai_edge_litert.__version__
    except Exception:
        litert = "unknown"
    try:
        import mediapipe
        mpv = mediapipe.__version__
    except Exception:
        mpv = "unknown"
    import cv2, numpy
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    records = []

    for s in strategies:
        for _ in range(args.warmup):
            cand.evaluate_clip(str(args.manifest.parent / rows[0]["path"]), s)

    for rep in range(args.repeats):
        for s in strategies:
            for row in rows:
                clip = args.manifest.parent / row["path"]
                entry = {"clip_id": row["clip_id"], "ground_truth": row["ground_truth"],
                         "signer_id": row.get("signer_id", ""), "strategy": s, "repeat": rep,
                         "started_at": datetime.now(timezone.utc).isoformat()}
                try:
                    started = time.perf_counter_ns()
                    result, timings = cand.evaluate_clip(str(clip), s)
                    timings["wall_ms"] = (time.perf_counter_ns() - started) / 1e6
                    entry.update({"status": "ok", "prediction": result, "timing": timings})
                except Exception as exc:
                    entry.update({"status": "error", "error_type": type(exc).__name__, "error": str(exc)})
                records.append(entry)

    per = {s: [r for r in records if r["strategy"] == s] for s in strategies}
    agg = {}
    for s, entries in per.items():
        ok = [e for e in entries if e["status"] == "ok"]
        agg[s] = {
            "runs": len(entries), "successful": len(ok),
            "decode": summary([e["timing"]["decode_ms"] for e in ok]),
            "tracking": summary([e["timing"]["tracking_ms"] for e in ok]),
            "features": summary([e["timing"]["features_ms"] for e in ok]),
            "classifier": summary([e["timing"]["classifier_ms"] for e in ok]),
            "postprocess": summary([e["timing"]["postprocess_ms"] for e in ok]),
            "total": summary([e["timing"]["total_ms"] for e in ok]),
            "wall": summary([e["timing"]["wall_ms"] for e in ok]),
            "correct": sum(1 for e in ok if str(e["prediction"]["prediction"]).upper() == str(e["ground_truth"]).upper()),
            "labelled": sum(1 for e in ok if e["ground_truth"]),
        }
        if agg[s]["labelled"]:
            agg[s]["top1"] = agg[s]["correct"] / agg[s]["labelled"]

    # All-pairs agreement against baseline (first strategy)
    baseline = strategies[0]
    paired_all = []
    if len(strategies) > 1:
        for s in strategies[1:]:
            paired = []
            for row in rows:
                for rep in range(args.repeats):
                    f = next((e for e in records if e["clip_id"] == row["clip_id"] and e["repeat"] == rep and e["strategy"] == baseline and e["status"] == "ok"), None)
                    s2 = next((e for e in records if e["clip_id"] == row["clip_id"] and e["repeat"] == rep and e["strategy"] == s and e["status"] == "ok"), None)
                    if f and s2:
                        paired.append({
                            "clip_id": row["clip_id"], "repeat": rep,
                            "candidate_minus_baseline_total_ms": s2["timing"]["total_ms"] - f["timing"]["total_ms"],
                            "candidate_minus_baseline_tracking_ms": s2["timing"]["tracking_ms"] - f["timing"]["tracking_ms"],
                            "same_prediction": f["prediction"]["prediction"] == s2["prediction"]["prediction"],
                            "same_top5": f["prediction"]["top5"] == s2["prediction"]["top5"],
                        })
            paired_all.append({
                "candidate": s,
                "compared_pairs": len(paired),
                "same_top1": sum(1 for e in paired if e["same_prediction"]),
                "same_top5": sum(1 for e in paired if e["same_top5"]),
                "paired_total_delta": summary([e["candidate_minus_baseline_total_ms"] for e in paired]),
                "paired_tracking_delta": summary([e["candidate_minus_baseline_tracking_ms"] for e in paired]),
                "pairs": paired,
            })

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task": "V7 latency lab: paired identical-clip full clip-to-JSON",
        "manifest": {"path": str(args.manifest.resolve()), "sha256": sha256(args.manifest.resolve())},
        "artifacts": {
            "fp16_classifier": {"path": str((ROOT / "models/landmark_classifier_fp16.tflite").resolve()),
                                "sha256": sha256(ROOT / "models/landmark_classifier_fp16.tflite")},
            "int8_classifier": {"path": str((ROOT / "models/landmark_classifier_int8.tflite").resolve()),
                                "sha256": sha256(ROOT / "models/landmark_classifier_int8.tflite")},
            "hand_landmarker": {"path": str((ROOT / "models/hand_landmarker.task").resolve()),
                                "sha256": sha256(ROOT / "models/hand_landmarker.task")},
            "labels": {"path": str((ROOT / "models/landmark_labels.json").resolve()),
                       "sha256": sha256(ROOT / "models/landmark_labels.json")},
            "frontend": {"path": str((ROOT / "src/landmark_frontend.py").resolve()),
                         "sha256": sha256(ROOT / "src/landmark_frontend.py")},
            "features": {"path": str((ROOT / "src/features.py").resolve()),
                         "sha256": sha256(ROOT / "src/features.py")},
            "main": {"path": str((ROOT / "src/main.py").resolve()),
                     "sha256": sha256(ROOT / "src/main.py")},
            "v7_candidates": {"path": str((HERE / "v7_candidates.py").resolve()),
                              "sha256": sha256(HERE / "v7_candidates.py")},
            "benchmark": {"path": str((HERE / "benchmark.py").resolve()),
                          "sha256": sha256(HERE / "benchmark.py")},
        },
        "environment": {
            "python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
            "cpu": platform.node(), "mediapipe": mpv, "litert": litert,
            "opencv": cv2.__version__, "numpy": numpy.__version__,
            "code_commit": git_sha(),
        },
        "hardware": {
            "brand": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True, stderr=subprocess.DEVNULL).strip(),
            "cpus": int(subprocess.check_output(["sysctl", "-n", "hw.ncpu"], text=True, stderr=subprocess.DEVNULL).strip()),
            "ram_bytes": int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True, stderr=subprocess.DEVNULL).strip()),
        },
        "model_contract": {
            "input_shape": [1, cand.MODEL_FRAMES, cand.MODEL_WIDTH], "input_dtype": "float32",
            "model_file_sha256": sha256(ROOT / "models/landmark_classifier_fp16.tflite"),
            "sampled_frames": cand.MAX_FRAMES, "frame_size": cand.FRAME_SIZE,
            "frame_budget_changed": False,
        },
        "configuration": {
            "repeats": args.repeats, "warmup_calls_per_strategy": args.warmup,
            "strategies": strategies, "serialized": True, "one_classifier_process": True,
            "state_lifecycle": "baseline=singleton IMAGE landmarker; parallel_2/parallel_4=persistent IMAGE tracker pools in threads with round-robin frame distribution; video_fresh=fresh VIDEO per clip ts=0,33,...; video_cached=singleton VIDEO monotonic timestamps across clips",
        },
        "aggregate": agg,
        "paired_vs_baseline": paired_all,
        "records": records,
    }
    (out / "benchmark.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (out / "predictions.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k not in ("records",)}, indent=2))


if __name__ == "__main__":
    main()
