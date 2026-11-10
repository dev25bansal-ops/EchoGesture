"""Independent validation-only evaluation of Dev's landmark candidate.

Runs the integrated `src/main.py` predictor on a labelled ASL Citizen
validation manifest, never the locked test split. Captures raw predictions,
confidence, per-clip diagnostics, and aggregate accuracy/confusion summaries.
No model tuning or parameter selection is performed here.
"""

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wilson(successes, total, z=1.96):
    if total == 0:
        return [None, None]
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return [centre - margin, centre + margin]


def main(argv=None):
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-model-sha",
                        default="a012f2b5f7a1110961079f19eb0e28c9abacb95e4d0787352e4157ebca7044a4")
    args = parser.parse_args(argv)

    manifest = args.manifest.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # The isolated interpreter prevents an adjacent untrusted module shadowing
    # imports. The predictor itself inserts its own src directory, as designed.
    sys.path.insert(0, str(root / "src"))
    import main as predictor

    model_path = root / "models/landmark_classifier_fp16.tflite"
    actual_sha = sha256(model_path)
    if actual_sha != args.expected_model_sha:
        raise SystemExit(f"Unexpected model SHA: {actual_sha}")

    with manifest.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))

    records = []
    failures = []
    for index, row in enumerate(rows, start=1):
        clip_id = row["clip_id"]
        path = Path(row["path"])
        if not path.is_absolute():
            path = manifest.parent / path
        truth = row["ground_truth"].upper()
        signer = row["signer_id"]
        record = {
            "clip_id": clip_id,
            "path": str(path),
            "ground_truth": truth,
            "signer_id": signer,
            "status": "ok",
        }
        try:
            result = predictor.predict(str(path), clip_id=clip_id, profile=True)
            profile = predictor.get_last_profile()
            tracking = profile["stages"].get("landmark_tracking", {})
            decode = profile["stages"].get("video_open_decode", {})
            record.update({
                "prediction": result["prediction"],
                "confidence": result["confidence"],
                "top5": result["top5"],
                "top1_correct": result["prediction"] == truth,
                "top5_correct": truth in result["top5"],
                "latency_ms": result["latency_ms"],
                "sampled_frames": decode.get("sampled_frames"),
                "frames_with_hands": tracking.get("frames_with_hands"),
                "hand_detection_rate": (
                    tracking.get("frames_with_hands", 0) / decode["sampled_frames"]
                    if decode.get("sampled_frames") else None
                ),
            })
        except Exception as exc:  # retain failures instead of dropping them
            record.update({
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "top1_correct": False,
                "top5_correct": False,
            })
            failures.append(record)
        records.append(record)
        print(f"[{index}/{len(rows)}] {clip_id}: {record['status']} "
              f"truth={truth} pred={record.get('prediction')} "
              f"conf={record.get('confidence')}", flush=True)

    n = len(records)
    correct = sum(r["top1_correct"] for r in records)
    top5_correct = sum(r["top5_correct"] for r in records)
    signers = sorted({r["signer_id"] for r in records})
    classes = sorted({r["ground_truth"] for r in records})
    predictions = [r for r in records if r["status"] == "ok"]

    per_signer = {}
    for signer in signers:
        subset = [r for r in records if r["signer_id"] == signer]
        c = sum(r["top1_correct"] for r in subset)
        per_signer[signer] = {
            "n": len(subset), "correct": c,
            "accuracy": c / len(subset),
            "wilson95": wilson(c, len(subset)),
            "failed": sum(r["status"] == "failed" for r in subset),
        }

    per_class = {}
    for label in classes:
        subset = [r for r in records if r["ground_truth"] == label]
        c = sum(r["top1_correct"] for r in subset)
        per_class[label] = {
            "n": len(subset), "correct": c, "accuracy": c / len(subset),
            "predictions": dict(Counter(r.get("prediction", "<FAILED>") for r in subset)),
        }

    confusion = Counter((r["ground_truth"], r.get("prediction", "<FAILED>"))
                        for r in records if not r["top1_correct"])
    confidence_bins = {
        "lt_0.25": lambda c: c < 0.25,
        "0.25_0.50": lambda c: 0.25 <= c < 0.50,
        "0.50_0.75": lambda c: 0.50 <= c < 0.75,
        "ge_0.75": lambda c: c >= 0.75,
    }
    confidence = {}
    for name, predicate in confidence_bins.items():
        subset = [r for r in predictions if predicate(r["confidence"])]
        c = sum(r["top1_correct"] for r in subset)
        confidence[name] = {
            "n": len(subset), "correct": c,
            "accuracy": c / len(subset) if subset else None,
            "mean_confidence": (sum(r["confidence"] for r in subset) / len(subset)
                                if subset else None),
        }

    summary = {
        "dataset": "ASL Citizen permitted validation subset (NOT locked test data)",
        "manifest": {"path": str(manifest), "sha256": sha256(manifest)},
        "model": {"path": str(model_path), "sha256": actual_sha},
        "labels_sha256": sha256(root / "models/landmark_labels.json"),
        "features_sha256": sha256(root / "src/features.py"),
        "frontend_sha256": sha256(root / "src/landmark_frontend.py"),
        "predictor_sha256": sha256(root / "src/main.py"),
        "n_clips": n,
        "n_signers": len(signers),
        "signers": signers,
        "n_classes": len(classes),
        "top1_correct": correct,
        "top1_accuracy": correct / n if n else None,
        "top1_wilson95": wilson(correct, n),
        "top5_correct": top5_correct,
        "top5_accuracy": top5_correct / n if n else None,
        "successful_predictions": len(predictions),
        "failed_predictions": len(failures),
        "per_signer": per_signer,
        "per_class": per_class,
        "confusion_pairs": [
            {"truth": truth, "prediction": pred, "count": count}
            for (truth, pred), count in sorted(confusion.items(), key=lambda x: (-x[1], x[0]))
        ],
        "confidence_stratification": confidence,
        "scope_caveat": (
            "Small, unbalanced 30-clip/5-signer validation batch. It cannot verify "
            "Dev's 62.61% claim or represent full held-out accuracy. Failures count "
            "as incorrect. No threshold tuning or model changes were performed."
        ),
    }

    with (out_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["clip_id", "ground_truth", "prediction", "top5", "signer_id", "status",
                  "top1_correct", "top5_correct", "confidence", "latency_ms",
                  "sampled_frames", "frames_with_hands", "hand_detection_rate",
                  "error_type", "error", "path"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["top5"] = "|".join(row.get("top5", []))
            writer.writerow({field: row.get(field) for field in fields})

    (out_dir / "predictions.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
