"""Paired clean/corrupted tracking comparison with FP16/INT8 parity."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def wilson(successes, n, z=1.96):
    if not n:
        return [None, None]
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [centre - margin, centre + margin]


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def summarize_arm(records, model):
    correct = sum(r["models"][model]["prediction"] == r["ground_truth"] for r in records)
    n = len(records)
    return {"correct": int(correct), "n": n, "top1": correct / n if n else None,
            "wilson95": wilson(correct, n) if n else [None, None],
            "mean_confidence": float(np.mean([r["models"][model]["confidence"] for r in records])) if n else None}


def compare_runs(clean_path, corrupt_path):
    clean = load(clean_path)
    corrupt = load(corrupt_path)
    c = {r["clip_id"]: r for r in clean["records"]}
    d = {r["clip_id"]: r for r in corrupt["records"]}
    all_ids = sorted(set(c) | set(d))
    result = {"clean": {}, "corrupted": {}, "pairs": {}, "changes": {}}
    for arm in ("clean", "corrupted"):
        for candidate in ("baseline", "slot", "gap", "slot_gap"):
            for model in ("fp16", "int8"):
                recs = []
                for clip_id in all_ids:
                    source = c if arm == "clean" else d
                    if clip_id not in source or "candidates" not in source[clip_id]:
                        continue
                    r = source[clip_id]
                    if candidate in r["candidates"]:
                        recs.append({"clip_id": clip_id, "ground_truth": r["ground_truth"],
                                     "models": r["candidates"][candidate]["models"]})
                result[arm][f"{candidate}_{model}"] = summarize_arm(recs, model)
    for candidate in ("baseline", "slot", "gap", "slot_gap"):
        for model in ("fp16", "int8"):
            pairs = []
            for clip_id in all_ids:
                if clip_id not in c or clip_id not in d:
                    continue
                a, b = c[clip_id], d[clip_id]
                if "candidates" not in a or "candidates" not in b:
                    continue
                x = a["candidates"][candidate]["models"][model]
                y = b["candidates"][candidate]["models"][model]
                pairs.append({"clip_id": clip_id, "truth": a["ground_truth"],
                              "clean_pred": x["prediction"], "corrupt_pred": y["prediction"],
                              "clean_correct": x["prediction"] == a["ground_truth"],
                              "corrupt_correct": y["prediction"] == a["ground_truth"]})
            c2w = sum(x["clean_correct"] and not x["corrupt_correct"] for x in pairs)
            w2c = sum(not x["clean_correct"] and x["corrupt_correct"] for x in pairs)
            same = sum(x["clean_pred"] == x["corrupt_pred"] for x in pairs)
            result["pairs"][f"{candidate}_{model}"] = {
                "n": len(pairs), "clean_correct_corrupted_wrong": int(c2w),
                "clean_wrong_corrupted_correct": int(w2c),
                "clean_corrupted_prediction_agreement": same / len(pairs) if pairs else None,
            }
    # Within one identical candidate tensor, compare FP16 and INT8 top-1.
    for candidate in ("baseline", "slot", "gap", "slot_gap"):
        for arm in ("clean", "corrupted"):
            pairs = []
            source = c if arm == "clean" else d
            for clip_id in sorted(source):
                r = source[clip_id]
                if "candidates" not in r or candidate not in r["candidates"]:
                    continue
                models = r["candidates"][candidate]["models"]
                pairs.append({"clip_id": clip_id, "fp16": models["fp16"], "int8": models["int8"],
                              "same": models["fp16"]["prediction"] == models["int8"]["prediction"],
                              "fp16_correct": models["fp16"]["prediction"] == r["ground_truth"],
                              "int8_correct": models["int8"]["prediction"] == r["ground_truth"],
                              "logits": models["fp16"]["logits"], "int8_logits": models["int8"]["logits"]})
            agreement = sum(x["same"] for x in pairs)
            l1 = [sum(abs(a - b) for a, b in zip(x["logits"], x["int8_logits"])) for x in pairs]
            result["changes"][f"{candidate}_{arm}"] = {
                "n": len(pairs), "fp16_int8_agreement": agreement / len(pairs) if pairs else None,
                "fp16_int8_disagreements": len(pairs) - agreement,
                "mean_logit_l1": float(np.mean(l1)) if l1 else None,
                "fp16_only_correct": sum(x["fp16_correct"] and not x["int8_correct"] for x in pairs),
                "int8_only_correct": sum(x["int8_correct"] and not x["fp16_correct"] for x in pairs),
            }
    # Per-clip changes are the useful failure report, not only aggregate counts.
    result["per_clip"] = []
    for clip_id in all_ids:
        if clip_id not in c or clip_id not in d:
            continue
        a, b = c[clip_id], d[clip_id]
        if "candidates" not in a or "candidates" not in b:
            continue
        entry = {"clip_id": clip_id, "ground_truth": a["ground_truth"],
                 "clean_reliability": a["raw_reliability"], "corrupted_reliability": b["raw_reliability"],
                 "candidate_effects": {}}
        for candidate in ("baseline", "slot", "gap", "slot_gap"):
            entry["candidate_effects"][candidate] = {}
            for model in ("fp16", "int8"):
                x = a["candidates"][candidate]["models"][model]
                y = b["candidates"][candidate]["models"][model]
                entry["candidate_effects"][candidate][model] = {
                    "clean_prediction": x["prediction"], "corrupted_prediction": y["prediction"],
                    "clean_correct": x["prediction"] == a["ground_truth"],
                    "corrupted_correct": y["prediction"] == a["ground_truth"],
                    "clean_confidence": x["confidence"], "corrupted_confidence": y["confidence"],
                }
        result["per_clip"].append(entry)
    result["meta"] = {"clean_manifest": clean["manifest"], "corrupted_manifest": corrupt["manifest"],
                      "clean_success": clean["n_success"], "clean_failed": clean["failed"],
                      "corrupted_success": corrupt["n_success"], "corrupted_failed": corrupt["failed"],
                      "corruption": corrupt["corruption"], "severity": corrupt["severity"], "seed": corrupt["seed"]}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--corrupted", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare_runs(args.clean, args.corrupted)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    compact = {k: v for k, v in result.items() if k in ("clean", "corrupted", "pairs", "changes", "meta")}
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
