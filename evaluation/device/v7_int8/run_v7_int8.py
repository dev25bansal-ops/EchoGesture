#!/usr/bin/env python3
"""Reproduce the pinned FP16/INT8 paired audit on cached real clips.

This module deliberately contains no model, detector, feature, training, or
export code.  It validates the two pinned classifier artifacts and all frozen
validation inputs, decodes/tracks each clip once, saves the resulting hashed
feature tensor, and supplies the identical tensor to both interpreters.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import statistics
import sys
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE_ROOT = Path("/tmp/echogesture-v7-agents/A06")
SOURCE_MANIFEST = SOURCE_ROOT / "evaluation/device/v5_accuracy/data/val_346/clips/fetched_manifest.csv"
SOURCE_PREDICTIONS = SOURCE_ROOT / "evaluation/device/v5_accuracy/results/val_346/predictions.json"
SOURCE_SUMMARY = SOURCE_ROOT / "evaluation/device/v5_accuracy/results/val_346/summary.json"
sys.path.insert(0, str(ROOT / "src"))

EXPECTED = {
    "fp16": (ROOT / "models/landmark_classifier_fp16.tflite", "a012f2b5f7a1110961079f19eb0e28c9abacb95e4d0787352e4157ebca7044a4", 832_888),
    "int8": (ROOT / "models/landmark_classifier_int8.tflite", "9d24640a23244a3cd00c9fa62257733819f86ed1da5a10194c9b1cb64323700b", 509_720),
    "labels": (ROOT / "models/landmark_labels.json", "e78f5028e97e2eaf86aff65af403af65061eebef010f0fbd41012293e8019a4a", 1_492),
    "detector": (ROOT / "models/hand_landmarker.task", "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1", 7_819_105),
    "features": (ROOT / "src/features.py", "baf28d05bbd36e2783ea7ef6f78e793b89ed832d0eaad5dffed23d35c775aa84", 8_312),
    "frontend": (ROOT / "src/landmark_frontend.py", "c73ef64b72f21a86638c4a126ad642b64c4ee29851cd2f8fd8ea4ed516fa97b0", 6_633),
    "manifest": (SOURCE_MANIFEST, "3f06798af00e69314da87414bf0c39bb2ec49e578f560c111754fa00966f6c82", 31_020),
    "source_predictions": (SOURCE_PREDICTIONS, None, None),
    "source_summary": (SOURCE_SUMMARY, None, None),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def verify_artifacts() -> dict[str, Any]:
    result = {}
    for name, (path, expected_sha, expected_bytes) in EXPECTED.items():
        if not path.is_file():
            raise ValueError(f"missing {name}: {path}")
        item = artifact(path)
        if expected_sha is not None and item["sha256"] != expected_sha:
            raise ValueError(f"{name} SHA mismatch: {item['sha256']}")
        if expected_bytes is not None and item["bytes"] != expected_bytes:
            raise ValueError(f"{name} bytes mismatch: {item['bytes']}")
        result[name] = item
    return result


def load_labels(path: Path) -> list[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    labels = [""] * len(value)
    for label, index in value.items():
        labels[int(index)] = str(label).upper()
    if sorted(value.values()) != list(range(len(value))) or any(not x for x in labels) or len(set(labels)) != len(labels):
        raise ValueError("labels are not a contiguous, unique vocabulary")
    return labels


def read_manifest() -> list[dict[str, Any]]:
    with SOURCE_MANIFEST.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 346 or len({row["clip_id"] for row in rows}) != 346:
        raise ValueError("expected 346 unique clips in the retained validation manifest")
    for row in rows:
        clip = SOURCE_MANIFEST.parent / row["path"]
        if not clip.is_file():
            raise ValueError(f"clip not found: {clip}")
        row["_clip_path"] = str(clip)
        row["ground_truth"] = row["ground_truth"].upper()
    return rows


def softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    shifted = z - np.max(z)
    exp = np.exp(shifted)
    return (exp / np.sum(exp)).astype(np.float64)


def run_and_parse(interpreter: Any, tensor: np.ndarray, labels: list[str]) -> dict[str, Any]:
    in_detail = interpreter.get_input_details()[0]
    out_detail = interpreter.get_output_details()[0]
    started = time.perf_counter_ns()
    interpreter.set_tensor(in_detail["index"], tensor)
    interpreter.invoke()
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
    logits = np.asarray(interpreter.get_tensor(out_detail["index"])[0], dtype=np.float32)
    if logits.shape != (100,) or not np.isfinite(logits).all():
        raise ValueError("classifier returned an invalid [1,100] float32 logit vector")
    probs = softmax(logits)
    order = np.argsort(-probs)
    top = order[:10]
    return {
        "prediction": labels[int(order[0])],
        "top1": [labels[int(top[0])]],
        "top5": [labels[int(index)] for index in top[:5]],
        "top10": [labels[int(index)] for index in top],
        "top5_probs": [float(probs[index]) for index in top[:5]],
        "top1_prob": float(probs[top[0]]),
        "top2_prob": float(probs[top[1]]),
        "margin_top1_top2": float(probs[top[0]] - probs[top[1]]),
        "logits": [float(x) for x in logits],
        "logits_sha256": hashlib.sha256(logits.astype("<f4").tobytes()).hexdigest(),
        "invoke_ms_including_copy_and_read": elapsed_ms,
    }


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * pct
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return float(ordered[lower] * (1 - fraction) + ordered[upper] * fraction)


def describe(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p10": percentile(values, 0.1),
        "p90": percentile(values, 0.9),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def wilson(correct: int, total: int) -> list[float | None]:
    if total == 0:
        return [None, None]
    z = 1.96
    p = correct / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    margin = z * (p * (1 - p) / total + z * z / (4 * total * total)) ** 0.5 / denom
    return [centre - margin, centre + margin]


def category(row: dict[str, Any]) -> str:
    f = row["fp16_correct"]
    i = row["int8_correct"]
    if f and i:
        return "both_correct"
    if f:
        return "fp16_only_correct"
    if i:
        return "int8_only_correct"
    return "both_wrong"


def group_analysis(rows: list[dict[str, Any]], key: str | Callable[[dict[str, Any]], Any]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(key(row) if callable(key) else row[key])].append(row)
    output = {}
    for name, subset in sorted(groups.items()):
        n = len(subset)
        fc = sum(row["fp16_correct"] for row in subset)
        ic = sum(row["int8_correct"] for row in subset)
        top1_agree = sum(row["fp16"]["prediction"] == row["int8"]["prediction"] for row in subset)
        output[name] = {
            "n": n,
            "fp16_correct": fc,
            "int8_correct": ic,
            "fp16_accuracy": fc / n,
            "int8_accuracy": ic / n,
            "fp16_minus_int8_correct_count": fc - ic,
            "fp16_minus_int8_percentage_points": (fc - ic) / n * 100,
            "top1_agreement": top1_agree / n,
            "both_correct": sum(row["paired_category"] == "both_correct" for row in subset),
            "fp16_only_correct": sum(row["paired_category"] == "fp16_only_correct" for row in subset),
            "int8_only_correct": sum(row["paired_category"] == "int8_only_correct" for row in subset),
            "both_wrong": sum(row["paired_category"] == "both_wrong" for row in subset),
            "fp16_top1_prob": describe([row["fp16"]["top1_prob"] for row in subset]),
            "int8_top1_prob": describe([row["int8"]["top1_prob"] for row in subset]),
            "fp16_margin_top1_top2": describe([row["fp16"]["margin_top1_top2"] for row in subset]),
            "int8_margin_top1_top2": describe([row["int8"]["margin_top1_top2"] for row in subset]),
            "logit_l2_diff": describe([row["logit_l2_diff"] for row in subset]),
            "prob_l1_diff": describe([row["prob_l1_diff"] for row in subset]),
            "hand_detection_rate": describe([row["hand_detection_rate"] for row in subset]),
            "frames_with_hands": describe([float(row["frames_with_hands"]) for row in subset]),
        }
    return output


def hand_bands(rows: list[dict[str, Any]]) -> dict[str, Any]:
    bands = {
        "no_frames_either_hand": lambda row: row["frames_with_hands"] == 0,
        "1_to_11_frames": lambda row: 1 <= row["frames_with_hands"] <= 11,
        "12_to_18_frames": lambda row: 12 <= row["frames_with_hands"] <= 18,
        "19_to_24_frames": lambda row: 19 <= row["frames_with_hands"] <= 24,
        "25_to_32_frames": lambda row: 25 <= row["frames_with_hands"] <= 32,
        "rate_le_0.50": lambda row: row["hand_detection_rate"] <= 0.5,
        "rate_0.50_to_0.75": lambda row: 0.5 < row["hand_detection_rate"] <= 0.75,
        "rate_gt_0.75": lambda row: row["hand_detection_rate"] > 0.75,
    }
    return group_analysis(rows, lambda row: next(name for name, test in bands.items() if test(row)))


def rank_of(values: Any, truth: str, labels: list[str]) -> int | None:
    array = np.asarray(values)
    if array.ndim != 1 or len(array) != len(labels) or not np.isfinite(array).all():
        raise ValueError("rank input must be a finite vector aligned with labels")
    order = np.argsort(-array, kind="stable").tolist()
    return order.index(labels.index(truth)) + 1


def write_atomic_json(path: Path, value: Any, rename_pairs: list[tuple[Path, Path]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        rename_pairs.append((Path(temp), path))
    except Exception:
        Path(temp).unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument("--frontend-cache", type=Path, default=HERE / "results/frontend_cache.json")
    parser.add_argument("--reuse-frontend-cache", action="store_true", help="currently unavailable: the cache contains hashes but not presence vectors")
    parser.add_argument("--skip-benchmark", action="store_true", help="run paired analysis only; use benchmark.py independently for timing")
    args = parser.parse_args()

    artifacts = verify_artifacts()
    rows = read_manifest()
    labels = load_labels(ROOT / "models/landmark_labels.json")
    if args.reuse_frontend_cache:
        raise ValueError("frontend cache cannot be reused without the exact presence vectors")
    from ai_edge_litert.interpreter import Interpreter
    from landmark_frontend import detect_hands, read_frames
    from features import build_features

    fp = Interpreter(model_path=str(EXPECTED["fp16"][0]))
    i8 = Interpreter(model_path=str(EXPECTED["int8"][0]))
    for model in (fp, i8):
        model.allocate_tensors()
        detail_in = model.get_input_details()[0]
        detail_out = model.get_output_details()[0]
        if list(detail_in["shape"]) != [1, 24, 255] or detail_in["dtype"] != np.float32:
            raise ValueError("model input contract changed")
        if list(detail_out["shape"]) != [1, 100] or detail_out["dtype"] != np.float32:
            raise ValueError("model output contract changed")

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    frontend_cache: list[dict[str, Any]] = []
    feature_hashes: list[bytes] = []

    for index, row in enumerate(rows, start=1):
        clip_identity = artifact(Path(row["_clip_path"]))
        rec: dict[str, Any] = {
            "clip_id": row["clip_id"],
            "clip_sha256": clip_identity["sha256"],
            "clip_bytes": clip_identity["bytes"],
            "path": row["path"],
            "ground_truth": row["ground_truth"],
            "signer_id": row["signer_id"],
            "split": row.get("split", ""),
            "source_status": row.get("status", ""),
            "detector_mode": "image",
            "feature_shape": [1, 24, 255],
            "feature_sha256": None,
            "landmarks_shape": None,
            "present_shape": None,
            "decoded_frames": 0,
            "frames_with_hands": 0,
            "hand_detection_rate": 0.0,
            "frame_presence": None,
            "frame_presence_left": None,
            "frame_presence_right": None,
            "frontend_ms": None,
            "feature_ms": None,
            "status": "ok",
            "error": None,
        }
        started = time.perf_counter_ns()
        try:
            frames = read_frames(row["_clip_path"], max_frames=32)
            rec["decoded_frames"] = len(frames)
            if not frames:
                raise ValueError("No valid frames decoded")
            landmarks, present = detect_hands(frames, mode="image")
            rec["landmarks_shape"] = [int(x) for x in landmarks.shape]
            rec["present_shape"] = [int(x) for x in present.shape]
            rec["frames_with_hands"] = int((present.max(axis=1) > 0.5).sum())
            rec["hand_detection_rate"] = float(rec["frames_with_hands"] / 32)
            rec["frame_presence"] = [float(x) for x in present.max(axis=1)]
            rec["frame_presence_left"] = [float(x) for x in present[:, 0]]
            rec["frame_presence_right"] = [float(x) for x in present[:, 1]]
            rec["frontend_ms"] = (time.perf_counter_ns() - started) / 1_000_000
            if landmarks.shape[0] == 0 or float(present.max()) < 0.5:
                raise ValueError("No hand detected in any frame")
            fstart = time.perf_counter_ns()
            features, mask = build_features(landmarks, present, expected_frames=24)
            rec["feature_ms"] = (time.perf_counter_ns() - fstart) / 1_000_000
            tensor = np.asarray(features[None, ...], dtype=np.float32)
            if tuple(tensor.shape) != (1, 24, 255):
                raise ValueError(f"unexpected feature shape {tensor.shape}")
            feature_bytes = tensor.astype("<f4").tobytes()
            feature_hashes.append(feature_bytes)
            rec["feature_sha256"] = hashlib.sha256(feature_bytes).hexdigest()
            rec["processed_frames"] = int(mask.sum())
            # One tensor feeds both models; no feature computation is repeated.
            fp_result = run_and_parse(fp, tensor, labels)
            i8_result = run_and_parse(i8, tensor, labels)
            fp_logits = np.asarray(fp_result["logits"], dtype=np.float64)
            i8_logits = np.asarray(i8_result["logits"], dtype=np.float64)
            fp_probs = softmax(fp_logits)
            i8_probs = softmax(i8_logits)
            truth_index = labels.index(row["ground_truth"])
            rec.update({
                "fp16": fp_result,
                "int8": i8_result,
                "fp16_correct": fp_result["prediction"] == row["ground_truth"],
                "int8_correct": i8_result["prediction"] == row["ground_truth"],
                "fp16_top5_correct": row["ground_truth"] in fp_result["top5"],
                "int8_top5_correct": row["ground_truth"] in i8_result["top5"],
                "fp16_rank_truth": rank_of(fp_logits, row["ground_truth"], labels),
                "int8_rank_truth": rank_of(i8_logits, row["ground_truth"], labels),
# Keep top-10 rank columns only for side-by-side debugging.  Do not call
# rank_of with the label list here: the full-logit ranks above are computed
# with aligned numeric logits and label indices.

                "fp16_true_class_prob": float(fp_probs[truth_index]),
                "int8_true_class_prob": float(i8_probs[truth_index]),
                "top1_agreement": fp_result["prediction"] == i8_result["prediction"],
                "top5_exact_agreement": fp_result["top5"] == i8_result["top5"],
                "top5_set_agreement": set(fp_result["top5"]) == set(i8_result["top5"]),
                "logit_mean_abs_diff": float(np.mean(np.abs(fp_logits - i8_logits))),
                "logit_max_abs_diff": float(np.max(np.abs(fp_logits - i8_logits))),
                "logit_l1_diff": float(np.sum(np.abs(fp_logits - i8_logits))),
                "logit_l2_diff": float(np.sqrt(np.sum((fp_logits - i8_logits) ** 2))),
                "logit_correlation": float(np.corrcoef(fp_logits, i8_logits)[0, 1]),
                "prob_l1_diff": float(np.sum(np.abs(fp_probs - i8_probs))),
                "prob_l2_diff": float(np.sqrt(np.sum((fp_probs - i8_probs) ** 2))),
                "probability_vector_sha256": {
                    "fp16": hashlib.sha256(fp_probs.astype("<f8").tobytes()).hexdigest(),
                    "int8": hashlib.sha256(i8_probs.astype("<f8").tobytes()).hexdigest(),
                },
            })
            rec["paired_category"] = category(rec)
            feature_rows.append({
                "clip_id": row["clip_id"],
                "feature_sha256": rec["feature_sha256"],
                "feature_shape": rec["feature_shape"],
                "dtype": "float32",
                "frames_with_hands": rec["frames_with_hands"],
                "processed_frames": rec["processed_frames"],
            })
        except Exception as exc:
            rec.update({
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "frontend_ms": (time.perf_counter_ns() - started) / 1_000_000,
                "fp16_correct": False,
                "int8_correct": False,
                "fp16_top5_correct": False,
                "int8_top5_correct": False,
                "paired_category": "frontend_failed",
            })
        records.append(rec)
        frontend_cache.append({
            "clip_id": rec["clip_id"],
            "clip_sha256": rec["clip_sha256"],
            "clip_bytes": rec["clip_bytes"],
            "status": rec["status"],
            "decoded_frames": rec["decoded_frames"],
            "landmarks_shape": rec["landmarks_shape"],
            "present_shape": rec["present_shape"],
            "frames_with_hands": rec["frames_with_hands"],
            "hand_detection_rate": rec["hand_detection_rate"],
            "frame_presence_has_frames": rec["frame_presence"] is not None,
            "frame_presence_sha256": hashlib.sha256(json.dumps(rec["frame_presence"], separators=(",", ":")).encode()).hexdigest() if rec["frame_presence"] is not None else None,
            "frame_presence_left_sha256": hashlib.sha256(json.dumps(rec["frame_presence_left"], separators=(",", ":")).encode()).hexdigest() if rec["frame_presence_left"] is not None else None,
            "frame_presence_right_sha256": hashlib.sha256(json.dumps(rec["frame_presence_right"], separators=(",", ":")).encode()).hexdigest() if rec["frame_presence_right"] is not None else None,
            "feature_sha256": rec["feature_sha256"],
            "feature_shape": rec["feature_shape"],
            "processed_frames": rec.get("processed_frames"),
        })
        print(f"[{index}/346] {rec['clip_id']} {rec['status']}", flush=True)

    ok = [row for row in records if row["status"] == "ok"]
    n = len(records)
    fc = sum(row["fp16_correct"] for row in records)
    ic = sum(row["int8_correct"] for row in records)
    fc5 = sum(row["fp16_top5_correct"] for row in records)
    ic5 = sum(row["int8_top5_correct"] for row in records)
    cats = Counter(row["paired_category"] for row in records)

    def direction(row: dict[str, Any]) -> str:
        if row["paired_category"] == "fp16_only_correct":
            return "fp16_correct_int8_wrong"
        if row["paired_category"] == "int8_only_correct":
            return "int8_correct_fp16_wrong"
        if row["paired_category"] == "both_correct":
            return "both_correct_same_or_different_prediction"
        if row["paired_category"] == "both_wrong":
            return "both_wrong_same_or_different_prediction"
        return "frontend_failed"

    disagreements = []
    for row in ok:
        if row["top1_agreement"]:
            continue
        disagreements.append({
            "clip_id": row["clip_id"],
            "signer_id": row["signer_id"],
            "ground_truth": row["ground_truth"],
            "direction": direction(row),
            "fp16_prediction": row["fp16"]["prediction"],
            "int8_prediction": row["int8"]["prediction"],
            "fp16_top5": row["fp16"]["top5"],
            "int8_top5": row["int8"]["top5"],
            "fp16_top1_prob": row["fp16"]["top1_prob"],
            "int8_top1_prob": row["int8"]["top1_prob"],
            "fp16_margin_top1_top2": row["fp16"]["margin_top1_top2"],
            "int8_margin_top1_top2": row["int8"]["margin_top1_top2"],
            "fp16_true_class_prob": row["fp16_true_class_prob"],
            "int8_true_class_prob": row["int8_true_class_prob"],
            "fp16_rank_truth": row["fp16_rank_truth"],
            "int8_rank_truth": row["int8_rank_truth"],
            "frames_with_hands": row["frames_with_hands"],
            "hand_detection_rate": row["hand_detection_rate"],
            "logit_l2_diff": row["logit_l2_diff"],
            "logit_max_abs_diff": row["logit_max_abs_diff"],
            "prob_l1_diff": row["prob_l1_diff"],
        })

    primary = {
        "status": "completed",
        "scope": "paired FP16 vs INT8 classifier audit on identical real features; official ASL Citizen val split only",
        "paired_input_method": "each clip decoded and landmark-tracked once; the identical hashed 24x255 feature tensor was supplied to both pinned classifiers",
        "split_policy": "validation only. Official test clips were not opened or evaluated.",
        "artifacts": artifacts,
        "fixture": {
            "manifest_sha256": artifacts["manifest"]["sha256"],
            "n_clips": n,
            "n_successful_feature_extractions": len(ok),
            "n_frontend_failures": n - len(ok),
            "all_successful_feature_sha256_in_manifest_order": hashlib.sha256(b"".join(feature_hashes)).hexdigest(),
            "feature_hashes": feature_rows,
        },
        "counts": {
            "fp16_top1_correct": fc,
            "fp16_top1_accuracy": fc / n,
            "fp16_top1_wilson95": wilson(fc, n),
            "int8_top1_correct": ic,
            "int8_top1_accuracy": ic / n,
            "int8_top1_wilson95": wilson(ic, n),
            "fp16_top5_correct": fc5,
            "fp16_top5_accuracy": fc5 / n,
            "int8_top5_correct": ic5,
            "int8_top5_accuracy": ic5 / n,
            "fp16_minus_int8_correct_count": fc - ic,
            "fp16_minus_int8_percentage_points": (fc - ic) / n * 100,
            "n_ok": len(ok),
            "top1_agreement": sum(row["top1_agreement"] for row in ok) / len(ok) if ok else None,
            "top5_exact_agreement": sum(row["top5_exact_agreement"] for row in ok) / len(ok) if ok else None,
            "top5_set_agreement": sum(row["top5_set_agreement"] for row in ok) / len(ok) if ok else None,
            "fp16_classifier_ms": describe([row["fp16"]["invoke_ms_including_copy_and_read"] for row in ok]),
            "int8_classifier_ms": describe([row["int8"]["invoke_ms_including_copy_and_read"] for row in ok]),
            "frontend_ms": describe([row["frontend_ms"] for row in ok]),
            "feature_ms": describe([row["feature_ms"] for row in ok]),
            "paired_categories": dict(cats),
            "direction_counts": dict(Counter(item["direction"] for item in disagreements)),
        },
        "category_diagnostics": {
            name: {
                "n": len([row for row in records if row["paired_category"] == name]),
                "fp16_top1_prob": describe([row["fp16"]["top1_prob"] for row in records if row["paired_category"] == name and row["status"] == "ok"]),
                "int8_top1_prob": describe([row["int8"]["top1_prob"] for row in records if row["paired_category"] == name and row["status"] == "ok"]),
                "fp16_margin": describe([row["fp16"]["margin_top1_top2"] for row in records if row["paired_category"] == name and row["status"] == "ok"]),
                "int8_margin": describe([row["int8"]["margin_top1_top2"] for row in records if row["paired_category"] == name and row["status"] == "ok"]),
                "fp16_true_class_prob": describe([row["fp16_true_class_prob"] for row in records if row["paired_category"] == name and row["status"] == "ok"]),
                "int8_true_class_prob": describe([row["int8_true_class_prob"] for row in records if row["paired_category"] == name and row["status"] == "ok"]),
                "logit_l2_diff": describe([row["logit_l2_diff"] for row in records if row["paired_category"] == name and row["status"] == "ok"]),
                "prob_l1_diff": describe([row["prob_l1_diff"] for row in records if row["paired_category"] == name and row["status"] == "ok"]),
                "hand_detection_rate": describe([row["hand_detection_rate"] for row in records if row["paired_category"] == name and row["status"] == "ok"]),
            }
            for name in sorted(cats)
        },
        "signer": group_analysis(ok, "signer_id"),
        "gloss": group_analysis(ok, "ground_truth"),
        "hand_detection_bands": hand_bands(ok),
        "disagreements": disagreements,
        "frontend_failure": [row for row in records if row["status"] != "ok"],
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }

    source = json.loads(SOURCE_PREDICTIONS.read_text(encoding="utf-8"))
    source_by_id = {row["clip_id"]: row for row in source}
    mismatches = []
    for row in records:
        old = source_by_id[row["clip_id"]]
        if old["status"] != row["status"]:
            mismatches.append({"clip_id": row["clip_id"], "field": "status", "old": old["status"], "new": row["status"]})
            continue
        if row["status"] == "ok":
            checks = {
                "fp16_prediction": (old["fp16_prediction"], row["fp16"]["prediction"]),
                "fp16_top5": (old["fp16_top5"], row["fp16"]["top5"]),
                "int8_prediction": (old["int8_prediction"], row["int8"]["prediction"]),
                "int8_top5": (old["int8_top5"], row["int8"]["top5"]),
                "frames_with_hands": (old["frames_with_hands"], row["frames_with_hands"]),
            }
            for field, (expected, actual) in checks.items():
                if expected != actual:
                    mismatches.append({"clip_id": row["clip_id"], "field": field, "old": expected, "new": actual})
    primary["cross_check"] = {
        "checked_clips": len(records),
        "mismatches": mismatches,
        "all_predictions_and_top5_match_source": not mismatches,
        "note": "probabilities are not reproduced because the source retained only top-1 probabilities and drift summaries, not all 100 logits",
    }

    renames: list[tuple[Path, Path]] = []
    write_atomic_json(args.frontend_cache, {
        "status": "completed",
        "identity": {
            "manifest_sha256": artifacts["manifest"]["sha256"],
            "clips_in_manifest_order": frontend_cache,
        },
    }, renames)
    write_atomic_json(out_dir / "predictions.json", records, renames)
    write_atomic_json(out_dir / "summary.json", primary, renames)
    # Benchmarking is intentionally separate (benchmark.py).  The completed
    # paired audit already measures real classifier invocation times below.
    # Benchmarking is intentionally separate (benchmark.py).  The completed
    # paired audit already measures real classifier invocation times below.
    try:
        for source, destination in renames:
            os.replace(source, destination)
    finally:
        for source, _ in renames:
            source.unlink(missing_ok=True)
    print(json.dumps(primary["counts"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
