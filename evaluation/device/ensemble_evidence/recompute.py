"""Recompute held-out test accuracy from frozen predictions, independently.

Joins a predictions file to the official signer-disjoint ground truth in this
checkout, restricted to the shipped 100-class vocabulary, and reports top-1,
top-5 and per-signer accuracy under BOTH denominators:

  * scored   -- denominator is the clips that produced a prediction (Dev's
                headline uses this one; 1249 of 1255 eligible).
  * eligible -- denominator is every eligible clip in the split, with a clip
                that was never scored counted as incorrect. This is the
                stricter, submission-safe reading.

Nothing here runs a model or touches the test split for tuning. The
predictions and the split CSV are frozen inputs.

Provenance check this script deliberately makes explicit:
  the frozen ensemble file has 1249 lines and is *not* the older single-model
  file (which also has 1249 lines but scores 62.61%). See RESULT.md.

Usage:
    .venv/bin/python -I evaluation/device/ensemble_evidence/recompute.py \
        --predictions evaluation/device/ensemble_evidence/predictions_ensemble.jsonl \
        --split-csv evaluation/device/v5_accuracy/splits/test.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SPLIT = ROOT / "evaluation/device/v5_accuracy/splits/test.csv"
DEFAULT_LABELS = ROOT / "models/landmark_labels.json"


def load_vocab(path: Path) -> set[str]:
    return {label.upper() for label in json.loads(path.read_text(encoding="utf-8"))}


def load_truth(split_csv: Path, vocab: set[str]) -> dict[str, tuple[str, str]]:
    """clip_id (video stem) -> (gloss, participant) for vocabulary-eligible rows."""
    if not split_csv.is_file():
        raise FileNotFoundError(
            f"Ground-truth split not found: {split_csv}\n"
            "The official ASL Citizen split CSVs are not committed (they carry "
            "participant identifiers). Supply the split locally to recompute "
            "accuracy; the ensemble's published figures are already recorded in "
            "RESULT.md.")
    truth: dict[str, tuple[str, str]] = {}
    with split_csv.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            gloss = (row.get("Gloss") or "").strip().upper()
            if gloss not in vocab:
                continue
            video = row["Video file"]
            stem = video[:-4] if video.endswith(".mp4") else video
            truth[stem] = (gloss, row["Participant ID"])
    return truth


def _pct(num: int, den: int) -> float:
    return round(100.0 * num / den, 2) if den else 0.0


def score(predictions: Path, split_csv: Path, labels_json: Path) -> dict:
    vocab = load_vocab(labels_json)
    truth = load_truth(split_csv, vocab)
    rows = [
        json.loads(line)
        for line in predictions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    correct = top5 = 0
    matched = 0
    unmatched: list[str] = []
    scored_ids: set[str] = set()
    per_signer: dict[str, list[int]] = {}

    for row in rows:
        candidates = [row["clip_id"]]
        if not row["clip_id"].endswith(".mp4"):
            candidates.append(f"{row['clip_id']}.mp4")
        key = next((c for c in candidates if c in truth), None)
        if key is None:
            unmatched.append(row["clip_id"])
            continue
        matched += 1
        scored_ids.add(key)
        gloss, signer = truth[key]
        hit = int(row["prediction"] == gloss)
        correct += hit
        top5 += int(gloss in row.get("top5", []))
        slot = per_signer.setdefault(signer, [0, 0])
        slot[0] += hit
        slot[1] += 1

    if matched == 0:
        raise ValueError("No predictions matched a ground-truth row")

    eligible = len(truth)
    # Eligible clips that never produced a prediction (no hand detected, etc.).
    # These are the only clips that can make the two denominators differ.
    unscored_eligible = sorted(set(truth) - scored_ids)

    rates = [c / t for c, t in per_signer.values()]
    return {
        "predictions": len(rows),
        "eligible_clips": eligible,
        "scored_clips": matched,
        "unmatched_predictions": len(unmatched),
        "unscored_eligible_clips": len(unscored_eligible),
        "unscored_eligible_ids": unscored_eligible,
        "top1_correct": correct,
        "top1_incorrect_scored": matched - correct,
        "top5_correct": top5,
        "top1_pct_scored": _pct(correct, matched),
        "top5_pct_scored": _pct(top5, matched),
        "top1_pct_all_eligible": _pct(correct, eligible),
        "top5_pct_all_eligible": _pct(top5, eligible),
        "signers": len(per_signer),
        "per_signer_mean_pct": round(100.0 * sum(rates) / len(rates), 2),
        "per_signer_min_pct": round(100.0 * min(rates), 2),
        "per_signer_max_pct": round(100.0 * max(rates), 2),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--split-csv", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    args = parser.parse_args(argv)
    try:
        result = score(args.predictions, args.split_csv, args.labels)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())