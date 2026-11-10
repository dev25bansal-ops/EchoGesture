"""Recompute the held-out test-split accuracy from raw predictions.

Joins Dev's test-split predictions against the official signer-disjoint
ground truth in this checkout, restricted to the shipped 100-class
vocabulary, and reports top-1, top-5 and per-signer accuracy. This is a
recomputation from frozen artifacts -- it never runs the model and never
touches the test split for tuning.

Provenance: predictions come from
``origin/hackathon/dev-model:results/predictions.jsonl`` (exported to a
local file by the caller); truth comes from
``evaluation/device/v5_accuracy/splits/test.csv``. Both are the official
ASL Citizen participant-disjoint splits audited in
``evaluation/device/v5_accuracy/results/split_audit.json``.

Usage:
    python3 -I evaluation/device/test_split_evidence.py --predictions <preds.jsonl>
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_vocabulary(path: Path) -> set[str]:
    return {label.upper() for label in json.loads(path.read_text(encoding="utf-8"))}


def load_truth(split_csv: Path, vocab: set[str]) -> dict[str, tuple[str, str]]:
    truth: dict[str, tuple[str, str]] = {}
    with split_csv.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            gloss = (row.get("Gloss") or "").strip().upper()
            if gloss in vocab:
                truth[row["Video file"]] = (gloss, row["Participant ID"])
    return truth


def resolve_key(clip_id: str, truth: dict) -> str | None:
    if clip_id in truth:
        return clip_id
    with_ext = f"{clip_id}.mp4"
    return with_ext if with_ext in truth else None


def score(predictions: Path, split_csv: Path, labels_json: Path) -> dict:
    vocab = load_vocabulary(labels_json)
    truth = load_truth(split_csv, vocab)
    rows = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines() if line.strip()]

    correct = top5 = matched = 0
    unmatched = []
    per_signer: dict[str, list[int]] = {}
    for row in rows:
        key = resolve_key(row["clip_id"], truth)
        if key is None:
            unmatched.append(row["clip_id"])
            continue
        matched += 1
        gloss, signer = truth[key]
        hit = int(row["prediction"] == gloss)
        correct += hit
        top5 += int(gloss in row.get("top5", []))
        slot = per_signer.setdefault(signer, [0, 0])
        slot[0] += hit
        slot[1] += 1

    if matched == 0:
        raise ValueError("No predictions matched a ground-truth row")

    signer_rates = {s: c / t for s, (c, t) in per_signer.items()}
    return {
        "predictions": len(rows),
        "matched": matched,
        "unmatched": len(unmatched),
        "vocab_eligible_truth_rows": len(truth),
        "top1_correct": correct,
        "top1_pct": round(100 * correct / matched, 2),
        "top5_correct": top5,
        "top5_pct": round(100 * top5 / matched, 2),
        "signers": len(per_signer),
        "per_signer_mean_pct": round(100 * sum(signer_rates.values()) / len(signer_rates), 2),
        "per_signer_min_pct": round(100 * min(signer_rates.values()), 2),
        "per_signer_max_pct": round(100 * max(signer_rates.values()), 2),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--split-csv", type=Path, default=ROOT / "evaluation/device/v5_accuracy/splits/test.csv")
    parser.add_argument("--labels", type=Path, default=ROOT / "models/landmark_labels.json")
    args = parser.parse_args(argv)
    result = score(args.predictions, args.split_csv, args.labels)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())