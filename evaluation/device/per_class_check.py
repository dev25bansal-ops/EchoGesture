"""Per-class recognition check across the whole shipped vocabulary.

Nothing here is hardcoded: the class list comes from the shipped vocabulary
file, the clip list comes from the organizer split CSV, and the label mapping
is derived by normalising both sides. Point it at a split and it will report
every class it can find clips for.

It answers one question: for each of the N classes, does the shipped pipeline
recognise a genuine clip of that class?

Usage:
    python3 evaluation/device/per_class_check.py --split /tmp/val.csv --clips /tmp/corpus
    python3 evaluation/device/per_class_check.py --split train.csv --clips DIR --min-clips 2

Clips are authorized research data and never enter the repository.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LABELS = ROOT / "models/landmark_labels.json"


def canon(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", s.upper())


def pct(n: int, d: int) -> str:
    return f"{100.0 * n / d:.1f}%" if d else "n/a"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", type=Path, required=True,
                    help="organizer split CSV with 'Video file' and 'Gloss' columns")
    ap.add_argument("--clips", type=Path, required=True,
                    help="directory containing <Video file> members")
    ap.add_argument("--min-clips", type=int, default=1,
                    help="only report classes with at least this many clips present")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args(argv)

    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    by_canon = {canon(l): l for l in labels}
    # Guard: every shipped label must be reachable, or the vocabulary and the
    # split use incompatible spellings and any table below would be misleading.
    if len(by_canon) != len(labels):
        print("vocabulary has labels that collide after normalisation", file=sys.stderr)
        return 2

    # class -> list of (clip_id, path, truth_label)
    per_class: dict[str, list[tuple[str, Path, str]]] = defaultdict(list)
    unmapped: dict[str, int] = defaultdict(int)
    missing = 0
    with args.split.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            name = row.get("Video file") or row.get("video") or ""
            gloss = row.get("Gloss") or row.get("true_gloss") or ""
            if not name or not gloss:
                continue
            label = by_canon.get(canon(gloss))
            if label is None:
                unmapped[gloss] += 1
                continue
            path = args.clips / name
            if not path.is_file():
                missing += 1
                continue
            per_class[label].append((Path(name).stem, path, label))

    if not per_class:
        print(f"no clips found under {args.clips} for {args.split}", file=sys.stderr)
        return 2

    sys.path.insert(0, str(ROOT / "src"))
    import main as model  # noqa: E402

    model._init_model()

    rows = []
    for label in sorted(per_class):
        clips = per_class[label]
        if len(clips) < args.min_clips:
            continue
        correct = 0
        scores = []
        misses = []
        for cid, path, truth in clips:
            try:
                r = model.predict(str(path), clip_id=cid)
            except ValueError:
                misses.append((cid, "NO_PREDICTION", 0.0))
                continue
            scores.append(r["confidence"])
            if r["prediction"] == truth:
                correct += 1
            else:
                misses.append((cid, r["prediction"], r["confidence"]))
        rows.append({
            "label": label,
            "n": len(clips),
            "correct": correct,
            "top1": correct / len(clips),
            "mean_score": statistics.mean(scores) if scores else 0.0,
            "misses": misses,
        })

    rows.sort(key=lambda r: (r["top1"], r["label"]))
    total_n = sum(r["n"] for r in rows)
    total_c = sum(r["correct"] for r in rows)

    print(f"vocabulary: {len(labels)} classes")
    print(f"classes with clips present: {len(per_class)}"
          f"  (reported: {len(rows)} with >= {args.min_clips} clip(s))")
    print(f"clips scored: {total_n}   correct: {total_c}   top-1: {pct(total_c, total_n)}")
    print(f"clips in split without a local file: {missing}")
    if unmapped:
        print(f"glosses outside the vocabulary: {len(unmapped)} distinct, "
              f"{sum(unmapped.values())} rows")
    print()

    perfect = [r for r in rows if r["correct"] == r["n"]]
    zero = [r for r in rows if r["correct"] == 0]
    print(f"classes at 100%: {len(perfect)}   at 0%: {len(zero)}   "
          f"in between: {len(rows) - len(perfect) - len(zero)}")
    print()

    print(f"{'class':<22}{'n':>3}{'ok':>4}{'top1':>8}{'score':>8}")
    for r in rows:
        flag = "" if r["top1"] == 1.0 else ("   <-- ZERO" if r["top1"] == 0 else "   <-- partial")
        print(f"{r['label']:<22}{r['n']:>3}{r['correct']:>4}"
              f"{100*r['top1']:>7.0f}%{r['mean_score']:>8.3f}{flag}")
        for cid, pred, sc in r["misses"][:3]:
            print(f"    miss: {cid[:38]:<40} -> {pred:<16} {sc:.3f}")

    if args.json_out:
        args.json_out.write_text(
            json.dumps([{k: v for k, v in r.items() if k != "misses"} for r in rows], indent=1),
            encoding="utf-8",
        )
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())