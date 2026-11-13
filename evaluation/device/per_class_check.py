"""Full per-class validation across the whole shipped vocabulary.

Every in-vocabulary clip in a split is evaluated — no per-class cap, and no
artificial filling when a class has fewer examples than others. Counts are
reported as they actually are.

Nothing is hardcoded: classes come from the shipped vocabulary file, clips from
an organizer split CSV passed on the command line, and the gloss->label mapping
is derived by normalising both sides.

Reports both aggregations, because they answer different questions:
  micro — every clip equally weighted; "how often is the system right?"
  macro — every class equally weighted; "is a typical sign recognised?"

Usage:
    python3 evaluation/device/per_class_check.py --split val.csv --clips DIR
    python3 evaluation/device/per_class_check.py --split val.csv --clips DIR --json-out out.json
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LABELS = ROOT / "models/landmark_labels.json"

# Acceptance rule the shipped app applies (top-1 minus top-2, with a score floor).
MARGIN_CUT = 0.15
SCORE_FLOOR = 0.20


def canon(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", s.upper())


def pct(n: float, d: float) -> str:
    return f"{100.0 * n / d:.1f}%" if d else "n/a"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% interval; honest about small n instead of a bare percentage."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--clips", type=Path, required=True)
    ap.add_argument("--min-clips", type=int, default=1)
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--max-clips-per-class", type=int, default=0,
                    help="0 (default) evaluates every available clip")
    args = ap.parse_args(argv)

    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    by_canon = {canon(l): l for l in labels}
    if len(by_canon) != len(labels):
        print("vocabulary has labels that collide after normalisation", file=sys.stderr)
        return 2

    per_class: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    unmapped: Counter[str] = Counter()
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
            per_class[label].append((Path(name).stem, path))

    if not per_class:
        print(f"no clips found under {args.clips}", file=sys.stderr)
        return 2

    if args.max_clips_per_class:
        for k in per_class:
            per_class[k] = per_class[k][: args.max_clips_per_class]

    sys.path.insert(0, str(ROOT / "src"))
    import main as model  # noqa: E402

    model._init_model()

    rows = []
    for label in sorted(per_class):
        clips = per_class[label]
        if len(clips) < args.min_clips:
            continue
        ok = top5_ok = no_hand = 0
        accepted = accepted_ok = 0
        scores: list[float] = []
        margins: list[float] = []
        confusions: Counter[str] = Counter()
        misses = []
        for cid, path in clips:
            try:
                r = model.predict(str(path), clip_id=cid)
            except ValueError:
                no_hand += 1
                misses.append({"clip": cid, "pred": None, "score": None,
                               "margin": None, "why": "no hand detected"})
                continue
            scores.append(r["confidence"])
            top = r["top5"]
            m = r.get("margin")
            if m is None and "top5_scores" in r:
                s = r["top5_scores"]
                m = s[0] - s[1] if len(s) > 1 else s[0]
            if m is not None:
                margins.append(float(m))
            if r["prediction"] == label:
                ok += 1
            if label in top:
                top5_ok += 1
            is_accepted = (m is not None and float(m) >= MARGIN_CUT
                           and r["confidence"] >= SCORE_FLOOR)
            if is_accepted:
                accepted += 1
                if r["prediction"] == label:
                    accepted_ok += 1
            if r["prediction"] != label:
                confusions[r["prediction"]] += 1
                misses.append({"clip": cid, "pred": r["prediction"],
                               "score": r["confidence"], "margin": m, "why": "wrong label"})
        n = len(clips)
        rows.append({
            "label": label, "n": n, "ok": ok, "top5_ok": top5_ok, "no_hand": no_hand,
            "scored": n - no_hand, "accepted": accepted, "accepted_ok": accepted_ok,
            "top1": ok / n, "top5": top5_ok / n,
            "scores": scores, "margins": margins,
            "mean_score": statistics.mean(scores) if scores else 0.0,
            "median_score": statistics.median(scores) if scores else 0.0,
            "mean_margin": statistics.mean(margins) if margins else 0.0,
            "confusions": confusions, "misses": misses,
            "wilson": wilson(ok, n),
        })

    total = sum(r["n"] for r in rows)
    total_ok = sum(r["ok"] for r in rows)
    total_top5 = sum(r["top5_ok"] for r in rows)
    total_nohand = sum(r["no_hand"] for r in rows)
    total_acc = sum(r["accepted"] for r in rows)
    total_acc_ok = sum(r["accepted_ok"] for r in rows)

    print(f"vocabulary: {len(labels)} classes")
    print(f"classes with >= {args.min_clips} clip(s): {len(rows)}")
    print(f"clips evaluated: {total}   (no-hand refusals: {total_nohand})")
    print(f"clips in split without a local file: {missing}")
    if unmapped:
        print(f"glosses outside the vocabulary: {len(unmapped)} distinct, {sum(unmapped.values())} rows")
    print()
    print("MICRO (per clip)")
    print(f"  top-1                {total_ok}/{total} = {pct(total_ok, total)}")
    print(f"  top-5                {total_top5}/{total} = {pct(total_top5, total)}")
    print(f"  accepted             {total_acc}/{total} = coverage {pct(total_acc, total)}")
    print(f"  accuracy if accepted {total_acc_ok}/{total_acc} = {pct(total_acc_ok, total_acc)}")
    print(f"  rejection rate       {pct(total - total_acc, total)}")
    print()
    macro1 = statistics.mean(r["top1"] for r in rows)
    macro5 = statistics.mean(r["top5"] for r in rows)
    print("MACRO (per class, equal weight)")
    print(f"  mean class top-1     {100*macro1:.1f}%")
    print(f"  mean class top-5     {100*macro5:.1f}%")
    print(f"  classes at 100%      {sum(1 for r in rows if r['ok'] == r['n'])}")
    print(f"  classes at 0%        {sum(1 for r in rows if r['ok'] == 0)}")
    print()
    print(f"{'class':<24}{'n':>3}{'ok':>4}{'top1':>7}{'95% CI':>16}{'top5':>7}{'acc':>5}{'mean':>7}")
    for r in sorted(rows, key=lambda r: (r["top1"], r["label"])):
        lo, hi = r["wilson"]
        print(f"{r['label']:<24}{r['n']:>3}{r['ok']:>4}{100*r['top1']:>6.0f}%"
              f"{100*lo:>7.0f}-{100*hi:<7.0f}{100*r['top5']:>6.0f}%{r['accepted']:>5}"
              f"{r['mean_score']:>7.3f}")
        if r["ok"] < r["n"]:
            top_conf = ", ".join(f"{k}({v})" for k, v in r["confusions"].most_common(3))
            print(f"    confusions: {top_conf or 'none'}")

    if args.json_out:
        out = []
        for r in rows:
            out.append({k: v for k, v in r.items() if k not in ("scores", "margins", "confusions", "misses")}
                       | {"confusions": dict(r["confusions"]), "misses": r["misses"]})
        args.json_out.write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())