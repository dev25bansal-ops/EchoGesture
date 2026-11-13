"""Turn a phone diagnostics log into accuracy, rejection and per-sign tables.

Joins `files/diagnostics.txt` (written by the app) with an annotations file that
supplies the sign the signer intended — the one thing the app cannot know.
Accuracy is only computable against that intent.

Usage:
    python3 evaluation/device/accuracy_report.py diagnostics.txt annotations.txt

Annotations: one line per capture, in capture order:
    <attempt#> <expected_label> <background> <lighting> <distance_cm>

Prints per-attempt rows, top-1/top-5, rejection rate, accuracy at accepted
coverage, a confusion matrix and a per-sign table. Writes nothing.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

# The capture-coverage floor the app itself uses to reject an unusable window.
MIN_HAND_FRACTION = 0.15
# The calibrated score cut the app uses to label a result uncertain.
SCORE_CUT = 0.35
MARGIN_CUT = 0.15


def parse_diagnostics(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        fields = {}
        for token in line.split():
            if "=" in token:
                k, v = token.split("=", 1)
                fields[k] = v
        if not fields:
            continue
        rows.append(
            {
                "frames": int(fields.get("frames", 0)),
                "dur_ms": int(fields.get("dur_ms", 0)),
                "fps": float(fields.get("fps", 0) or 0),
                "hand_frames": int(fields.get("hand_frames", 0)),
                "pred": fields.get("pred", ""),
                "score": float(fields["score"]) if fields.get("score") else 0.0,
                "margin": float(fields["margin"]) if fields.get("margin") else None,
                "top5": [t for t in fields.get("top5", "").split("|") if t],
                "infer_ms": int(fields.get("infer_ms", -1)),
            }
        )
    return rows


def parse_annotations(path: Path) -> list[dict]:
    out = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            raise ValueError(f"{path}:{lineno}: need at least '<attempt> <label>'")
        out.append(
            {
                "attempt": parts[0],
                "expected": parts[1],
                "condition": " ".join(parts[2:]) or "-",
            }
        )
    return out


def categorize(row: dict, expected: str) -> str:
    """One bucket per attempt. Order matters: earlier tests are more specific."""
    if row["frames"] == 0:
        return "failed capture"
    coverage = row["hand_frames"] / row["frames"]
    if coverage < MIN_HAND_FRACTION:
        return "failed capture"
    if not row["pred"]:
        return "failed capture"
    if row["pred"] == expected:
        return "correct"
    if row["score"] < SCORE_CUT:
        return "incorrect (rejected)"
    return "incorrect (accepted)"


def pct(n: int, d: int) -> str:
    return f"{100.0 * n / d:.1f}%" if d else "n/a"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("diagnostics", type=Path)
    ap.add_argument("annotations", type=Path)
    args = ap.parse_args(argv)

    rows = parse_diagnostics(args.diagnostics)
    anns = parse_annotations(args.annotations)
    if len(rows) != len(anns):
        print(
            f"refusing to report: {len(rows)} diagnostic captures but "
            f"{len(anns)} annotations. Fix the pairing; do not guess which "
            f"row belongs to which attempt.",
            file=sys.stderr,
        )
        return 2
    if not rows:
        print("no captures to report", file=sys.stderr)
        return 2

    joined = list(zip(rows, anns))
    print(f"{'#':>3} {'expected':<14} {'predicted':<14} {'score':>6} "
          f"{'cov':>6} {'dur':>6} {'category':<22} condition")
    for row, ann in joined:
        cov = row["hand_frames"] / row["frames"] if row["frames"] else 0.0
        print(
            f"{ann['attempt']:>3} {ann['expected']:<14} "
            f"{(row['pred'] or '-'):<14} {row['score']:6.3f} "
            f"{cov:5.0%} {row['dur_ms']:5d}ms "
            f"{categorize(row, ann['expected']):<22} {ann['condition']}"
        )

    n = len(joined)
    top1 = sum(1 for r, a in joined if r["pred"] == a["expected"] and r["pred"])
    top5 = sum(1 for r, a in joined if a["expected"] in r["top5"])

    # The app accepts on the top-1/top-2 margin. Older logs have no margin
    # field, so fall back to the score cut and say which rule was used.
    use_margin = any(r["margin"] is not None for r, _ in joined)

    def accepts(r):
        if not r["pred"]:
            return False
        if use_margin and r["margin"] is not None:
            return r["margin"] >= MARGIN_CUT
        return r["score"] >= SCORE_CUT

    accepted = [(r, a) for r, a in joined if accepts(r)]
    accepted_ok = sum(1 for r, a in accepted if r["pred"] == a["expected"])

    print()
    print(f"captures                 {n}")
    print(f"top-1                    {top1}/{n} = {pct(top1, n)}")
    print(f"top-5                    {top5}/{n} = {pct(top5, n)}")
    rule = f"margin>={MARGIN_CUT:.2f}" if use_margin else f"score>={SCORE_CUT:.2f} (no margin in log)"
    print(f"accepted ({rule})   {len(accepted)}/{n} = coverage {pct(len(accepted), n)}")
    print(f"accuracy at accepted     {accepted_ok}/{len(accepted)} = {pct(accepted_ok, len(accepted))}")
    print(f"rejection rate           {pct(n - len(accepted), n)}")

    cats = Counter(categorize(r, a["expected"]) for r, a in joined)
    print()
    print("outcome breakdown")
    for k, v in sorted(cats.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<22} {v:>3}  {pct(v, n)}")

    print()
    print("per-sign")
    print(f"  {'sign':<16} {'n':>3} {'top-1':>7} {'top-5':>7}")
    bysign: dict[str, list] = defaultdict(list)
    for r, a in joined:
        bysign[a["expected"]].append((r, a))
    for sign in sorted(bysign):
        group = bysign[sign]
        c1 = sum(1 for r, a in group if r["pred"] == sign)
        c5 = sum(1 for r, a in group if sign in r["top5"])
        print(f"  {sign:<16} {len(group):>3} {pct(c1, len(group)):>7} {pct(c5, len(group)):>7}")

    wrong = [(r, a) for r, a in joined if r["pred"] and r["pred"] != a["expected"]]
    if wrong:
        print()
        print("confusion (expected -> predicted)")
        cm = Counter((a["expected"], r["pred"]) for r, a in wrong)
        for (exp, pred), count in cm.most_common():
            print(f"  {exp:<16} -> {pred:<16} {count}")

    low_cov = [a["expected"] for r, a in joined
               if r["frames"] and r["hand_frames"] / r["frames"] < MIN_HAND_FRACTION]
    if low_cov:
        print()
        print(f"{len(low_cov)} capture(s) below the {MIN_HAND_FRACTION:.0%} coverage floor "
              f"were not classified by the app; they are failed captures, not model errors.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())