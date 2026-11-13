"""Is a failing class consistently failing, or was one clip unlucky?

For each class it runs every available genuine TRAIN-split clip through the
shipped pipeline and reports the distribution of outcomes. A class that is 0/4
on training clips is a model gap; one that is 3/4 is a clip-quality or
signer-variance question. The single validation clip that failed in
PER_CLASS_RESULTS.md cannot distinguish those.

Nothing is hardcoded: an index TSV supplies (label, clip filename, signer) and
the clips live beside it under <label>/<filename>.

Usage:
    python3 evaluation/device/train_clip_consistency.py --index /tmp/trainclips/index.tsv
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MARGIN_CUT = 0.15
SCORE_FLOOR = 0.20


def pct(n: float, d: float) -> str:
    return f"{100.0 * n / d:.0f}%" if d else "n/a"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", type=Path, required=True,
                    help="TSV of label<TAB>filename<TAB>signer; clips under sibling <label>/")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args(argv)

    root = args.index.parent
    entries = []
    for line in args.index.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        label, name = parts[0], parts[1]
        signer = parts[2] if len(parts) > 2 else "-"
        path = root / label / name
        if path.is_file():
            entries.append((label, path, signer))
    if not entries:
        print(f"no clips found under {root}", file=sys.stderr)
        return 2

    sys.path.insert(0, str(ROOT / "src"))
    import main as model  # noqa: E402

    model._init_model()

    by_class: dict[str, list[dict]] = defaultdict(list)
    for label, path, signer in entries:
        try:
            r = model.predict(str(path))
        except ValueError:
            by_class[label].append({"clip": path.name, "signer": signer,
                                    "pred": None, "score": None, "ok": False,
                                    "why": "no hand"})
            continue
        m = r.get("margin")
        accepted = (m is not None and float(m) >= MARGIN_CUT
                    and r["confidence"] >= SCORE_FLOOR)
        by_class[label].append({
            "clip": path.name, "signer": signer,
            "pred": r["prediction"], "score": r["confidence"],
            "margin": m, "ok": r["prediction"] == label,
            "accepted": accepted,
        })

    print(f"{'class':<24}{'n':>3}{'ok':>5}{'top1':>7}{'asserts':>10}  pattern")
    summary = []
    for label in sorted(by_class):
        rs = by_class[label]
        n = len(rs)
        ok = sum(1 for r in rs if r["ok"])
        signs = {}
        for r in rs:
            signs.setdefault(r["signer"], [0, 0])
            signs[r["signer"]][1] += 1
            if r["ok"]:
                signs[r["signer"]][0] += 1
        # A class that fails on EVERY signer is a model gap; one that fails on
        # a single signer is closer to a signer/domain question.
        bad_signers = sum(1 for s, (c, tot) in signs.items() if c == 0)
        good_signers = len(signs) - bad_signers
        if ok == n:
            pattern = "consistent PASS"
        elif ok == 0:
            pattern = f"consistent FAIL ({len(signs)} signers)"
        elif good_signers and bad_signers:
            pattern = f"mixed: {good_signers} signer(s) ok, {bad_signers} all-wrong"
        else:
            pattern = "mixed within signer(s)"
        print(f"{label:<24}{n:>3}{ok:>5}{pct(ok, n):>7}{len(signs):>10}  {pattern}")
        scores = [r["score"] for r in rs if r["score"] is not None]
        summary.append({
            "label": label, "n": n, "ok": ok, "top1": ok / n,
            "signers": len(signs), "bad_signers": bad_signers,
            "mean_score": statistics.mean(scores) if scores else 0.0,
            "pattern": pattern,
            "misses": [r for r in rs if not r["ok"]],
        })

    total = sum(s["n"] for s in summary)
    correct = sum(s["ok"] for s in summary)
    print()
    print(f"TRAIN-SPLIT clips: {total}   correct {correct}   top-1 {pct(correct, total)}")
    cons_fail = [s["label"] for s in summary if s["ok"] == 0]
    cons_pass = [s["label"] for s in summary if s["ok"] == s["n"]]
    print(f"classes failing on EVERY clip: {len(cons_fail)} -> {cons_fail}")
    print(f"classes passing on EVERY clip: {len(cons_pass)}")

    if args.json_out:
        args.json_out.write_text(json.dumps(summary, indent=1), encoding="utf-8")
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())