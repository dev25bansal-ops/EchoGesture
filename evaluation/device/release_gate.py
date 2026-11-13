"""Release gate: compare a candidate pipeline against the frozen baseline.

Runs the shipped `predict()` over authorized ASL Citizen validation clips and
compares it with Dev's frozen `predictions_ensemble_val.jsonl` on the SAME
population. Both are scored against the official signer-disjoint split.

The comparison is only valid on clips present in BOTH — the frozen baseline
covers 345 of the 346 in-vocabulary validation clips. Anything else is reported
separately rather than pooled.

Usage:
    python3 evaluation/device/release_gate.py --clips /path/to/clips

Nothing is written; the report goes to stdout.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LABELS = ROOT / "models/landmark_labels.json"
# The frozen baseline lives outside Git (it names private clip ids), so its
# location is a flag rather than a fixed path.
DEFAULT_FROZEN = Path(
    "/Users/abhijeetmiskin/AppData/EchoGesture/"
    "evaluation/device/ensemble_evidence/predictions_ensemble_val.jsonl"
)

SCORE_CUT = 0.35


def canon(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", s.upper())


def load_labels() -> dict[str, str]:
    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    return {canon(l): l for l in labels}


def load_truth(split_csv: Path, lab: dict[str, str]) -> dict[str, str]:
    """clip id (without .mp4) -> shipped label, only for in-vocabulary glosses."""
    out = {}
    with split_csv.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            v = canon(row["Gloss"])
            if v in lab:
                out[Path(row["Video file"]).stem] = lab[v]
    return out


def load_frozen(path: Path) -> dict[str, dict]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            out[d["clip_id"]] = d
    return out


def pct(n: int, d: int) -> str:
    return f"{100.0 * n / d:.2f}%" if d else "n/a"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clips", type=Path, required=True, help="directory of .mp4 clips")
    ap.add_argument("--split", type=Path, default=Path("/tmp/val.csv"))
    ap.add_argument("--frozen", type=Path, default=DEFAULT_FROZEN)
    args = ap.parse_args(argv)

    sys.path.insert(0, str(ROOT / "src"))
    import main as model  # noqa: E402

    lab = load_labels()
    truth = load_truth(args.split, lab)
    frozen = load_frozen(args.frozen) if args.frozen.exists() else {}
    if not frozen:
        print(f"note: frozen baseline not found at {args.frozen}; reporting candidate only", file=sys.stderr)
    model._init_model()

    rows = []
    for path in sorted(args.clips.glob("*.mp4")):
        cid = path.stem
        t = truth.get(cid)
        if t is None:
            continue
        try:
            r = model.predict(str(path))
        except ValueError as exc:
            # A clip with no detectable hand is a rejected capture, not a
            # crash. The baseline records exactly this clip as unscored, so
            # counting it as a rejection keeps the populations comparable.
            r = {"prediction": "", "confidence": 0.0, "top5": [], "total_ms": 0.0}
            print(f"  no prediction: {cid} ({exc})", file=sys.stderr)
        rows.append(
            {
                "clip_id": cid,
                "truth": t,
                "pred": r["prediction"],
                "score": r["confidence"],
                "top5": r["top5"],
                "latency_ms": r.get("total_ms") or r.get("latency_ms") or 0.0,
                "frozen": frozen.get(cid),
            }
        )

    n = len(rows)
    if not n:
        print("no scorable clips found", file=sys.stderr)
        return 2

    both = [r for r in rows if r["frozen"]]
    print(f"scored {n} clips; {len(both)} also present in the frozen baseline\n")

    cand_top1 = sum(1 for r in rows if r["pred"] == r["truth"])
    cand_top5 = sum(1 for r in rows if r["truth"] in r["top5"])
    print(f"{'candidate top-1':<28}{cand_top1:>7}/{n:<4}{pct(cand_top1, n):>12}")
    print(f"{'candidate top-5':<28}{cand_top5:>7}/{n:<4}{pct(cand_top5, n):>12}")

    if both:
        m = len(both)
        bt1 = sum(1 for r in both if r["frozen"]["prediction"] == r["truth"])
        ct1 = sum(1 for r in both if r["pred"] == r["truth"])
        bt5 = sum(1 for r in both if r["truth"] in r["frozen"]["top5"])
        ct5 = sum(1 for r in both if r["truth"] in r["top5"])
        print()
        print(f"paired on the same {m} clips")
        print(f"{'':<24}{'candidate':>12}{'baseline':>12}{'delta':>10}")
        for name, c, b in (("top-1", ct1, bt1), ("top-5", ct5, bt5)):
            print(f"{name:<24}{pct(c, m):>12}{pct(b, m):>12}{('+' if c >= b else '')}{100.0*(c-b)/m:>8.2f}pt")

        flipped_gain = [r for r in both if r["pred"] == r["truth"] and r["frozen"]["prediction"] != r["truth"]]
        flipped_loss = [r for r in both if r["pred"] != r["truth"] and r["frozen"]["prediction"] == r["truth"]]
        print()
        print(f"fixed by candidate: {len(flipped_gain)}   broken by candidate: {len(flipped_loss)}")
        for r in flipped_loss[:5]:
            print(f"  regressed {r['clip_id']}: {r['frozen']['prediction']} -> {r['pred']}")

    accepted = [r for r in rows if r["score"] >= SCORE_CUT]
    ok = sum(1 for r in accepted if r["pred"] == r["truth"])
    print()
    print(f"accepted (score>={SCORE_CUT}): {len(accepted)}/{n} coverage {pct(len(accepted), n)}")
    print(f"accuracy at accepted:        {ok}/{len(accepted)} = {pct(ok, len(accepted))}")
    print(f"rejection rate:              {pct(n - len(accepted), n)}")

    lat = sorted(r["latency_ms"] for r in rows if r["latency_ms"])
    if lat:
        print(f"clip-to-JSON latency p50/p95: {lat[len(lat)//2]:.1f} / {lat[min(len(lat)-1, int(len(lat)*0.95))]:.1f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())