"""Derive `results/predictions.json` from `results/predictions.jsonl`.

Deck p.11 requires a single `predictions.jsonl`; deck p.12's bundle diagram
shows `results/predictions.json`. Emitting both from one source of truth
satisfies either reading without risking the two files disagreeing.

The JSON form is the same records as an array, in the same order, with the same
five contract fields. No values are recomputed or rounded differently.

Usage:
    python3 scripts/release/predictions_json_from_jsonl.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results/predictions.jsonl"
DST = ROOT / "results/predictions.json"
REQUIRED = ("clip_id", "prediction", "top5", "confidence", "latency_ms")


def main(argv: list[str] | None = None) -> int:
    if not SRC.is_file():
        print(f"missing {SRC}", file=sys.stderr)
        return 2

    records = []
    seen = set()
    for lineno, line in enumerate(SRC.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        rec = json.loads(line)
        missing = [k for k in REQUIRED if k not in rec]
        if missing:
            print(f"{SRC}:{lineno}: missing {missing}", file=sys.stderr)
            return 2
        cid = rec["clip_id"]
        if cid in seen:
            print(f"{SRC}:{lineno}: duplicate clip_id {cid!r}", file=sys.stderr)
            return 2
        seen.add(cid)
        if rec["prediction"] != rec["prediction"].upper():
            print(f"{SRC}:{lineno}: prediction not uppercase", file=sys.stderr)
            return 2
        if rec["prediction"] not in rec["top5"]:
            print(f"{SRC}:{lineno}: prediction absent from top5", file=sys.stderr)
            return 2
        records.append(rec)

    DST.write_text(json.dumps(records, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {DST} with {len(records)} records ({DST.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())