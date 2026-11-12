"""Derive the app's confidence cut from held-out evidence.

Reads the frozen ensemble validation predictions, maps each clip's gloss to a
shipped class label, and reports top-1 accuracy within score bands. The cut the
app uses is read off this table; it is not chosen by eye.

Usage:
    .venv/bin/python -I evaluation/device/recognition_diagnosis/calibrate.py
"""
from __future__ import annotations
import json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PRED = ROOT / "evaluation/device/ensemble_evidence/predictions_ensemble_val.jsonl"
LABELS = ROOT / "models/landmark_labels.json"


def canon(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", s.upper())


def truth_label(name: str, labels: list[str]) -> str | None:
    """Clip gloss -> shipped label.

    Class labels are the gloss plus an optional disambiguating suffix, so an
    exact hit wins and a unique prefix match is the fallback. Anything else is
    left unmapped rather than guessed.
    """
    t = canon(name)
    exact = [l for l in labels if canon(l) == t]
    if exact:
        return exact[0]
    pref = [l for l in labels if canon(l).startswith(t)]
    return pref[0] if len(pref) == 1 else None


def main() -> int:
    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    rows, unmapped = [], 0
    for line in PRED.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        p = json.loads(line)
        tl = truth_label(p["clip_id"].split("-", 1)[1], labels)
        if tl is None:
            unmapped += 1
            continue
        rows.append((p["confidence"], p["prediction"] == tl))

    total = len(rows)
    acc = sum(r[1] for r in rows) / total
    print(f"clips: {total} mapped, {unmapped} unmapped")
    print(f"top-1 overall: {acc*100:.2f}%")
    print(f"{'score >=':>9}  {'kept':>10}  {'coverage':>9}  {'top-1 on kept':>14}")
    for thr in (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50):
        kept = [r for r in rows if r[0] >= thr]
        if not kept:
            continue
        k = sum(r[1] for r in kept) / len(kept)
        print(f"{thr:9.2f}  {len(kept):10d}  {len(kept)/total*100:8.1f}%  {k*100:13.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
