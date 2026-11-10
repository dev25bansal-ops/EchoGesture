#!/usr/bin/env python3
"""Frozen-prediction split integrity recomputation (read-only reconciliation).

Independently recomputes from official train/val/test CSVs:
- full row counts and signer sets/counts
- vocabulary-eligible counts (100-class labels, upper-cased gloss match)
- exact overlaps (signer / filename / clip-ID stem) full + eligible subsets
- within-split duplicate filenames/clip IDs
- label index permutation check (0..99 exactly once)
Cross-checks vs saved v5 split_audit.json and v6 leakage_audit.json.
No video bytes read; test CSV used only for overlap/counting, never predictions.
"""
from __future__ import annotations
import argparse, csv, hashlib, json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set

REQUIRED = {"Participant ID", "Video file", "Gloss", "ASL-LEX Code"}

def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def clip_id(vf: str) -> str:
    return Path(vf).stem

def read_rows(p: Path) -> List[Dict[str, str]]:
    with p.open(newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        assert r.fieldnames and REQUIRED.issubset(set(r.fieldnames)), p
        rows = list(r)
    for i, row in enumerate(rows, start=2):
        assert all(row[c] not in (None, "") and row[c].strip() != "" for c in REQUIRED), (p, i)
    return rows

def summary(rows: List[Dict[str, str]], vocab: Set[str]) -> Dict[str, Any]:
    elig = [r for r in rows if r["Gloss"].strip().upper() in vocab]
    fn = [r["Video file"] for r in rows]; ci = [clip_id(x) for x in fn]
    efn = [r["Video file"] for r in elig]; eci = [clip_id(x) for x in efn]
    return {
        "full_rows": len(rows), "full_signers": len({r["Participant ID"] for r in rows}),
        "within_dup_filenames": {k: v for k, v in Counter(fn).items() if v > 1},
        "within_dup_clip_ids": {k: v for k, v in Counter(ci).items() if v > 1},
        "eligible_rows": len(elig),
        "eligible_signers": sorted({r["Participant ID"] for r in elig}),
        "eligible_signer_count": len({r["Participant ID"] for r in elig}),
        "eligible_signer_counts": dict(sorted(Counter(r["Participant ID"] for r in elig).items())),
        "eligible_gloss_count": len({r["Gloss"].strip().upper() for r in elig}),
        "eligible_gloss_counts": dict(sorted(Counter(r["Gloss"].strip().upper() for r in elig).items())),
        "eligible_dup_filenames": {k: v for k, v in Counter(efn).items() if v > 1},
        "eligible_dup_clip_ids": {k: v for k, v in Counter(eci).items() if v > 1},
        "eligible_missing": sorted(vocab - {r["Gloss"].strip().upper() for r in elig}),
    }

def overlaps(a: List[Dict[str, str]], b: List[Dict[str, str]]) -> Dict[str, Any]:
    return {"signer_overlap": sorted({r["Participant ID"] for r in a} & {r["Participant ID"] for r in b}),
            "filename_overlap": sorted({r["Video file"] for r in a} & {r["Video file"] for r in b}),
            "clip_id_overlap": sorted({clip_id(r["Video file"]) for r in a} & {clip_id(r["Video file"]) for r in b})}

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-csv", type=Path, required=True)
    ap.add_argument("--val-csv", type=Path, required=True)
    ap.add_argument("--test-csv", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--v5-audit", type=Path, required=True)
    ap.add_argument("--v6-audit", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    assert isinstance(labels, dict) and len(labels) == 100 and set(labels.values()) == set(range(100))
    assert all(isinstance(v, int) and not isinstance(v, bool) for v in labels.values())
    vocab = {k.strip().upper() for k in labels}
    assert len(vocab) == 100

    train, val, test = read_rows(args.train_csv), read_rows(args.val_csv), read_rows(args.test_csv)
    splits = {"train": summary(train, vocab), "val": summary(val, vocab), "test": summary(test, vocab)}
    ov = {"train_vs_val": overlaps(train, val), "train_vs_test": overlaps(train, test), "val_vs_test": overlaps(val, test)}
    etr = {k: [r for r in rows if r["Gloss"].strip().upper() in vocab]
           for k, rows in (("train", train), ("val", val), ("test", test))}
    eov = {"train_vs_val": overlaps(etr["train"], etr["val"]), "train_vs_test": overlaps(etr["train"], etr["test"]),
           "val_vs_test": overlaps(etr["val"], etr["test"])}

    # exact expected counts
    assert splits["train"]["full_rows"] == 40154 and splits["train"]["full_signers"] == 35
    assert splits["val"]["full_rows"] == 10304 and splits["test"]["full_rows"] == 32941
    assert (splits["train"]["eligible_rows"], splits["val"]["eligible_rows"], splits["test"]["eligible_rows"]) == (1789, 346, 1255)
    assert (splits["train"]["eligible_signer_count"], splits["val"]["eligible_signer_count"],
            splits["test"]["eligible_signer_count"]) == (32, 5, 11)
    for pair in (ov, eov):
        for k, v in pair.items():
            assert v["signer_overlap"] == [] and v["filename_overlap"] == [] and v["clip_id_overlap"] == [], (k, v)
    for name, s in splits.items():
        assert s["within_dup_filenames"] == {} and s["within_dup_clip_ids"] == {}, name
        assert s["eligible_gloss_count"] == 100 and s["eligible_missing"] == [], name

    v5 = json.loads(args.v5_audit.read_text(encoding="utf-8"))
    v6 = json.loads(args.v6_audit.read_text(encoding="utf-8"))
    assert v5["vocab_eligible"]["val"]["eligible"] == 346 and v5["vocab_eligible"]["test"]["eligible"] == 1255
    assert v5["official_split_signers"]["overlap"] == []
    assert v5["requires_dev_split"]["eligible_clips"] == 1789
    assert v5["requires_dev_split"]["train_val_signer_disjoint"] is True
    assert v5["requires_dev_split"]["train_test_signer_disjoint"] is True
    assert v6["status"] == "PASS_WITH_LIMITATIONS" and all(v6["exact_dataset_checks"].values())
    assert v6["inputs"]["train_csv"]["sha256"] == sha256(args.train_csv)
    assert v6["inputs"]["val_csv"]["sha256"] == sha256(args.val_csv)
    assert v6["inputs"]["test_csv"]["sha256"] == sha256(args.test_csv)

    out = {"analysis": "V7 split integrity reconciliation (frozen CSVs, exact string comparisons only)",
           "status": "PASS_WITH_LIMITATIONS",
           "inputs": {"train_csv": {"path": str(args.train_csv), "sha256": sha256(args.train_csv), "rows": len(train)},
                      "val_csv": {"path": str(args.val_csv), "sha256": sha256(args.val_csv), "rows": len(val)},
                      "test_csv": {"path": str(args.test_csv), "sha256": sha256(args.test_csv), "rows": len(test)},
                      "labels": {"path": str(args.labels), "sha256": sha256(args.labels), "classes": 100}},
           "splits": splits, "overlaps_full": ov, "overlaps_eligible": eov,
           "checks": {"counts_match_expected": True, "all_exact_overlaps_empty": True,
                      "all_eligible_splits_cover_100_classes": True,
                      "labels_are_0_99_permutation": True, "v5_v6_cross_checks_match": True},
           "verified": ["Exact participant/filename/clip-ID cross-split disjointness on all three official manifests (full + eligible subsets).",
                        "Eligible counts train=1789 / val=346 / test=1255; eligible signers 32/5/11; each eligible split covers all 100 classes.",
                        "No within-split duplicate filenames or clip IDs."],
           "not_verified": ["Near-duplicate, re-encoded, truncated, semantically duplicate, or renamed video content.",
                            "Signer aliases / participant identity errors.",
                            "Whether training actually consumed only train rows (provenance).",
                            "Test-split accuracy (test untouched)."],
           "limitations": ["Manifest string comparisons only; no video-byte hashing/inspection.",
                           "Stripped/upper-cased gloss matching; ASL-LEX codes not independently validated.",
                           "Training-pipeline provenance cannot be established from dataset rows alone."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": out["status"], "eligible": {k: v["eligible_rows"] for k, v in splits.items()},
                      "checks": out["checks"]}, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
