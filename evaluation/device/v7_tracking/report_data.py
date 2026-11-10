#!/usr/bin/env python3
"""Produce a compact, human-readable V7 tracking verdict from measured JSON."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/tmp/echogesture-v7-agents/A03/evaluation/device/v7_tracking")
RESULTS = ROOT / "results"


def fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def main():
    rows = []
    for path in sorted(RESULTS.glob("compare_*346.json")):
        if "verified" in path.name:
            continue
        d = json.loads(path.read_text())
        rows.append({
            "file": path.name,
            "corruption": d["meta"]["corruption"],
            "clean": {c: {m: {"correct": d["clean"][f"{c}_{m}"]["correct"],
                               "n": d["clean"][f"{c}_{m}"]["n"]} for m in ("fp16", "int8")}
                      for c in ("baseline", "slot", "gap", "slot_gap")},
            "corrupted": {c: {m: {"correct": d["corrupted"][f"{c}_{m}"]["correct"],
                                   "n": d["corrupted"][f"{c}_{m}"]["n"]} for m in ("fp16", "int8")}
                          for c in ("baseline", "slot", "gap", "slot_gap")},
            "baseline_pairs": {m: d["pairs"][f"baseline_{m}"] for m in ("fp16", "int8")},
            "gap_pairs": {m: d["pairs"][f"gap_{m}"] for m in ("fp16", "int8")},
            "baseline_parity_clean": d["changes"]["baseline_clean"],
            "gap_parity_clean": d["changes"]["gap_clean"],
            "baseline_parity_corrupted": d["changes"]["baseline_corrupted"],
            "gap_parity_corrupted": d["changes"]["gap_corrupted"],
            "meta": d["meta"],
        })
    text = {
        "scope": "V7 landmark candidate evaluation; 346 labelled validation clips (one clean failure), with 30-clip frame-drop control.",
        "model_shas": {
            "fp16": "a012f2b5f7a1110961079f19eb0e28c9abacb95e4d0787352e4157ebca7044a4",
            "int8": "9d24640a23244a3cd00c9fa62257733819f86ed1da5a10194c9b1cb64323700b",
        },
        "runs": rows,
        "clean_30": json.loads((RESULTS / "clean_30_rerun/run.json").read_text())["summary"],
        "clean_346": json.loads((RESULTS / "clean_346_verified/run.json").read_text())["summary"],
        "failures": {},
    }
    for name in ("clean_346_verified", "corrupt_occlusion_346", "corrupt_motion_blur_346", "corrupt_frame_drop_346", "corrupt_low_light_346", "corrupt_crop_346"):
        d = json.loads((RESULTS / name / "run.json").read_text())
        text["failures"][name] = {"n_requested": d["n_requested"], "n_success": d["n_success"], "failed": d["failed"],
                                  "errors": [(r["clip_id"], r.get("error")) for r in d["records"] if "error" in r]}
    (RESULTS / "verdict_data.json").write_text(json.dumps(text, indent=2) + "\n")
    print(json.dumps(text, indent=2))


if __name__ == "__main__":
    main()
