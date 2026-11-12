#!/usr/bin/env python3
"""Summarize paired accuracy, drop, agreement, and FP16/INT8 parity."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/tmp/echogesture-v7-agents/A03/evaluation/device/v7_tracking")
RESULTS = ROOT / "results"


def summarize():
    out = {}
    for path in sorted(RESULTS.glob("compare_*.json")):
        data = json.loads(path.read_text())
        out[path.name] = {"clean": data["clean"], "corrupted": data["corrupted"],
                          "pairs": data["pairs"], "parity": data["changes"], "meta": data["meta"]}
    return out


if __name__ == "__main__":
    print(json.dumps(summarize(), indent=2))
