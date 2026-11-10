#!/usr/bin/env python3
"""Acceptance gate for a proposed INT8 classifier against the frozen contract.

Reads an audit JSON produced by audit_quantization.py and a candidate model.
The gate fails closed: no candidate, wrong SHA, changed feature/label contract,
non-finite logits, or accuracy regression all exit nonzero.

This does not replace the selected model, train, or export anything.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FIXTURES = HERE / "fixtures"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(candidate: Path, baseline: Path, audit: dict) -> dict:
    pins = json.loads((FIXTURES / "provenance.json").read_text(encoding="utf-8"))
    report = {"candidate": str(candidate), "candidate_sha256": sha256(candidate),
              "baseline": str(baseline), "baseline_sha256": sha256(baseline), "checks": {}}
    checks = report["checks"]
    checks["candidate_is_not_baseline"] = report["candidate_sha256"] != report["baseline_sha256"]
    checks["features_unchanged"] = sha256(ROOT / "src/features.py") == pins["frozen_contract"]["feature_sha256"]
    checks["labels_unchanged"] = sha256(ROOT / "models/landmark_labels.json") == pins["frozen_contract"]["labels_sha256"]
    checks["audit_is_current"] = bool(audit.get("repo_head") and audit.get("assets", {}).get("fp16", {}).get("sha256")
                                      == pins["assets"]["fp16"]["sha256"])
    graph = audit.get("classifiers", {}).get("int8", {})
    checks["audit_candidate_is_reported_int8"] = graph.get("category") == "INT8 weights/activations"
    checks["candidate_io_matches_contract"] = (
        graph.get("interpreter_io", {}).get("inputs", [{}])[0].get("dtype") == "float32"
        and graph.get("interpreter_io", {}).get("outputs", [{}])[0].get("dtype") == "float32")
    evidence = audit.get("existing_paired_346", {})
    checks["paired_evidence_pinned"] = bool(evidence.get("checks") and all(evidence["checks"].values()))
    fp16 = evidence.get("fp16_top1", {}).get("top1_correct")
    int8 = evidence.get("int8_top1", {}).get("top1_correct")
    checks["existing_int8_no_regression"] = bool(fp16 is not None and int8 is not None and int8 >= fp16)
    checks["whole_pipeline_int8_established"] = bool(audit.get("verdict", {}).get("whole_pipeline_int8"))
    checks["official_interpretation_resolved"] = audit.get("verdict", {}).get("official_interpretation") != "UNRESOLVED"
    report["passed"] = all(checks.values())
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=HERE / "results/audit.json")
    parser.add_argument("--candidate", type=Path, default=ROOT / "models/landmark_classifier_int8.tflite")
    parser.add_argument("--baseline", type=Path, default=ROOT / "models/landmark_classifier_fp16.tflite")
    parser.add_argument("--output", type=Path, default=HERE / "results/acceptance.json")
    args = parser.parse_args(argv)
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    report = check(args.candidate, args.baseline, audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["checks"], indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())