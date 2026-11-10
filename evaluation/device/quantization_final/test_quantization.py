"""Deterministic graph-classification and inference tests.

These tests use only pinned artifacts and synthetic graph objects, never train,
download, or benchmark, and never modify another owner's file.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FIXTURES = HERE / "fixtures"
sys.path.insert(0, str(HERE))

import audit_quantization as audit  # noqa: E402

PINS = json.loads((FIXTURES / "provenance.json").read_text(encoding="utf-8"))


class TestGraphClassification(unittest.TestCase):
    def test_classifier_bytes_and_hashes_are_pinned(self):
        for name in ("fp16", "int8", "detector", "labels", "meta", "features", "frontend"):
            self.assertEqual(audit.sha256_bytes((ROOT / PINS["assets"][name]["path"]).read_bytes()),
                             PINS["assets"][name]["sha256"], name)

    def test_graph_inspection_of_flatbuffer(self):
        graphs = {"fp16": audit.inspect_graph((ROOT / PINS["assets"]["fp16"]["path"]).read_bytes(), "fp16"),
                  "int8": audit.inspect_graph((ROOT / PINS["assets"]["int8"]["path"]).read_bytes(), "int8")}
        self.assertEqual(graphs["fp16"]["category"], "FP16")
        self.assertEqual(graphs["int8"]["category"], "INT8 weights/activations")
        self.assertEqual(graphs["fp16"]["quantized_tensor_count"], 0)
        self.assertGreater(graphs["int8"]["quantized_tensor_count"], 0)
        self.assertLess(graphs["int8"]["bytes"], graphs["fp16"]["bytes"])
        self.assertTrue(graphs["int8"]["scale_zero_point_checks"]["all_scales_finite_positive"])

    def test_detector_members_are_fp16(self):
        import zipfile
        with zipfile.ZipFile(ROOT / PINS["assets"]["detector"]["path"]) as archive:
            for info in archive.infolist():
                member = audit.inspect_graph(archive.read(info.filename), info.filename)
                self.assertEqual(member["category"], "FP16", info.filename)
                self.assertEqual(member["quantized_tensor_count"], 0, info.filename)

    def test_interpreter_io_matches_frozen_contract(self):
        for name in ("fp16", "int8"):
            io = audit.interpreter_io(ROOT / PINS["assets"][name]["path"])
            self.assertEqual(io["inputs"][0]["shape"], [1, 24, 255], name)
            self.assertEqual(io["outputs"][0]["shape"], [1, 100], name)
            self.assertEqual(io["inputs"][0]["dtype"], "float32", name)
            self.assertEqual(io["outputs"][0]["dtype"], "float32", name)
            self.assertEqual(io["inputs"][0]["scale"], 0.0)
            self.assertEqual(io["outputs"][0]["scale"], 0.0)


class TestFeatureAndLabelContract(unittest.TestCase):
    def test_ordered_labels_contiguous_unique_uppercase(self):
        labels = audit.ordered_labels()
        self.assertEqual(len(labels), 100)
        self.assertEqual(len(set(labels)), 100)
        self.assertTrue(all(label and label == label.upper() for label in labels))

    def test_feature_contract_pinned(self):
        contract = audit.feature_contract()
        self.assertEqual(contract["pinned_sha256"], PINS["frozen_contract"]["feature_sha256"])

    def test_missing_hand_feature_probe(self):
        contract = audit.feature_contract()
        self.assertTrue(contract["missing_hand_probe"]["finite"])
        # The mask is not applied to features before inference: an all-missing
        # frame still carries lagged velocity/acceleration from prior frames.
        self.assertEqual(contract["missing_hand_probe"]["missing_frame_rows_are_all_zero"], False)


class TestExistingEvidence(unittest.TestCase):
    def test_paired_evidence_verifies(self):
        evidence = audit.verify_existing_pair()
        self.assertTrue(all(evidence["checks"].values()), evidence["checks"])
        self.assertEqual(evidence["n"], 346)
        self.assertGreater(evidence["fp16_top1"]["top1_correct"], evidence["int8_top1"]["top1_correct"])


if __name__ == "__main__":
    unittest.main(verbosity=2)