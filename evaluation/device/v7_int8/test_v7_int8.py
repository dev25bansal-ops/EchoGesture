#!/usr/bin/env python3
"""Targeted V6 INT8 qualification and paired-analysis tests.

These tests use the pinned artifacts and delivered JSON evidence.  They do not
call frontend/model inference and therefore are not a substitute for the
measurements in run_v7_int8.py, audit_tflite.py, and benchmark.py.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUNNER = HERE / "run_v7_int8.py"
AUDIT = HERE / "audit_tflite.py"
RESULTS = HERE / "results"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class TestV7GraphAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit = load_json(HERE / "tflite_audit.json")

    def test_pinned_hashes_and_categories(self):
        self.assertEqual(
            self.audit["classifiers"]["fp16"]["sha256"],
            "a012f2b5f7a1110961079f19eb0e28c9abacb95e4d0787352e4157ebca7044a4",
        )
        self.assertEqual(
            self.audit["classifiers"]["int8"]["sha256"],
            "9d24640a23244a3cd00c9fa62257733819f86ed1da5a10194c9b1cb64323700b",
        )
        self.assertEqual(self.audit["classifiers"]["fp16"]["strict_category"], "FP16")
        self.assertEqual(
            self.audit["classifiers"]["int8"]["strict_category"],
            "INT8 weights/activations with float32 IO",
        )
        self.assertFalse(self.audit["verdict"]["classifier_full_int8_io"])
        self.assertFalse(self.audit["verdict"]["whole_pipeline_int8"])

    def test_int8_weight_dtype_and_external_io(self):
        int8 = self.audit["classifiers"]["int8"]
        self.assertEqual(int8["weight_dtype_hist"], {"INT32": 35, "INT8": 21})
        self.assertNotIn("FLOAT16", int8["weight_dtype_hist"])
        self.assertNotIn("FLOAT32", int8["weight_dtype_hist"])
        self.assertEqual(int8["interpreter_io"]["inputs"][0]["dtype"], "float32")
        self.assertEqual(int8["interpreter_io"]["outputs"][0]["dtype"], "float32")
        self.assertEqual(int8["interpreter_io"]["inputs"][0]["scale"], 0.0)
        self.assertEqual(int8["interpreter_io"]["outputs"][0]["scale"], 0.0)

    def test_all_quantization_scale_pairs_are_present(self):
        int8 = self.audit["classifiers"]["int8"]
        self.assertEqual(int8["quantized_tensor_count"], 117)
        self.assertEqual(len(int8["quantization_scale_pairs"]), 117)
        for pair in int8["quantization_scale_pairs"]:
            self.assertTrue(pair["scale"])
            self.assertEqual(len(pair["scale"]), len(pair["zero_point"]))
        self.assertEqual(
            sorted({pair["scale"][0] for pair in int8["quantization_scale_pairs"] if len(pair["scale"]) == 1 and pair["index"] == 57}),
            [0.5725842118263245],
        )
        self.assertIn(0, [zero_point for pair in int8["quantization_scale_pairs"] for zero_point in pair["zero_point"]])
        self.assertIn(-1, [zero_point for pair in int8["quantization_scale_pairs"] for zero_point in pair["zero_point"]])

    def test_exact_float_side_path_is_mapped(self):
        int8 = self.audit["classifiers"]["int8"]
        signatures = [
            (op["index"], op["op"], op["input_indices"], op["output_indices"], op["input_dtypes"], op["output_dtypes"])
            for op in int8["operators"]
            if op["index"] in (0, 63, 64)
        ]
        self.assertEqual(signatures[0][:4], (0, "QUANTIZE", [0], [57]))
        self.assertEqual(signatures[1][:4], (63, "ZEROS_LIKE", [119], [120]))
        self.assertEqual(signatures[2][:4], (64, "QUANTIZE", [120], [121]))
        self.assertEqual(len(int8["float_input_operators"]), 3)
        self.assertTrue(
            all(
                op["op"] in {"QUANTIZE", "ZEROS_LIKE"}
                for op in int8["float_input_operators"]
            )
        )

    def test_detector_remains_fp16(self):
        detector = self.audit["detector"]
        self.assertEqual(detector["strict_category"], "FP16")
        self.assertEqual(set(detector["members"]), {"hand_detector.tflite", "hand_landmarks_detector.tflite"})
        for member in detector["members"].values():
            self.assertEqual(member["weight_dtype_hist"]["FLOAT16"], int(member["weight_dtype_hist"].get("FLOAT16", 0)))
            self.assertEqual(member["quantized_tensor_count"], 0)


class TestV7RunnerHelpers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load_module("v7_runner", RUNNER)

    def test_softmax_and_rank_use_complete_logits(self):
        import numpy as np

        # Fake 100-logit output matching the real [1,100] contract.
        logits = np.zeros(100, dtype=np.float32)
        logits[0] = 0.1
        logits[1] = 0.5
        logits[2] = -1.0
        probs = self.runner.softmax(logits)
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=12)
        parsed = self.runner.run_and_parse(
            type(
                "FakeInterpreter",
                (),
                {
                    "get_input_details": lambda self: [{"index": 0}],
                    "get_output_details": lambda self: [{"index": 1}],
                    "set_tensor": lambda self, *_: None,
                    "invoke": lambda self: None,
                    "get_tensor": lambda self, *_: logits[None, ...],
                },
            )(),
            np.zeros((1, 24, 255), dtype=np.float32),
            ["L" + str(i) for i in range(100)],
        )
        self.assertEqual(parsed["prediction"], "L1")
        self.assertGreater(parsed["margin_top1_top2"], 0.0)
        # rank_of itself works on arbitrary-length vectors.
        short = np.array([0.1, 0.5, -1.0], dtype=np.float32)
        rank = self.runner.rank_of(short, "A", ["A", "B", "C"])
        self.assertEqual(rank, 2)
        self.assertEqual(self.runner.rank_of(short, "C", ["A", "B", "C"]), 3)

    def test_category_and_hand_bands(self):
        ok = {"fp16_correct": True, "int8_correct": False}
        self.assertEqual(self.runner.category(ok), "fp16_only_correct")
        for frames, expected_band in (
            (0, "no_frames_either_hand"),
            (1, "1_to_11_frames"),
            (15, "12_to_18_frames"),
            (20, "19_to_24_frames"),
            (32, "25_to_32_frames"),
        ):
            row = {
                "frames_with_hands": frames,
                "hand_detection_rate": frames / 32,
                "fp16_correct": False,
                "int8_correct": False,
                "paired_category": "both_wrong",
                "fp16": {"prediction": "A", "top1_prob": 0.1, "margin_top1_top2": 0.01},
                "int8": {"prediction": "A", "top1_prob": 0.1, "margin_top1_top2": 0.01},
                "logit_l2_diff": 1.0,
                "prob_l1_diff": 0.1,
            }
            bands = self.runner.hand_bands([row])
            self.assertIn(expected_band, bands)


class TestV7PairedEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = load_json(RESULTS / "summary.json")
        cls.records = load_json(RESULTS / "predictions.json")
        cls.benchmark = load_json(HERE / "benchmark.json")

    def test_counts_are_exact(self):
        counts = self.summary["counts"]
        self.assertEqual(counts["fp16_top1_correct"], 272)
        self.assertEqual(counts["int8_top1_correct"], 254)
        self.assertEqual(counts["paired_categories"]["both_correct"], 242)
        self.assertEqual(counts["paired_categories"]["fp16_only_correct"], 30)
        self.assertEqual(counts["paired_categories"]["int8_only_correct"], 12)
        self.assertEqual(counts["paired_categories"]["both_wrong"], 61)
        self.assertEqual(counts["paired_categories"]["frontend_failed"], 1)
        self.assertAlmostEqual(counts["fp16_minus_int8_percentage_points"], 5.202312138728324, places=8)

    def test_records_cover_all_clips_and_identical_feature_inputs(self):
        self.assertEqual(len(self.records), 346)
        self.assertEqual(len({row["clip_id"] for row in self.records}), 346)
        hashes = [row["feature_sha256"] for row in self.records if row["status"] == "ok"]
        self.assertEqual(len(hashes), 345)
        self.assertEqual(len(set(hashes)), 345)
        for row in self.records:
            if row["status"] == "ok":
                self.assertEqual(len(row["fp16"]["logits"]), 100)
                self.assertEqual(len(row["int8"]["logits"]), 100)
                self.assertEqual(row["fp16"]["logits_sha256"], hashlib.sha256(
                    __import__("numpy").array(row["fp16"]["logits"], dtype="<f4").tobytes()
                ).hexdigest())

    def test_frontend_failure_and_cross_check(self):
        failed = [row for row in self.records if row["status"] != "ok"]
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0]["clip_id"], "5989539775823847-INTUITIVE")
        self.assertTrue(failed[0]["signer_id"].startswith(("signer_", "P")))
        self.assertIn("No hand detected", failed[0]["error"])
        self.assertTrue(self.summary["cross_check"]["all_predictions_and_top5_match_source"])

    def test_signer_and_gloss_keys_cover_all_observed_groups(self):
        # Signer keys are pseudonymized (signer_<hash>) in published evidence.
        self.assertEqual(len(self.summary["signer"]), 5)
        self.assertTrue(all(k.startswith("signer_") for k in self.summary["signer"]))
        self.assertEqual(sum(value["n"] for value in self.summary["gloss"].values()), 345)
        self.assertEqual(self.summary["gloss"]["CHEEK"]["fp16_only_correct"], 2)
        self.assertEqual(self.summary["gloss"]["CHEEK"]["int8_only_correct"], 0)

    def test_benchmark_footprint_and_sizes(self):
        footprint = self.benchmark["footprint"]
        self.assertEqual(footprint["classifier_saving_bytes"], 323168)
        self.assertEqual(footprint["fp16_plus_detector_bytes"], 8651993)
        self.assertEqual(footprint["int8_plus_detector_bytes"], 8328825)
        self.assertTrue(
            all(
                config["times_set_invoke_get_tensor"]["n"] == 300
                for config in self.benchmark["configs"].values()
            )
        )
        for key in ("fp16_threads1_AUTO", "int8_threads1_AUTO", "fp16_threads1_BUILTIN_WITHOUT_DEFAULT_DELEGATES", "int8_threads1_BUILTIN_WITHOUT_DEFAULT_DELEGATES"):
            self.assertIn(key, self.benchmark["configs"])


if __name__ == "__main__":
    unittest.main()
