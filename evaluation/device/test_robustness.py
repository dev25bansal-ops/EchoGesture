import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from corruptions import CorruptionSpec, corrupt_frames, decode_video, write_video
from robustness import paired_evaluate
from robustness_lab import build_config, config_hash, derive_clip_seed, aggregate_pairs, wilson_interval
from robustness_lab.reporting import render_markdown, pairs_to_csv


def _make_items(root, n, signers=None):
    items = []
    for index in range(n):
        path = root / f"{index}.mp4"
        write_video(path, [np.full((20, 20, 3), (index * 37) % 256, np.uint8)] * 3, 10)
        signer = (signers[index] if signers else f"S{index % 2}")
        items.append((f"clip {index}", path, "A" if index % 2 == 0 else "B", signer))
    return items


def _fake_predict(corrupt_pred=None):
    def predict(path, *, clip_id=None):
        corrupted = Path(path).parent.name == "work"
        if corrupted and corrupt_pred is not None:
            prediction = corrupt_pred
        else:
            prediction = "A" if clip_id == "clip 0" else "B"
        top5 = ["A", "B"] if prediction == "A" else ["B", "A"]
        return {"clip_id": clip_id, "prediction": prediction, "top5": top5,
                "confidence": 0.8, "latency_ms": 1}
    return predict


class ConfigTest(unittest.TestCase):
    def test_default_severity(self):
        for kind, expected in (("occlusion", 0.5), ("low_light", 0.5), ("motion_blur", 0.5),
                               ("crop", 0.3), ("frame_drop", 0.5)):
            self.assertEqual(build_config(kind).severity, expected)

    def test_unknown_corruption_rejected(self):
        with self.assertRaises(ValueError):
            build_config("noise")

    def test_severity_bounds(self):
        for bad in (0, -0.1, 1.01):
            with self.assertRaises(ValueError):
                build_config("crop", severity=bad)
        build_config("crop", severity=1.0)

    def test_hash_stable_and_seed_sensitive(self):
        a = build_config("crop", seed=42)
        b = build_config("crop", seed=42)
        c = build_config("crop", seed=43)
        self.assertEqual(config_hash(a), config_hash(b))
        self.assertNotEqual(config_hash(a), config_hash(c))
        self.assertEqual(len(config_hash(a)), 64)

    def test_derive_clip_seed_deterministic_and_position_sensitive(self):
        s1 = derive_clip_seed(7, 0, "clip-0")
        s2 = derive_clip_seed(7, 0, "clip-0")
        s3 = derive_clip_seed(7, 1, "clip-0")
        s4 = derive_clip_seed(7, 0, "clip-1")
        self.assertEqual(s1, s2)
        self.assertNotEqual(s1, s3)
        self.assertNotEqual(s1, s4)


class StatsTest(unittest.TestCase):
    def test_wilson_zero_trials(self):
        self.assertEqual(wilson_interval(0, 0), (None, None))

    def test_wilson_perfect_score(self):
        lo, hi = wilson_interval(10, 10)
        self.assertGreater(lo, 0.6)
        self.assertLessEqual(hi, 1.0)

    def test_wilson_wider_for_small_n(self):
        lo10, hi10 = wilson_interval(5, 10)
        lo100, hi100 = wilson_interval(50, 100)
        self.assertGreater(hi10 - lo10, hi100 - lo100)

    def test_aggregate_empty(self):
        agg = aggregate_pairs([])
        self.assertEqual(agg["sample_count"], 0)
        self.assertIsNone(agg["clean_top1"])

    def test_aggregate_flips_and_failures(self):
        pairs = [
            {"clip_id": "a", "signer_id": "S1", "ground_truth": "A",
             "clean_prediction": "A", "corrupted_prediction": "B",
             "clean_correct": True, "corrupted_correct": False},
            {"clip_id": "b", "signer_id": "S1", "ground_truth": "B",
             "clean_prediction": None, "corrupted_prediction": "B",
             "clean_correct": False, "corrupted_correct": True},
            {"clip_id": "c", "signer_id": "S2", "ground_truth": "A",
             "clean_prediction": "A", "corrupted_prediction": "A",
             "clean_correct": True, "corrupted_correct": True},
        ]
        agg = aggregate_pairs(pairs)
        self.assertEqual(agg["sample_count"], 3)
        self.assertEqual(agg["paired_flip_count"], 1)
        self.assertEqual(agg["failed_pair_count"], 1)
        self.assertEqual(agg["failure_breakdown"].get("clean_prediction_failed"), 1)
        self.assertEqual(agg["signer_count"], 2)
        self.assertEqual(agg["per_signer"]["S1"]["sample_count"], 2)
        self.assertAlmostEqual(agg["accuracy_drop_percentage_points"], 0.0)


class PairedEvaluateTest(unittest.TestCase):
    def test_pairing_and_accuracy_arithmetic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            items = _make_items(root, 2)
            # Fake predictor is correct clean, always wrong on corrupted.
            def predict(path, *, clip_id=None):
                corrupted = Path(path).parent.name == "work"
                clean_pred = "A" if clip_id == "clip 0" else "B"
                prediction = "B" if corrupted else clean_pred
                top5 = ["A", "B"] if prediction == "A" else ["B", "A"]
                return {"clip_id": clip_id, "prediction": prediction, "top5": top5,
                        "confidence": 0.8, "latency_ms": 1}
            summary = paired_evaluate(items, predict, {"A", "B"},
                                      CorruptionSpec("low_light", 0.2, 0), root / "out")
            self.assertEqual(summary["sample_count"], 2)
            self.assertEqual(summary["clean_correct"], 2)
            self.assertEqual(summary["corrupted_correct"], 1)
            self.assertEqual(summary["accuracy_drop_percentage_points"], 50)
            self.assertEqual(summary["affected_signer_count"], 2)
            self.assertEqual(summary["signer_count"], 2)
            self.assertEqual(len(summary["pairs"]), 2)
            self.assertFalse((root / "out/work").exists())
            # Per-clip seeds recorded and distinct.
            seeds = [p["corruption_seed"] for p in summary["pairs"]]
            self.assertEqual(len(set(seeds)), 2)

    def test_failed_predictions_counted_not_dropped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            items = _make_items(root, 2)
            def predict(path, *, clip_id=None):
                if clip_id == "clip 1":
                    raise RuntimeError("boom")
                return {"clip_id": clip_id, "prediction": "A", "top5": ["A", "B"],
                        "confidence": 0.8, "latency_ms": 1}
            summary = paired_evaluate(items, predict, {"A", "B"},
                                      CorruptionSpec("low_light", 0.2, 0), root / "out")
            self.assertEqual(summary["sample_count"], 2)
            self.assertGreaterEqual(summary["failed_pair_count"], 1)
            self.assertTrue(any(p["clean_prediction"] is None or p["corrupted_prediction"] is None
                                for p in summary["pairs"]))

    def test_decode_failure_counted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "0.mp4"
            write_video(path, [np.full((20, 20, 3), 10, np.uint8)] * 3, 10)
            missing = root / "missing.mp4"  # never created
            items = [("clip 0", path, "A", "S1"), ("clip 1", missing, "B", "S1")]
            summary = paired_evaluate(items, _fake_predict(), {"A", "B"},
                                      CorruptionSpec("crop", 0.3, 0), root / "out")
            self.assertEqual(summary["sample_count"], 2)
            self.assertGreaterEqual(summary["failure_breakdown"].get("decode_error", 0), 1)

    def test_workdir_cleaned_on_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            items = _make_items(root, 1)
            def predict(path, *, clip_id=None):
                raise RuntimeError("always fails")
            summary = paired_evaluate(items, predict, {"A", "B"},
                                      CorruptionSpec("crop", 0.3, 0), root / "out")
            self.assertEqual(summary["failed_pair_count"], 1)
            self.assertFalse((root / "out/work").exists())

    def test_missing_labels_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(Exception, "ground_truth"):
                paired_evaluate([("x", root / "x.mp4", "", "")], lambda _: {}, set(),
                                CorruptionSpec("crop"), root / "out")

    def test_partial_labels_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            items = _make_items(root, 2)
            items[1] = (items[1][0], items[1][1], "", items[1][3])
            with self.assertRaisesRegex(Exception, "ground_truth"):
                paired_evaluate(items, _fake_predict(), {"A", "B"},
                                CorruptionSpec("crop"), root / "out")

    def test_not_representative_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            items = _make_items(root, 2)
            summary = paired_evaluate(items, _fake_predict(), {"A", "B"},
                                      CorruptionSpec("crop"), root / "out")
            self.assertFalse(summary["representative_robustness_measured"])


class RepresentativeGateTest(unittest.TestCase):
    def test_representative_requires_size_and_signers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            items = _make_items(root, 20)  # 2 signers
            summary = paired_evaluate(items, _fake_predict(), {"A", "B"},
                                      CorruptionSpec("low_light"), root / "out",
                                      representative=True)
            self.assertTrue(summary["representative_robustness_measured"])

    def test_single_signer_never_representative(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            items = _make_items(root, 20, signers=["S1"] * 20)
            summary = paired_evaluate(items, _fake_predict(), {"A", "B"},
                                      CorruptionSpec("low_light"), root / "out",
                                      representative=True)
            self.assertFalse(summary["representative_robustness_measured"])

    def test_failures_block_representative(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            items = _make_items(root, 20)
            def predict(path, *, clip_id=None):
                if clip_id == "clip 5":
                    raise RuntimeError("boom")
                return {"clip_id": clip_id, "prediction": "A", "top5": ["A", "B"],
                        "confidence": 0.8, "latency_ms": 1}
            summary = paired_evaluate(items, predict, {"A", "B"},
                                      CorruptionSpec("low_light"), root / "out",
                                      representative=True)
            self.assertFalse(summary["representative_robustness_measured"])


class CorruptionTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(5)
        self.frames = [rng.integers(0, 255, (40, 60, 3), dtype=np.uint8) for _ in range(10)]

    def test_all_corruptions_are_seeded(self):
        for kind in ("occlusion", "low_light", "motion_blur", "crop", "frame_drop"):
            spec = CorruptionSpec(kind, 0.5, 21)
            first = corrupt_frames(self.frames, spec)
            second = corrupt_frames(self.frames, spec)
            self.assertEqual(len(first), len(second))
            for left, right in zip(first, second):
                self.assertTrue(np.array_equal(left, right), kind)
            self.assertTrue(any(not np.array_equal(a, b) for a, b in zip(first, self.frames)), kind)

    def test_unknown_kind_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            CorruptionSpec("gaussian_noise", 0.5, 0)

    def test_empty_clip_rejected(self):
        with self.assertRaises(ValueError):
            corrupt_frames([], CorruptionSpec("crop"))

    def test_video_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.mp4"
            write_video(path, self.frames, 20)
            frames, fps = decode_video(path)
            self.assertEqual(len(frames), 10)
            self.assertEqual(fps, 20)
            self.assertEqual(frames[0].shape, self.frames[0].shape)


class ReportingTest(unittest.TestCase):
    def test_report_labels_transform_check_when_not_representative(self):
        summary = {
            "corruption_type": "low_light", "corruption_severity": 0.5, "seed": 0,
            "split": "unspecified", "representative_robustness_measured": False,
            "sample_count": 4, "clean_correct": 3, "corrupted_correct": 2,
            "clean_top1": 0.75, "corrupted_top1": 0.5,
            "accuracy_drop_percentage_points": 25.0, "paired_flip_count": 1,
            "paired_flip_rate": 0.25, "failed_pair_count": 0, "failure_breakdown": {},
            "clean_top1_ci95": [0.3, 0.95], "corrupted_top1_ci95": [0.15, 0.85],
            "per_signer": {"S1": {"sample_count": 4, "clean_top1": 0.75, "corrupted_top1": 0.5,
                                   "accuracy_drop_percentage_points": 25.0, "failed": 0}},
            "signer_count": 1, "prediction_agreement_rate": 0.75,
            "model_sha256": "abc", "manifest_sha256": "def",
        }
        md = render_markdown(summary)
        self.assertIn("TRANSFORM CHECK", md)
        self.assertIn("S1", md)

    def test_pairs_csv_has_header(self):
        csv_text = pairs_to_csv([{"clip_id": "c1", "signer_id": "S1", "ground_truth": "A",
                                  "clean_prediction": "A", "corrupted_prediction": "B",
                                  "clean_correct": True, "corrupted_correct": False}])
        lines = csv_text.strip().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertIn("clip_id", lines[0])


if __name__ == "__main__":
    unittest.main()
