"""Unit tests for the ensemble robustness aggregation math (no inference).

Run with:

    .venv/bin/python -m unittest evaluation/device/robustness_ensemble/test_robustness_ensemble.py

These tests exercise only the pure functions in
``run_ensemble_robustness`` (:func:`accuracy`,
:func:`percentage_point_drop`, :func:`failure_stage`, :func:`summarise`) on
synthetic numbers.  They never import the model or touch video.
"""

import importlib.util
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parent / "run_ensemble_robustness.py"
_spec = importlib.util.spec_from_file_location("ens_robustness", MODULE_PATH)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def row(stage):
    """One synthetic per-clip record as run_condition emits it."""
    return {
        "clip_id": "x",
        "truth": "A",
        "prediction": "A" if stage == "correct" else None,
        "correct": stage == "correct",
        "stage": stage,
        "error": "ValueError: No hand detected" if stage == "no_hand" else None,
    }


class AggregationMath(unittest.TestCase):
    def test_accuracy_empty_denominator_is_none(self):
        self.assertIsNone(mod.accuracy(0, 0))
        self.assertIsNone(mod.accuracy(3, 0))

    def test_accuracy_basic(self):
        self.assertAlmostEqual(mod.accuracy(22, 30), 22 / 30)

    def test_drop_zero_when_identical(self):
        self.assertEqual(mod.percentage_point_drop(22, 22, 30), 0.0)

    def test_drop_is_percentage_points_not_fraction(self):
        # 22/30 -> 21/30 is 3.33 percentage points, not 0.033.
        self.assertAlmostEqual(mod.percentage_point_drop(22, 21, 30), 100 / 30)

    def test_drop_counts_failures_wrong_via_shared_denominator(self):
        self.assertAlmostEqual(mod.percentage_point_drop(5, 1, 20), 20.0)

    def test_drop_negative_when_corruption_helped(self):
        self.assertAlmostEqual(mod.percentage_point_drop(10, 12, 20), -10.0)

    def test_drop_empty_denominator_is_none(self):
        self.assertIsNone(mod.percentage_point_drop(0, 0, 0))


class FailureStage(unittest.TestCase):
    CASES = [
        ("A", None, "A", "correct"),
        ("B", None, "A", "wrong_prediction"),
        (None, "ValueError: No hand detected in any frame of: x.mp4", "A", "no_hand"),
        (None, "RuntimeError: Could not open video", "A", "error"),
        (None, None, "A", "error"),
    ]

    def test_stage_classification(self):
        for prediction, error, truth, expected in self.CASES:
            with self.subTest(prediction=prediction, error=error):
                self.assertEqual(mod.failure_stage(prediction, error, truth), expected)


class Summarise(unittest.TestCase):
    def test_clean_all_correct(self):
        summary = mod.summarise([row("correct")] * 30, "clean")
        self.assertEqual(summary["n_eligible"], 30)
        self.assertEqual(summary["n_scored"], 30)
        self.assertAlmostEqual(summary["accuracy_scored"], 1.0)
        self.assertAlmostEqual(summary["accuracy_eligible"], 1.0)
        self.assertEqual(summary["n_no_hand"], 0)
        self.assertEqual(summary["n_wrong"], 0)
        self.assertEqual(summary["n_error"], 0)

    def test_no_hand_shrinks_scored_not_eligible(self):
        # 20 correct, 2 wrong, 8 detection failures.
        rows = [row("correct")] * 20 + [row("wrong_prediction")] * 2 + [row("no_hand")] * 8
        summary = mod.summarise(rows, "motion_blur")
        self.assertEqual(summary["n_eligible"], 30)
        self.assertEqual(summary["n_scored"], 22)       # 8 detection failures excluded
        self.assertEqual(summary["n_correct"], 20)
        self.assertEqual(summary["n_no_hand"], 8)
        self.assertAlmostEqual(summary["accuracy_scored"], 20 / 22)
        self.assertAlmostEqual(summary["accuracy_eligible"], 20 / 30)

    def test_separates_no_hand_from_wrong_prediction(self):
        rows = [row("correct")] * 10 + [row("wrong_prediction")] * 3 + [row("no_hand")] * 2
        summary = mod.summarise(rows, "occlusion")
        self.assertEqual(summary["n_wrong"], 3)
        self.assertEqual(summary["n_no_hand"], 2)
        self.assertEqual(summary["n_error"], 0)
        # Both kinds of failure reduce eligible accuracy; only wrong ones
        # survive in the scored denominator.
        self.assertAlmostEqual(summary["accuracy_scored"], 10 / 13)

    def test_all_failed_is_zero_eligible_and_no_scored_accuracy(self):
        summary = mod.summarise([row("no_hand")] * 5, "low_light")
        self.assertEqual(summary["n_scored"], 0)
        self.assertIsNone(summary["accuracy_scored"])
        self.assertAlmostEqual(summary["accuracy_eligible"], 0.0)

    def test_report_arithmetic_reference_numbers(self):
        # clean 22/30, corrupted 21/30 with one detection failure counted wrong.
        clean = [row("correct")] * 22 + [row("wrong_prediction")] * 8
        corrupted = [row("correct")] * 21 + [row("wrong_prediction")] * 8 + [row("no_hand")] * 1
        clean_sum = mod.summarise(clean, "clean")
        corr_sum = mod.summarise(corrupted, "frame_drop")
        self.assertAlmostEqual(
            mod.percentage_point_drop(
                clean_sum["n_correct"], corr_sum["n_correct"], corr_sum["n_eligible"]),
            100 * (22 - 21) / 30)
        self.assertAlmostEqual(corr_sum["accuracy_eligible"], 21 / 30)
        # 21 correct out of 29 scored because one clip produced no hand at all.
        self.assertAlmostEqual(corr_sum["accuracy_scored"], 21 / 29)
        self.assertEqual(corr_sum["n_no_hand"], 1)


if __name__ == "__main__":
    unittest.main()