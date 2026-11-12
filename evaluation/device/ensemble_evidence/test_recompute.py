"""Pin the ensemble's held-out accuracy so drift fails loudly.

The predictions file is a frozen artifact -- no model is run here. If a later
edit to the predictions, the split CSV, or the vocabulary changes these
numbers, this test fails instead of the submission claim silently moving.

Run:
    .venv/bin/python -I -m unittest evaluation/device/ensemble_evidence/test_recompute.py -v
"""

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

from recompute import score  # noqa: E402

FROZEN = HERE / "predictions_ensemble.jsonl"
SPLIT = ROOT / "evaluation/device/v5_accuracy/splits/test.csv"
LABELS = ROOT / "models/landmark_labels.json"


@unittest.skipUnless(FROZEN.is_file(), "frozen ensemble predictions not present")
class EnsembleAccuracyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = score(FROZEN, SPLIT, LABELS)

    def test_scored_population(self):
        self.assertEqual(self.r["predictions"], 1249)
        self.assertEqual(self.r["scored_clips"], 1249)
        self.assertEqual(self.r["unmatched_predictions"], 0)

    def test_top1_matches_dev_claim(self):
        self.assertEqual(self.r["top1_correct"], 909)
        self.assertEqual(self.r["top1_incorrect_scored"], 340)
        self.assertEqual(self.r["top1_pct_scored"], 72.78)

    def test_top5_matches_dev_claim(self):
        self.assertEqual(self.r["top5_correct"], 1093)
        self.assertEqual(self.r["top5_pct_scored"], 87.51)

    def test_eleven_signers(self):
        self.assertEqual(self.r["signers"], 11)

    def test_per_signer_spread(self):
        self.assertEqual(self.r["per_signer_mean_pct"], 73.36)
        self.assertEqual(self.r["per_signer_min_pct"], 53.52)
        self.assertEqual(self.r["per_signer_max_pct"], 87.63)

    def test_denominator_gap_is_disclosed(self):
        # 6 eligible clips produced no prediction; the stricter denominator
        # treats them as incorrect and is therefore lower.
        self.assertEqual(self.r["eligible_clips"], 1255)
        self.assertEqual(self.r["unscored_eligible_clips"], 6)
        self.assertEqual(self.r["top1_pct_all_eligible"], 72.43)
        self.assertEqual(self.r["top5_pct_all_eligible"], 87.09)
        self.assertLess(self.r["top1_pct_all_eligible"], self.r["top1_pct_scored"])

    def test_beats_previous_single_model(self):
        # Prior single-model held-out figure was 62.61% on the same 1249 clips.
        self.assertGreater(self.r["top1_pct_scored"], 62.61)


@unittest.skipUnless((HERE / "predictions_ensemble_val.jsonl").is_file(),
                     "frozen validation predictions not present")
class EnsembleValidationAccuracyTest(unittest.TestCase):
    """Pins the validation figure and the denominator used for it."""

    @classmethod
    def setUpClass(cls):
        cls.r = score(HERE / "predictions_ensemble_val.jsonl",
                      ROOT / "evaluation/device/v5_accuracy/splits/val.csv",
                      LABELS)

    def test_validation_top1_is_75_36(self):
        self.assertEqual(self.r["top1_correct"], 260)
        self.assertEqual(self.r["scored_clips"], 345)
        self.assertEqual(self.r["top1_pct_scored"], 75.36)

    def test_validation_top5(self):
        self.assertEqual(self.r["top5_pct_scored"], 87.25)

    def test_one_unscored_eligible_clip(self):
        self.assertEqual(self.r["eligible_clips"], 346)
        self.assertEqual(self.r["unscored_eligible_clips"], 1)

    def test_validation_is_optimistic_vs_test(self):
        # Members were selected on validation, so val must exceed test (72.78%).
        self.assertGreater(self.r["top1_pct_scored"], 72.78)

if __name__ == "__main__":
    unittest.main()
