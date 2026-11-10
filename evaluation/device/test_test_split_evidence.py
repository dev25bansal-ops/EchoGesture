"""Verifies the in-repo held-out test-split evidence is reproducible.

The test predictions are frozen artifacts (no model run here); this test
pins the recomputed held-out numbers so a later edit to the predictions,
the split, or the vocabulary that changes them fails loudly instead of
silently drifting the headline submission metric.
"""

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_split_evidence import score  # noqa: E402

PREDICTIONS = ROOT / "evaluation/device/v5_accuracy/results/test_split/predictions.jsonl"
SPLIT = ROOT / "evaluation/device/v5_accuracy/splits/test.csv"
LABELS = ROOT / "models/landmark_labels.json"


@unittest.skipUnless(PREDICTIONS.is_file(), "frozen test predictions not present")
class HeldOutTestEvidenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = score(PREDICTIONS, SPLIT, LABELS)

    def test_all_predictions_match_a_ground_truth_row(self):
        self.assertEqual(self.result["predictions"], 1249)
        self.assertEqual(self.result["matched"], 1249)
        self.assertEqual(self.result["unmatched"], 0)

    def test_held_out_top1_is_62_61_percent(self):
        self.assertEqual(self.result["top1_correct"], 782)
        self.assertEqual(self.result["top1_pct"], 62.61)

    def test_held_out_top5_is_80_46_percent(self):
        self.assertEqual(self.result["top5_correct"], 1005)
        self.assertEqual(self.result["top5_pct"], 80.46)

    def test_eleven_unseen_signers(self):
        self.assertEqual(self.result["signers"], 11)
        self.assertEqual(self.result["per_signer_mean_pct"], 63.36)
        self.assertEqual(self.result["per_signer_min_pct"], 45.77)
        self.assertEqual(self.result["per_signer_max_pct"], 80.41)

    def test_below_the_stated_internal_target(self):
        # The honest headline is below target; this pins that fact so no one
        # later reports the optimistic validation rate as the held-out score.
        self.assertLess(self.result["top1_pct"], 70)


if __name__ == "__main__":
    unittest.main()