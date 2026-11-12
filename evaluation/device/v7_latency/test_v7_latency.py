"""Contract tests for v7_latency."""
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v7_candidates as cand


class TestV7Contracts(unittest.TestCase):
    def test_constants(self):
        self.assertEqual(cand.MAX_FRAMES, 32)
        self.assertEqual(cand.FRAME_SIZE, 256)
        self.assertEqual(cand.MODEL_FRAMES, 24)
        self.assertEqual(cand.MODEL_WIDTH, 255)
        self.assertEqual(cand.STRATEGIES,
                         ("baseline", "parallel_2", "parallel_4", "video_cached", "video_fresh"))

    def test_invalid_strategy(self):
        with self.assertRaises(ValueError):
            cand.evaluate_clip("/nonexistent.mp4", "unknown_strategy")

    def test_parallel_workers_bounded(self):
        self.assertIn("parallel_2", cand.STRATEGIES)
        self.assertIn("parallel_4", cand.STRATEGIES)
        # No other worker levels allowed through strategy names
        self.assertNotIn("parallel_8", cand.STRATEGIES)

    def test_evaluate_rejects_missing_file(self):
        with self.assertRaises((ValueError, RuntimeError, Exception)):
            cand.evaluate_clip("/nonexistent/path.mp4", "baseline")


if __name__ == "__main__":
    unittest.main()
