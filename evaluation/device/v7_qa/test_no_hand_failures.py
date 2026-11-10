"""V7 QA: no-hand / predictor-failure accounting in profiled and paired runs.

Verified behavior:
  * evaluate_profiled() records a no-hand ValueError as a typed failure,
    drops that clip from predictions/wall_times/profiles, and nulls top1
    (failures make the run non-representative).
  * robustness.paired_evaluate() records a clean-leg no-hand failure as
    failed_pair_count >= 1 with failure_breakdown clean_prediction_failed,
    keeps the clip in the denominator, and removes the work/ directory.
  * robustness.predict_clip() never raises on a no-hand failure -- it
    returns {"prediction": None, "error": "..."}.
  * A clean-leg decode failure in paired evaluation is counted as
    decode_error in the breakdown, and work/ is still removed (D-2
    regression; current production already cleans via try/finally).
"""

import sys
import tempfile
import types
import unittest
from pathlib import Path

DEVICE_DIR = Path(__file__).resolve().parents[1]
ROOT = DEVICE_DIR.parents[1]
for _p in (str(DEVICE_DIR), str(ROOT / "scripts" / "release")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

import run_manifest  # noqa: E402
from corruptions import CorruptionSpec, write_video  # noqa: E402
from robustness import paired_evaluate, predict_clip  # noqa: E402


class _Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.labels = {"CANDY", "RIGHT"}

    def _clip(self, name):
        path = self.root / name
        path.write_bytes(b"local test fixture")
        return path

    def _video(self, name, frames=3):
        path = self.root / name
        write_video(path, [np.full((20, 20, 3), 128, np.uint8)] * frames, 10)
        return path

    def _manifest(self, rows):
        manifest = self.root / "manifest.csv"
        with manifest.open("w", encoding="utf-8") as stream:
            stream.write("clip_id,path,ground_truth\n")
            for clip_id, path, truth in rows:
                stream.write(f"{clip_id},{Path(path).name},{truth}\n")
        return manifest


class ProfiledNoHandTest(_Base):
    def _profiled_namespace(self):
        namespace = types.SimpleNamespace(__module__="v7_fake_profile")
        namespace.get_last_profile = lambda: {"stages": {"model_inference": {"milliseconds": 2}}}
        sys.modules[namespace.__module__] = namespace
        self.addCleanup(lambda: sys.modules.pop(namespace.__module__, None))
        return namespace

    def test_no_hand_failure_records_and_nulls_top1(self):
        namespace = self._profiled_namespace()
        manifest = self._manifest([
            ("clip 01", self._clip("c1.mp4"), "CANDY"),
            ("clip 02", self._clip("c2.mp4"), "CANDY"),
        ])

        def predict(path, *, clip_id=None, profile=False):
            if clip_id == "clip 02":
                raise ValueError("No hand detected in any sampled frame")
            return {"clip_id": clip_id, "prediction": "CANDY", "top5": ["CANDY", "RIGHT"],
                    "confidence": 0.9, "latency_ms": 1}

        predict.__module__ = namespace.__module__
        predictions, wall_times, accuracy, details = run_manifest.evaluate_profiled(
            run_manifest.read_manifest(manifest), predict, self.labels)

        self.assertEqual([p["clip_id"] for p in predictions], ["clip 01"])
        self.assertEqual(len(wall_times), 1)
        self.assertEqual(len(details["profiles"]), 1)
        self.assertIsNone(accuracy["top1"])
        self.assertFalse(accuracy["representative_accuracy_measured"])
        self.assertEqual(len(details["failures"]), 1)
        failure = details["failures"][0]
        self.assertEqual(failure["clip_id"], "clip 02")
        self.assertEqual(failure["error_type"], "ValueError")
        self.assertIn("No hand detected", failure["error"])

    def test_predict_clip_swallows_no_hand_error(self):
        def fail(path, *, clip_id=None):
            raise ValueError("No hand detected in any sampled frame")

        result = predict_clip("clip 01", self._clip("c.mp4"), fail, self.labels)
        self.assertIsNone(result["prediction"])
        self.assertIn("No hand detected", result["error"])
        self.assertIn("ValueError", result["error"])


class PairedNoHandTest(_Base):
    def test_clean_leg_failure_keeps_denominator_and_cleans_work(self):
        good = self._video("good.mp4")
        out = self.root / "out"

        def fail(path, *, clip_id=None):
            raise ValueError("No hand detected")

        summary = paired_evaluate(
            [("clip x", good, "CANDY", "s1")], fail, self.labels,
            CorruptionSpec("crop", 0.5, 0), out)
        self.assertEqual(summary["clean_correct"], 0)
        self.assertEqual(summary["corrupted_correct"], 0)
        self.assertEqual(summary["clean_top1"], 0.0)
        self.assertEqual(summary["failed_pair_count"], 1)
        self.assertEqual(summary["failure_breakdown"],
                         {"clean_prediction_failed": 1, "corrupted_prediction_failed": 1})
        self.assertEqual(summary["sample_count"], 1)
        self.assertFalse((out / "work").exists())

    def test_decode_failure_records_decode_error_and_cleans_work(self):
        good = self._video("good.mp4")
        broken = self.root / "broken.mp4"
        broken.write_bytes(b"not a real encoded video")
        out = self.root / "out"

        def predict(path, *, clip_id=None):
            return {"clip_id": clip_id, "prediction": "CANDY", "top5": ["CANDY"],
                    "confidence": 1.0, "latency_ms": 1}

        summary = paired_evaluate(
            [("good", good, "CANDY", "s1"), ("broken", broken, "CANDY", "s1")],
            predict, self.labels, CorruptionSpec("crop", 0.5, 0), out)
        self.assertEqual(summary["failure_breakdown"].get("decode_error"), 1)
        self.assertEqual(summary["failed_pair_count"], 1)
        self.assertFalse((out / "work").exists())

    def test_predictor_exception_on_one_leg_counts_as_failure(self):
        good = self._video("good.mp4")
        out = self.root / "out"

        def predict(path, *, clip_id=None):
            if Path(path).parent.name == "work":
                raise RuntimeError("simulated model fault")
            return {"clip_id": clip_id, "prediction": "CANDY", "top5": ["CANDY"],
                    "confidence": 0.9, "latency_ms": 1}

        summary = paired_evaluate(
            [("clip x", good, "CANDY", "s1")], predict, self.labels,
            CorruptionSpec("low_light", 0.5, 0), out)
        self.assertEqual(summary["clean_correct"], 1)
        self.assertEqual(summary["corrupted_correct"], 0)
        self.assertEqual(summary["accuracy_drop_percentage_points"], 100.0)
        self.assertEqual(summary["failure_breakdown"], {"corrupted_prediction_failed": 1})
        self.assertFalse((out / "work").exists())


if __name__ == "__main__":
    unittest.main()
