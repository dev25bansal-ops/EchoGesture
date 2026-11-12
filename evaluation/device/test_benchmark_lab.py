"""Tests for benchmark_lab matrix logic (no heavy inference)."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "benchmark_lab"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matrix  # noqa: E402
import run_manifest  # noqa: E402


def _fake_predict_factory(fail_on=None):
    calls = {"n": 0}

    def predict(video_path, *, clip_id=None, profile=False):
        calls["n"] += 1
        cid = clip_id or Path(video_path).stem
        if fail_on and cid in fail_on:
            raise ValueError(f"Synthetic failure for {cid}")
        # simulate stage profile via module attribute
        import types

        # attach to this function's module namespace lookup done in matrix via predict.__module__
        return {
            "clip_id": cid,
            "prediction": "CANDY",
            "top5": ["CANDY", "RIGHT"],
            "confidence": 0.9,
            "latency_ms": 5.0,
        }

    predict.calls = calls
    # matrix looks up get_last_profile on predict's module; install it here
    def get_last_profile():
        return {"stages": {
            "video_open_decode": {"milliseconds": 2.0},
            "landmark_tracking": {"milliseconds": 100.0},
            "feature_preparation": {"milliseconds": 0.3},
            "model_inference": {"milliseconds": 0.2},
            "output_postprocessing": {"milliseconds": 0.05},
            "other": {"milliseconds": 1.0, "frames": 24},
        }}

    mod = sys.modules.get(predict.__module__)
    mod.get_last_profile = get_last_profile
    return predict


class MatrixLogicTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.labels = {"CANDY", "RIGHT"}
        self.clip = self.root / "clip.mp4"
        self.clip.write_bytes(b"fake clip bytes")
        self.items = [("clip 01", self.clip, "CANDY")]

    def tearDown(self):
        self.temp.cleanup()

    def test_timing_summary_fields(self):
        s = matrix._timing_summary([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(s["count"], 4)
        self.assertEqual(s["p50_ms"], 2.0)
        self.assertEqual(s["p95_ms"], 4.0)
        self.assertIsNotNone(s["mean_ms"])
        self.assertIsNotNone(s["median_ms"])

    def test_cold_warm_split(self):
        cold, warm = matrix._split_cold_warm([10.0, 20.0, 30.0])
        self.assertEqual(cold, [10.0])
        self.assertEqual(warm, [20.0, 30.0])
        cold, warm = matrix._split_cold_warm([])
        self.assertEqual(cold, [])
        self.assertEqual(warm, [])

    def test_stage_aggregation(self):
        profiles = [{"clip_id": "a", "stages": {
            "landmark_tracking": {"milliseconds": 10.0},
            "model_inference": {"milliseconds": 1.0}}},
            {"clip_id": "b", "stages": {
                "landmark_tracking": {"milliseconds": 20.0},
                "model_inference": {"milliseconds": 3.0}}}]
        agg, extras = matrix._aggregate_stage_profiles(profiles)
        self.assertEqual(agg["landmark_tracking"]["count"], 2)
        self.assertAlmostEqual(agg["landmark_tracking"]["mean_ms"], 15.0)
        self.assertAlmostEqual(agg["model_inference"]["p95_ms"], 3.0)

    def test_run_cell_records_failures_separately(self):
        predict = _fake_predict_factory(fail_on={"clip 01"})
        # need real profile getter present; fake raises before returning so no profile
        cell = matrix.run_cell(
            self.items, predict, self.labels,
            matrix.MatrixCellConfig(name="t", repeats=1, warmup=0, profile=True),
            predict_module_name=predict.__module__)
        self.assertEqual(cell["successful_predictions"], 0)
        self.assertEqual(len(cell["failures"]), 1)
        self.assertEqual(cell["failures"][0]["error_type"], "ValueError")
        self.assertIsNone(cell["accuracy"]["top1"])

    def test_run_cell_success_and_stages(self):
        predict = _fake_predict_factory()
        cell = matrix.run_cell(
            self.items, predict, self.labels,
            matrix.MatrixCellConfig(name="t", repeats=1, warmup=0, profile=True),
            predict_module_name=predict.__module__)
        self.assertEqual(cell["successful_predictions"], 1)
        self.assertIn("landmark_tracking", cell["stage_aggregates"])
        self.assertGreater(cell["stage_aggregates"]["landmark_tracking"]["mean_ms"], 50.0)
        self.assertEqual(cell["accuracy"]["correct"], 1)

    def test_same_clip_repeats(self):
        predict = _fake_predict_factory()
        cell = matrix.run_cell(
            self.items, predict, self.labels,
            matrix.MatrixCellConfig(name="t", repeats=3, warmup=0, profile=False,
                                    same_clip_id="clip 01"),
            predict_module_name=predict.__module__)
        self.assertEqual(cell["successful_predictions"], 3)
        self.assertEqual(cell["config"]["repeat_mode"], "identical_clip")

    def test_build_summary_exact_fields(self):
        cell = matrix.run_cell(
            self.items, _fake_predict_factory(), self.labels,
            matrix.MatrixCellConfig(name="t", repeats=1, warmup=0, profile=True))
        manifest = self.root / "manifest.csv"
        manifest.write_text("clip_id,path\nclip 01,clip.mp4\n")
        summary = matrix.build_summary(
            [cell], manifest_path=manifest, artifact_paths={},
            fixture_type="real", split="validation",
            hardware_note="Mac laptop — NOT target device")
        for key in ("status", "representative_performance_measured", "cells",
                    "artifacts", "manifest", "code_commit", "environment",
                    "recorded_at", "fixture_type", "split", "dataset",
                    "hardware_is_target_device", "hardware_note", "power"):
            self.assertIn(key, summary)
        self.assertFalse(summary["hardware_is_target_device"])
        self.assertIn("NOT_MEASURED", summary["power"])
        self.assertIn("manifest", summary)
        self.assertIn("sha256", summary["manifest"])
        # latency fields
        lat = cell["latency_all_calls"]
        for k in ("count", "p50_ms", "p90_ms", "p95_ms", "p99_ms", "mean_ms",
                  "median_ms", "min_ms", "max_ms"):
            self.assertIn(k, lat)

    def test_stage_ranking_orders_by_mean(self):
        cell = matrix.run_cell(
            self.items, _fake_predict_factory(), self.labels,
            matrix.MatrixCellConfig(name="t", repeats=1, warmup=0, profile=True))
        manifest = self.root / "manifest.csv"
        manifest.write_text("clip_id,path\nclip 01,clip.mp4\n")
        summary = matrix.build_summary([cell], manifest_path=manifest,
                                       artifact_paths={}, fixture_type="real")
        ranking = matrix.stage_ranking(summary)
        self.assertGreater(len(ranking), 0)
        self.assertEqual(ranking[0][0], "landmark_tracking")


if __name__ == "__main__":
    unittest.main()
