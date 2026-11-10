import csv
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_manifest


class ManifestEvaluatorTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.manifest = self.root / "manifest.csv"
        self.clip = self.root / "clip.mp4"
        self.clip.write_bytes(b"local test fixture")
        self.labels = {"CANDY", "RIGHT"}
        self.write_manifest([("clip 01", "clip.mp4", "candy")])

    def tearDown(self):
        self.temp.cleanup()

    def write_manifest(self, rows):
        with self.manifest.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["clip_id", "path", "ground_truth"])
            writer.writerows(rows)

    def clip_with_id(self, clip_id):
        path = self.root / clip_id
        path.write_bytes(b"local test fixture")
        return str(path)

    def prediction(self, **updates):
        value = {
            "clip_id": "clip 01",
            "prediction": "CANDY",
            "top5": ["CANDY", "RIGHT"],
            "confidence": 0.9,
            "latency_ms": 10,
        }
        value.update(updates)
        return value

    def test_valid_manifest_and_accuracy(self):
        items = run_manifest.read_manifest(self.manifest)
        predictions, times, accuracy = run_manifest.evaluate(
            items, lambda _: self.prediction(), self.labels)
        self.assertEqual(predictions[0]["clip_id"], "clip 01")
        self.assertEqual(accuracy["clips_with_ground_truth"], 1)
        self.assertEqual(accuracy["correct"], 1)
        self.assertEqual(accuracy["top1"], 1.0)
        self.assertFalse(accuracy["representative_accuracy_measured"])
        self.assertEqual(len(times), 1)

    def test_duplicate_ids_rejected(self):
        self.write_manifest([("same", "clip.mp4", ""), ("same", "clip.mp4", "")])
        with self.assertRaisesRegex(run_manifest.EvaluationError, "duplicate"):
            run_manifest.read_manifest(self.manifest)

    def test_missing_clip_rejected(self):
        self.write_manifest([("clip 01", "missing.mp4", "")])
        with self.assertRaisesRegex(run_manifest.EvaluationError, "clip not found"):
            run_manifest.read_manifest(self.manifest)

    def test_empty_manifest_rejected(self):
        self.manifest.write_text("clip_id,path\n", encoding="utf-8")
        with self.assertRaisesRegex(run_manifest.EvaluationError, "no clips"):
            run_manifest.read_manifest(self.manifest)

    def test_missing_output_field_rejected(self):
        value = self.prediction()
        del value["confidence"]
        with self.assertRaisesRegex(run_manifest.EvaluationError, "confidence"):
            run_manifest.validate_prediction(value, "clip 01", self.labels)

    def test_nan_confidence_rejected(self):
        with self.assertRaisesRegex(run_manifest.EvaluationError, "confidence"):
            run_manifest.validate_prediction(self.prediction(confidence=float("nan")), "clip 01", self.labels)

    def test_negative_latency_rejected(self):
        with self.assertRaisesRegex(run_manifest.EvaluationError, "latency"):
            run_manifest.validate_prediction(self.prediction(latency_ms=-1), "clip 01", self.labels)

    def test_lowercase_prediction_rejected(self):
        with self.assertRaisesRegex(run_manifest.EvaluationError, "uppercase"):
            run_manifest.validate_prediction(self.prediction(prediction="candy"), "clip 01", self.labels)

    def test_wrong_id_rejected(self):
        with self.assertRaisesRegex(run_manifest.EvaluationError, "returned clip_id"):
            run_manifest.validate_prediction(self.prediction(clip_id="other"), "clip 01", self.labels)

    def test_partial_ground_truth_rejected(self):
        self.write_manifest([
            ("clip 01", "clip.mp4", "CANDY"),
            ("clip 02", self.clip_with_id("clip 02"), ""),
        ])
        ids = iter(["clip 01", "clip 02"])
        with self.assertRaisesRegex(run_manifest.EvaluationError, "every clip or none"):
            run_manifest.evaluate(
                run_manifest.read_manifest(self.manifest),
                lambda _: self.prediction(clip_id=next(ids)), self.labels)

    def test_predictor_failure_aborts_run(self):
        def fail(_):
            raise ValueError("unreadable clip")
        with self.assertRaisesRegex(run_manifest.EvaluationError, "predict.*failed"):
            run_manifest.evaluate(run_manifest.read_manifest(self.manifest), fail, self.labels)

    def test_file_import_loads_actual_predictor(self):
        source = self.root / "predictor.py"
        source.write_text("def predict(video_path):\n    return {'clip_id': Path(video_path).stem}\n", encoding="utf-8")
        with self.assertRaisesRegex(NameError, "Path"):
            run_manifest.load_predictor(source)("clip.mp4")

    def test_no_ground_truth_means_no_accuracy(self):
        self.write_manifest([("clip 01", "clip.mp4", "")])
        _, _, accuracy = run_manifest.evaluate(
            run_manifest.read_manifest(self.manifest), lambda _: self.prediction(), self.labels)
        self.assertIsNone(accuracy["top1"])

    def test_nearest_rank_percentiles(self):
        self.assertEqual(run_manifest.percentile([1, 2, 3, 4], 0.95), 4)
        self.assertEqual(run_manifest.percentile([1, 2, 3, 4], 0.50), 2)
        self.assertIsNone(run_manifest.percentile([], 0.95))

    def test_optional_clip_id_preserves_arbitrary_manifest_ids(self):
        def predict(_, *, clip_id=None):
            return self.prediction(clip_id=clip_id)

        self.write_manifest([("clip 01", "clip.mp4", "CANDY")])
        predictions, _, _ = run_manifest.evaluate(
            run_manifest.read_manifest(self.manifest), predict, self.labels)
        self.assertEqual(predictions[0]["clip_id"], "clip 01")

    def test_positional_clip_id_parameter_is_not_used(self):
        self.assertFalse(run_manifest._accepts_clip_id(lambda _, clip_id=None: None))

    def test_cli_rejects_invalid_labels_and_does_not_write_outputs(self):
        for name, content in (("model.tflite", b"model"), ("labels.json", b"[]"),
                              ("predictor.py", b"def predict(path): pass\n")):
            (self.root / name).write_bytes(content)
        output = self.root / "output"
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as result:
            run_manifest.main([
                str(self.manifest), "--output-dir", str(output),
                "--predictor", str(self.root / "predictor.py"),
                "--labels", str(self.root / "labels.json"),
                "--model", str(self.root / "model.tflite"),
            ])
        self.assertEqual(result.exception.code, 2)
        self.assertFalse((output / "predictions.jsonl").exists())
        self.assertFalse((output / "benchmark.json").exists())

    def test_manifest_ids_keep_outer_whitespace(self):
        self.write_manifest([(" clip 01 ", "clip.mp4", "CANDY")])
        self.assertEqual(run_manifest.read_manifest(self.manifest)[0][0], " clip 01 ")

    def test_profiled_predictions_capture_stages(self):
        source = self.root / "profile_predictor.py"
        source.write_text(
            "def predict(_, *, clip_id=None, profile=False):\n"
            "    return {'clip_id': clip_id, 'prediction': 'CANDY', 'top5': ['CANDY', 'RIGHT'], 'confidence': .9, 'latency_ms': 1}\n"
            "def get_last_profile():\n"
            "    return {'stages': {'model_inference': {'milliseconds': 1}}}\n")
        predict = run_manifest.load_predictor(source)
        predictions, times, _, details = run_manifest.evaluate_profiled(
            run_manifest.read_manifest(self.manifest), predict, self.labels)
        self.assertEqual(predictions[0]["clip_id"], "clip 01")
        self.assertEqual(details["profiles"][0]["stages"]["model_inference"]["milliseconds"], 1)
        self.assertEqual(details["failures"], [])

    def test_profiled_failures_are_explicit(self):
        import types
        predictor = types.SimpleNamespace(__module__="fake_profile_failure")
        predictor.get_last_profile = lambda: {"stages": {}}
        sys.modules[predictor.__module__] = predictor

        def predict(_, *, profile=False):
            raise ValueError("No hand detected")

        predictions, _, accuracy, details = run_manifest.evaluate_profiled(
            run_manifest.read_manifest(self.manifest), predict, self.labels)
        self.assertEqual(predictions, [])
        self.assertIsNone(accuracy["top1"])
        self.assertEqual(details["failures"][0]["error_type"], "ValueError")

    def test_unicode_clip_id_preserved(self):
        self.write_manifest([("sign-✋-01", "clip.mp4", "")])
        items = run_manifest.read_manifest(self.manifest)
        predictions, _, _ = run_manifest.evaluate(
            items, lambda _, *, clip_id=None: self.prediction(clip_id=clip_id), self.labels)
        self.assertEqual(predictions[0]["clip_id"], "sign-✋-01")

    @patch("run_manifest.time.perf_counter_ns", side_effect=[1_000_000_000, 1_012_000_000, 2_000_000_000, 2_020_000_000])
    def test_cold_and_warm_calls_are_distinct(self, _):
        self.write_manifest([
            ("clip 01", "clip.mp4", ""),
            ("clip 02", self.clip_with_id("clip 02"), ""),
        ])
        ids = iter(["clip 01", "clip 02"])
        predictions, times, _ = run_manifest.evaluate(
            run_manifest.read_manifest(self.manifest),
            lambda _: self.prediction(clip_id=next(ids)), self.labels)
        self.assertEqual(len(times), 2)
        self.assertAlmostEqual(times[0], 12.0)
        self.assertAlmostEqual(times[1], 20.0)
        self.assertAlmostEqual(run_manifest.timings(times[:1])["p95_ms"], 12.0)
        self.assertAlmostEqual(run_manifest.timings(times[1:])["p95_ms"], 20.0)
        self.assertEqual(len(predictions), 2)


if __name__ == "__main__":
    unittest.main()
