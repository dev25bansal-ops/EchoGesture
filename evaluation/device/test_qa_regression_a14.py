"""A14 regression expansion for EchoGesture public contracts.

Coverage added (additive only; the 39 pre-existing tests are untouched):
 * repeated-path distinct exact clip IDs and Unicode/CSV-edge clip IDs
 * profiled runs with a no-hand predictor failure among successes
 * corrupt/empty/missing video behaviour at the decode layer
 * model/config rejection and stale-evidence rejection in release generation
 * release-path safety (zip traversal, symlink, absolute entries, size cap)
 * atomic output publication and temporary-file cleanup
 * paired robustness accounting when the predictor fails on clean/corrupt legs

Known production defects are exercised by narrow reproductions in this
directory (reproduce_repeated_load.py, reproduce_paired_cleanup.py); the
behaviour-pinning tests here assert the *intended* contract and will FAIL
while a defect is open, so they are deliberately not included here.
"""

import csv
import hashlib
import json
import stat
import sys
import tempfile
import types
import unittest
import zipfile
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from unittest import mock

import numpy as np

DEVICE_DIR = Path(__file__).resolve().parent
ROOT = DEVICE_DIR.parents[1]
sys.path.insert(0, str(DEVICE_DIR))
sys.path.insert(0, str(ROOT / "scripts" / "release"))

import run_manifest
from corruptions import CorruptionSpec, corrupt_frames, decode_video, write_video
from model_contract import CompatibilityError, validate_bundle
from robustness import paired_evaluate, predict_clip
import build_bundle as release_build
from generate_results import generate


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

class A14Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.clip = self.root / "clip.mp4"
        self.clip.write_bytes(b"local test fixture")
        self.labels = {"CANDY", "RIGHT"}

    def write_manifest(self, rows, header=("clip_id", "path", "ground_truth")):
        manifest = self.root / "manifest.csv"
        with manifest.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(rows)
        return manifest

    def clip_file(self, name):
        path = self.root / name
        path.write_bytes(b"local test fixture")
        return str(path)

    def prediction(self, clip_id="clip 01", **updates):
        value = {"clip_id": clip_id, "prediction": "CANDY", "top5": ["CANDY", "RIGHT"],
                 "confidence": 0.9, "latency_ms": 10}
        value.update(updates)
        return value

    def video(self, name, frames=3):
        path = self.root / name
        write_video(path, [np.full((20, 20, 3), 128, np.uint8)] * frames, 10)
        return path


# --------------------------------------------------------------------------
# repeated runs with distinct exact IDs
# --------------------------------------------------------------------------

class RepeatedPathDistinctIDsTest(A14Base):
    """Two consecutive evaluate() runs over the same manifest must each
    return their own exact, ordered clip IDs with no bleed-over."""

    def test_two_evaluate_runs_keep_distinct_exact_ids(self):
        # unicode, comma and whitespace-bearing IDs on the same path
        rows = [("clip 01", "clip.mp4", ""),
                ("sign-✋-01", self.clip_file("u.mp4"), ""),
                ("a,b", self.clip_file("comma.mp4"), ""),
                (" clip 01 ", self.clip_file("sp.mp4"), "")]
        self.write_manifest(rows)
        items = run_manifest.read_manifest(self.root / "manifest.csv")
        reversed_items = list(reversed(items))

        first, _, _ = run_manifest.evaluate(
            items, lambda _, *, clip_id=None: self.prediction(clip_id=clip_id), self.labels)
        second, _, _ = run_manifest.evaluate(
            reversed_items, lambda _, *, clip_id=None: self.prediction(clip_id=clip_id), self.labels)

        self.assertEqual([p["clip_id"] for p in first],
                         ["clip 01", "sign-✋-01", "a,b", " clip 01 "])
        self.assertEqual([p["clip_id"] for p in second],
                         [" clip 01 ", "a,b", "sign-✋-01", "clip 01"])
        self.assertEqual(len({p["clip_id"] for p in first}), 4)

    def test_comma_and_combining_clip_ids_roundtrip(self):
        self.write_manifest([("é", "clip.mp4", ""), ("标题，1", self.clip_file("c.mp4"), "")])
        items = run_manifest.read_manifest(self.root / "manifest.csv")
        predictions, _, _ = run_manifest.evaluate(
            items, lambda _, *, clip_id=None: self.prediction(clip_id=clip_id), self.labels)
        self.assertEqual([p["clip_id"] for p in predictions], ["é", "标题，1"])

    def test_ground_truth_lowercase_is_normalised(self):
        self.write_manifest([("clip 01", "clip.mp4", "candy")])
        items = run_manifest.read_manifest(self.root / "manifest.csv")
        self.assertEqual(items[0][2], "CANDY")
        _, _, accuracy = run_manifest.evaluate(
            items, lambda _: self.prediction(), self.labels)
        self.assertEqual(accuracy["correct"], 1)

    def test_signer_id_and_optional_columns_preserved(self):
        self.write_manifest(
            [("clip 01", "clip.mp4", "", "signer-7")],
            header=("clip_id", "path", "ground_truth", "signer_id"))
        items = run_manifest.read_manifest(self.root / "manifest.csv")
        self.assertEqual(items[0][3], "signer-7")

    def test_whitespace_variant_ids_are_distinct_items(self):
        self.write_manifest([("a", "clip.mp4", ""), (" a", self.clip_file("b.mp4"), "")])
        items = run_manifest.read_manifest(self.root / "manifest.csv")
        self.assertEqual([i[0] for i in items], ["a", " a"])


# --------------------------------------------------------------------------
# profiled runs: no-hand failure among successes
# --------------------------------------------------------------------------

class ProfiledFailureTest(A14Base):
    def _profile_namespace(self):
        namespace = types.SimpleNamespace(__module__="a14fake_profile")
        namespace.get_last_profile = lambda: {"stages": {"model_inference": {"milliseconds": 2}}}
        sys.modules[namespace.__module__] = namespace
        self.addCleanup(lambda: sys.modules.pop(namespace.__module__, None))
        return namespace

    def test_mixed_success_and_no_hand_failure(self):
        namespace = self._profile_namespace()
        self.write_manifest([("clip 01", "clip.mp4", "CANDY"),
                             ("clip 02", self.clip_file("clip02.mp4"), "CANDY")])

        def predict(path, *, clip_id=None, profile=False):
            if clip_id == "clip 01":
                return self.prediction()
            raise ValueError("No hand detected in any sampled frame")

        predict.__module__ = namespace.__module__
        predictions, wall_times, accuracy, details = run_manifest.evaluate_profiled(
            run_manifest.read_manifest(self.root / "manifest.csv"), predict, self.labels)

        self.assertEqual([p["clip_id"] for p in predictions], ["clip 01"])
        self.assertEqual(len(details["failures"]), 1)
        failure = details["failures"][0]
        self.assertEqual(failure["clip_id"], "clip 02")
        self.assertEqual(failure["error_type"], "ValueError")
        self.assertIn("No hand detected", failure["error"])
        # one failure nulls top1 even though the successful clip was correct
        self.assertIsNone(accuracy["top1"])
        self.assertEqual(accuracy["clips_with_ground_truth"], 2)
        self.assertEqual(accuracy["correct"], 1)
        self.assertFalse(accuracy["representative_accuracy_measured"])
        self.assertEqual(len(details["profiles"]), 1)
        self.assertEqual(len(wall_times), 1)

    def test_profiled_without_profile_kwarg_rejected(self):
        self.write_manifest([("clip 01", "clip.mp4", "")])
        with self.assertRaisesRegex(run_manifest.EvaluationError, "profile=True"):
            run_manifest.evaluate_profiled(
                run_manifest.read_manifest(self.root / "manifest.csv"),
                lambda _: self.prediction(), self.labels)

    def test_profiled_without_stage_profile_rejected(self):
        module = types.SimpleNamespace(__module__="a14fake_nostage")
        module.get_last_profile = lambda: {"no_stages": {}}
        sys.modules[module.__module__] = module
        self.addCleanup(lambda: sys.modules.pop(module.__module__, None))

        def predict(_, *, profile=False):
            return self.prediction()

        predict.__module__ = module.__module__
        self.write_manifest([("clip 01", "clip.mp4", "")])
        with self.assertRaisesRegex(run_manifest.EvaluationError, "stage profile"):
            run_manifest.evaluate_profiled(
                run_manifest.read_manifest(self.root / "manifest.csv"), predict, self.labels)

    def test_profiled_wrong_id_failure_recorded(self):
        namespace = self._profile_namespace()

        def predict(_, *, clip_id=None, profile=False):
            return self.prediction(clip_id="other")

        predict.__module__ = namespace.__module__
        self.write_manifest([("clip 01", "clip.mp4", "")])
        predictions, _, _, details = run_manifest.evaluate_profiled(
            run_manifest.read_manifest(self.root / "manifest.csv"), predict, self.labels)
        self.assertEqual(predictions, [])
        self.assertEqual(details["failures"][0]["error_type"], "EvaluationError")


# --------------------------------------------------------------------------
# corrupt / empty video behaviour
# --------------------------------------------------------------------------

class CorruptVideoTest(A14Base):
    def test_empty_file_rejected(self):
        empty = self.root / "empty.mp4"
        empty.write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "Could not open video"):
            decode_video(empty)

    def test_corrupt_bytes_rejected(self):
        corrupt = self.root / "corrupt.mp4"
        corrupt.write_bytes(b"\x00" * 128)
        with self.assertRaisesRegex(ValueError, "Could not open video"):
            decode_video(corrupt)

    def test_missing_file_rejected(self):
        with self.assertRaisesRegex(ValueError, "Could not open video"):
            decode_video(self.root / "missing.mp4")

    def test_corrupt_frames_raises_for_empty_and_unknown(self):
        with self.assertRaisesRegex(ValueError, "empty clip"):
            corrupt_frames([], CorruptionSpec("crop"))
        with self.assertRaisesRegex(ValueError, "Unsupported corruption"):
            corrupt_frames([np.zeros((4, 4, 3), np.uint8)], CorruptionSpec("bogus"))

    def test_severity_bounds(self):
        frame = [np.zeros((4, 4, 3), np.uint8)]
        for bad in (0, -0.5, 1.5):
            with self.assertRaisesRegex(ValueError, "severity"):
                corrupt_frames(frame, CorruptionSpec("crop", bad))

    def test_decode_roundtrip_preserves_fps_and_frames(self):
        path = self.video("roundtrip.mp4", frames=5)
        frames, fps = decode_video(path)
        self.assertEqual(len(frames), 5)
        self.assertEqual(fps, 10)
        self.assertEqual(frames[0].shape, (20, 20, 3))

    def test_predict_clip_swallows_no_hand_failure(self):
        def fail(path, *, clip_id=None):
            raise ValueError("No hand detected")

        result = predict_clip("clip 01", str(self.clip), fail, self.labels)
        self.assertIsNone(result["prediction"])
        self.assertIn("No hand detected", result["error"])


# --------------------------------------------------------------------------
# paired robustness accounting
# --------------------------------------------------------------------------

class PairedAccountingTest(A14Base):
    def test_predictor_failure_on_both_legs_counts_as_zero_correct(self):
        good = self.video("good.mp4")
        out = self.root / "out"

        def fail(path, *, clip_id=None):
            raise ValueError("No hand detected")

        summary = paired_evaluate([("clip x", good, "CANDY", "s1")], fail,
                                  self.labels, CorruptionSpec("crop", 0.5, 0), out)
        self.assertEqual(summary["clean_correct"], 0)
        self.assertEqual(summary["corrupted_correct"], 0)
        self.assertEqual(summary["clean_top1"], 0.0)
        self.assertEqual(summary["corrupted_top1"], 0.0)
        self.assertEqual(summary["accuracy_drop_percentage_points"], 0.0)
        self.assertEqual(summary["failure_breakdown"], {"clean_prediction_failed": 1, "corrupted_prediction_failed": 1})
        self.assertEqual(summary["pairs"][0]["clean_prediction"], None)
        self.assertEqual(summary["affected_signer_count"], 1)
        self.assertFalse((out / "work").exists())

    def test_corrupt_leg_only_failure(self):
        good = self.video("good.mp4")
        out = self.root / "out"

        def predict(path, *, clip_id=None):
            if Path(path).parent.name == "work":
                # returns a well-formed response whose prediction disagrees
                # with ground truth, so the pair counts as a corrupted failure
                return self.prediction(clip_id=clip_id, prediction="RIGHT", top5=["RIGHT", "CANDY"])
            return self.prediction(clip_id=clip_id)

        summary = paired_evaluate([("clip x", good, "CANDY", "")], predict,
                                  self.labels, CorruptionSpec("low_light", 0.5, 0), out)
        self.assertEqual(summary["clean_correct"], 1)
        self.assertEqual(summary["corrupted_correct"], 0)
        self.assertEqual(summary["accuracy_drop_percentage_points"], 100.0)
        self.assertEqual(summary["failure_breakdown"], {})
        # failure_breakdown is explicit; an empty string signer is counted as
        # unknown rather than ignored.
        self.assertIsNone(summary["affected_signer_count"])

    def test_unicode_clip_id_in_pairs(self):
        good = self.video("good.mp4")
        out = self.root / "out"
        summary = paired_evaluate(
            [("sign-✋-01", good, "CANDY", "signer-ü")],
            lambda path, *, clip_id=None: self.prediction(clip_id=clip_id),
            self.labels, CorruptionSpec("crop", 0.5, 0), out)
        self.assertEqual(summary["pairs"][0]["clip_id"], "sign-✋-01")
        self.assertEqual(summary["affected_signer_count"], 1)


# --------------------------------------------------------------------------
# model / config rejection
# --------------------------------------------------------------------------

class ModelContractRejectionTest(A14Base):
    def setUp(self):
        super().setUp()
        self.bundle = self.root / "bundle"
        self.bundle.mkdir()
        (self.bundle / "classifier.tflite").write_bytes(b"test model")
        self.manifest = {
            "model_sha256": hashlib.sha256(b"test model").hexdigest(),
            "model_size_bytes": 10, "model_format": "tflite",
            "input_tensor_shape": [1, 24, 255], "input_dtype": "float32",
            "input_scale": 0.0, "input_zero_point": 0,
            "output_tensor_shape": [1, 2], "output_dtype": "float32",
            "output_scale": 0.0, "output_zero_point": 0,
            "output_is_logits_or_probabilities": "logits",
            "ordered_vocabulary": ["A", "B"],
            "feature_config_version": "v1", "feature_shape": [24, 255],
            "frame_count": 24, "sampling_method": "uniform full clip",
            "normalization": "documented", "handedness": "left/right",
            "presence_masks": True, "coordinate_system": "normalized image xy",
            "missing_landmark_policy": "explicit presence mask",
        }
        self.write()

    def write(self, **updates):
        manifest = dict(self.manifest)
        manifest.update(updates)
        (self.bundle / "integration_manifest.json").write_text(json.dumps(manifest))

    def test_non_tflite_rejected(self):
        self.write(model_format="onnx")
        with self.assertRaisesRegex(CompatibilityError, "Only TFLite"):
            validate_bundle(self.bundle)

    def test_invalid_output_semantics_rejected(self):
        self.write(output_is_logits_or_probabilities="softmax")
        with self.assertRaisesRegex(CompatibilityError, "Output semantics"):
            validate_bundle(self.bundle)

    def test_vocabulary_must_be_unique_uppercase(self):
        for vocab in ([], ["a", "B"], ["A", "A"]):
            self.write(ordered_vocabulary=vocab)
            with self.assertRaisesRegex(CompatibilityError, "uppercase"):
                validate_bundle(self.bundle)

    def test_frame_and_feature_shape_consistency(self):
        self.write(frame_count=0)
        with self.assertRaisesRegex(CompatibilityError, "frame_count"):
            validate_bundle(self.bundle)
        self.write(feature_shape=[12, 255])
        with self.assertRaisesRegex(CompatibilityError, "feature_shape"):
            validate_bundle(self.bundle)

    def test_size_mismatch_rejected(self):
        self.write(model_size_bytes=11)
        with self.assertRaisesRegex(CompatibilityError, "byte size"):
            validate_bundle(self.bundle)

    def test_invalid_json_manifest_rejected(self):
        (self.bundle / "integration_manifest.json").write_text("not json")
        with self.assertRaisesRegex(CompatibilityError, "Invalid integration manifest"):
            validate_bundle(self.bundle)

    def test_missing_bundle_files_rejected(self):
        (self.bundle / "integration_manifest.json").unlink()
        with self.assertRaisesRegex(CompatibilityError, "required"):
            validate_bundle(self.bundle)

    def test_wrong_shape_dtypes_rejected(self):
        class Int8:
            def get_input_details(self):
                return [{"shape": np.array([1, 24, 255]), "dtype": np.uint8,
                         "quantization": (0.02, 128)}]
            def get_output_details(self):
                return [{"shape": np.array([1, 2]), "dtype": np.float32,
                         "quantization": (0.0, 0)}]
        with self.assertRaisesRegex(CompatibilityError, "input_dtype mismatch"):
            validate_bundle(self.bundle, Int8())

    def test_two_inputs_rejected(self):
        class Two:
            def get_input_details(self):
                d = {"shape": np.array([1, 24, 255]), "dtype": np.float32,
                     "quantization": (0.0, 0)}
                return [d, d]
            def get_output_details(self):
                return [{"shape": np.array([1, 2]), "dtype": np.float32,
                         "quantization": (0.0, 0)}]
        with self.assertRaisesRegex(CompatibilityError, "exactly one"):
            validate_bundle(self.bundle, Two())


# --------------------------------------------------------------------------
# stale evidence and release-path safety
# --------------------------------------------------------------------------

class ReleaseSafetyTest(A14Base):
    def make_source(self):
        src = self.root / "src_tree"
        (src / "src").mkdir(parents=True)
        (src / "models").mkdir()
        (src / "scripts").mkdir()
        for name in ("main.py", "requirements.txt", "README.md"):
            (src / "src" / name).write_text("# fixture\n")
        (src / "models" / "classifier.tflite").write_bytes(b"model")
        (src / "models" / "labels.json").write_text('{"A": 0, "B": 1}')
        (src / "scripts" / "export_model.py").write_text("# conversion fixture\n")
        (src / "results.csv").write_text(
            "metric,target,measured,unit,device,runtime,pass,notes\n"
            "avg_power,<=500,NOT_MEASURED,mW,n/a,n/a,NOT_MEASURED,no device\n")
        return src

    def test_zip_traversal_rejected(self):
        path = self.root / "trav.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("../evil.txt", "x")
        with self.assertRaisesRegex(ValueError, "Unsafe bundle path"):
            release_build.verify_zip(path)

    def test_zip_absolute_rejected(self):
        path = self.root / "abs.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("/abs/file", "x")
        with self.assertRaisesRegex(ValueError, "Unsafe bundle path"):
            release_build.verify_zip(path)

    def test_zip_symlink_rejected(self):
        link = self.root / "link.zip"
        with zipfile.ZipFile(link, "w") as archive:
            info = zipfile.ZipInfo("entry")
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info, "src/main.py")
        with self.assertRaisesRegex(ValueError, "Unsafe bundle path|Unsafe ZIP member type"):
            release_build.verify_zip(link)

    def test_build_rejects_missing_export_script(self):
        src = self.make_source()
        with self.assertRaisesRegex(ValueError, "conversion/export|model conversion"):
            release_build.build(
                src, "models/classifier.tflite", "models/labels.json",
                src / "results.csv", self.root / "out.zip",
                extra=[src / "models" / "classifier.tflite", src / "models" / "labels.json"])

    def test_build_rejects_outside_model(self):
        src = self.make_source()
        outside = self.root / "outside.tflite"
        outside.write_bytes(b"model")
        with self.assertRaises(ValueError):
            release_build.build(
                src, "../outside.tflite", "models/labels.json",
                src / "results.csv", self.root / "out.zip", extra=[])

    def _benchmark(self, name="benchmark.json", sha="right", **updates):
        doc = {
            "artifacts": {"model": {"sha256": sha, "bytes": 10}},
            "environment": {"device_model": "d", "inference_runtime": "r"},
            "code_commit": "c", "manifest": {"sha256": "m"},
            "accuracy": {"top1": None},
            "latency": {"all_calls": {"p95_ms": 10}},
            "representative_performance_measured": False,
        }
        doc.update(updates)
        path = self.root / name
        path.write_text(json.dumps(doc))
        return path

    def test_generate_rejects_stale_robustness(self):
        bench = self._benchmark()
        stale = self.root / "rb.json"
        stale.write_text(json.dumps({
            "model_sha256": "stale", "accuracy_drop_percentage_points": 3,
            "corruption_type": "crop", "representative_robustness_measured": False,
            "sample_count": 3}))
        with self.assertRaisesRegex(ValueError, "Stale robustness"):
            generate(bench, [stale], self.root / "r.csv", "right")

    def test_generate_without_measurements_is_not_measured(self):
        rows = generate(self._benchmark(), [], self.root / "r.csv", "right")
        by_metric = {row["metric"]: row for row in rows}
        self.assertEqual(by_metric["top1_accuracy"]["pass"], "NOT_MEASURED")
        self.assertEqual(by_metric["latency_p95"]["pass"], "NOT_MEASURED")
        self.assertEqual(by_metric["model_size"]["pass"], "PASS")
        self.assertEqual(by_metric["avg_power"]["pass"], "NOT_MEASURED")

    def test_generate_rejects_incomplete_benchmark(self):
        bench = self.root / "bench2.json"
        bench.write_text(json.dumps({"artifacts": {"model": {"sha256": "right"}}}))
        with self.assertRaises(KeyError):
            generate(bench, [], self.root / "r.csv", "right")

    def test_cli_invalid_labels_write_no_outputs(self):
        self.write_manifest([("clip 01", "clip.mp4", "")])
        for name, content in (("model.tflite", b"model"), ("labels.json", b"{not json"),
                              ("predictor.py", b"def predict(path): pass\n")):
            (self.root / name).write_bytes(content)
        output = self.root / "output"
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as result:
            run_manifest.main([
                str(self.root / "manifest.csv"), "--output-dir", str(output),
                "--predictor", str(self.root / "predictor.py"),
                "--labels", str(self.root / "labels.json"),
                "--model", str(self.root / "model.tflite"),
            ])
        self.assertEqual(result.exception.code, 2)
        self.assertFalse((output / "predictions.jsonl").exists())
        self.assertFalse((output / "benchmark.json").exists())


# --------------------------------------------------------------------------
# atomic publication and temp cleanup
# --------------------------------------------------------------------------

class AtomicOutputTest(A14Base):
    def test_nan_prediction_never_publishes(self):
        output = self.root / "out"
        with self.assertRaises(ValueError):
            run_manifest.write_pair_atomically(
                output, [{"clip_id": "a", "confidence": float("nan")}], {})
        self.assertFalse((output / "predictions.jsonl").exists())
        self.assertFalse((output / "benchmark.json").exists())
        leftovers = list(output.iterdir()) if output.exists() else []
        self.assertEqual(leftovers, [])

    def test_atomic_pair_written_and_unicode_safe(self):
        output = self.root / "out"
        run_manifest.write_pair_atomically(
            output, [self.prediction(clip_id="sign-✋-01")], {"status": "passed_schema"})
        lines = (output / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(json.loads(lines[0])["clip_id"], "sign-✋-01")
        self.assertTrue((output / "benchmark.json").exists())
        self.assertEqual([p.name for p in output.iterdir()],
                         ["predictions.jsonl", "benchmark.json"])

    def test_failed_predictor_load_cleans_module_registry(self):
        bad = self.root / "bad.py"
        bad.write_text("raise ImportError('boom')\n")
        with self.assertRaisesRegex(run_manifest.EvaluationError, "Cannot import"):
            run_manifest.load_predictor(bad)
        self.assertNotIn("echogesture_predictor", sys.modules)


# --------------------------------------------------------------------------
# prediction-shape contract around boundaries
# --------------------------------------------------------------------------

class PredictionBoundaryTest(A14Base):
    def test_bool_confidence_and_latency_rejected(self):
        with self.assertRaisesRegex(run_manifest.EvaluationError, "invalid confidence"):
            run_manifest.validate_prediction(self.prediction(confidence=True), "clip 01", self.labels)
        with self.assertRaisesRegex(run_manifest.EvaluationError, "invalid latency"):
            run_manifest.validate_prediction(self.prediction(latency_ms=False), "clip 01", self.labels)

    def test_confidence_boundaries(self):
        at_zero = self.prediction(confidence=0)
        self.assertIs(run_manifest.validate_prediction(at_zero, "clip 01", self.labels), at_zero)
        with self.assertRaisesRegex(run_manifest.EvaluationError, "confidence"):
            run_manifest.validate_prediction(self.prediction(confidence=1.000001), "clip 01", self.labels)
        with self.assertRaisesRegex(run_manifest.EvaluationError, "confidence"):
            run_manifest.validate_prediction(self.prediction(confidence=float("inf")), "clip 01", self.labels)

    def test_top5_contract_variants(self):
        for bad in ([], ["CANDY"] * 6, ["CANDY", "CANDY"], ["RIGHT", "CANDY"], ["CANDY", "right"]):
            with self.assertRaisesRegex(run_manifest.EvaluationError, "top5"):
                run_manifest.validate_prediction(self.prediction(top5=bad), "clip 01", self.labels)
        single = self.prediction(top5=["CANDY"])
        self.assertIs(run_manifest.validate_prediction(single, "clip 01", self.labels), single)


if __name__ == "__main__":
    unittest.main()
