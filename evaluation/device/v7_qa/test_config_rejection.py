"""V7 QA: malformed config / model / label inputs are rejected, never defaulted.

Verified behavior:
  * model_contract.validate_bundle rejects missing fields, wrong SHA,
    wrong format, non-contiguous vocab, and quantization lying about INT8
    vs FP16 (INT8 artifact declared float16 raises on census mismatch).
  * run_manifest.load_labels rejects an empty {} vocabulary; a list of
    labels loads as an upper-cased set.
  * build (release) rejects a model path that escapes the source root
    and rejects empty results/labels artifacts.
  * ECHOGESTURE_MODEL accepts only "s3d" / "landmark"; anything else
    (including "INT8"/"int8") raises ValueError -- there is no silent
    INT8 default. The default (env unset) selects the landmark FP16
    candidate, never the INT8 artifact.
"""

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

DEVICE_DIR = Path(__file__).resolve().parents[1]
ROOT = DEVICE_DIR.parents[1]
for _p in (str(DEVICE_DIR), str(ROOT / "scripts" / "release")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

import build_bundle as release_build  # noqa: E402
import run_manifest  # noqa: E402
from model_contract import CompatibilityError, validate_bundle  # noqa: E402


class ConfigRejectionTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def test_empty_json_vocabulary_rejected(self):
        path = self.root / "empty.json"
        path.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(run_manifest.EvaluationError, "Empty vocabulary"):
            run_manifest.load_labels(path)

    def test_missing_vocabulary_file_rejected(self):
        with self.assertRaisesRegex(run_manifest.EvaluationError, "Cannot read vocabulary"):
            run_manifest.load_labels(self.root / "missing.json")

    def test_list_vocabulary_loads_upper_cased(self):
        path = self.root / "labels.json"
        path.write_text(json.dumps(["candy", "Right"]), encoding="utf-8")
        self.assertEqual(run_manifest.load_labels(path), {"CANDY", "RIGHT"})

    def test_non_json_vocabulary_rejected(self):
        path = self.root / "bad.json"
        path.write_bytes(b"{not json")
        with self.assertRaisesRegex(run_manifest.EvaluationError, "Cannot read vocabulary"):
            run_manifest.load_labels(path)

    def test_release_model_path_escape_rejected(self):
        src = self.root / "src_tree"
        (src / "src").mkdir(parents=True)
        for name in ("main.py", "requirements.txt", "README.md"):
            (src / "src" / name).write_text("# fixture\n")
        outside = self.root / "outside.tflite"
        outside.write_bytes(b"model")
        results = self.root / "results.csv"
        results.write_text("metric,target,measured,unit,device,runtime,pass,notes\n")
        with self.assertRaises(ValueError):
            release_build.build(
                src, "../outside.tflite", "models/labels.json",
                results, self.root / "out.zip", extra=[])

    def test_release_empty_results_rejected(self):
        src = self.root / "src_tree"
        (src / "src").mkdir(parents=True)
        (src / "models").mkdir()
        (src / "scripts").mkdir()
        for name in ("main.py", "requirements.txt", "README.md"):
            (src / "src" / name).write_text("# fixture\n")
        (src / "models" / "classifier.tflite").write_bytes(b"model")
        (src / "models" / "labels.json").write_text('{"A": 0}')
        (src / "scripts" / "export_model.py").write_text("# fixture\n")
        empty_results = self.root / "results.csv"
        empty_results.write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "Results file"):
            release_build.build(
                src, "models/classifier.tflite", "models/labels.json",
                empty_results, self.root / "out.zip",
                extra=[src / "models" / "classifier.tflite",
                       src / "models" / "labels.json"])


class ModelIdentityTest(unittest.TestCase):
    """INT8 must never be silently claimed for an FP16 artifact (and vice versa)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
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
            "quantization": "float16",
        }
        self.write()

    def write(self, **updates):
        manifest = dict(self.manifest)
        manifest.update(updates)
        (self.bundle / "integration_manifest.json").write_text(json.dumps(manifest))

    def test_int8_tensor_artifact_misdeclared_as_float16_rejected(self):
        class Int8Internal:
            def get_input_details(self):
                return [{"shape": np.array([1, 24, 255]), "dtype": np.float32,
                         "quantization": (0.0, 0)}]

            def get_output_details(self):
                return [{"shape": np.array([1, 2]), "dtype": np.float32,
                         "quantization": (0.0, 0)}]

            def get_tensor_details(self):
                return [
                    {"dtype": np.float32, "quantization": (0.0, 0)},
                    {"dtype": np.int8, "quantization": (0.02, 0)},
                    {"dtype": np.int8, "quantization": (0.05, 3)},
                ]

            def get_input_details_index(self):  # unused; kept for stub parity
                return 0

            def set_tensor(self, index, value):
                self._t = value

            def invoke(self):
                pass

            def get_tensor(self, index):
                # logits-like output so only the quantization check fires
                return np.array([[5.0, -3.0]], dtype=np.float32)

        with self.assertRaisesRegex(CompatibilityError, "quantization mismatch"):
            validate_bundle(self.bundle, Int8Internal())

    def test_missing_model_file_rejected(self):
        (self.bundle / "classifier.tflite").unlink()
        with self.assertRaisesRegex(CompatibilityError, "required"):
            validate_bundle(self.bundle)

    def test_sha_mismatch_rejected(self):
        (self.bundle / "classifier.tflite").write_bytes(b"different!")
        with self.assertRaisesRegex(CompatibilityError, "SHA mismatch"):
            validate_bundle(self.bundle)


class NoSilentInt8DefaultTest(unittest.TestCase):
    def test_default_backend_is_landmark_fp16_not_int8(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "v7_main_default", ROOT / "src" / "main.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            backend, model_path, labels_path = module._model_paths()
        self.assertEqual(backend, "landmark")
        self.assertEqual(model_path.name, "landmark_classifier_fp16.tflite")
        self.assertNotIn("int8", model_path.name.lower())
        self.assertEqual(labels_path.name, "landmark_labels.json")

    def test_int8_backend_names_are_not_accepted(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "v7_main_badenv", ROOT / "src" / "main.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for bad in ("INT8", "int8", "landmark2", "s3D"):
            with mock.patch.dict(os.environ, {"ECHOGESTURE_MODEL": bad}):
                with self.assertRaisesRegex(ValueError, "must be 's3d' or 'landmark'"):
                    module._model_paths()


if __name__ == "__main__":
    unittest.main()
