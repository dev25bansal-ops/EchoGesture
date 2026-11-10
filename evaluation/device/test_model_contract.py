import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_contract import (
    CANDIDATE_SPEC,
    CompatibilityError,
    identify_quantization,
    probe_output_semantics,
    quantization_census,
    tensor_contract,
    validate_bundle,
    validate_candidate_bundle,
)

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


def _load_interpreter(path):
    from ai_edge_litert.interpreter import Interpreter

    interpreter = Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    return interpreter


def _has_real_models():
    return (MODELS_DIR / CANDIDATE_SPEC["model_filename"]).is_file()


class _FakeInterpreter:
    """Minimal interpreter stub for the legacy validate_bundle path.

    Supplies get_input_details/get_output_details only; census/probe
    helpers are not exercised through this stub.
    """

    def __init__(self, input_shape=(1, 24, 255), output_shape=(1, 2)):
        self._in = [{"shape": np.array(input_shape), "dtype": np.float32, "quantization": (0.0, 0)}]
        self._out = [{"shape": np.array(output_shape), "dtype": np.float32, "quantization": (0.0, 0)}]

    def get_input_details(self):
        return self._in

    def get_output_details(self):
        return self._out


class ModelContractTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "classifier.tflite").write_bytes(b"test model")
        import hashlib

        self.manifest = {
            "model_sha256": hashlib.sha256(b"test model").hexdigest(),
            "model_size_bytes": 10,
            "model_format": "tflite",
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
        (self.root / "integration_manifest.json").write_text(json.dumps(self.manifest))

    def tearDown(self):
        self.temp.cleanup()

    def test_valid_contract(self):
        self.assertEqual(validate_bundle(self.root)["status"], "compatible")

    def test_hash_mismatch(self):
        (self.root / "classifier.tflite").write_bytes(b"wrong model")
        with self.assertRaisesRegex(CompatibilityError, "SHA mismatch"):
            validate_bundle(self.root)

    def test_missing_field(self):
        del self.manifest["feature_config_version"]
        (self.root / "integration_manifest.json").write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(CompatibilityError, "feature_config_version"):
            validate_bundle(self.root)

    def test_tensor_mismatch(self):
        with self.assertRaisesRegex(CompatibilityError, "input_tensor_shape"):
            validate_bundle(self.root, _FakeInterpreter(input_shape=(1, 24, 254)))

    def test_required_asset_missing_is_error(self):
        self.manifest["required_assets"] = ["hand_landmarker.task"]
        (self.root / "integration_manifest.json").write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(CompatibilityError, "Missing required asset"):
            validate_bundle(self.root)

    def test_labels_file_mismatch(self):
        # label map orders A=1, B=0, but manifest vocabulary is ["A", "B"]
        labels = {"b": 0, "a": 1}
        (self.root / "labels.json").write_text(json.dumps(labels))
        self.manifest["labels_file"] = "labels.json"
        (self.root / "integration_manifest.json").write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(CompatibilityError, "Labels file ordering"):
            validate_bundle(self.root)

    def test_real_dev_fp16_contract(self):
        candidate = Path("/tmp/echogesture-dev-candidate/models")
        if not candidate.exists():
            self.skipTest("Dev verification worktree not available")
        labels = json.loads((candidate / "landmark_labels.json").read_text())
        ordered = [None] * len(labels)
        for label, index in labels.items():
            ordered[index] = label.upper()
        self.manifest.update({
            "model_sha256": "a012f2b5f7a1110961079f19eb0e28c9abacb95e4d0787352e4157ebca7044a4",
            "model_size_bytes": 832888, "ordered_vocabulary": ordered,
            "output_tensor_shape": [1, 100],
        })
        (candidate / "classifier.tflite").write_bytes((candidate / "landmark_classifier_fp16.tflite").read_bytes())
        (candidate / "integration_manifest.json").write_text(json.dumps(self.manifest))
        result = validate_bundle(candidate, _load_interpreter(candidate / "landmark_classifier_fp16.tflite"))
        self.assertEqual(result["classes"], 100)


class RealArtifactIdentificationTest(unittest.TestCase):
    """Identification/rejection tests against the real artifacts in models/.

    These use the actual FP16 and INT8 files present in the checkout --
    no fabricated tensors, no synthetic logits.
    """

    def setUp(self):
        if not _has_real_models():
            self.skipTest("Real model artifacts not available")

    def test_fp16_actual_tensor_contract(self):
        interpreter = _load_interpreter(MODELS_DIR / "landmark_classifier_fp16.tflite")
        actual = tensor_contract(interpreter)
        self.assertEqual(actual["input_tensor_shape"], [1, 24, 255])
        self.assertEqual(actual["output_tensor_shape"], [1, 100])
        self.assertEqual(actual["input_dtype"], "float32")
        self.assertEqual(actual["output_dtype"], "float32")
        self.assertEqual(actual["input_scale"], 0.0)
        self.assertEqual(actual["output_scale"], 0.0)

    def test_fp16_identified_as_fp16_not_int8(self):
        interpreter = _load_interpreter(MODELS_DIR / "landmark_classifier_fp16.tflite")
        identified = identify_quantization(interpreter)
        self.assertEqual(identified["classification"], "fp16_float32_io")
        self.assertGreater(identified["census"]["float16_tensors"], 0)
        self.assertEqual(identified["census"]["int8_quantized_tensors"], 0)

    def test_int8_identified_as_int8_mixed(self):
        interpreter = _load_interpreter(MODELS_DIR / "landmark_classifier_int8.tflite")
        identified = identify_quantization(interpreter)
        self.assertEqual(identified["classification"], "int8_mixed_float_io")
        self.assertGreater(identified["census"]["int8_quantized_tensors"], 0)
        self.assertEqual(identified["census"]["float16_tensors"], 0)
        # Both expose float32 IO -- IO alone cannot distinguish them,
        # which is why the census exists.
        self.assertEqual(identified["input_dtype"], "float32")
        self.assertEqual(identified["output_dtype"], "float32")

    def test_fp16_and_int8_are_distinct_artifacts(self):
        fp16 = _load_interpreter(MODELS_DIR / "landmark_classifier_fp16.tflite")
        int8 = _load_interpreter(MODELS_DIR / "landmark_classifier_int8.tflite")
        self.assertNotEqual(
            identify_quantization(fp16)["classification"],
            identify_quantization(int8)["classification"],
        )
        c_fp16 = quantization_census(fp16)
        c_int8 = quantization_census(int8)
        self.assertNotEqual(c_fp16["dtype_counts"], c_int8["dtype_counts"])

    def test_s3d_baseline_is_int8_io(self):
        path = MODELS_DIR / "s3d_conv1d_qat_int8.tflite"
        if not path.is_file():
            self.skipTest("S3D baseline artifact not available")
        identified = identify_quantization(_load_interpreter(path))
        self.assertEqual(identified["classification"], "int8_io")
        self.assertEqual(identified["input_dtype"], "int8")

    def test_fp16_output_semantics_are_logits(self):
        interpreter = _load_interpreter(MODELS_DIR / "landmark_classifier_fp16.tflite")
        semantics = probe_output_semantics(interpreter)
        self.assertEqual(semantics["looks_like"], "logits")
        self.assertTrue(semantics["has_negative"])
        self.assertTrue(semantics["finite"])

    def test_int8_output_semantics_are_logits(self):
        interpreter = _load_interpreter(MODELS_DIR / "landmark_classifier_int8.tflite")
        semantics = probe_output_semantics(interpreter)
        self.assertEqual(semantics["looks_like"], "logits")

    def test_candidate_bundle_accepts_fp16(self):
        interpreter = _load_interpreter(MODELS_DIR / "landmark_classifier_fp16.tflite")
        result = validate_candidate_bundle(MODELS_DIR, interpreter)
        self.assertEqual(result["status"], "compatible")
        self.assertEqual(result["classes"], 100)
        self.assertEqual(result["quantization"]["classification"], "fp16_float32_io")
        self.assertEqual(result["output_semantics"]["looks_like"], "logits")
        self.assertEqual(len(result["ordered_vocabulary"]), 100)
        self.assertEqual(result["ordered_vocabulary"][43], "PULLCONVINCE")

    def test_candidate_bundle_rejects_int8_as_fp16(self):
        """The INT8 bytes must never pass as the FP16 candidate: its SHA
        differs, so strict validation raises rather than substituting."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("landmark_labels.json", "landmark_model_meta.json",
                         "hand_landmarker.task"):
                (root / name).write_bytes((MODELS_DIR / name).read_bytes())
            # INT8 bytes under the FP16 filename -> SHA must fail
            (root / "landmark_classifier_fp16.tflite").write_bytes(
                (MODELS_DIR / "landmark_classifier_int8.tflite").read_bytes())
            interpreter = _load_interpreter(MODELS_DIR / "landmark_classifier_int8.tflite")
            with self.assertRaisesRegex(CompatibilityError, "SHA mismatch"):
                validate_candidate_bundle(root, interpreter)

    def test_candidate_bundle_rejects_int8_via_census_when_sha_pinned(self):
        """Even with the INT8 file's own SHA pinned in the spec, the
        census must reject it as the FP16 candidate."""
        spec = dict(CANDIDATE_SPEC)
        spec["model_sha256"] = "9d24640a23244a3cd00c9fa62257733819f86ed1da5a10194c9b1cb64323700b"
        spec["model_size_bytes"] = 509720
        spec["model_filename"] = "landmark_classifier_int8.tflite"
        interpreter = _load_interpreter(MODELS_DIR / "landmark_classifier_int8.tflite")
        with self.assertRaisesRegex(CompatibilityError, "quantization mismatch"):
            validate_candidate_bundle(MODELS_DIR, interpreter, spec=spec)

    def test_candidate_bundle_missing_detector_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("landmark_classifier_fp16.tflite", "landmark_labels.json",
                         "landmark_model_meta.json"):
                (root / name).write_bytes((MODELS_DIR / name).read_bytes())
            # detector intentionally absent -- must raise, not fall back
            with self.assertRaisesRegex(CompatibilityError, "Missing candidate asset"):
                validate_candidate_bundle(root)

    def test_candidate_bundle_labels_parity(self):
        labels = json.loads((MODELS_DIR / "landmark_labels.json").read_text())
        self.assertEqual(len(labels), 100)
        self.assertEqual(sorted(int(v) for v in labels.values()), list(range(100)))
        interpreter = _load_interpreter(MODELS_DIR / "landmark_classifier_fp16.tflite")
        self.assertEqual(tensor_contract(interpreter)["output_tensor_shape"][-1], len(labels))

    def test_candidate_bundle_meta_parity(self):
        meta = json.loads((MODELS_DIR / "landmark_model_meta.json").read_text())
        self.assertEqual(meta["frames"], 24)
        self.assertEqual(meta["feature_dim"], 255)
        self.assertEqual(meta["classes"], 100)
        self.assertEqual(meta["quantization"], "float16")
        interpreter = _load_interpreter(MODELS_DIR / "landmark_classifier_fp16.tflite")
        actual = tensor_contract(interpreter)
        self.assertEqual(actual["input_tensor_shape"], [1, meta["frames"], meta["feature_dim"]])

    def test_fp16_int8_logit_drift_is_real(self):
        """FP16 and INT8 disagree on real inputs (measured, not claimed
        as parity). This pins that the two artifacts are not interchangeable."""
        rng = np.random.RandomState(42)
        x = rng.randn(1, 24, 255).astype(np.float32)
        fp16 = _load_interpreter(MODELS_DIR / "landmark_classifier_fp16.tflite")
        int8 = _load_interpreter(MODELS_DIR / "landmark_classifier_int8.tflite")
        outs = []
        for interpreter in (fp16, int8):
            interpreter.set_tensor(interpreter.get_input_details()[0]["index"], x)
            interpreter.invoke()
            outs.append(interpreter.get_tensor(interpreter.get_output_details()[0]["index"])[0])
        self.assertGreater(float(np.abs(outs[0] - outs[1]).max()), 0.1)


if __name__ == "__main__":
    unittest.main()
