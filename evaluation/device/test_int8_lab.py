"""Unit tests for the A06 INT8 lab: real tensor classification and lab scripts.

These tests use real artifact tensor inspection, never filenames. They are
deliberately isolated to this file and evaluation/device/int8_lab/; they do
not modify models, scripts or src. Run:
  .venv/bin/python -m unittest evaluation/device/test_int8_lab.py -v
"""
import importlib.util
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LAB = Path(__file__).resolve().parent / "int8_lab"
sys.path.insert(0, str(LAB))
import inspect_tflite  # noqa: E402

FP16 = ROOT / "models/landmark_classifier_fp16.tflite"
INT8 = ROOT / "models/landmark_classifier_int8.tflite"
DET = ROOT / "models/hand_landmarker.task"


def _load(path):
    fb = inspect_tflite.inspect_flatbuffer(path)
    io = inspect_tflite.inspect_interpreter(path)
    return fb, io, inspect_tflite.classify(fb, io)


@unittest.skipUnless(
    FP16.is_file() and INT8.is_file() and DET.is_file(),
    "trained model weights are not committed (see NOTICE); supply them locally")
class TestRealArtifacts(unittest.TestCase):
    """Acceptance: classify FP16 as non-INT8; INT8 alt as INT8; detector FP16."""

    def test_fp16_is_non_int8(self):
        self.assertTrue(FP16.is_file())
        fb, io, cls = _load(FP16)
        self.assertEqual(io["inputs"][0]["shape"], [1, 24, 255])
        self.assertEqual(io["inputs"][0]["dtype"], "float32")
        self.assertEqual(io["outputs"][0]["shape"], [1, 100])
        self.assertIn("FLOAT16", cls["weight_dtype_hist"])
        self.assertNotIn("INT8", cls["weight_dtype_hist"])
        self.assertEqual(cls["quantized_tensors"], 0)
        self.assertEqual(cls["ops_touching_int8"], 0)
        self.assertFalse(cls["is_int8_by_tensor_test"])
        self.assertTrue(cls["classification"].startswith("FP16"))

    def test_int8_alternative_is_int8_float_io(self):
        self.assertTrue(INT8.is_file())
        fb, io, cls = _load(INT8)
        self.assertEqual(io["inputs"][0]["shape"], [1, 24, 255])
        self.assertEqual(io["inputs"][0]["dtype"], "float32")
        self.assertEqual(io["outputs"][0]["shape"], [1, 100])
        self.assertEqual(cls["weight_dtype_hist"].get("INT8"), 21)
        self.assertGreater(cls["quantized_tensors"], 0)
        self.assertGreater(cls["ops_touching_int8"], 0)
        self.assertTrue(cls["is_int8_by_tensor_test"])
        self.assertTrue(cls["classification"].startswith("INT8"))
        # Same external contract as FP16: drop-in at IO level.
        self.assertEqual(io["inputs"][0]["scale"], 0.0)
        self.assertEqual(io["outputs"][0]["scale"], 0.0)

    def test_detector_members_are_fp16(self):
        import zipfile, tempfile
        self.assertTrue(DET.is_file())
        with zipfile.ZipFile(DET) as z:
            names = sorted(z.namelist())
            self.assertEqual(names, ["hand_detector.tflite", "hand_landmarks_detector.tflite"])
            for name in names:
                data = z.read(name)
                tmp = Path(tempfile.mkdtemp()) / name
                tmp.write_bytes(data)
                fb = inspect_tflite.inspect_flatbuffer(tmp)
                cls = inspect_tflite.classify(fb, None)
                tmp.unlink(missing_ok=True)
                self.assertEqual(cls["quantized_tensors"], 0, name)
                self.assertFalse(cls["is_int8_by_tensor_test"], name)
                self.assertIn("FLOAT16", cls["weight_dtype_hist"], name)

    def test_meta_declares_float16(self):
        meta = json.loads((ROOT / "models/landmark_model_meta.json").read_text())
        self.assertEqual(meta["quantization"], "float16")

    def test_fp16_int8_same_io_contract(self):
        a = inspect_tflite.inspect_interpreter(FP16)
        b = inspect_tflite.inspect_interpreter(INT8)
        self.assertEqual(a, b)


@unittest.skipUnless(INT8.is_file() and FP16.is_file(),
                     "trained weights are not committed (see NOTICE)")
class TestComparisonScript(unittest.TestCase):
    def test_compare_runs_and_reports_no_accuracy_on_synthetic(self):
        spec = importlib.util.spec_from_file_location("cmp", LAB / "compare_fp16_int8.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        out = []
        sys.stdout = type("T", (), {"write": lambda s, t: out.append(t), "flush": lambda s: None})()
        try:
            m.main([])
        finally:
            sys.stdout = sys.__stdout__
        text = "".join(out)
        result = json.loads(text.split("NOTE:")[0])
        self.assertEqual(result["n"], 700)
        self.assertIn("argmax_agreement", result)
        self.assertNotIn("fp16_top1", result)  # synthetic rows must not imply accuracy
        self.assertIn("not accuracy", text)


@unittest.skipUnless(FP16.is_file() and INT8.is_file() and DET.is_file(),
                     "trained weights are not committed (see NOTICE)")
class TestFootprint(unittest.TestCase):
    def test_combined_bytes_under_limit_and_int8_smaller(self):
        fp, i8, det = FP16.stat().st_size, INT8.stat().st_size, DET.stat().st_size
        self.assertLess(i8, fp)
        for total in (fp + det, i8 + det):
            self.assertLess(total, 20 * 1024 * 1024)
            self.assertLess(total, 20_000_000)


if __name__ == "__main__":
    unittest.main()
