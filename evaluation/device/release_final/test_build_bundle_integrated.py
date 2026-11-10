"""Tests for the integrated landmark closure in scripts/release/build_bundle.py.

These complement (and do not replace) test_release_final.py, which covers the
earlier side-script workaround. The defect under test here:

  ``build_bundle.build()`` copied only ``src/main.py`` + ``requirements.txt``
  + ``README.md``, the selected model/labels, one conversion script and
  ``results.csv``. It did NOT stage ``src/features.py``,
  ``src/landmark_frontend.py``, ``models/hand_landmarker.task`` or
  ``models/landmark_model_meta.json``, so a clean extract failed the moment
  ``main.predict`` imported them.

The tests exercise the real production entry point end to end: ``build()``
produces a ZIP, then ``verify_zip`` extracts it into a fresh temp dir and runs
the real predictor on a real clip via the same subprocess shape. When the
project interpreter or clip is unavailable the end-to-end tests skip rather
than pretend to pass.
"""

import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "scripts" / "release"))

import build_bundle as bb  # noqa: E402

PYTHON = REPO / ".venv" / "bin" / "python"
CLIP = REPO / "evaluation/device/v5_accuracy/data/val_30/0283886564670357-WHAT FOR.mp4"
MODEL = "models/landmark_classifier_fp16.tflite"
LABELS = "models/landmark_labels.json"
CONVERSION = "scripts/ml/export_tflite_tf.py"
EXPECTED_CLOSURE = (
    "src/features.py",
    "src/landmark_frontend.py",
    "models/hand_landmarker.task",
    "models/landmark_model_meta.json",
)


class BackendDetectionTest(unittest.TestCase):
    def test_landmark_artifact_returns_closure(self):
        self.assertEqual(bb._landmark_runtime_files(MODEL, LABELS), EXPECTED_CLOSURE)

    def test_s3d_artifact_returns_no_closure(self):
        self.assertEqual(bb._landmark_runtime_files(
            "models/s3d_conv1d_qat_int8.tflite", "models/label100.json"), ())

    def test_closure_paths_are_safe_members(self):
        for rel in EXPECTED_CLOSURE:
            self.assertTrue(bb.safe_member(rel), f"unsafe closure path: {rel}")


class ResultsValidatorTest(unittest.TestCase):
    """The real results.csv uses PASS/FAIL/NOT_MEASURED and must be accepted."""

    def test_current_results_csv_is_accepted(self):
        digest = bb.sha256(REPO / MODEL)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "results").mkdir()
            (root / "results/results.csv").write_text(
                (REPO / "results/results.csv").read_text(encoding="utf-8"),
                encoding="utf-8")
            bb._validate_results(root, digest)  # must not raise

    def test_unknown_status_still_rejected(self):
        digest = bb.sha256(REPO / MODEL)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "results").mkdir()
            header = "metric,target,measured,unit,device,runtime,pass,notes\n"
            (root / "results/results.csv").write_text(
                header + "m,1,1,u,d,r,MAYBE,note\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                bb._validate_results(root, digest)

    def test_absolute_path_leak_still_rejected(self):
        digest = bb.sha256(REPO / MODEL)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "results").mkdir()
            header = "metric,target,measured,unit,device,runtime,pass,notes\n"
            (root / "results/results.csv").write_text(
                header + "m,1,1,u,d,r,PASS,/Users/example/private.csv\n",
                encoding="utf-8")
            with self.assertRaises(ValueError):
                bb._validate_results(root, digest)


class BuildClosureTest(unittest.TestCase):
    def _build(self, output):
        return bb.build(REPO, MODEL, LABELS, REPO / "results/results.csv",
                        output, extra=[REPO / CONVERSION])

    def test_build_stages_full_landmark_closure(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "bundle.zip"
            info = self._build(output)
            self.assertEqual(info["status"], "structure_verified")
            with zipfile.ZipFile(output) as archive:
                names = set(archive.namelist())
            for rel in EXPECTED_CLOSURE:
                self.assertIn(rel, names, f"not staged: {rel}")
            self.assertIn("src/main.py", names)
            self.assertIn("results/results.csv", names)

    def test_missing_runtime_asset_fails_naming_file(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "src"
            (source / "src").mkdir(parents=True)
            (source / "models").mkdir()
            (source / "results").mkdir()
            (source / "scripts/ml").mkdir(parents=True)
            for rel in ("src/main.py", "src/requirements.txt", "src/README.md",
                        "src/landmark_frontend.py"):
                (source / rel).write_text("x", encoding="utf-8")
            # features.py deliberately absent
            (source / MODEL).write_bytes((REPO / MODEL).read_bytes())
            (source / LABELS).write_bytes((REPO / LABELS).read_bytes())
            (source / "models/hand_landmarker.task").write_bytes(b"x")
            (source / "models/landmark_model_meta.json").write_bytes(b"{}")
            (source / "results/results.csv").write_bytes(
                (REPO / "results/results.csv").read_bytes())
            (source / CONVERSION).write_text("# export", encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                bb.build(source, MODEL, LABELS, source / "results/results.csv",
                         Path(directory) / "bundle.zip", extra=[source / CONVERSION])
            self.assertIn("src/features.py", str(ctx.exception))

    def test_s3d_build_does_not_require_landmark_assets(self):
        """Backward compatibility: an s3d selection needs no landmark files."""
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "repo"
            (source / "src").mkdir(parents=True)
            (source / "models").mkdir()
            (source / "results").mkdir()
            (source / "scripts/ml").mkdir(parents=True)
            for name in ("main.py", "requirements.txt", "README.md"):
                (source / "src" / name).write_text("x", encoding="utf-8")
            (source / "models/s3d_conv1d_qat_int8.tflite").write_bytes(b"model")
            (source / "models/label100.json").write_text(
                json.dumps({"A": 0, "B": 1}), encoding="utf-8")
            (source / "results/results.csv").write_bytes(
                (REPO / "results/results.csv").read_bytes())
            (source / "scripts/ml/export_tflite_tf.py").write_text(
                "# export", encoding="utf-8")
            info = bb.build(source, "models/s3d_conv1d_qat_int8.tflite",
                            "models/label100.json", source / "results/results.csv",
                            Path(directory) / "bundle.zip",
                            extra=[source / "scripts/ml/export_tflite_tf.py"])
            self.assertEqual(info["status"], "structure_verified")

    def test_official_still_blocked_without_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError) as ctx:
                bb.build(REPO, MODEL, LABELS, REPO / "results/results.csv",
                         Path(directory) / "bundle.zip",
                         official=True, extra=[REPO / CONVERSION])
            self.assertIn("organizer-generated predictions", str(ctx.exception))


class CleanExtractionIntegrationTest(unittest.TestCase):
    def test_built_zip_predicts_after_clean_extraction(self):
        if not PYTHON.is_file():
            self.skipTest("project interpreter not available")
        if not CLIP.is_file():
            self.skipTest("real clip not available")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "bundle.zip"
            bb.build(REPO, MODEL, LABELS, REPO / "results/results.csv",
                     output, extra=[REPO / CONVERSION])
            prediction = bb.verify_zip(output, clip=CLIP)

            self.assertEqual(prediction["clip_id"], CLIP.stem)
            self.assertEqual(prediction["prediction"], prediction["prediction"].upper())
            labels = json.loads((REPO / LABELS).read_text(encoding="utf-8"))
            self.assertIn(prediction["prediction"],
                          {str(label).upper() for label in labels})
            self.assertTrue(0 <= prediction["confidence"] <= 1)
            self.assertGreaterEqual(prediction["latency_ms"], 0)


class NoExitSafetyTest(unittest.TestCase):
    """main() must exit 2 on rejected/broken input, never traceback."""

    def test_build_without_conversion_script_exits_2(self):
        with self.assertRaises(SystemExit) as ctx:
            bb.main(["build", "--source", str(REPO), "--model", MODEL,
                     "--labels", LABELS, "--results",
                     str(REPO / "results/results.csv"), "--output",
                     str(Path(tempfile.gettempdir()) / "unused-bundle.zip")])
        self.assertEqual(ctx.exception.code, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)