"""Deterministic tests for release_build.py (A07 release engineering).

Covers the specific packaging-gap risks this task is about:
  - complete runtime closure assembly (the build_bundle.build() gap where
    src/features.py, src/landmark_frontend.py, and detector/meta assets were
    dropped),
  - missing-asset detection (omit one closure file -> build refuses),
  - path-traversal / symlink safety via the imported build_bundle.safe_member,
  - hash-mismatch rejection (selected model must be the pinned FP16 bytes),
  - model-size accounting (models/ total must stay under 20 MB).

The clean-extraction + real-predictor path is exercised by the test that
builds the actual bundle and, when the real clip and interpreter are present,
runs the predictor via the same subprocess shape as verify_zip. Tests never
download anything and leave no half-built zip behind.
"""

import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "scripts" / "release"))

import release_build as rb  # noqa: E402
from build_bundle import safe_member  # noqa: E402

PYTHON = "/Users/abhijeetmiskin/AppData/EchoGesture/.venv/bin/python"
CLIP = REPO / "evaluation/device/v5_accuracy/data/val_30/0283886564670357-WHAT FOR.mp4"


class ClosureAssemblyTest(unittest.TestCase):
    def test_closure_is_complete_and_honest(self):
        self.assertEqual(len(rb.CLOSURE), 11)
        for rel in rb.CLOSURE:
            self.assertTrue(safe_member(rel), f"closure path not safe: {rel}")
            full = REPO / rel
            self.assertTrue(full.is_file(), f"closure asset missing: {rel}")
            self.assertGreater(full.stat().st_size, 0, f"closure asset empty: {rel}")

    def test_selected_model_sha_matches_pinned_artifact(self):
        self.assertEqual(rb.sha256(REPO / rb.SELECTED_MODEL_REL),
                         rb.EXPECTED_MODEL_SHA)
        self.assertEqual((REPO / rb.SELECTED_MODEL_REL).stat().st_size,
                         rb.EXPECTED_MODEL_BYTES)

    def test_assemble_stages_every_closure_asset(self):
        with tempfile.TemporaryDirectory() as directory:
            staging = Path(directory) / "bundle"
            rb.assemble(staging)
            for rel in rb.CLOSURE:
                self.assertTrue((staging / rel).is_file(), f"not staged: {rel}")
            self.assertTrue((staging / "release.json").is_file())
            self.assertTrue((staging / "Dockerfile").is_file())

    def test_omitting_features_fails_and_names_the_file(self):
        with tempfile.TemporaryDirectory() as directory:
            staging = Path(directory) / "bundle"
            rb.assemble(staging)
            feature = staging / "src" / "features.py"
            feature.unlink()
            with self.assertRaises(ValueError) as ctx:
                rb.check_complete(staging)
            self.assertIn("src/features.py", str(ctx.exception))

    def test_detector_and_meta_assets_are_staged(self):
        with tempfile.TemporaryDirectory() as directory:
            staging = Path(directory) / "bundle"
            rb.assemble(staging)
            self.assertTrue((staging / "models/hand_landmarker.task").is_file())
            self.assertTrue((staging / "models/landmark_model_meta.json").is_file())


class HashAndSizeTest(unittest.TestCase):
    def _stage_with_model_bytes(self, directory, payload):
        staging = Path(directory) / "bundle"
        rb.assemble(staging)
        (staging / rb.SELECTED_MODEL_REL).write_bytes(payload)
        # release.json was written before the mutation; refresh so declared/actual align.
        return staging

    def test_hash_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            staging = Path(directory) / "bundle"
            rb.assemble(staging)
            model = staging / rb.SELECTED_MODEL_REL
            payload = bytearray(model.read_bytes())
            payload[0] ^= 0xFF
            model.write_bytes(bytes(payload))
            with self.assertRaises(ValueError) as ctx:
                rb.check_release_hash(staging)
            self.assertIn("does not match", str(ctx.exception))

    def test_wrong_model_rejected_even_if_self_consistent(self):
        with tempfile.TemporaryDirectory() as directory:
            staging = Path(directory) / "bundle"
            rb.assemble(staging)
            model = staging / rb.SELECTED_MODEL_REL
            other = b"0" * rb.EXPECTED_MODEL_BYTES
            model.write_bytes(other)
            rb.write_release_json(staging)  # re-emit so declared == actual (wrong model)
            with self.assertRaises(ValueError) as ctx:
                rb.check_release_hash(staging)
            self.assertIn("pinned", str(ctx.exception))

    def test_model_size_accounting_reports_detector_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            staging = Path(directory) / "bundle"
            rb.assemble(staging)
            total = rb.model_dir_bytes(staging)
            detector = (staging / "models/hand_landmarker.task").stat().st_size
            self.assertGreaterEqual(total, detector)
            self.assertLess(total, rb._MAX_MODEL_BYTES)

    def test_oversized_model_dir_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            staging = Path(directory) / "bundle"
            rb.assemble(staging)
            blob = staging / "models" / "oversized_blob.bin"
            blob.write_bytes(b"x" * (rb._MAX_MODEL_BYTES + 1))
            with self.assertRaises(ValueError) as ctx:
                rb.model_dir_bytes(staging)
            self.assertIn("exceed 20 MB", str(ctx.exception))


class SafeMemberTest(unittest.TestCase):
    def test_clean_relative_posix_member_is_safe(self):
        self.assertTrue(safe_member("src/main.py"))
        self.assertTrue(safe_member("models/landmark_classifier_fp16.tflite"))
        self.assertTrue(safe_member("src/"))

    def test_traversal_and_absolute_forms_rejected(self):
        for bad in ("../etc/passwd", "/etc/passwd", "a/../../b",
                    "src/./main.py", "src//main.py", "C:\\evil", "C:evil",
                    "file:stream", "dir/file.mp4:stream", ".."):
            self.assertFalse(safe_member(bad), f"should be unsafe: {bad}")

    def test_backslash_and_nul_rejected(self):
        for bad in ("src\\main.py", "a\x00b"):
            self.assertFalse(safe_member(bad), f"should be unsafe: {bad!r}")

    def test_symlink_member_rejected_on_extract(self):
        with tempfile.TemporaryDirectory() as directory:
            staging = Path(directory) / "bundle"
            rb.assemble(staging)
            zip_path = Path(directory) / "evil.zip"
            rb.zip_staging(staging, zip_path)
            # Inject a symlink member directly into a fresh zip.
            evil = zip_path.with_name("links.zip")
            with zipfile.ZipFile(evil, "w") as archive:
                info = zipfile.ZipInfo("link")
                info.external_attr = (0o120777 << 16)  # S_ISLNK in the high bits
                archive.writestr(info, "../etc/passwd")
            with self.assertRaises(ValueError) as ctx:
                rb.safe_extract(evil, Path(directory) / "out")
            self.assertIn("member type", str(ctx.exception))


class CleanExtractionTest(unittest.TestCase):
    def test_real_bundle_builds_and_predicts_cleanly(self):
        if not Path(PYTHON).is_file():
            self.skipTest("project interpreter not available")
        if not CLIP.is_file():
            self.skipTest("real clip not available")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "echogesture_landmark_release.zip"
            info = rb.build_release(output=output, clip_path=CLIP, python=PYTHON, verify=True)
            self.assertTrue(output.is_file())
            self.assertEqual(info["zip_sha256"], rb.sha256(output))
            self.assertEqual(info["zip_bytes"], output.stat().st_size)
            pred = info["extraction"]["prediction"]
            self.assertEqual(pred["clip_id"], CLIP.stem)
            self.assertEqual(pred["prediction"], pred["prediction"].upper())
            with (REPO / rb.SELECTED_LABELS_REL).open(encoding="utf-8") as stream:
                labels = json.load(stream)
            self.assertIn(pred["prediction"], {str(k).upper() for k in labels})
            self.assertTrue(0 <= pred["confidence"] <= 1)
            self.assertGreaterEqual(pred["latency_ms"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)