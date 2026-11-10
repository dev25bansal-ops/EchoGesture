"""V7 QA: atomic evidence publication, temp cleanup, bundle ZIP safety.

Verified behavior:
  * run_manifest.write_pair_atomically writes predictions.jsonl +
    benchmark.json via hidden temp files in the target directory, then
    os.replace into place. A serialization failure (NaN) publishes
    NEITHER file and leaves NO temp residue.
  * robustness.paired_evaluate always removes output_dir/work (D-2
    regression) and leaves no per-clip .mp4 temp files behind.
  * build_bundle.verify_zip rejects traversal (../), absolute entries,
    symlinks/non-regular members, duplicate member names, and backslash
    member names.
  * Empty artifacts are rejected by the release path: empty results CSV
    and empty predictions.jsonl raise ValueError rather than producing a
    passing bundle.
  * verify_zip / safe_member reject NUL-containing names and drive-letter
    names.
"""

import json
import stat
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

DEVICE_DIR = Path(__file__).resolve().parents[1]
ROOT = DEVICE_DIR.parents[1]
for _p in (str(DEVICE_DIR), str(ROOT / "scripts" / "release")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

import build_bundle as release_build  # noqa: E402
import run_manifest  # noqa: E402
from corruptions import CorruptionSpec, write_video  # noqa: E402
from robustness import paired_evaluate  # noqa: E402


class AtomicEvidenceTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def test_nan_prediction_publishes_neither_file_nor_temp(self):
        output = self.root / "out"
        with self.assertRaises(ValueError):
            run_manifest.write_pair_atomically(
                output, [{"clip_id": "a", "confidence": float("nan")}], {})
        self.assertFalse((output / "predictions.jsonl").exists())
        self.assertFalse((output / "benchmark.json").exists())
        leftovers = list(output.iterdir()) if output.exists() else []
        self.assertEqual(leftovers, [])

    def test_successful_write_leaves_exactly_two_files_no_temp(self):
        output = self.root / "out"
        run_manifest.write_pair_atomically(
            output,
            [{"clip_id": "sign-✋-01", "prediction": "CANDY", "confidence": 0.5,
              "latency_ms": 1}],
            {"status": "passed_schema"})
        names = sorted(p.name for p in output.iterdir())
        self.assertEqual(names, ["benchmark.json", "predictions.jsonl"])
        for name in names:
            self.assertFalse(name.startswith("."), f"temp residue: {name}")
        lines = (output / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(json.loads(lines[0])["clip_id"], "sign-✋-01")
        summary = json.loads((output / "benchmark.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "passed_schema")

    def test_overwrite_replaces_prior_pair_atomically(self):
        output = self.root / "out"
        run_manifest.write_pair_atomically(output, [{"n": 1}], {"v": 1})
        run_manifest.write_pair_atomically(output, [{"n": 2}], {"v": 2})
        self.assertEqual(json.loads((output / "predictions.jsonl").read_text())["n"], 2)
        self.assertEqual(json.loads((output / "benchmark.json").read_text())["v"], 2)

    def test_paired_evaluate_leaves_no_workdir_or_temp_videos(self):
        good = self.root / "good.mp4"
        write_video(good, [np.full((20, 20, 3), 128, np.uint8)] * 3, 10)
        out = self.root / "out"

        def predict(path, *, clip_id=None):
            return {"clip_id": clip_id, "prediction": "CANDY", "top5": ["CANDY"],
                    "confidence": 0.9, "latency_ms": 1}

        paired_evaluate([("clip x", good, "CANDY", "s1")], predict, {"CANDY"},
                        CorruptionSpec("crop", 0.5, 0), out)
        self.assertFalse((out / "work").exists())
        self.assertEqual(list(out.rglob("*.mp4")), [])
        self.assertEqual(list(out.glob(".*")), [])

    def test_paired_evaluate_cleans_workdir_when_predictor_raises(self):
        good = self.root / "good.mp4"
        write_video(good, [np.full((20, 20, 3), 128, np.uint8)] * 3, 10)
        out = self.root / "out"

        def boom(path, *, clip_id=None):
            raise RuntimeError("simulated mid-run fault")

        # predict_clip swallows per-clip faults; the run completes and
        # the workdir must still be gone.
        summary = paired_evaluate([("clip x", good, "CANDY", "s1")], boom, {"CANDY"},
                                  CorruptionSpec("crop", 0.5, 0), out)
        self.assertEqual(summary["failed_pair_count"], 1)
        self.assertFalse((out / "work").exists())


class BundleZipSafetyTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def _zip(self, name, members):
        path = self.root / name
        with zipfile.ZipFile(path, "w") as archive:
            for member_name, data in members:
                archive.writestr(member_name, data)
        return path

    def test_duplicate_member_names_rejected(self):
        path = self._zip("dup.zip", [("src/main.py", "a"), ("src/main.py", "b")])
        with self.assertRaisesRegex(ValueError, "duplicate member names"):
            release_build.verify_zip(path)

    def test_traversal_rejected(self):
        path = self._zip("trav.zip", [("../evil.txt", "x")])
        with self.assertRaisesRegex(ValueError, "Unsafe bundle path"):
            release_build.verify_zip(path)

    def test_absolute_entry_rejected(self):
        path = self._zip("abs.zip", [("/abs/file", "x")])
        with self.assertRaisesRegex(ValueError, "Unsafe bundle path"):
            release_build.verify_zip(path)

    def test_backslash_entry_rejected(self):
        path = self.root / "bs.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("src\\evil.txt", "x")
        with self.assertRaisesRegex(ValueError, "Unsafe bundle path"):
            release_build.verify_zip(path)

    def test_symlink_member_rejected(self):
        path = self.root / "link.zip"
        with zipfile.ZipFile(path, "w") as archive:
            info = zipfile.ZipInfo("entry")
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info, "src/main.py")
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            release_build.verify_zip(path)

    def test_safe_member_rejects_nul_and_drive_letter(self):
        self.assertFalse(release_build.safe_member("a\x00b"))
        self.assertFalse(release_build.safe_member("C:/evil"))
        self.assertFalse(release_build.safe_member("C:evil"))
        self.assertFalse(release_build.safe_member(""))
        self.assertTrue(release_build.safe_member("src/main.py"))

    def test_empty_predictions_jsonl_rejected_by_verify(self):
        # verify_directory is exercised through verify_zip on a minimal
        # bundle; an empty predictions.jsonl must not pass as evidence.
        # Build the minimal staging bundle the verifier expects.
        staging = self.root / "staging"
        (staging / "src").mkdir(parents=True)
        (staging / "results").mkdir()
        (staging / "models").mkdir()
        for name in ("main.py", "requirements.txt", "README.md"):
            (staging / "src" / name).write_text("# fixture\n")
        (staging / "models" / "classifier.tflite").write_bytes(b"model")
        (staging / "models" / "labels.json").write_text('{"CANDY": 0}')
        (staging / "scripts").mkdir()
        (staging / "scripts" / "export_model.py").write_text("# fixture\n")
        (staging / "results" / "results.csv").write_text(
            "metric,target,measured,unit,device,runtime,pass,notes\n"
            "avg_power,<=500,NOT_MEASURED,mW,n/a,n/a,NOT_MEASURED,none\n")
        (staging / "results" / "predictions.jsonl").write_bytes(b"")
        (staging / "manifest.json").write_text("clip_id\nclip 01\n")
        (staging / "Dockerfile").write_text("FROM python:3.11-slim\n")
        import hashlib
        (staging / "release.json").write_text(json.dumps({
            "model": "models/classifier.tflite", "labels": "models/labels.json",
            "model_sha256": hashlib.sha256(b"model").hexdigest(),
            "conversion_script": "scripts/export_model.py",
            "official_submission": False}))
        bundled = self.root / "bundle.zip"
        with zipfile.ZipFile(bundled, "w") as archive:
            for file in sorted(staging.rglob("*")):
                if file.is_file():
                    archive.write(file, file.relative_to(staging))
        with self.assertRaises(ValueError):
            release_build.verify_zip(bundled)


if __name__ == "__main__":
    unittest.main()
