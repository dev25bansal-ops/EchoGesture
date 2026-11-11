"""Regression tests for the hardened bundle-member check (D3 residual).

The original ``safe_member`` rejects colons only in the first
component, so ``foo/bar:baz`` and ``dir/file.mp4:stream`` pass.  The
hardened version must reject a colon in *any* component while keeping
legitimate members working.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # evaluation/device (contains v8_privacy + run_manifest)

from v8_privacy.secure_bundle import safe_member, safe_relative  # noqa: E402


class SafeMemberTest(unittest.TestCase):
    def test_colon_in_any_component_rejected(self):
        for name in (
            "data:file",
            "a:b",
            "foo/bar:baz",
            "results/predictions.jsonl:extra",
            "a/b:c/d",
            "dir/file.mp4:stream",
            "models/model.tflite:ads",
            "file:",
        ):
            self.assertFalse(safe_member(name), name)

    def test_legitimate_members_accepted(self):
        for name in (
            "src/main.py",
            "models/hand_landmarker.task",
            "results/results.csv",
            "release.json",
            "Dockerfile",
            "dir/file.mp4",
        ):
            self.assertTrue(safe_member(name), name)

    def test_traversal_and_absolute_rejected(self):
        for name in (
            "../x",
            "a/../b",
            "/absolute",
            "C:/evil",
            "C:evil",
            "//server/share/file",
            "a/./b",
            "",
            "a\x00b",
            "back\\slash",
        ):
            self.assertFalse(safe_member(name), name)

    def test_directory_entries(self):
        self.assertTrue(safe_member("models/"))
        self.assertFalse(safe_member("/"))


class SafeRelativeTest(unittest.TestCase):
    def test_colon_relative_rejected_before_fs_access(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                safe_relative(td, "foo/bar:baz")

    def test_escape_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "inside.txt").write_text("x", encoding="utf-8")
            self.assertEqual(safe_relative(root, "inside.txt"), (root / "inside.txt").resolve())
            with self.assertRaises(ValueError):
                safe_relative(root, "../escape.txt")
            with self.assertRaises(ValueError):
                safe_relative(root, "/etc/passwd")


if __name__ == "__main__":
    unittest.main()
