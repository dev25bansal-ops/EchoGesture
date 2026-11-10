"""Targeted regression tests for secure_manifest (D2).

Covers the A15 D2 reproduction (``../`` escape must now be rejected),
absolute-path escapes, symlink escapes, and the legitimate flows that
must keep working:

* sibling clips beside the manifest (``fetch_validation_batch`` layout),
* absolute paths inside the manifest subtree (existing tests),
* absolute/relative judge paths inside an explicit ``allowed_roots``
  entry (``ECHOGESTURE_INPUT_ROOT`` or the kwarg).
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # evaluation/device (contains v8_privacy + run_manifest)

from v8_privacy.secure_manifest import (  # noqa: E402
    SecureManifestError,
    effective_roots,
    parse_allowed_roots,
    read_manifest_secure,
    resolve_clip_path,
)

# Also verify the parent module's surface still exists read-only.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_manifest  # noqa: E402


def _write_manifest(manifest: Path, rows) -> None:
    import csv as _csv

    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = _csv.writer(stream)
        writer.writerow(["clip_id", "path", "ground_truth", "signer_id"])
        for row in rows:
            writer.writerow(row)


class ManifestTraversalTest(unittest.TestCase):
    """Attacks: every escape vector must be rejected."""

    def test_dotdot_escape_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sub = root / "sub"
            sub.mkdir()
            (root / "outside.mp4").write_bytes(b"x")
            manifest = sub / "m.csv"
            _write_manifest(manifest, [("c1", "../outside.mp4", "", "")])
            with self.assertRaisesRegex(SecureManifestError, "escapes|line 2"):
                read_manifest_secure(manifest)

    def test_dotdot_in_subdir_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            nested = root / "a" / "b"
            nested.mkdir(parents=True)
            (root / "a" / "clip.mp4").write_bytes(b"x")
            manifest = nested / "m.csv"
            _write_manifest(manifest, [("c1", "../clip.mp4", "", "")])
            with self.assertRaises(SecureManifestError):
                read_manifest_secure(manifest)

    def test_absolute_outside_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            clips = root / "clips"
            clips.mkdir()
            other = root / "other"
            other.mkdir()
            (other / "outside.mp4").write_bytes(b"x")
            manifest = clips / "m.csv"
            _write_manifest(manifest, [("c1", str(other / "outside.mp4"), "", "")])
            with self.assertRaisesRegex(SecureManifestError, "escapes|line 2"):
                read_manifest_secure(manifest)

    def test_absolute_etc_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = root / "m.csv"
            _write_manifest(manifest, [("c1", "/etc/passwd", "", "")])
            with self.assertRaises(SecureManifestError):
                read_manifest_secure(manifest)

    def test_tilde_expand_outside_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = root / "m.csv"
            _write_manifest(manifest, [("c1", "~/outside.mp4", "", "")])
            with self.assertRaises(SecureManifestError):
                read_manifest_secure(manifest)

    def test_symlink_escape_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            clips = root / "clips"
            clips.mkdir()
            outside_dir = root / "outside"
            outside_dir.mkdir()
            (outside_dir / "secret.mp4").write_bytes(b"x")
            link = clips / "link.mp4"
            try:
                link.symlink_to(outside_dir / "secret.mp4")
            except (OSError, NotImplementedError):
                self.skipTest("symlinks unavailable")
            manifest = clips / "m.csv"
            _write_manifest(manifest, [("c1", "link.mp4", "", "")])
            with self.assertRaises(SecureManifestError):
                read_manifest_secure(manifest)

    def test_nested_symlink_dir_escape_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            clips = root / "clips"
            clips.mkdir()
            outside_dir = root / "outside"
            outside_dir.mkdir()
            (outside_dir / "secret.mp4").write_bytes(b"x")
            linkdir = clips / "linkdir"
            try:
                linkdir.symlink_to(outside_dir, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("symlinks unavailable")
            manifest = clips / "m.csv"
            _write_manifest(manifest, [("c1", "linkdir/secret.mp4", "", "")])
            with self.assertRaises(SecureManifestError):
                read_manifest_secure(manifest)

    def test_nul_byte_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = root / "m.csv"
            _write_manifest(manifest, [("c1", "clip\x00.mp4", "", "")])
            with self.assertRaises(SecureManifestError):
                read_manifest_secure(manifest)

    def test_windows_drive_path_rejected_on_posix(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = root / "m.csv"
            _write_manifest(manifest, [("c1", "C:/evil.mp4", "", "")])
            # On POSIX this resolves as a relative filename "C:/evil.mp4"
            # under the manifest dir; the file does not exist there so
            # either the containment or the not-found error must fire.
            with self.assertRaises(SecureManifestError):
                read_manifest_secure(manifest)


class LegitimatePathsTest(unittest.TestCase):
    """Legitimate local/absolute/judge paths must still work."""

    def test_sibling_clip_accepted(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "clip.mp4").write_bytes(b"x")
            manifest = root / "m.csv"
            _write_manifest(manifest, [("c1", "clip.mp4", "CANDY", "P15")])
            items = read_manifest_secure(manifest)
            self.assertEqual(len(items), 1)
            self.assertTrue(items[0][1].is_file())
            self.assertEqual(items[0][0], "c1")

    def test_subdir_clip_accepted(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sub = root / "clips" / "set1"
            sub.mkdir(parents=True)
            (sub / "clip.mp4").write_bytes(b"x")
            manifest = root / "m.csv"
            _write_manifest(manifest, [("c1", "clips/set1/clip.mp4", "", "")])
            items = read_manifest_secure(manifest)
            self.assertTrue(items[0][1].is_file())

    def test_absolute_inside_manifest_subtree_accepted(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            clip = root / "clip.mp4"
            clip.write_bytes(b"x")
            manifest = root / "m.csv"
            _write_manifest(manifest, [("c1", str(clip), "", "")])
            items = read_manifest_secure(manifest)
            self.assertTrue(items[0][1].is_file())

    def test_fetch_validation_batch_layout_accepted(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            out = root / "batch"
            out.mkdir()
            clips = out / "clips"
            clips.mkdir()
            for i in range(3):
                (clips / f"clip{i}.mp4").write_bytes(b"x")
            manifest = out / "manifest.csv"
            _write_manifest(
                manifest,
                [(f"c{i}", f"clips/clip{i}.mp4", "", "") for i in range(3)],
            )
            items = read_manifest_secure(manifest)
            self.assertEqual(len(items), 3)

    def test_absolute_judge_path_via_allowed_roots_kwarg(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            judge_clips = root / "judge" / "clips"
            judge_clips.mkdir(parents=True)
            clip = judge_clips / "judge01.mp4"
            clip.write_bytes(b"x")
            manifest_dir = root / "eval"
            manifest_dir.mkdir()
            manifest = manifest_dir / "m.csv"
            _write_manifest(manifest, [("c1", str(clip), "", "")])
            with self.assertRaises(SecureManifestError):
                read_manifest_secure(manifest)  # no root configured: rejected
            items = read_manifest_secure(manifest, allowed_roots=[judge_clips])
            self.assertTrue(items[0][1].is_file())

    def test_absolute_judge_path_via_env(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            judge_clips = root / "organizer"
            judge_clips.mkdir()
            clip = judge_clips / "clip.mp4"
            clip.write_bytes(b"x")
            manifest = root / "work" / "m.csv"
            manifest.parent.mkdir()
            _write_manifest(manifest, [("c1", str(clip), "", "")])
            env = {os.environ.get("ECHOGESTURE_INPUT_ROOT", ""): ""}
            env = {"ECHOGESTURE_INPUT_ROOT": str(judge_clips)}
            items = read_manifest_secure(manifest, env=env)
            self.assertTrue(items[0][1].is_file())

    def test_relative_judge_path_via_input_root(self):
        # Manifest in one dir, clips in a sibling dir under a shared
        # judge root; relative "../judge/clip.mp4" resolves inside the
        # judge root and must be accepted only when that root is set.
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            judge_root = root / "judge"
            eval_dir = judge_root / "eval"
            clip_dir = judge_root / "clips"
            eval_dir.mkdir(parents=True)
            clip_dir.mkdir()
            (clip_dir / "clip.mp4").write_bytes(b"x")
            manifest = eval_dir / "m.csv"
            _write_manifest(manifest, [("c1", "../clips/clip.mp4", "", "")])
            with self.assertRaises(SecureManifestError):
                read_manifest_secure(manifest)
            items = read_manifest_secure(manifest, allowed_roots=[judge_root])
            self.assertTrue(items[0][1].is_file())

    def test_missing_file_inside_root_still_rejected_as_not_found(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = root / "m.csv"
            _write_manifest(manifest, [("c1", "missing.mp4", "", "")])
            with self.assertRaisesRegex(SecureManifestError, "clip not found"):
                read_manifest_secure(manifest)

    def test_duplicate_and_missing_columns_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "clip.mp4").write_bytes(b"x")
            manifest = root / "m.csv"
            _write_manifest(manifest, [("c1", "clip.mp4", "", ""), ("c1", "clip.mp4", "", "")])
            with self.assertRaisesRegex(SecureManifestError, "duplicate"):
                read_manifest_secure(manifest)


class ResolveHelperTest(unittest.TestCase):
    def test_resolve_clip_path_direct(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "a.mp4").write_bytes(b"x")
            manifest = root / "m.csv"
            self.assertTrue(resolve_clip_path("a.mp4", manifest).is_file())
            with self.assertRaises(SecureManifestError):
                resolve_clip_path("../a.mp4", manifest)

    def test_effective_roots_includes_manifest_dir_and_parsed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = root / "sub" / "m.csv"
            roots = effective_roots(manifest, allowed_roots=[root / "judge"])
            self.assertIn((root / "sub").resolve(), roots)
            self.assertIn((root / "judge").resolve(), roots)

    def test_secure_error_is_value_error(self):
        # Callers catching ValueError keep working during migration.
        self.assertTrue(issubclass(SecureManifestError, ValueError))

    def test_parent_module_readable(self):
        self.assertTrue(callable(run_manifest.read_manifest))
        self.assertTrue(callable(run_manifest.evaluate))


if __name__ == "__main__":
    unittest.main()
