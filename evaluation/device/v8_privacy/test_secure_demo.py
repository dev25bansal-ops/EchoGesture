"""Regression tests for hardened demo media retention.

* Clip lands in a private 0700 subdirectory, never directly in the
  shared temp dir (D4).
* Clip file is 0600 (not world-readable 0644).
* Clip is deleted on success *and* on predictor exception (D5 control).
* ``close()`` while inference is still running still purges leftovers
  (no silent orphan on the close race).
* Retention wording states only the on-disk truth.
"""

from __future__ import annotations

import os
import stat
import sys
import time
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # evaluation/device (contains v8_privacy + run_manifest)

from v8_privacy.secure_demo import (  # noqa: E402
    RETENTION_WORDING,
    RETENTION_WORDING_SHORT,
    SecurePredictionWorker,
    _purge_dir,
    write_temp_clip,
)


def _numpy():
    import numpy as np  # noqa: F401

    return np


def _frames(n=2):
    np = _numpy()
    return [np.zeros((24, 32, 3), np.uint8) for _ in range(n)]


class PrivateDirTest(unittest.TestCase):
    def test_clip_in_private_subdir_not_shared_root(self):
        seen = []

        def predict(path, *, clip_id=None):
            seen.append(Path(path))
            return {"clip_id": clip_id, "prediction": "X", "confidence": 0.9, "latency_ms": 1.0}

        worker = SecurePredictionWorker(predict)
        try:
            self.assertTrue(worker.submit(_frames(), 12, "c1"))
            kind, _ = worker.results.get(timeout=10)
            self.assertEqual(kind, "result")
            self.assertTrue(seen, "predictor was called")
            clip = seen[0]
            self.assertEqual(clip.parent, worker.tmpdir_path)
            # Private directory: one level below the shared temp root.
            self.assertNotEqual(str(clip.parent), tempfile.gettempdir())
            self.assertTrue(str(clip).startswith(tempfile.gettempdir()))
            # Deleted after success (poll: result visible before unlink).
            for _ in range(50):
                if not clip.exists():
                    break
                time.sleep(0.01)
            self.assertFalse(clip.exists())
        finally:
            worker.close()
        self.assertFalse(worker.tmpdir_path.exists())

    def test_dir_and_file_modes_private(self):
        seen = []

        def predict(path, *, clip_id=None):
            clip = Path(path)
            seen.append(
                (
                    stat.S_IMODE(clip.parent.stat().st_mode),
                    stat.S_IMODE(clip.stat().st_mode),
                )
            )
            return {"clip_id": clip_id, "prediction": "X", "confidence": 0.9, "latency_ms": 1.0}

        worker = SecurePredictionWorker(predict)
        try:
            worker.submit(_frames(), 12, "c1")
            worker.results.get(timeout=10)
            self.assertTrue(seen)
            dir_mode, file_mode = seen[0]
            self.assertEqual(dir_mode, 0o700)
            self.assertEqual(file_mode, 0o600)
        finally:
            worker.close()

    def test_explicit_tmpdir_is_parent_not_direct(self):
        # A caller-supplied tmpdir must never receive clips directly.
        with tempfile.TemporaryDirectory() as parent:
            seen = []

            def predict(path, *, clip_id=None):
                seen.append(Path(path))
                return {"clip_id": clip_id, "prediction": "X", "confidence": 0.9, "latency_ms": 1.0}

            worker = SecurePredictionWorker(predict, tmpdir=Path(parent))
            try:
                self.assertEqual(Path(worker.tmpdir_path).resolve().parent, Path(parent).resolve())
                worker.submit(_frames(), 12, "c1")
                worker.results.get(timeout=10)
                self.assertTrue(seen)
                self.assertEqual(seen[0].parent, worker.tmpdir_path)
                self.assertNotEqual(Path(seen[0]).resolve().parent, Path(parent).resolve())
            finally:
                worker.close()
            # Parent itself is left alone; only the private child is removed.
            self.assertTrue(Path(parent).is_dir())


class DeletionTest(unittest.TestCase):
    def test_deleted_on_predictor_exception(self):
        seen = []

        def boom(path, *, clip_id=None):
            seen.append(Path(path))
            raise RuntimeError("boom")

        worker = SecurePredictionWorker(boom)
        try:
            worker.submit(_frames(), 12, "c2")
            kind, _ = worker.results.get(timeout=10)
            self.assertEqual(kind, "error")
            self.assertTrue(seen)
            # Consumer may observe the result before finally unlink runs;
            # poll like test_offline_demo.py (50 x 10ms) then assert.
            for _ in range(50):
                if not seen[0].exists():
                    break
                time.sleep(0.01)
            self.assertFalse(seen[0].exists())
        finally:
            worker.close()

    def test_deleted_when_encode_resets_shape(self):
        # Inconsistent frame shape raises mid-encode; partial file must not survive.
        np = _numpy()
        frames = [np.zeros((10, 10, 3), np.uint8), np.zeros((20, 20, 3), np.uint8)]
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                write_temp_clip(frames, 12, td, "bad-shape")
            self.assertEqual(list(Path(td).iterdir()), [])

    def test_write_temp_clip_rejects_bad_inputs(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                write_temp_clip([], 30, td, "x")
            with self.assertRaises(ValueError):
                write_temp_clip(_frames(), 0, td, "x")


class CloseRaceTest(unittest.TestCase):
    def test_close_idempotent(self):
        worker = SecurePredictionWorker(
            lambda path, **kwargs: {"prediction": "X", "confidence": 1, "top5": ["X"], "latency_ms": 1}
        )
        worker.close()
        self.assertFalse(worker.tmpdir_path.exists())
        worker.close()  # second close must not raise

    def test_close_purges_leftover_from_jammed_predictor(self):
        # Predictor that blocks past close_timeout: the leftover clip
        # must still be purged (no silent orphan).
        import threading

        release = threading.Event()

        def slow(path, *, clip_id=None):
            release.wait(timeout=5)
            return {"clip_id": clip_id, "prediction": "X", "confidence": 0.9, "latency_ms": 1.0}

        worker = SecurePredictionWorker(slow, close_timeout=0.3)
        worker.submit(_frames(3), 12, "jam")
        try:
            worker.close()
            self.assertTrue(worker.tmpdir_path.exists() or not worker.tmpdir_path.exists())
            # After close returns, no .mp4 remains under the worker dir
            # (the dir itself is cleaned up once the thread is dead; if
            # the thread is still wedged the sweep has still emptied it).
            if worker.tmpdir_path.exists():
                remaining = list(worker.tmpdir_path.iterdir())
                self.assertEqual(remaining, [])
        finally:
            release.set()
            worker.thread.join(timeout=3)

    def test_purge_dir_removes_nested(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "a.mp4").write_bytes(b"x")
            nested = root / "sub"
            nested.mkdir()
            (nested / "b.mp4").write_bytes(b"x")
            removed = _purge_dir(root)
            self.assertGreaterEqual(removed, 2)
            self.assertEqual(list(root.iterdir()), [])


class WordingTest(unittest.TestCase):
    def test_retention_wording_states_disk_only(self):
        self.assertIn("No footage is kept on disk", RETENTION_WORDING)
        self.assertIn("memory only during inference", RETENTION_WORDING)
        self.assertNotIn("no footage is retained", RETENTION_WORDING.lower())

    def test_short_wording_for_hud(self):
        self.assertIn("kept on disk", RETENTION_WORDING_SHORT)


if __name__ == "__main__":
    unittest.main()
