"""Frame-geometry contract.

The shipped path is NATIVE geometry, unconditionally. This is not a preference:
the nine FP16 members were trained on native-geometry landmarks, and the Android
app feeds native frames, so Python and Android only agree when Python is native
too (measured: native 84.94%, 512-letterbox 85.10%, 256-squash 81.51% on the full
test set — `src/main.py`'s comment block carries the numbers and the script).

This file previously tested `_frame_size()` and the `ECHOGESTURE_FRAME_SIZE`
override. Both were DELETED in 8740f27 when the squash default was removed, so
those five tests were asserting against a function that no longer exists and the
whole class failed with AttributeError. They are replaced by a guard on the live
contract: `_predict_locked` must call `clip_to_landmarks` with `size=None`.

The `read_frames` tests below are genuine API tests and still apply — the
frontend still accepts a `size` argument even though `predict()` no longer
passes one.
"""

from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import landmark_frontend as LF  # noqa: E402

CLIPS = Path("/Users/abhijeetmiskin/AppData/EchoGesture/"
             "evaluation/device/v5_accuracy/data/val_30")

MAIN_PY = ROOT / "src" / "main.py"


def _call_kwarg(func: ast.AST, callee: str, kwarg: str):
    """Return the literal value of `callee(..., kwarg=<literal>)` in func, or None."""
    for node in ast.walk(func):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        name = target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", None)
        if name != callee:
            continue
        for kw in node.keywords:
            if kw.arg == kwarg:
                return kw.value
    return None


class PredictGeometryContractTest(unittest.TestCase):
    """The shipped predict() path must run native geometry, not a squash."""

    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(MAIN_PY.read_text(encoding="utf-8"))
        fns = {n.name: n for n in cls.tree.body if isinstance(n, ast.FunctionDef)}
        if "_predict_locked" not in fns:
            raise AssertionError("_predict_locked not found in src/main.py — the "
                                 "contract test needs updating, not deleting")
        cls.fn = fns["_predict_locked"]

    def test_predict_passes_size_none(self):
        value = _call_kwarg(self.fn, "clip_to_landmarks", "size")
        self.assertIsNotNone(value, "predict() no longer calls clip_to_landmarks "
                                    "with an explicit size= — the geometry is now "
                                    "implicit and this contract cannot be checked")
        self.assertIsInstance(value, ast.Constant)
        self.assertIsNone(value.value,
                          "predict() must pass size=None (native). A squash here "
                          "puts the Python path on different geometry from the "
                          "Android app and silently costs ~3.4 points.")

    def test_no_frame_size_env_override_remains(self):
        """The deleted env var must not creep back in without the docs changing."""
        src = MAIN_PY.read_text(encoding="utf-8")
        self.assertNotIn("ECHOGESTURE_FRAME_SIZE", src,
                         "ECHOGESTURE_FRAME_SIZE was removed in 8740f27; every "
                         "document now states that predict() is native. If it "
                         "returns, it needs a test of its own and the docs fixed.")


class ReadFramesGeometryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not CLIPS.is_dir():
            raise unittest.SkipTest("val_30 clips not present")

    def _clip(self):
        return str(sorted(CLIPS.glob("*.mp4"))[0])

    def test_default_matches_explicit_256(self):
        c = self._clip()
        a = LF.read_frames(c, max_frames=8)
        b = LF.read_frames(c, max_frames=8, size=256)
        self.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            self.assertTrue((x == y).all())

    def test_native_keeps_source_resolution(self):
        c = self._clip()
        frames = LF.read_frames(c, max_frames=4, size=None)
        self.assertTrue(frames, "no frames decoded")
        h, w = frames[0].shape[:2]
        self.assertEqual((w, h), (640, 480),
                         "native must not resize; source clips are 640x480")

    def test_native_differs_from_squash(self):
        """The whole point: native and squash must not produce the same input."""
        c = self._clip()
        a = LF.read_frames(c, max_frames=4, size=256)[0]
        b = LF.read_frames(c, max_frames=4, size=None)[0]
        self.assertNotEqual(a.shape, b.shape)
        self.assertEqual(a.shape[:2], (256, 256))
        self.assertEqual(b.shape[:2], (480, 640))

    def test_square_override_still_applies(self):
        c = self._clip()
        frames = LF.read_frames(c, max_frames=2, size=512)
        self.assertEqual(frames[0].shape[:2], (512, 512))


if __name__ == "__main__":
    unittest.main(verbosity=2)