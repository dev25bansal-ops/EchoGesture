"""Frame-geometry contract.

The default must stay byte-identical to the historical 256 squash, because every
published result was measured there. Native resolution is opt-in and must
actually change the pixels — an earlier version silently resolved to 256 while
reporting "native", which a shape check alone would not have caught.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import landmark_frontend as LF  # noqa: E402

CLIPS = Path("/Users/abhijeetmiskin/AppData/EchoGesture/"
             "evaluation/device/v5_accuracy/data/val_30")


def _reset_main():
    """Reimport main so module-level state does not leak between cases."""
    for name in ("main",):
        sys.modules.pop(name, None)
    import main as M  # noqa: F811
    return M


class FrameSizeEnvTest(unittest.TestCase):
    def tearDown(self):
        os.environ.pop("ECHOGESTURE_FRAME_SIZE", None)

    def test_default_is_256(self):
        os.environ.pop("ECHOGESTURE_FRAME_SIZE", None)
        self.assertEqual(_reset_main()._frame_size(), 256)

    def test_native_sentinel_is_none(self):
        os.environ["ECHOGESTURE_FRAME_SIZE"] = "native"
        self.assertIsNone(_reset_main()._frame_size())
        os.environ["ECHOGESTURE_FRAME_SIZE"] = "NATIVE"
        self.assertIsNone(_reset_main()._frame_size())

    def test_integer_override(self):
        os.environ["ECHOGESTURE_FRAME_SIZE"] = "512"
        self.assertEqual(_reset_main()._frame_size(), 512)

    def test_invalid_raises(self):
        os.environ["ECHOGESTURE_FRAME_SIZE"] = "bogus"
        with self.assertRaises(ValueError):
            _reset_main()._frame_size()

    def test_nonpositive_raises(self):
        os.environ["ECHOGESTURE_FRAME_SIZE"] = "0"
        with self.assertRaises(ValueError):
            _reset_main()._frame_size()


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