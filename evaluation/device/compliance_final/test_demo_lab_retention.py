"""Demo-media retention tests for app/demo_lab (A2 Fix 3).

The HUD PNG embeds a rendered video frame, so the demo lab must not write it
to a persistent shared default location. These tests patch ``run_video`` so no
model, camera, or real clip is needed, then exercise the CLI contract:

* no ``--outdir``  -> the output lands in a private per-run directory that is
  gone once ``main()`` returns (nothing persists);
* explicit ``--outdir`` -> the operator opted in and output persists.

Also a static guard that the old ``tempfile.gettempdir()/echogesture-demo-lab``
default is gone.
"""

from __future__ import annotations

import io
import contextlib
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent          # evaluation/device/compliance_final/
_REPO = HERE.parent.parent.parent               # repo root
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import app.demo_lab as demo_lab  # noqa: E402


class DemoLabRetentionTests(unittest.TestCase):
    def setUp(self):
        self.seen_outdirs: list[Path] = []
        self._original_run_video = demo_lab.run_video

        def fake_run_video(video, outdir, threshold=0.25):
            outdir = Path(outdir)
            outdir.mkdir(parents=True, exist_ok=True)
            (outdir / "hud-clip.png").write_bytes(b"\x89PNG\r\n\x1a\n")
            (outdir / "hud-clip.json").write_text("{}", encoding="utf-8")
            self.seen_outdirs.append(outdir)
            return {"render_written": True, "render_path": str(outdir / "hud-clip.png")}

        demo_lab.run_video = fake_run_video

    def tearDown(self):
        demo_lab.run_video = self._original_run_video

    def _main(self, *args):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = demo_lab.main(list(args))
        return code, out.getvalue()

    def test_default_run_writes_only_to_ephemeral_dir(self):
        code, stdout = self._main("--video", "clip.mp4")
        self.assertEqual(code, 0)
        self.assertEqual(len(self.seen_outdirs), 1)
        outdir = self.seen_outdirs[0]
        # The per-run directory is cleaned up when main() returns.
        self.assertFalse(outdir.exists(), f"persistent demo media left at {outdir}")
        self.assertIn('"retained": false', stdout.replace("False", "false"))

    def test_default_run_never_uses_shared_demo_lab_location(self):
        self._main("--video", "clip.mp4")
        shared = Path(tempfile.gettempdir()) / "echogesture-demo-lab"
        self.assertNotEqual(self.seen_outdirs[0], shared)
        # And nothing was created there by this run.
        self.assertFalse(shared.exists() and any(shared.iterdir()
                                                 if shared.is_dir() else []))

    def test_explicit_outdir_persists(self):
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "operator-chosen"
            code, _ = self._main("--video", "clip.mp4", "--outdir", str(target))
            self.assertEqual(code, 0)
            self.assertTrue((target / "hud-clip.png").is_file())
            self.assertTrue((target / "hud-clip.json").is_file())

    def test_source_has_no_persistent_shared_default(self):
        source = (_REPO / "app" / "demo_lab" / "__init__.py").read_text(encoding="utf-8")
        self.assertNotIn("gettempdir", source)
        self.assertIn("ephemeral_outdir", source)


if __name__ == "__main__":
    unittest.main()