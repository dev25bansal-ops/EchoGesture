"""Headless tests for the A10 V7 offline demo wrappers.

Covers: stale-result discard, reopen epoch bump, worker close mid-
inference with temp-purge, width-fitted HUD text, single-decode file runs
with proper exit codes, and model-identity pinning to the FP16 SHA.
No camera, no GUI, no network.
"""

import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.v7_demo.reliable import (
    EXPECTED_FP16_SHA256,
    ReliableCaptionState,
    ReliableHUD,
    ReliableWorker,
    fit_text,
    run_file,
    run_session,
)
from app.offline_demo import CaptionState, CameraUnavailableError
from evaluation.device.v7_demo.identity import check_model_identity


class StaleResultTest(unittest.TestCase):
    def test_older_sequence_is_discarded_as_stale(self):
        state = ReliableCaptionState(0.25)
        seq1 = state.submit()
        seq2 = state.submit()
        # Late result for the older capture must not overwrite the newer.
        outcome = state.apply(seq1, 0, "result",
                              {"prediction": "OLD", "confidence": 0.9, "top5": ["OLD"], "latency_ms": 1})
        self.assertEqual(outcome, "STALE")
        self.assertEqual(state.state, "PROCESSING")
        self.assertIsNone(state.result)
        outcome = state.apply(seq2, 0, "result",
                              {"prediction": "NEW", "confidence": 0.9, "top5": ["NEW"], "latency_ms": 1})
        self.assertEqual(outcome, "CAPTION")
        self.assertEqual(state.result["prediction"], "NEW")
        self.assertEqual(state.snapshot()["stale_count"], 1)

    def test_reopen_epoch_discards_inflight_result(self):
        state = ReliableCaptionState(0.25)
        seq = state.submit()
        epoch = state.snapshot()["epoch"]
        state.reopen()
        outcome = state.apply(seq, epoch, "result",
                              {"prediction": "BOOK", "confidence": 0.9, "top5": ["BOOK"], "latency_ms": 1})
        self.assertEqual(outcome, "STALE")
        # State after reopen is a fresh WAITING, not the old caption.
        self.assertEqual(state.state, "WAITING")
        self.assertIsNone(state.result)

    def test_error_result_also_seq_guarded(self):
        state = ReliableCaptionState(0.25)
        seq1 = state.submit()
        seq2 = state.submit()
        outcome = state.apply(seq1, 0, "error", "old failure")
        self.assertEqual(outcome, "STALE")
        self.assertEqual(state.state, "PROCESSING")


class WorkerCloseTest(unittest.TestCase):
    def test_close_mid_inference_purges_temp_and_does_not_raise(self):
        started = threading_event = None
        gate = mock.Mock()
        gate.wait = lambda timeout=None: True
        import threading
        entered = threading.Event()

        def slow_predict(path, *, clip_id=None):
            entered.set()
            time.sleep(1.5)
            return {"prediction": "X", "confidence": 1, "top5": ["X"], "latency_ms": 1}

        state = ReliableCaptionState(0.25)
        worker = ReliableWorker(slow_predict, state, close_timeout=0.2)
        try:
            frames = [np.zeros((24, 32, 3), np.uint8)] * 3
            seq = worker.submit(frames, 12, "slow")
            self.assertIsNotNone(seq)
            self.assertTrue(entered.wait(timeout=5))
            # Close while inference is still running: must not raise.
            worker.close()
            # No temp clips remain.
            leftovers = list(worker.tmpdir_path.iterdir()) if worker.tmpdir_path.exists() else []
            self.assertEqual(leftovers, [])
            # Second close is idempotent.
            worker.close()
        finally:
            worker.close()

    def test_close_after_drain_deletes_clip(self):
        seen = []

        def predict(path, *, clip_id=None):
            seen.append(Path(path))
            return {"prediction": "BOOK", "confidence": 0.8, "top5": ["BOOK"], "latency_ms": 3.2}

        state = ReliableCaptionState(0.25)
        worker = ReliableWorker(predict, state, close_timeout=5.0)
        try:
            frames = [np.zeros((24, 32, 3), np.uint8)] * 3
            seq = worker.submit(frames, 12, "fast")
            self.assertIsNotNone(seq)
            for _ in range(50):
                if seen and not seen[0].exists():
                    break
                time.sleep(0.05)
            self.assertFalse(seen[0].exists())
        finally:
            worker.close()
        # close() deletes the tmpdir itself; no directory, no leftovers.
        self.assertFalse(worker.tmpdir_path.exists())


class FitTextTest(unittest.TestCase):
    def test_short_text_keeps_base_scale(self):
        text, scale = fit_text("BOOK", 640, 1.1, 2)
        self.assertEqual(text, "BOOK")
        self.assertAlmostEqual(scale, 1.1)

    def test_long_text_shrinks_to_fit_width(self):
        long_text = "A" * 120
        text, scale = fit_text(long_text, 200, 1.1, 1)
        self.assertLessEqual(scale, 1.1)
        (w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        self.assertLessEqual(w, 200)

    def test_missing_text_yields_empty(self):
        text, _ = fit_text(None, 200, 1.1, 1)
        self.assertEqual(text, "")


class HUDTest(unittest.TestCase):
    def test_narrow_frame_hud_still_fits(self):
        state = CaptionState(0.25)
        state.update({"prediction": "WHATFOR", "confidence": 0.9,
                      "top5": ["WHATFOR"] * 5, "latency_ms": 3.5})
        hud = ReliableHUD(0.25)
        image = hud.render(np.zeros((240, 320, 3), np.uint8), state, "model", 0.25)
        self.assertEqual(image.shape, (240, 320, 3))

    def test_error_state_renders_message(self):
        state = CaptionState(0.25)
        state.error("Camera read failed - press R to reopen")
        hud = ReliableHUD(0.25)
        image = hud.render(np.zeros((240, 640, 3), np.uint8), state, "model", 0.25)
        self.assertEqual(image.shape, (240, 640, 3))


def _write_clip(path, frames=6):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 12.0, (32, 24))
    assert writer.isOpened(), "test video encoder unavailable"
    for _ in range(frames):
        writer.write(np.zeros((24, 32, 3), np.uint8))
    writer.release()
    assert Path(path).stat().st_size > 0


class FileRunTest(unittest.TestCase):
    def test_missing_file_exits_1_with_note(self):
        with tempfile.TemporaryDirectory() as td:
            report = run_file(Path(td) / "nope.mp4", Path("src/main.py"), 0.25)
            self.assertEqual(report.exit_code, 1)
            self.assertTrue(any("not found" in n for n in report.notes))

    def test_undecodable_file_exits_1(self):
        with tempfile.TemporaryDirectory() as td:
            bad = Path(td) / "bad.mp4"
            bad.write_bytes(b"not a video")
            report = run_file(bad, Path("src/main.py"), 0.25)
            self.assertEqual(report.exit_code, 1)

    def test_real_predict_on_synthetic_clip(self):
        with tempfile.TemporaryDirectory() as td:
            clip = Path(td) / "sign.mp4"
            _write_clip(clip)
            predictor = Path(td) / "fake_predictor.py"
            predictor.write_text(
                "def predict(video_path, clip_id=None):\n"
                "    assert __import__('pathlib').Path(video_path).exists()\n"
                "    return {'clip_id': clip_id or 'x', 'prediction': 'BOOK',"
                " 'top5': ['BOOK'], 'confidence': 0.8, 'latency_ms': 1.0}\n")
            out = Path(td) / "hud.png"
            report = run_file(clip, predictor, 0.25, out)
            self.assertEqual(report.exit_code, 0)
            self.assertEqual(report.captures[0].state, "CAPTION")
            self.assertEqual(report.captures[0].result["prediction"], "BOOK")
            self.assertTrue(out.exists())
            self.assertFalse(report.live_camera_verified)
            # No leftover temp clips in the system temp prefix for this run.
            leftovers = list(Path(tempfile.gettempdir()).glob("echogesture-v7-*"))
            self.assertEqual(leftovers, [])

    def test_predict_failure_exits_1(self):
        with tempfile.TemporaryDirectory() as td:
            clip = Path(td) / "sign.mp4"
            _write_clip(clip)
            bad = Path(td) / "bad_predictor.py"
            bad.write_text(
                "def predict(video_path, clip_id=None):\n"
                "    raise RuntimeError('model exploded')\n")
            report = run_file(clip, bad, 0.25)
            self.assertEqual(report.exit_code, 1)
            self.assertTrue(any("model exploded" in n for n in report.notes))


class CameraSessionTest(unittest.TestCase):
    def test_camera_denied_propagates_clear_error(self):
        # Headless session must surface CameraUnavailableError, never infer success.
        with mock.patch("app.v7_demo.reliable.open_camera",
                        side_effect=CameraUnavailableError("Camera 0 unavailable (denied)")):
            with self.assertRaises(CameraUnavailableError):
                run_session(0, 1.0, Path("src/main.py"), 0.25, headless=True)


class IdentityTest(unittest.TestCase):
    @unittest.skipUnless(
        (Path(__file__).resolve().parents[3] / "models/landmark_classifier_fp16.tflite").is_file(),
        "trained weights are not committed (see NOTICE)")
    def test_fp16_sha_matches_expected_candidate(self):
        result = check_model_identity()
        self.assertTrue(result["ok"], result["report"])
        report = result["report"]
        self.assertEqual(report["model_full_sha256"], EXPECTED_FP16_SHA256)
        self.assertEqual(report["backend"], "landmark")
        self.assertIn("provisional", report["threshold_status"])


if __name__ == "__main__":
    unittest.main()
