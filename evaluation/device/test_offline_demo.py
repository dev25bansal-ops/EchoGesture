import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.offline_demo import (
    CameraUnavailableError,
    CaptionState,
    PredictionWorker,
    model_identity,
    model_label_from_identity,
    open_camera,
    probe_camera,
    render,
    sanitize_result,
    write_temp_clip,
)


class DemoTest(unittest.TestCase):
    def test_retry_and_caption_states(self):
        state = CaptionState(0.4)
        result = {"prediction": "BOOK", "confidence": 0.2, "top5": ["BOOK"], "latency_ms": 2}
        self.assertEqual(state.update(result), "RETRY")
        result["confidence"] = 0.7
        self.assertEqual(state.update(result), "CAPTION")
        state.error("camera unavailable")
        self.assertEqual(state.state, "ERROR")

    def test_invalid_threshold(self):
        with self.assertRaises(ValueError):
            CaptionState(-0.1)

    def test_real_worker_runs_predictor_and_deletes_clip(self):
        seen = []

        def predict(path, *, clip_id=None):
            clip = Path(path)
            seen.append(clip)
            self.assertTrue(clip.exists())
            return {"clip_id": clip_id, "prediction": "BOOK", "top5": ["BOOK"],
                    "confidence": 0.8, "latency_ms": 3.2}

        worker = PredictionWorker(predict)
        try:
            self.assertTrue(worker.submit([np.zeros((24, 32, 3), np.uint8)] * 3, 12, "unicode-👋"))
            kind, result = worker.results.get(timeout=10)
            self.assertEqual(kind, "result")
            self.assertEqual(result["clip_id"], "unicode-👋")
            self.assertGreaterEqual(result["capture_to_result_ms"], 0)
            self.assertIn("queue_wait_ms", result)
            self.assertIn("encode_ms", result)
            for _ in range(50):
                if not seen[0].exists():
                    break
                time.sleep(0.01)
            self.assertFalse(seen[0].exists())
            self.assertEqual(list(worker.tmpdir_path.iterdir()), [])
        finally:
            worker.close()

    def test_worker_close_idempotent_and_cleans_tmpdir(self):
        worker = PredictionWorker(lambda path, **kwargs: {"prediction": "X", "confidence": 1, "top5": ["X"], "latency_ms": 1})
        tmpdir = worker.tmpdir_path
        worker.close()
        self.assertFalse(tmpdir.exists())
        worker.close()  # second close must not raise

    def test_worker_rejects_bad_inputs(self):
        worker = PredictionWorker(lambda path, **kwargs: {"prediction": "X", "confidence": 1, "top5": ["X"], "latency_ms": 1})
        try:
            self.assertFalse(worker.submit([], 30, "empty"))
            kind, message = self._submit_and_get(worker, [np.zeros((10, 10, 3), np.uint8)], 0, "bad-fps")
            self.assertEqual(kind, "error")
            self.assertIn("fps", message.lower())
        finally:
            worker.close()

    def _submit_and_get(self, worker, frames, fps, clip_id):
        self.assertTrue(worker.submit(frames, fps, clip_id))
        return worker.results.get(timeout=10)

    def test_worker_surfaces_model_error(self):
        worker = PredictionWorker(lambda path, **kwargs: (_ for _ in ()).throw(ValueError("No hand detected")))
        try:
            worker.submit([np.zeros((12, 12, 3), np.uint8)], 10, "x")
            kind, message = worker.results.get(timeout=10)
            self.assertEqual(kind, "error")
            self.assertIn("No hand detected", message)
        finally:
            worker.close()

    def test_sanitize_rejects_malformed_results(self):
        for bad in [
            {"confidence": 0.9, "top5": ["X"], "latency_ms": 1},  # no prediction
            {"prediction": "X", "confidence": 42, "top5": ["X"], "latency_ms": 1},
            {"prediction": "X", "confidence": float("nan"), "top5": ["X"], "latency_ms": 1},
            {"prediction": "X", "confidence": "high", "top5": ["X"], "latency_ms": 1},
        ]:
            with self.assertRaises(ValueError):
                sanitize_result(bad)

    def test_caption_state_recovers_from_error(self):
        state = CaptionState(0.25)
        state.error("Camera read failed")
        self.assertEqual(state.state, "ERROR")
        state.reset()
        self.assertEqual(state.state, "WAITING")
        self.assertIsNone(state.result)

    def test_caption_state_snapshot_is_thread_safe(self):
        state = CaptionState(0.25)
        errors = []

        def writer():
            try:
                for i in range(200):
                    state.update({"prediction": "BOOK", "confidence": 0.9, "top5": ["BOOK"], "latency_ms": 1})
                    state.processing()
                    state.reset()
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        def reader():
            try:
                for _ in range(200):
                    snap = state.snapshot()
                    assert snap["state"] in ("WAITING", "PROCESSING", "RETRY", "CAPTION", "ERROR")
                    assert isinstance(snap["message"], str)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        self.assertEqual(errors, [])

    def test_write_temp_clip_validates_input(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                write_temp_clip([], 30, tmpdir, "x")
            with self.assertRaises(ValueError):
                write_temp_clip([np.zeros((10, 10, 3), np.uint8)], 0, tmpdir, "x")
            path = write_temp_clip([np.zeros((24, 32, 3), np.uint8)] * 2, 12, tmpdir, "unicode-👋")
            self.assertTrue(Path(path).exists())
            self.assertGreater(Path(path).stat().st_size, 0)

    def test_probe_camera_requires_a_real_frame(self):
        blocked = mock.Mock()
        blocked.read.return_value = (False, None)
        self.assertEqual(probe_camera(blocked, attempts=3), 0)

        working = mock.Mock()
        working.read.return_value = (True, np.zeros((10, 10, 3), np.uint8))
        self.assertGreaterEqual(probe_camera(working), 1)

    def test_open_camera_gives_tcc_guidance_when_blocked(self):
        with mock.patch("app.offline_demo.cv2.VideoCapture") as factory:
            capture = factory.return_value
            capture.isOpened.return_value = True
            capture.read.return_value = (False, None)
            with self.assertRaises(CameraUnavailableError) as ctx:
                open_camera(9)
            message = str(ctx.exception)
            self.assertIn("Camera 9 unavailable", message)
            self.assertIn("Privacy & Security", message)
            self.assertIn("tccutil", message)
            self.assertIn("--video", message)

    def test_open_camera_gives_tcc_guidance_when_unopened(self):
        with mock.patch("app.offline_demo.cv2.VideoCapture") as factory:
            factory.return_value.isOpened.return_value = False
            with self.assertRaises(CameraUnavailableError) as ctx:
                open_camera(9)
            self.assertIn("not opened", str(ctx.exception))

    def test_model_identity_names_selected_artifact(self):
        identity = model_identity()
        label = model_label_from_identity(identity)
        self.assertIn(identity["backend"], ("landmark", "s3d"))
        self.assertIn(identity["model"]["name"], str(label))
        self.assertRegex(identity["model"]["sha256_prefix"], r"^[0-9a-f]{12}$")
        self.assertIn("provisional", identity["threshold_status"])

    def test_model_identity_picks_s3d_when_env_set(self):
        with mock.patch.dict("os.environ", {"ECHOGESTURE_MODEL": "s3d"}):
            identity = model_identity()
            self.assertEqual(identity["backend"], "s3d")
            self.assertIn("s3d", identity["model"]["name"])

    def test_hud_marks_baseline_and_offline(self):
        state = CaptionState()
        image = render(np.zeros((240, 640, 3), np.uint8), state, "model", 0.25)
        self.assertEqual(image.shape, (240, 640, 3))

    def test_hud_uses_state_snapshot_fields(self):
        state = CaptionState()
        state.update({"prediction": "BOOK", "confidence": 0.9, "top5": ["BOOK", "A"], "latency_ms": 3.5})
        image = render(np.zeros((240, 640, 3), np.uint8), state, "model", 0.25)
        self.assertEqual(image.shape, (240, 640, 3))


if __name__ == "__main__":
    unittest.main()
