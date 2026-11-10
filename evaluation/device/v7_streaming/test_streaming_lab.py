"""Tests for the reliable streaming session.

Run with the project interpreter:

    $REPO/.venv/bin/python -I \
      -m unittest discover -s . -p 'test_*.py' -v
"""

import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from capture_states import CAPTURING, COMMITTED, ERROR, IDLE, CaptureSession, CaptureStateError
from ring_buffer import RingBuffer

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_MODEL_PATH = _PROJECT_ROOT / "models/landmark_classifier_fp16.tflite"
_FIXTURE = Path("$TMP/echogesture-.../0009701491968876486-PULL CONVINCE.mp4")


def _frame(index, present_hand=True):
    landmarks = np.zeros((2, 21, 3), np.float32)
    landmarks[1, 0] = [0.5, 0.5, 0.0]
    landmarks[1, 9] = [0.5, 0.3, 0.0]
    present = np.array([0.0, 1.0 if present_hand else 0.0], np.float32)
    return landmarks, present


class RingBufferTest(unittest.TestCase):
    def test_bounded_drop_oldest_reports_counts(self):
        buffer = RingBuffer(capacity=3)
        for index in range(5):
            landmarks, present = _frame(index)
            result = buffer.push({"frame_index": index, "landmarks": landmarks, "present": present})
            self.assertEqual(result.accepted, True)
            self.assertEqual(result.dropped, 0 if index < 3 else 1)
        self.assertEqual([item["frame_index"] for item in buffer.snapshot()], [2, 3, 4])
        self.assertEqual(buffer.stats()["dropped_overflow"], 2)

    def test_capacity_must_be_positive(self):
        with self.assertRaises(ValueError):
            RingBuffer(capacity=0)


class CaptureSessionTest(unittest.TestCase):
    def test_new_signs_have_monotonic_ids(self):
        session = CaptureSession()
        first = session.begin_sign()
        self.assertEqual((first, session.state), (1, CAPTURING))
        session.transition("SIGN_HOLD")
        session.transition(COMMITTED)
        second = session.begin_sign()
        self.assertEqual(second, first + 1)

    def test_illegal_transition_raises(self):
        session = CaptureSession()
        with self.assertRaises(CaptureStateError):
            session.transition(COMMITTED)


class _FakeClassifier:
    frames = 24
    warmup_ms = 0.0

    def classify_window(self, landmarks, present):
        return "PULLCONVINCE", 0.75, ["PULLCONVINCE", "LOCK1"], 0.17, 0.20


class _BlockingClassifier:
    frames = 24

    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()

    def classify_window(self, landmarks, present):
        self.entered.set()
        self.release.wait(timeout=2)
        return "PULLCONVINCE", 0.75, ["PULLCONVINCE", "LOCK1"], 0.17, 0.20


class StreamingSessionTest(unittest.TestCase):
    def test_committed_caption_does_not_cross_signs(self):
        from streaming_runtime import StreamingSession

        session = StreamingSession(capacity=96, classifier=_FakeClassifier())
        for index in range(3):
            landmarks, present = _frame(index)
            session.ingest_landmark_frame(index, 1_000_000 + index * 33_000_000, landmarks, present)
        sequence = session.sign_seq
        caption = session.flush_sign(t_sign_end_ns=2_000_000_000)
        self.assertEqual(caption["sign_seq"], sequence)
        self.assertIsNotNone(session.current_caption())
        self.assertEqual(session.resume(), sequence + 1)
        self.assertIsNone(session.current_caption())

    def test_gap_can_be_flushed_then_next_sign_commits_independently(self):
        from streaming_runtime import StreamingSession

        session = StreamingSession(
            classifier=_FakeClassifier(), max_capture_gap_ms=100, timestamp_clock="external_ns"
        )
        landmarks, present = _frame(0)
        session.ingest_landmark_frame(0, 1_000_000, landmarks, present)
        self.assertEqual(
            session.ingest_landmark_frame(1, 151_000_000, landmarks, present),
            "timestamp_gap",
        )
        first = session.flush_sign(t_sign_end_ns=150_000_000)
        self.assertEqual(first["sign_seq"], 1)
        session.resume()
        self.assertEqual(session.sign_seq, 2)
        session.ingest_landmark_frame(1, 151_000_000, landmarks, present)
        second = session.flush_sign(t_sign_end_ns=200_000_000)
        self.assertEqual(second["sign_seq"], 2)
        self.assertEqual(second["window_frames"], 1)

    def test_stop_and_reset_clear_caption(self):
        from streaming_runtime import StreamingSession

        session = StreamingSession(classifier=_FakeClassifier())
        landmarks, present = _frame(0)
        session.ingest_landmark_frame(0, 1_000_000, landmarks, present)
        session.flush_sign(t_sign_end_ns=2_000_000)
        self.assertIsNotNone(session.current_caption())
        session.stop()
        self.assertIsNone(session.current_caption())
        session.reset()
        self.assertIsNone(session.current_caption())

    def test_external_replay_requires_explicit_end_timestamp(self):
        from streaming_runtime import StreamingSession

        session = StreamingSession(classifier=_FakeClassifier(), timestamp_clock="external_ns")
        landmarks, present = _frame(0)
        session.ingest_landmark_frame(0, 1_000_000, landmarks, present)
        with self.assertRaises(CaptureStateError):
            session.flush_sign()
        self.assertIsNone(session.current_caption())
        self.assertEqual(session.state, IDLE)

    def test_begin_sign_does_not_silently_discard_active_window(self):
        from streaming_runtime import StreamingSession

        session = StreamingSession(classifier=_FakeClassifier())
        landmarks, present = _frame(0)
        session.ingest_landmark_frame(0, 1_000_000, landmarks, present)
        with self.assertRaises(CaptureStateError):
            session.begin_sign()
        self.assertEqual(len(session.buffer), 1)

    def test_caption_expires(self):
        from streaming_runtime import StreamingSession

        session = StreamingSession(
            capacity=96, classifier=_FakeClassifier(), caption_ttl_seconds=0.02
        )
        landmarks, present = _frame(0)
        session.ingest_landmark_frame(0, 1_000_000, landmarks, present)
        session.flush_sign()
        time.sleep(0.03)
        self.assertIsNone(session.current_caption())

    def test_timestamp_gap_requires_disjoint_windows(self):
        from streaming_runtime import StreamingSession

        session = StreamingSession(capacity=96, classifier=_FakeClassifier(), max_capture_gap_ms=100)
        landmarks, present = _frame(0)
        session.ingest_landmark_frame(0, 1_000_000, landmarks, present)
        gap_frame = session.ingest_landmark_frame(1, 1_000_000 + 150_000_000, landmarks, present)
        self.assertEqual(gap_frame, "timestamp_gap")
        self.assertEqual(len(session.buffer), 1)
        self.assertEqual(session.dropped_timestamp_gaps, 1)

    def test_out_of_order_and_duplicate_are_explicit(self):
        from streaming_runtime import StreamingSession

        session = StreamingSession(capacity=96, classifier=_FakeClassifier())
        landmarks, present = _frame(0)
        session.ingest_landmark_frame(0, 5_000_000_000, landmarks, present)
        self.assertEqual(
            session.ingest_landmark_frame(1, 5_000_000_000, landmarks, present),
            "dropped_duplicate",
        )
        self.assertEqual(
            session.ingest_landmark_frame(2, 4_000_000_000, landmarks, present),
            "dropped_stale",
        )
        self.assertEqual(session.dropped_duplicate, 1)
        self.assertEqual(session.dropped_out_of_order, 1)

    def test_backpressure_counts_dropped_frames(self):
        from streaming_runtime import StreamingSession

        session = StreamingSession(capacity=4, classifier=_FakeClassifier())
        for index in range(10):
            landmarks, present = _frame(index)
            status = session.ingest_landmark_frame(
                index, 1_000_000 + index * 1_000, landmarks, present
            )
            self.assertIn(status, ("ok", "ok_dropped_oldest"))
        self.assertEqual(session.buffer.stats()["dropped_overflow"], 6)
        self.assertEqual(len(session.buffer), 4)

    def test_empty_and_no_hand_flush_fail_without_caption_or_sign_increment(self):
        from streaming_runtime import StreamingSession

        session = StreamingSession(capacity=96, classifier=_FakeClassifier())
        landmarks, present = _frame(0, present_hand=False)
        session.ingest_landmark_frame(0, 1_000_000, landmarks, present)
        sequence = session.sign_seq
        with self.assertRaises(CaptureStateError):
            session.flush_sign(t_sign_end_ns=2_000_000)
        self.assertEqual(session.sign_seq, sequence)
        self.assertEqual(session.state, IDLE)
        self.assertIsNone(session.current_caption())
        self.assertEqual(len(session.buffer), 0)

    def test_failure_recovery_does_not_reuse_previous_timestamp(self):
        from streaming_runtime import StreamingSession

        session = StreamingSession(classifier=_FakeClassifier())
        landmarks, present = _frame(0, present_hand=False)
        session.ingest_landmark_frame(0, 100_000_000, landmarks, present)
        with self.assertRaises(CaptureStateError):
            session.flush_sign(t_sign_end_ns=100_000_000)
        self.assertIsNone(session.last_capture_ns)
        self.assertIsNotNone(session.last_error)
        landmarks, present = _frame(1)
        self.assertEqual(session.ingest_landmark_frame(1, 0, landmarks, present), "ok")
        self.assertEqual(len(session.buffer), 1)
        session.reset()
        self.assertIsNone(session.last_error)

    def test_reset_discards_in_flight_classifier_result(self):
        from streaming_runtime import StreamingSession

        classifier = _BlockingClassifier()
        session = StreamingSession(capacity=96, classifier=classifier)
        landmarks, present = _frame(0)
        session.ingest_landmark_frame(0, 1_000_000, landmarks, present)

        result = []
        thread = threading.Thread(target=lambda: result.append(session.flush_sign()))
        thread.start()
        self.assertTrue(classifier.entered.wait(timeout=2))
        session.reset()
        classifier.release.set()
        thread.join(timeout=2)
        self.assertEqual(result, [None])
        self.assertEqual(session.discarded_stale_results, 1)
        self.assertIsNone(session.current_caption())


@unittest.skipUnless(_MODEL_PATH.exists(), "FP16 model is absent")
class RealModelTest(unittest.TestCase):
    def test_classifier_shape_and_timing_separation(self):
        from streaming_runtime import StreamingClassifier

        classifier = StreamingClassifier()
        warmup = classifier.warmup()
        landmarks = np.zeros((24, 2, 21, 3), np.float32)
        landmarks[:, 1, :, 0] = 0.5
        landmarks[:, 1, :, 1] = np.linspace(0.4, 0.6, 21)[None, :]
        present = np.zeros((24, 2), np.float32)
        present[:, 1] = 1.0
        label, confidence, top5, invoke_ms, total_ms = classifier.classify_window(
            landmarks, present
        )
        self.assertGreaterEqual(warmup, 0)
        self.assertEqual(label, top5[0])
        self.assertTrue(0.0 <= confidence <= 1.0)
        self.assertGreaterEqual(invoke_ms, 0)
        self.assertGreaterEqual(total_ms, invoke_ms)

    def test_real_fixture_landmark_replay_does_not_claim_live_latency(self):
        if not _FIXTURE.exists():
            self.skipTest("real fixture is absent")
        try:
            sys.path.insert(0, str(_PROJECT_ROOT / "src"))
            from landmark_frontend import detect_hands, read_frames
        except Exception as exc:
            self.skipTest(f"landmark front end is unavailable: {exc}")
        frames = read_frames(str(_FIXTURE), max_frames=32)
        if not frames:
            self.skipTest("no frames were decoded from the fixture")
        landmarks, present = detect_hands(frames, mode="image")
        if landmarks.shape[0] == 0 or float(present.max()) < 0.5:
            self.skipTest("no hand was detected in the fixture")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "landmarks.npz"
            start_ns = time.monotonic_ns()
            timestamps = np.array(
                [start_ns + i * 33_000_000 for i in range(len(landmarks))], dtype=np.int64
            )
            np.savez(path, timestamps_ns=timestamps, landmarks=landmarks, present=present)
            from streaming_runtime import run_landmark_file

            result, _session = run_landmark_file(path)
            self.assertEqual(result["frames"], len(landmarks))
            self.assertEqual(result["frame_processing_ms"]["n"], 0)
            self.assertIsNone(result["sign_end_to_caption_ms"])
            self.assertFalse(result["live_capture"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
