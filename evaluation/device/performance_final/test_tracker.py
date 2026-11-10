"""Tests for the persistent parallel IMAGE tracker.

These tests cover the required behaviours:

  * worker startup/reuse (construct exactly `workers` landmarkers, reuse them)
  * per-clip isolation (results for one clip never leak into the next)
  * ordered results (frame order restored after round-robin split)
  * handedness/presence parity (bit-identical to `detect_hands` on real frames)
  * worker exception / timeout recovery (replace once, propagate on second)
  * malformed video / malformed frames (reject, never fabricate)
  * repeated calls (same result and no state growth between calls)
  * bounded resource cleanup (close() idempotent, closes all landmarkers)

Most tests use injected fake landmarkers so they run without CPU-heavy MediaPipe
inference.  A small number of real-clip parity tests are skipped automatically
when the model asset or the permitted validation clips are absent.
"""

from __future__ import annotations

import os
import sys
import threading
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
ROOT = HERE.parents[2]
SRC = ROOT / "src"
MODELS = ROOT / "models"
VAL30 = ROOT / "evaluation" / "device" / "v5_accuracy" / "data" / "val_30"

import tracker as t  # noqa: E402


# --------------------------------------------------------------------------- #
# Fake landmarkers                                                             #
# --------------------------------------------------------------------------- #

def _px(image):
    """First blue-channel pixel of a MediaPipe Image (shared buffer with frame)."""
    return image.numpy_view()[0, 0, 0]


def fake_frames(n: int, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 255, size=(256, 256, 3), dtype=np.uint8) for _ in range(n)]


class FakeLandmarker:
    """MediaPipe-like landmarker whose output encodes the frame index.

    `detect` returns one present Right hand whose x coordinate equals the
    first byte of the frame, so tests can prove order restoration and
    per-worker isolation.
    """

    instances = 0
    closed = []

    def __init__(self, tracker_owned=False, prefix=""):
        type(self).instances += 1
        self.closed = False
        self.prefix = prefix
        self.tracker_owned = tracker_owned

    def detect(self, image):
        # image pixel buffer is shared with the input frame.
        tag = float(_px(image))
        landmark = np.zeros((21, 3), np.float32)
        landmark[:, 0] = tag
        # Returned as Right hand, present.
        return SResult(landmark, "Right")

    def close(self):
        self.closed = True
        type(self).closed.append(self)


class _Point:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


class _HandLandmarks:
    """Iterable (like MediaPipe's) over landmark points."""

    def __init__(self, pts):
        self.points = [_Point(float(p[0]), float(p[1]), float(p[2])) for p in pts]

    def __iter__(self):
        return iter(self.points)


class _Handedness:
    def __init__(self, category):
        self.category_name = category


class SResult:
    """Minimal stand-in for a MediaPipe detection result.

    Mirrors the two attributes `src/landmark_frontend.detect_hands` reads:
    `hand_landmarks` (list of iterable landmark groups) and `handedness`
    (list of lists, where `handedness[i][0].category_name` is the label).
    """

    def __init__(self, landmark, category):
        self.hand_landmarks = [_HandLandmarks(landmark)]
        self.handedness = [[_Handedness(category)]]


# --------------------------------------------------------------------------- #
# Real-work helpers                                                           #
# --------------------------------------------------------------------------- #

def have_model() -> bool:
    return (MODELS / "hand_landmarker.task").is_file()


def real_clips() -> list[Path]:
    manifest = VAL30 / "manifest.csv"
    if not manifest.is_file():
        return []
    clips = []
    for line in manifest.read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split(",")
        if parts:
            clips.append(VAL30 / parts[1])
    return [c for c in clips if c.is_file()]


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #

class TestWorkerStartupAndReuse(unittest.TestCase):
    def test_builds_exactly_workers_landmarkers(self):
        orig = t._create_image_landmarker
        made = []
        t._create_image_landmarker = lambda task_path: made.append(task_path) or FakeLandmarker()
        try:
            pool = t.ParallelImageTracker(workers=3, task_path=MODELS / "hand_landmarker.task")
        finally:
            t._create_image_landmarker = orig
        self.assertEqual(len(made), 3)
        pool.close()

    def test_reuses_same_workers_across_detects(self):
        made = []

        class CountingLM:
            def __init__(self):
                made.append(self)
                self.closed = False
            def detect(self, image):
                lm = np.zeros((21, 3), np.float32)
                return SResult(lm, "Right")
            def close(self):
                self.closed = True

        orig = t._create_image_landmarker
        t._create_image_landmarker = lambda task_path: CountingLM()
        try:
            pool = t.ParallelImageTracker(workers=2, task_path=MODELS / "hand_landmarker.task")
            pool.detect(fake_frames(4))
            pool.detect(fake_frames(4, seed=1))
            self.assertEqual(len(made), 2, "workers must be constructed once and reused")
        finally:
            t._create_image_landmarker = orig
            pool.close()


class TestOrderedResults(unittest.TestCase):
    """Use fake landmarkers tagged by pixel value to verify order restoration."""

    def _patched_pool(self, workers):
        made = []

        def fake_lm():
            lm = _TaggedLM()
            made.append(lm)
            return lm
        orig = t._create_image_landmarker
        t._create_image_landmarker = lambda task_path: fake_lm()
        pool = t.ParallelImageTracker(workers=workers,
                                      task_path=MODELS / "hand_landmarker.task")
        t._create_image_landmarker = orig
        return pool

    def test_round_robin_order_restored(self):
        n = 7
        frames = []
        for i in range(n):
            f = np.zeros((256, 256, 3), np.uint8)
            f[0, 0, 0] = i + 1
            frames.append(f)
        for workers in (1, 2, 4):
            pool = self._patched_pool(workers)
            lm, present = pool.detect(frames)
            pool.close()
            # Right-hand slot x-coordinate must match each input frame index + 1.
            tags = lm[:, 1, 0, 0].astype(int)
            self.assertEqual(tags.tolist(), [i + 1 for i in range(n)])


class _TaggedLM:
    def __init__(self):
        self.closed = False
    def detect(self, image):
        v = float(_px(image))
        lm = np.zeros((21, 3), np.float32)
        lm[:, 0] = v
        return SResult(lm, "Right")
    def close(self):
        self.closed = True


class TestPerClipIsolation(unittest.TestCase):
    def test_two_detects_do_not_share_buffers(self):
        # Real implementation should return fresh arrays, not views.
        made = []

        class LM:
            def __init__(self):
                made.append(self)
                self.closed = False
            def detect(self, image):
                v = float(_px(image))
                lm = np.zeros((21, 3), np.float32)
                lm[:, 0] = v
                return SResult(lm, "Right")
            def close(self):
                self.closed = True

        orig = t._create_image_landmarker
        t._create_image_landmarker = lambda task_path: LM()
        try:
            pool = t.ParallelImageTracker(workers=2, task_path=MODELS / "hand_landmarker.task")
            a, pa = pool.detect(fake_frames(3, seed=10))
            b, pb = pool.detect(fake_frames(3, seed=11))
            a[0, 0, 0, 0] = 999.0
            # Mutating the first result must not change the second (per-clip isolation).
            self.assertEqual(pool.detect(fake_frames(3, seed=11))[0][0, 0, 0, 0],
                             b[0, 0, 0, 0])
        finally:
            t._create_image_landmarker = orig
            pool.close()


class TestWorkerRecovery(unittest.TestCase):
    """Exercise the worker-level recovery path.

    `_Worker.detect` calls Dev's `landmark_frontend.detect_hands`, which itself
    swallows per-frame landmarker exceptions, so recovery is reachable only when
    `detect_hands` raises.  We patch that boundary: a landmarker tagged `_boom`
    makes `detect_hands` raise; the replacement (a fresh, untagged landmarker)
    then succeeds.
    """

    def _make_pool(self, workers):
        """Patch both the factory and detect_hands; return (pool, made, restore)."""
        made = []
        orig_factory = t._create_image_landmarker

        class TaggedLM:
            def __init__(self, boom=False):
                made.append(self)
                self.closed = False
                self._boom = boom
            def detect(self, image):
                lm = np.zeros((21, 3), np.float32)
                return SResult(lm, "Right")
            def close(self):
                self.closed = True

        t._create_image_landmarker = lambda task_path: TaggedLM()
        pool = t.ParallelImageTracker(workers=workers,
                                      task_path=MODELS / "hand_landmarker.task")

        orig_detect = t._frontend.detect_hands

        def patched_detect(frames, landmarker=None, mode="image"):
            if getattr(landmarker, "_boom", False):
                raise RuntimeError("worker crash")
            lm = np.zeros((len(frames), 2, 21, 3), np.float32)
            present = np.ones((len(frames), 2), np.float32)
            return lm, present

        t._frontend.detect_hands = patched_detect

        def restore():
            t._create_image_landmarker = orig_factory
            t._frontend.detect_hands = orig_detect
            pool.close()

        return pool, made, restore

    def test_exception_recovered_by_replacing_worker(self):
        pool, made, restore = self._make_pool(2)
        try:
            # Mark worker 0's landmarker to crash on first detect.
            pool._workers[0]._landmarker._boom = True
            lm, present = pool.detect(fake_frames(4), timeout=5.0)
            self.assertEqual(lm.shape, (4, 2, 21, 3))
            self.assertEqual(present.shape, (4, 2))
        finally:
            restore()
        # Initial 2 + one replacement = 3 landmarkers constructed.
        self.assertEqual(len(made), 3)
        # The doomed worker was closed during replacement.
        self.assertTrue(made[0].closed)

    def test_double_failure_propagates(self):
        pool, made, restore = self._make_pool(1)
        # Make detect_hands always raise so the replacement also crashes,
        # promoting the bounded recovery to a TrackerError.
        t._frontend.detect_hands = lambda frames, landmarker=None, mode="image": (
            _always_raise())
        try:
            with self.assertRaises(t.TrackerError):
                pool.detect(fake_frames(1), timeout=5.0)
        finally:
            restore()
        # Original + replacement = 2 constructed; second failure propagated.
        self.assertEqual(len(made), 2)


def _always_raise():
    raise RuntimeError("persistent crash")


class TestTimeoutRecovery(unittest.TestCase):
    def test_timeout_recovers_then_propagates(self):
        made = []
        orig_factory = t._create_image_landmarker

        class FakeLM:
            def __init__(self):
                made.append(self)
                self.closed = False
            def detect(self, image):
                return SResult(np.zeros((21, 3), np.float32), "Right")
            def close(self):
                self.closed = True

        t._create_image_landmarker = lambda task_path: FakeLM()
        pool = t.ParallelImageTracker(workers=1, task_path=MODELS / "hand_landmarker.task")

        orig_detect = t._frontend.detect_hands

        def slow_detect(frames, landmarker=None, mode="image"):
            import time
            time.sleep(0.5)
            return (np.zeros((len(frames), 2, 21, 3), np.float32),
                    np.zeros((len(frames), 2), np.float32))

        t._frontend.detect_hands = slow_detect
        try:
            with self.assertRaises(t.TrackerError):
                pool.detect(fake_frames(1), timeout=0.05)
        finally:
            t._frontend.detect_hands = orig_detect
            t._create_image_landmarker = orig_factory
            pool.close()
        # Initial + one replacement = 2 constructed; both timed out.
        self.assertEqual(len(made), 2)


def fake_frames4():
    return fake_frames(4)


class TestMalformedInputs(unittest.TestCase):
    def setUp(self):
        made = []

        class LM:
            def __init__(self):
                made.append(self); self.closed = False
            def detect(self, image):
                return SResult(np.zeros((21, 3), np.float32), "Right")
            def close(self):
                self.closed = True
        orig = t._create_image_landmarker
        t._create_image_landmarker = lambda task_path: LM()
        self.pool = t.ParallelImageTracker(workers=2,
                                           task_path=MODELS / "hand_landmarker.task")
        t._create_image_landmarker = orig
        self.made = made

    def tearDown(self):
        self.pool.close()

    def test_none_frames_rejected(self):
        with self.assertRaises(ValueError):
            self.pool.detect(None)

    def test_too_many_frames_rejected(self):
        with self.assertRaises(ValueError):
            self.pool.detect(fake_frames(33))

    def test_wrong_shape_rejected(self):
        with self.assertRaises(ValueError):
            self.pool.detect([np.zeros((10, 10, 3), np.uint8)])

    def test_wrong_dtype_rejected(self):
        with self.assertRaises(ValueError):
            self.pool.detect([np.zeros((256, 256, 3), np.float32)])

    def test_missing_task_path_rejected(self):
        with self.assertRaises(FileNotFoundError):
            t.ParallelImageTracker(workers=1, task_path=HERE / "does-not-exist.task")

    def test_zero_workers_rejected(self):
        with self.assertRaises(ValueError):
            t.ParallelImageTracker(workers=0, task_path=MODELS / "hand_landmarker.task")

    def test_malformed_video_yields_no_frames(self):
        import sys
        sys.path.insert(0, str(SRC))
        import landmark_frontend as lf
        # A garbage file must decode to zero frames, and the tracker must reject
        # an empty frame list with an empty (0, ...) result, never fabricate.
        garbage = HERE / "_garbage.mp4"
        garbage.write_bytes(b"not a real video")
        try:
            frames = lf.read_frames(str(garbage))
            self.assertEqual(frames, [])
            lm, present = self.pool.detect(frames)
            self.assertEqual(lm.shape, (0, 2, 21, 3))
        finally:
            garbage.unlink()


class TestRepeatedCallsAndCleanup(unittest.TestCase):
    def test_repeated_calls_are_consistent(self):
        made = []

        class LM:
            def __init__(self):
                made.append(self); self.closed = False
            def detect(self, image):
                v = float(_px(image))
                lm = np.zeros((21, 3), np.float32)
                lm[:, 0] = v
                return SResult(lm, "Right")
            def close(self):
                self.closed = True
        orig = t._create_image_landmarker
        t._create_image_landmarker = lambda task_path: LM()
        pool = t.ParallelImageTracker(workers=2, task_path=MODELS / "hand_landmarker.task")
        t._create_image_landmarker = orig
        try:
            frames = fake_frames(6, seed=3)
            expected = pool.detect(frames)[0]
            for _ in range(3):
                np.testing.assert_array_equal(pool.detect(frames)[0], expected)
            self.assertEqual(pool.frames_processed, 6 * 4)
        finally:
            pool.close()

    def test_close_is_idempotent_and_closes_workers(self):
        made = []

        class LM:
            def __init__(self):
                made.append(self); self.closed = False
            def detect(self, image):
                return SResult(np.zeros((21, 3), np.float32), "Right")
            def close(self):
                self.closed = True
        orig = t._create_image_landmarker
        t._create_image_landmarker = lambda task_path: LM()
        pool = t.ParallelImageTracker(workers=3, task_path=MODELS / "hand_landmarker.task")
        t._create_image_landmarker = orig
        pool.close()
        pool.close()  # idempotent
        self.assertTrue(all(lm.closed for lm in made))
        # Using after close raises.
        with self.assertRaises(RuntimeError):
            pool.detect(fake_frames(2))

    def test_context_manager_closes(self):
        made = []

        class LM:
            def __init__(self):
                made.append(self); self.closed = False
            def detect(self, image):
                return SResult(np.zeros((21, 3), np.float32), "Right")
            def close(self):
                self.closed = True
        orig = t._create_image_landmarker
        t._create_image_landmarker = lambda task_path: LM()
        t._create_image_landmarker = orig
        with t.ParallelImageTracker(workers=2, task_path=MODELS / "hand_landmarker.task") as pool:
            self.assertFalse(pool._closed)
        self.assertTrue(pool._closed)


# --------------------------------------------------------------------------- #
# Real-clip parity tests (skipped when assets are absent)                      #
# --------------------------------------------------------------------------- #

@unittest.skipUnless(have_model(), "hand_landmarker.task not present")
class TestRealParity(unittest.TestCase):
    def test_parity_on_empty_clip_exercise_duplicate_frames(self):
        # Deterministic input: duplicate blank frames must agree with
        # `detect_hands` (both use Dev's frontend).
        import sys
        sys.path.insert(0, str(SRC))
        import landmark_frontend as lf
        frames = [np.zeros((256, 256, 3), np.uint8) for _ in range(3)]
        baseline_lm, baseline_present = lf.detect_hands(frames)
        with t.ParallelImageTracker(workers=4, task_path=MODELS / "hand_landmarker.task") as pool:
            par_lm, par_present = pool.detect(frames)
        np.testing.assert_array_equal(par_lm, baseline_lm)
        np.testing.assert_array_equal(par_present, baseline_present)

    def test_parity_on_real_clip_if_available(self):
        import sys
        sys.path.insert(0, str(SRC))
        import landmark_frontend as lf
        clips = real_clips()
        if not clips:
            self.skipTest("no permitted validation clips available")
        frames = lf.read_frames(str(clips[0]), max_frames=32)
        if not frames:
            self.skipTest("clip decoded to no frames")
        baseline_lm, baseline_present = lf.detect_hands(frames)
        with t.ParallelImageTracker(workers=4, task_path=MODELS / "hand_landmarker.task") as pool:
            par_lm, par_present = pool.detect(frames)
        np.testing.assert_array_equal(par_lm, baseline_lm)
        np.testing.assert_array_equal(par_present, baseline_present)


if __name__ == "__main__":
    unittest.main(verbosity=2)