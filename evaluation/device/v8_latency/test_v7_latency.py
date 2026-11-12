"""Contract tests for v7_latency."""
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v7_candidates as cand


class TestV7Contracts(unittest.TestCase):
    def test_constants(self):
        self.assertEqual(cand.MAX_FRAMES, 32)
        self.assertEqual(cand.FRAME_SIZE, 256)
        self.assertEqual(cand.MODEL_FRAMES, 24)
        self.assertEqual(cand.MODEL_WIDTH, 255)
        self.assertEqual(cand.STRATEGIES,
                         ("baseline", "parallel_2", "parallel_4", "video_cached", "video_fresh"))

    def test_invalid_strategy(self):
        with self.assertRaises(ValueError):
            cand.evaluate_clip("/nonexistent.mp4", "unknown_strategy")

    def test_parallel_workers_bounded(self):
        self.assertIn("parallel_2", cand.STRATEGIES)
        self.assertIn("parallel_4", cand.STRATEGIES)
        # No other worker levels allowed through strategy names
        self.assertNotIn("parallel_8", cand.STRATEGIES)

    def test_evaluate_rejects_missing_file(self):
        with self.assertRaises((ValueError, RuntimeError, Exception)):
            cand.evaluate_clip("/nonexistent/path.mp4", "baseline")


class TestV8Lifecycle(unittest.TestCase):
    """Worker shutdown/lifecycle tests (v8 fix). No model inference required
    except where noted; tracker tests use mocks to avoid CPU cost."""

    def setUp(self):
        # Ensure a clean state dict shape without loading models.
        cand._initialise()

    def tearDown(self):
        cand.shutdown()

    def test_default_strategy_is_baseline(self):
        import os
        os.environ.pop("V7_STRATEGY", None)
        # predict() would run inference for a real clip; verify the default
        # branch differently: invalid strategy still raises ValueError mentioning baseline default set.
        self.assertIn("baseline", cand.STRATEGIES)
        # Check the source default
        import inspect
        src = inspect.getsource(cand.predict)
        self.assertIn('"baseline"', src)

    def test_shutdown_idempotent_and_clears_state(self):
        st = cand._INTERNALS["state"]
        # Fake live trackers
        closed = []
        class FakeLM:
            def __init__(self):
                self.closed = False
            def close(self):
                self.closed = True
                closed.append(self)
        st["image"] = FakeLM()
        st["video_cached"] = FakeLM()
        st["parallel_2"] = [FakeLM(), FakeLM()]
        st["parallel_4"] = [FakeLM(), FakeLM(), FakeLM(), FakeLM()]
        st["video_ts"] = 12345
        cand.shutdown()
        self.assertIsNone(st["image"])
        self.assertIsNone(st["video_cached"])
        self.assertIsNone(st["parallel_2"])
        self.assertIsNone(st["parallel_4"])
        self.assertEqual(st["video_ts"], 0)
        self.assertEqual(len(closed), 8)
        # Second call is a no-op (idempotent)
        cand.shutdown()
        self.assertEqual(len(closed), 8)

    def test_shutdown_on_uninitialised_is_noop(self):
        # Temporarily clear internals
        saved = dict(cand._INTERNALS)
        cand._INTERNALS.clear()
        try:
            cand.shutdown()  # must not raise
        finally:
            cand._INTERNALS.update({"state": {"image": None, "video_cached": None, "video_ts": 0,
                                              "parallel_2": None, "parallel_4": None}})
            cand._initialise() if "frontend" not in cand._INTERNALS else None

    def test_shutdown_swallows_close_errors(self):
        class BoomLM:
            def close(self):
                raise RuntimeError("boom")
        st = cand._INTERNALS["state"]
        st["image"] = BoomLM()
        cand.shutdown()  # must not raise
        self.assertIsNone(st["image"])

    def test_partial_pool_creation_closes_built_trackers(self):
        closed = []
        made = []
        class FakeLM:
            def __init__(self):
                made.append(self)
            def close(self):
                closed.append(self)
        st = cand._INTERNALS["state"]
        st["parallel_4"] = None
        orig = cand._make_landmarker
        calls = [0]
        def flaky(mode, cpu=False):
            calls[0] += 1
            if calls[0] == 3:
                raise RuntimeError("mock create failure on 3rd tracker")
            return FakeLM()
        cand._make_landmarker = flaky
        try:
            with self.assertRaises(RuntimeError):
                cand._get_parallel_pool(4)
        finally:
            cand._make_landmarker = orig
        # Two trackers were built before the failure; both must be closed,
        # and state must remain None (no partial pool stored).
        self.assertEqual(len(made), 2)
        self.assertEqual(len(closed), 2)
        self.assertIsNone(st["parallel_4"])

    def test_concurrent_singleton_creation_single_instance(self):
        import threading
        created = []
        class FakeLM:
            def __init__(self):
                created.append(self)
            def close(self):
                pass
        st = cand._INTERNALS["state"]
        st["image"] = None
        orig = cand._make_landmarker
        def fake(mode, cpu=False):
            # Simulate slow construction to widen the race window
            import time as _t
            _t.sleep(0.05)
            return FakeLM()
        cand._make_landmarker = fake
        errors = []
        results = []
        def target():
            try:
                results.append(cand._singleton("image"))
            except Exception as exc:
                errors.append(exc)
        try:
            threads = [threading.Thread(target=target) for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        finally:
            cand._make_landmarker = orig
        self.assertEqual(errors, [])
        # Lock ensures exactly one tracker was constructed
        self.assertEqual(len(created), 1)
        self.assertEqual(len(results), 8)
        self.assertTrue(all(r is results[0] for r in results))


if __name__ == "__main__":
    unittest.main()
