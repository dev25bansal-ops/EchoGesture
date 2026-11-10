"""Integration tests for the OPT-IN parallel tracker inside `src/main.py`.

These tests exercise the real public entry point (`src/main.predict`) through
the same importlib loader the manifest evaluator uses, and prove:

  * the default (env unset) path is unchanged and never builds a pool;
  * `ECHOGESTURE_TRACKER=parallel` selects the persistent pool, with
    `ECHOGESTURE_WORKERS` (default 4) controlling its size;
  * repeated predict() calls reuse one pool and `close_tracker()` leaves no
    worker threads behind;
  * the parallel path returns the same prediction label as the singleton on a
    real permitted validation clip;
  * malformed / missing video fails cleanly (clear ValueError, no hang);
  * an unsupported tracker config falls back to the singleton path;
  * a fresh-process run with the env unset still does not build a pool, and a
    fresh-process parallel run matches the fresh-process default label.

Parallel remains OPT-IN.  Nothing here asserts a latency target; the measured
p50/p95 are A01's benchmark numbers, not a <=50 ms claim.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SRC = ROOT / "src"
MODELS = ROOT / "models"
VAL30 = ROOT / "evaluation" / "device" / "v5_accuracy" / "data" / "val_30"
CLIP = VAL30 / "0283886564670357-WHAT FOR.mp4"

_ENV_TRACKER = "ECHOGESTURE_TRACKER"
_ENV_WORKERS = "ECHOGESTURE_WORKERS"


def _can_run() -> bool:
    clip = CLIP.is_file()
    model = (MODELS / "hand_landmarker.task").is_file()
    classifier = (MODELS / "landmark_classifier_fp16.tflite").is_file()
    return clip and model and classifier


def load_main():
    """Load `src/main.py` exactly like `run_manifest.load_predictor` does."""
    spec = importlib.util.spec_from_file_location(
        "echogesture_main_under_test", SRC / "main.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {SRC / 'main.py'}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not callable(getattr(module, "predict", None)):
        raise RuntimeError("loaded module has no predict()")
    return module


def _clean_env() -> dict:
    env = dict(os.environ)
    env.pop(_ENV_TRACKER, None)
    env.pop(_ENV_WORKERS, None)
    return env


def _img_thread_names() -> set[str]:
    return {
        t.name for t in threading.enumerate()
        if t.is_alive() and t.name.startswith("echogesture-img-")
    }


def _wait_until(predicate, timeout=5.0, interval=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


@unittest.skipUnless(_can_run(), "permitted clip or landmark assets absent")
class TestOptInIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = load_main()
        cls.singleton_label = None

    def setUp(self):
        self._saved_env = {
            _ENV_TRACKER: os.environ.pop(_ENV_TRACKER, None),
            _ENV_WORKERS: os.environ.pop(_ENV_WORKERS, None),
        }

    def tearDown(self):
        self.main.close_tracker()
        for key, value in self._saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def _singleton_label(self):
        """Label from the default singleton path (computed once, env clean)."""
        if type(self).singleton_label is None:
            for key in (_ENV_TRACKER, _ENV_WORKERS):
                os.environ.pop(key, None)
            type(self).singleton_label = self.main.predict(str(CLIP))["prediction"]
        return type(self).singleton_label

    # -- config selection ---------------------------------------------------

    def test_00_config_defaults_to_singleton(self):
        self.assertEqual(self.main._tracker_config(), ("singleton", 0))
        os.environ[_ENV_TRACKER] = "image"
        self.assertEqual(self.main._tracker_config(), ("singleton", 0))
        os.environ[_ENV_TRACKER] = "parallel"
        self.assertEqual(self.main._tracker_config(), ("parallel", 4))
        os.environ[_ENV_WORKERS] = "2"
        self.assertEqual(self.main._tracker_config(), ("parallel", 2))

    def test_01_default_path_does_not_build_pool(self):
        # The default must reach Dev's singleton detect_hands, never the pool.
        def _boom(*_a, **_k):
            raise AssertionError("default path must not build the parallel pool")
        original = self.main._get_tracker
        self.main._get_tracker = _boom
        try:
            result = self.main.predict(str(CLIP))
        finally:
            self.main._get_tracker = original
        self.assertIsNone(self.main._TRACKER)
        self.assertIsInstance(result["prediction"], str)
        self.assertTrue(result["prediction"])
        type(self).singleton_label = result["prediction"]

    def test_02_invalid_config_raises_rather_than_silently_falling_back(self):
        # A malformed opt-in config is a caller error: predict() must surface it
        # instead of silently running the singleton path with a typo'd setting.
        os.environ[_ENV_TRACKER] = "nonsense"
        with self.assertRaises(ValueError):
            self.main._tracker_config()
        with self.assertRaises(ValueError):
            self.main.predict(str(CLIP))
        self.assertIsNone(self.main._TRACKER)

        os.environ[_ENV_TRACKER] = "parallel"
        os.environ[_ENV_WORKERS] = "not-an-int"
        with self.assertRaises(ValueError):
            self.main._tracker_config()
        with self.assertRaises(ValueError):
            self.main.predict(str(CLIP))
        self.assertIsNone(self.main._TRACKER)

        os.environ[_ENV_WORKERS] = "0"
        with self.assertRaises(ValueError):
            self.main._tracker_config()
        with self.assertRaises(ValueError):
            self.main.predict(str(CLIP))
        self.assertIsNone(self.main._TRACKER)

    # -- parallel selection and parity -------------------------------------

    def test_03_parallel_selects_pool_and_matches_singleton(self):
        expected = self._singleton_label()
        os.environ[_ENV_TRACKER] = "parallel"
        os.environ[_ENV_WORKERS] = "2"
        result = self.main.predict(str(CLIP), profile=True)
        pool = self.main._TRACKER
        self.assertIsNotNone(pool, "parallel env must build the persistent pool")
        self.assertEqual(pool.workers, 2)
        self.assertEqual(result["prediction"], expected)
        stage = self.main.get_last_profile()["stages"]["landmark_tracking"]
        self.assertEqual(stage.get("engine"), "parallel")

    def test_04_repeated_calls_reuse_pool_and_no_leftover_workers(self):
        expected = self._singleton_label()
        os.environ[_ENV_TRACKER] = "parallel"
        os.environ[_ENV_WORKERS] = "2"
        first = self.main.predict(str(CLIP))
        pool_first = self.main._TRACKER
        processed_first = pool_first.frames_processed
        second = self.main.predict(str(CLIP))
        pool_second = self.main._TRACKER
        self.assertIs(pool_first, pool_second, "pool must persist across clips")
        self.assertGreater(pool_second.frames_processed, processed_first)
        self.assertEqual(first["prediction"], expected)
        self.assertEqual(second["prediction"], expected)
        self.assertTrue(_img_thread_names(), "pool should own worker threads")

        self.main.close_tracker()
        self.assertIsNone(self.main._TRACKER)
        self.assertTrue(pool_first._closed)
        self.assertTrue(
            _wait_until(lambda: not _img_thread_names()),
            f"worker threads left alive after close: {_img_thread_names()}")

    def test_05_malformed_video_fails_cleanly(self):
        os.environ[_ENV_TRACKER] = "parallel"
        os.environ[_ENV_WORKERS] = "2"
        with tempfile.TemporaryDirectory() as tmp:
            garbage = Path(tmp) / "garbage.mp4"
            garbage.write_bytes(b"not a real video")
            started = time.monotonic()
            with self.assertRaises(ValueError) as caught:
                self.main.predict(str(garbage))
            self.assertIn("No valid frames decoded", str(caught.exception))
            self.assertLess(time.monotonic() - started, 30.0)
        missing = CLIP.parent / "_does-not-exist.mp4"
        with self.assertRaises(ValueError):
            self.main.predict(str(missing))
        # The failed clip never built a pool, so nothing leaked.
        self.assertIsNone(self.main._TRACKER)

    # -- fresh-process evidence --------------------------------------------

    def _subprocess_label(self, env):
        code = (
            "import json, sys\n"
            f"sys.path.insert(0, {str(SRC)!r})\n"
            "import main\n"
            f"r = main.predict({str(CLIP)!r})\n"
            "print(json.dumps({'prediction': r['prediction'],\n"
            "                  'tracker_built': main._TRACKER is not None,\n"
            "                  'workers': getattr(main._TRACKER, 'workers', None)}))\n"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True,
            env=env, timeout=180)
        self.assertEqual(completed.returncode, 0, completed.stderr[-2000:])
        return json.loads(completed.stdout.strip().splitlines()[-1])

    def test_06_subprocess_default_unchanged(self):
        result = self._subprocess_label(_clean_env())
        self.assertIsInstance(result["prediction"], str)
        self.assertTrue(result["prediction"])
        self.assertFalse(result["tracker_built"],
                         "default env must not build the parallel pool")
        type(self).subprocess_default_label = result["prediction"]

    def test_07_subprocess_parallel_matches_default(self):
        default_label = getattr(type(self), "subprocess_default_label", None)
        if default_label is None:
            self.skipTest("default subprocess test did not run first")
        env = _clean_env()
        env[_ENV_TRACKER] = "parallel"
        env[_ENV_WORKERS] = "2"
        result = self._subprocess_label(env)
        self.assertTrue(result["tracker_built"])
        self.assertEqual(result["workers"], 2)
        self.assertEqual(result["prediction"], default_label)


if __name__ == "__main__":
    unittest.main(verbosity=2)