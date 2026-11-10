"""V7 QA: repeated model/predictor load behavior (defect D-1 family).

Verified production behavior pinned here:
  * run_manifest.load_predictor registers every predictor under the fixed
    name "echogesture_predictor". Loading a second predictor replaces the
    module object in sys.modules for *new* lookups (verified), while a
    function returned by the first load keeps its own globals (verified
    with a TOKEN probe).
  * Import failure during load_predictor cleans the fixed-name registry
    entry; a predictor file missing predict() raises EvaluationError.
  * Repeated loads of the same file produce distinct module objects that
    both answer predict() identically.
  * Known-open interaction (reproduced, NOT asserted as passing): a
    profiled predictor's get_last_profile() is resolved through
    sys.modules[predict.__module__] at call time, so loading an
    unprofiled predictor second makes evaluate_profiled() for the first
    predictor raise "does not expose get_last_profile()". This is
    repro_d1_profile_crosstalk.py; no test here asserts that flow works,
    and no production fix is fabricated.
"""

import sys
import tempfile
import unittest
from pathlib import Path

DEVICE_DIR = Path(__file__).resolve().parents[1]
ROOT = DEVICE_DIR.parents[1]
for _p in (str(DEVICE_DIR), str(ROOT / "scripts" / "release")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_manifest  # noqa: E402


class RepeatedLoadTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def _write(self, name, text):
        path = self.root / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_second_load_replaces_registry_entry_but_keeps_first_globals(self):
        first = self._write(
            "predictor_a.py",
            "TOKEN = 'alpha'\n"
            "def predict(path):\n"
            "    return {'clip_id': 'a', 'prediction': 'CANDY', 'top5': ['CANDY'],\n"
            "            'confidence': 1.0, 'latency_ms': 1}\n"
            "def get_token():\n"
            "    return TOKEN\n",
        )
        second = self._write(
            "predictor_b.py",
            "TOKEN = 'bravo'\n"
            "def predict(path):\n"
            "    return {'clip_id': 'b', 'prediction': 'CANDY', 'top5': ['CANDY'],\n"
            "            'confidence': 1.0, 'latency_ms': 1}\n",
        )
        predict_a = run_manifest.load_predictor(first)
        self.assertEqual(predict_a.__globals__["TOKEN"], "alpha")
        run_manifest.load_predictor(second)
        # Fixed-name entry now points at the second module.
        self.assertEqual(sys.modules["echogesture_predictor"].TOKEN, "bravo")
        self.assertIsNot(sys.modules["echogesture_predictor"].predict, predict_a)
        # But the retained function still resolves through its own globals.
        self.assertEqual(predict_a.__globals__["TOKEN"], "alpha")
        self.assertEqual(predict_a("/nonexistent.mp4")["clip_id"], "a")

    def test_repeated_load_same_file_distinct_objects_both_work(self):
        good = self._write(
            "good.py",
            "def predict(path):\n"
            "    return {'clip_id': 'x', 'prediction': 'CANDY', 'confidence': 1.0, 'latency_ms': 1}\n",
        )
        p1 = run_manifest.load_predictor(good)
        p2 = run_manifest.load_predictor(good)
        self.assertIsNot(p1, p2)
        self.assertIsNot(p1.__globals__, p2.__globals__)
        self.assertEqual(p1("anything")["prediction"], "CANDY")
        self.assertEqual(p2("anything")["prediction"], "CANDY")

    def test_missing_predict_callable_rejected(self):
        no_pred = self._write("nopred.py", "X = 1\n")
        with self.assertRaisesRegex(run_manifest.EvaluationError, "Missing callable predict"):
            run_manifest.load_predictor(no_pred)

    def test_missing_predictor_file_rejected(self):
        with self.assertRaisesRegex(run_manifest.EvaluationError, "Predictor not found"):
            run_manifest.load_predictor(self.root / "missing.py")

    def test_import_failure_cleans_module_registry(self):
        boom = self._write("boom.py", "raise ImportError('boom')\n")
        with self.assertRaisesRegex(run_manifest.EvaluationError, "Cannot import"):
            run_manifest.load_predictor(boom)
        self.assertNotIn("echogesture_predictor", sys.modules)

    def test_predictor_returning_wrong_id_rejected_by_evaluate(self):
        self._write(
            "good.py",
            "def predict(path, *, clip_id=None):\n"
            "    return {'clip_id': 'WRONG', 'prediction': 'CANDY',\n"
            "            'confidence': 1.0, 'latency_ms': 1}\n",
        )
        clip = self.root / "clip.mp4"
        clip.write_bytes(b"fixture")
        manifest = self.root / "manifest.csv"
        manifest.write_text("clip_id,path\na,clip.mp4\n", encoding="utf-8")
        items = run_manifest.read_manifest(manifest)
        predict = run_manifest.load_predictor(self.root / "good.py")
        with self.assertRaisesRegex(run_manifest.EvaluationError, "returned clip_id"):
            run_manifest.evaluate(items, predict, {"CANDY"})


if __name__ == "__main__":
    unittest.main()
