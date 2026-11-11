"""Runtime tests for the landmark ensemble in `src/main.py`.

These prove the production entry point loads the four FP16 members from the
meta sidecar, keeps per-member tensor handles, averages logits before one
softmax, and preserves every pre-ensemble capability (explicit single-model
fallback, s3d backend selection, strict config errors, parallel tracker).

The label/confidence expectations on the real clip were MEASURED on this
machine and recorded here; they are regression guards, not accuracy claims.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SRC = ROOT / "src"
MODELS = ROOT / "models"
VAL30 = ROOT / "evaluation" / "device" / "v5_accuracy" / "data" / "val_30"
CLIP = VAL30 / "0283886564670357-WHAT FOR.mp4"
META = MODELS / "landmark_model_meta.json"

# Measured on this machine (see the module docstring). The ensemble averages
# four FP16 members; the single-model fallback returns a different confidence
# and top5 on the same clip.
ENSEMBLE_TOP1 = "WHATFOR1"
SINGLE_TOP1 = "WHATFOR1"

_ENV_MODEL = "ECHOGESTURE_MODEL"
_ENV_TRACKER = "ECHOGESTURE_TRACKER"
_ENV_WORKERS = "ECHOGESTURE_WORKERS"

_LOAD_COUNTER = 0


def load_main():
    """Load `src/main.py` under a fresh module name (fresh interpreter cache)."""
    global _LOAD_COUNTER
    _LOAD_COUNTER += 1
    name = f"echogesture_main_ensemble_{_LOAD_COUNTER}"
    spec = importlib.util.spec_from_file_location(name, SRC / "main.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {SRC / 'main.py'}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _can_run() -> bool:
    return (
        CLIP.is_file()
        and (MODELS / "hand_landmarker.task").is_file()
        and META.is_file()
    )


def _meta_members() -> list[str]:
    return list(json.loads(META.read_text(encoding="utf-8")).get("members", []))


def _make_fixture(members=None, *, drop_member=None, meta_override=None,
                  copy_single=True):
    """Build a temp repo layout (src + models) for load-failure tests.

    Returns the root directory. The caller is responsible for cleanup.
    """
    root = Path(tempfile.mkdtemp(prefix="echogesture-ens-"))
    (root / "src").mkdir()
    for name in ("main.py", "features.py", "landmark_frontend.py"):
        shutil.copy2(SRC / name, root / "src" / name)
    models = root / "models"
    models.mkdir()
    shutil.copy2(MODELS / "hand_landmarker.task", models / "hand_landmarker.task")
    shutil.copy2(MODELS / "landmark_labels.json", models / "landmark_labels.json")

    meta = json.loads(META.read_text(encoding="utf-8"))
    meta.pop("members", None)
    if members is not None:
        meta["members"] = list(members)
    if meta_override:
        meta.update(meta_override)
    (models / "landmark_model_meta.json").write_text(
        json.dumps(meta), encoding="utf-8")

    if copy_single:
        shutil.copy2(MODELS / "landmark_classifier_fp16.tflite",
                     models / "landmark_classifier_fp16.tflite")
    for member in (members or []):
        if member == drop_member or member == "../x.tflite":
            continue
        source = MODELS / member
        if source.is_file():
            shutil.copy2(source, models / member)
    return root


class EnsembleRuntimeTest(unittest.TestCase):
    """The shipped ensemble: loading, IO, aggregation and the public contract."""

    @classmethod
    def setUpClass(cls):
        cls.members = _meta_members()

    def test_meta_lists_four_ordered_members(self):
        self.assertEqual(
            self.members,
            [
                "landmark_ens_tcn21_fp16.tflite",
                "landmark_ens_tcnm61_fp16.tflite",
                "landmark_ens_tcnm62_fp16.tflite",
                "landmark_ens_rob71_fp16.tflite",
            ],
        )

    def test_init_loads_every_member_with_own_tensor_indices(self):
        main = load_main()
        main._init_model()
        self.assertEqual(main._MEMBER_NAMES, self.members)
        self.assertEqual(len(main._INTERPRETERS), 4)
        self.assertEqual(len(main._MEMBER_IO), 4)

        # Indices are resolved per interpreter; architectures disagree.
        out_indices = [io[1] for io in main._MEMBER_IO]
        self.assertGreater(len(set(out_indices)), 1,
                           "expected per-architecture output indices")
        for interp, (in_idx, out_idx, *_rest) in zip(
                main._INTERPRETERS, main._MEMBER_IO):
            self.assertEqual(interp.get_input_details()[0]["index"], in_idx)
            self.assertEqual(interp.get_output_details()[0]["index"], out_idx)

        # All members agree on the tensor contract.
        for interp in main._INTERPRETERS:
            self.assertEqual(list(interp.get_input_details()[0]["shape"]),
                             [1, 24, 255])
            self.assertEqual(list(interp.get_output_details()[0]["shape"]),
                             [1, 100])
        self.assertEqual(main._INPUT_SCALE, 0.0)
        self.assertEqual(main._OUTPUT_SCALE, 0.0)

    def test_load_is_cached(self):
        main = load_main()
        main._init_model()
        first = list(main._INTERPRETERS)
        main._init_model()
        self.assertEqual(first, list(main._INTERPRETERS))
        self.assertIs(first[0], main._INTERPRETERS[0])

    def test_averaged_logits_match_manual_mean(self):
        """predict()'s ensemble path equals a hand-computed mean of 4 raw outputs."""
        from features import build_features
        from landmark_frontend import read_frames, detect_hands

        main = load_main()
        main._init_model()

        frames = read_frames(str(CLIP), max_frames=max(32, main._FRAMES))
        landmarks, present = detect_hands(frames)
        feats, _ = build_features(landmarks, present,
                                  expected_frames=main._FRAMES)
        tensor = feats[None, ...].astype(np.float32)

        # 1. The helper predict() calls.
        aggregated, _member_ms = main._run_ensemble(tensor)

        # 2. The same four invokes done by hand in the test, via the module's
        #    cached interpreters but NOT via _run_ensemble.
        manual = []
        for interp, (in_idx, out_idx, in_scale, in_zp, out_scale, out_zp) in zip(
                main._INTERPRETERS, main._MEMBER_IO):
            if in_scale > 0:
                q = np.clip(np.round(tensor / in_scale) + in_zp,
                            -128, 127).astype(np.int8)
            else:
                q = tensor.astype(np.float32)
            interp.set_tensor(in_idx, q)
            interp.invoke()
            raw = interp.get_tensor(out_idx)[0].astype(np.float32)
            manual.append((raw - out_zp) * out_scale if out_scale > 0 else raw)

        expected = np.mean(manual, axis=0)
        np.testing.assert_allclose(aggregated, expected, rtol=1e-6, atol=1e-6)

        # And the mean is not any single member's output (sanity).
        for raw in manual:
            self.assertFalse(np.allclose(aggregated, raw))
        # The public prediction is the argmax of that mean after softmax.
        probs = np.exp(aggregated - aggregated.max())
        probs /= probs.sum()
        self.assertEqual(main._IDX_TO_LABEL[int(np.argmax(probs))], ENSEMBLE_TOP1)

    def test_predict_real_clip_contract(self):
        main = load_main()
        result = main.predict(str(CLIP))
        self.assertEqual(
            set(result), {"clip_id", "prediction", "top5", "confidence",
                          "latency_ms"})
        self.assertEqual(result["prediction"], ENSEMBLE_TOP1)
        self.assertEqual(result["top5"][0], result["prediction"])
        self.assertEqual(len(result["top5"]), 5)
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)
        self.assertEqual(result["clip_id"], CLIP.stem)

    def test_repeated_calls_reuse_interpreters(self):
        main = load_main()
        first = main.predict(str(CLIP))
        cached = list(main._INTERPRETERS)
        second = main.predict(str(CLIP))
        self.assertEqual(first["prediction"], second["prediction"])
        self.assertEqual(first["confidence"], second["confidence"])
        self.assertIs(cached[0], main._INTERPRETERS[0])

    def test_profile_records_ensemble_timing(self):
        main = load_main()
        main.predict(str(CLIP), profile=True)
        stages = main.get_last_profile()["stages"]
        self.assertIn("model_inference", stages)
        inference = stages["model_inference"]
        self.assertEqual(inference["members"], 4)
        self.assertEqual(set(inference["member_ms"]), set(self.members))
        # Existing stage keys are preserved.
        for key in ("video_open_decode", "landmark_tracking",
                    "feature_preparation", "output_postprocessing",
                    "model_initialization"):
            self.assertIn(key, stages)


class EnsembleFailureTest(unittest.TestCase):
    """Misconfiguration must fail loudly and name the offender."""

    def setUp(self):
        self._saved_env = {
            k: os.environ.pop(k, None)
            for k in (_ENV_MODEL, _ENV_TRACKER, _ENV_WORKERS)
        }
        self._roots = []

    def tearDown(self):
        for key, value in self._saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        for root in self._roots:
            shutil.rmtree(root, ignore_errors=True)

    def _fixture(self, *args, **kwargs):
        root = _make_fixture(*args, **kwargs)
        self._roots.append(root)
        return root

    def _load(self, root):
        spec = importlib.util.spec_from_file_location(
            f"echogesture_main_fixture_{id(root)}", root / "src" / "main.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def test_missing_member_raises_file_not_found_naming_it(self):
        missing = "landmark_ens_tcnm62_fp16.tflite"
        root = self._fixture(_meta_members(), drop_member=missing)
        main = self._load(root)
        with self.assertRaises(FileNotFoundError) as ctx:
            main._init_model()
        self.assertIn(missing, str(ctx.exception))

    def test_unsafe_member_name_raises_value_error(self):
        root = self._fixture(["../x.tflite"])
        main = self._load(root)
        with self.assertRaises(ValueError) as ctx:
            main._init_model()
        self.assertIn("../x.tflite", str(ctx.exception))

    def test_empty_members_list_raises(self):
        root = self._fixture([])
        main = self._load(root)
        with self.assertRaises(ValueError):
            main._init_model()

    def test_wrong_feature_dim_raises_clear_error(self):
        root = self._fixture(meta_override={"feature_dim": 256})
        main = self._load(root)
        with self.assertRaises(ValueError) as ctx:
            main._init_model()
        self.assertIn("feature_dim", str(ctx.exception))

    def test_no_members_key_uses_single_model_explicitly(self):
        main = self._load(self._fixture())
        main._init_model()
        self.assertEqual(main._MEMBER_NAMES,
                         ["landmark_classifier_fp16.tflite"])
        self.assertEqual(len(main._INTERPRETERS), 1)

        result = main.predict(str(CLIP))
        self.assertEqual(result["prediction"], SINGLE_TOP1)
        self.assertEqual(result["top5"][0], result["prediction"])


class BackendSelectionTest(unittest.TestCase):
    """s3d selection and strict env validation are unchanged."""

    def setUp(self):
        self._saved = {
            k: os.environ.pop(k, None)
            for k in (_ENV_MODEL, _ENV_TRACKER, _ENV_WORKERS)
        }

    def tearDown(self):
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_bogus_model_env_raises(self):
        os.environ[_ENV_MODEL] = "nonsense"
        main = load_main()
        with self.assertRaises(ValueError):
            main._init_model()

    def test_s3d_env_selects_s3d_and_skips_ensemble(self):
        os.environ[_ENV_MODEL] = "s3d"
        main = load_main()
        main._init_model()
        self.assertEqual(main._MODEL_BACKEND, "s3d")
        self.assertEqual(main._MEMBER_NAMES, [])
        self.assertEqual(main._MEMBER_IO, [])
        self.assertIsNotNone(main._INTERPRETER)


@unittest.skipUnless(_can_run(), "real clip or landmark assets unavailable")
class TrackerParityTest(unittest.TestCase):
    """The opt-in parallel tracker still returns the singleton label."""

    def setUp(self):
        self._saved = {
            k: os.environ.pop(k, None)
            for k in (_ENV_TRACKER, _ENV_WORKERS)
        }
        os.environ.pop(_ENV_MODEL, None)

    def tearDown(self):
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_parallel_matches_singleton_label(self):
        singleton_main = load_main()
        singleton = singleton_main.predict(str(CLIP))

        os.environ[_ENV_TRACKER] = "parallel"
        os.environ[_ENV_WORKERS] = "2"
        parallel_main = load_main()
        try:
            parallel = parallel_main.predict(str(CLIP))
        finally:
            parallel_main.close_tracker()

        self.assertEqual(parallel["prediction"], singleton["prediction"])
        self.assertEqual(parallel["top5"], singleton["top5"])


if __name__ == "__main__":
    unittest.main()