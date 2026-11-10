"""Reproduction for the D-1 profile cross-talk interaction.

run_manifest.load_predictor() registers every predictor module under one
fixed name, "echogesture_predictor". run_manifest.evaluate_profiled()
resolves get_last_profile() via sys.modules[predict.__module__] at call
time. So after loading a second (unprofiled) predictor, calling
evaluate_profiled() for the first predictor looks up get_last_profile()
on the *second* module and raises EvaluationError.

OBSERVED (current production, 2026-11-09):
    fresh A alone            -> evaluate_profiled() returns stage profile
    load A, then load B      -> evaluate_profiled(A) raises
                                "Predictor does not expose get_last_profile()"

This is a behavior/contract gap, not an asserted regression test:
production code is outside A14 ownership and no fix is fabricated here.
Exit code 0 means the cross-talk reproduces; 1 means evaluate_profiled()
survived loading a second predictor (behavior is safe for this flow).

Run with the project venv:
    .venv/bin/python evaluation/device/v7_qa/repro_d1_profile_crosstalk.py
"""

import sys
import tempfile
from pathlib import Path

DEVICE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEVICE_DIR))

import run_manifest  # noqa: E402


def main():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        a = root / "a.py"
        a.write_text(
            "def predict(path, *, clip_id=None, profile=False):\n"
            "    return {'clip_id': clip_id, 'prediction': 'CANDY',\n"
            "            'confidence': 1.0, 'latency_ms': 1}\n"
            "def get_last_profile():\n"
            "    return {'stages': {'model_inference': {}}}\n"
        )
        b = root / "b.py"
        b.write_text(
            "def predict(path):\n"
            "    return {'clip_id': 'b', 'prediction': 'CANDY',\n"
            "            'confidence': 1.0, 'latency_ms': 1}\n"
        )
        clip = root / "c.mp4"
        clip.write_bytes(b"fixture")
        manifest = root / "manifest.csv"
        manifest.write_text("clip_id,path\na,c.mp4\n", encoding="utf-8")
        items = run_manifest.read_manifest(manifest)
        labels = {"CANDY"}

        predict_a = run_manifest.load_predictor(a)
        run_manifest.load_predictor(b)

        try:
            run_manifest.evaluate_profiled(items, predict_a, labels)
        except run_manifest.EvaluationError as exc:
            print(f"REPRODUCED: evaluate_profiled(A) after loading B raised: {exc}")
            return 0
        print("OK: evaluate_profiled(A) survived loading B.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
