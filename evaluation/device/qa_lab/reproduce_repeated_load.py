"""Narrow reproduction for defect D-1.

OBSERVED
--------
run_manifest.load_predictor() registers the imported predictor module under one
fixed module name, ``echogesture_predictor``.  Loading a second predictor file
therefore REPLACES the module object, and functions already returned from the
first predictor resolve their globals through the second module.  In a worker
that loads predictors more than once (the offline demo harness reloads the
predictor file for each captured session via app/offline_demo.load_predictor,
which uses its own name), a first predictor that caches/returns module-level
state can be corrupted, and any consumer retaining the first ``predict``
function observes the second module's globals.

Run with the project venv (numpy/opencv importable):
    /Users/abhijeetmiskin/AppData/EchoGesture/.venv/bin/python reproduce_repeated_load.py
Exit code 0 means the defect reproduces, 1 means behaviour is safe.
"""

import sys
import tempfile
from pathlib import Path

ROOT = Path("/tmp/echogesture-agents/A14")
sys.path.insert(0, str(ROOT / "evaluation" / "device"))
import run_manifest


def main():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        first = root / "predictor_a.py"
        first.write_text(
            "TOKEN = 'alpha'\n"
            "def predict(path):\n"
            "    return {'clip_id': 'a', 'prediction': 'CANDY', 'top5': ['CANDY'],\n"
            "            'confidence': 1.0, 'latency_ms': 1}\n"
            "def get_token():\n"
            "    return TOKEN\n"
        )
        second = root / "predictor_b.py"
        second.write_text(
            "TOKEN = 'bravo'\n"
            "def predict(path):\n"
            "    return {'clip_id': 'b', 'prediction': 'CANDY', 'top5': ['CANDY'],\n"
            "            'confidence': 1.0, 'latency_ms': 1}\n"
        )
        predict_a = run_manifest.load_predictor(first)
        stale_predict = predict_a  # retained by an earlier consumer
        _ = run_manifest.load_predictor(second)

        module_globals = sys.modules["echogesture_predictor"].predict.__globals__
        leaked = module_globals.get("TOKEN")
        print(f"retained predict_a.__globals__['TOKEN'] = {leaked!r} (expected 'alpha')")

        # A retained function from load #1 must still see load #1's globals.
        if leaked == "bravo":
            print("DEFECT REPRODUCED: loading a second predictor replaced the "
                  "module object behind the fixed name 'echogesture_predictor'.")
            return 0
        print("OK: retained predictor kept its own module globals.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
