"""Narrow reproduction for defect D-2.

OBSERVED
--------
robustness.paired_evaluate() creates ``output_dir/work`` and removes it with
``shutil.rmtree(workdir)`` after the loop.  The removal is NOT in a
``try/finally``, so when any clip fails to decode (corrupt/empty video, a
class of input production clips can contain), the raw ``ValueError`` from
``corruptions.decode_video`` propagates out of ``paired_evaluate`` and the
intermediate ``work/`` directory with its encoded corrupted clips is left on
disk inside the run output.  A subsequent run in the same output directory
then reuses the same ``work/`` paths.

Run with the project venv (numpy/opencv importable):
    /Users/abhijeetmiskin/AppData/EchoGesture/.venv/bin/python reproduce_paired_cleanup.py
Exit code 0 means the defect reproduces, 1 means work/ is cleaned up.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path("/tmp/echogesture-agents/A14")
sys.path.insert(0, str(ROOT / "evaluation" / "device"))
import corruptions
import robustness


def main():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        good = root / "good.mp4"
        corruptions.write_video(good, [np.zeros((20, 20, 3), np.uint8)] * 3, 10)
        broken = root / "broken.mp4"
        broken.write_bytes(b"not an encoded video")
        out = root / "out"

        def predict(path, *, clip_id=None):
            return {"clip_id": clip_id, "prediction": "CANDY", "top5": ["CANDY"],
                    "confidence": 1.0, "latency_ms": 1}

        raised = None
        try:
            robustness.paired_evaluate(
                [("good", good, "CANDY", "s1"), ("broken", broken, "CANDY", "s1")],
                predict, {"CANDY"},
                corruptions.CorruptionSpec("crop", 0.5, 0), out)
        except ValueError as exc:
            raised = exc
        print(f"paired_evaluate raised: {raised!r}")
        print(f"work/ left behind: {(out / 'work').exists()}")
        if raised is not None and (out / "work").exists():
            print("DEFECT REPRODUCED: corrupt clip skipped cleanup of work/.")
            return 0
        print("OK: work/ cleaned up even on decode failure.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
