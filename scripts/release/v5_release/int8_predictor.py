"""INT8 evaluation-only predictor wrapper.

Paired A/B harness (owned by Agent D). Loads the untouched `src/main.py`
predictor and, in this process only, points `_model_paths()` at
`models/landmark_classifier_int8.tflite`. This is an explicit evaluation
override — it is NOT a default-selection change and must not be read as one.
`src/main.py` still ships FP16 for `ECHOGESTURE_MODEL=landmark`.

Why the wrapper exists: `src/main.py._model_paths()` hardcodes the FP16
filename for the landmark backend, so a real paired FP16-vs-INT8 run through
the actual inference path needs a process-local override. The `robustness.py`
`--model` argument records which artifact was evaluated (hash/config binding);
it does not itself swap the predictor's model.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import main  # noqa: E402


def _int8_paths():
    return (
        "landmark",
        ROOT / "models" / "landmark_classifier_int8.tflite",
        ROOT / "models" / "landmark_labels.json",
    )


main._model_paths = _int8_paths


def predict(video_path: str, *, clip_id: str | None = None, profile: bool = False):
    main._init_model()  # binds INT8 before first call
    return main.predict(video_path, clip_id=clip_id, profile=profile)
