"""
EchoGesture: Sign Language Prediction on Edge Device
Track 2 — Samsung XR Hackathon 2026 (Samsung R&D Institute India, Bangalore)

Fully on-device, exocentric sign language translation pipeline.
Evaluator contract:
    def predict(video_path: str) -> dict
"""

import os
import sys
import time
import json
from pathlib import Path
import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from features import build_features, FEATURE_DIM  # noqa: E402
from landmark_frontend import clip_to_landmarks  # noqa: E402

# Universal TFLite interpreter loader
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
        Interpreter = tflite.Interpreter
    except ImportError:
        try:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter
        except ImportError:
            Interpreter = None

# Global model state (singleton pattern to avoid reloading per clip).
# The shipped artifact is an ENSEMBLE: several independently-trained models
# whose logits are averaged. Each member makes different errors on unseen
# signers, and the mean cancels part of that variance - measured 70.22% top-1
# versus 67.65% for the best single member, on 11 held-out signers.
_INTERPRETERS = []
_MEMBER_IO = []
_INPUT_DETAILS = None
_OUTPUT_DETAILS = None
_INPUT_SCALE = 1.0
_INPUT_ZP = 0
_OUTPUT_SCALE = 1.0
_OUTPUT_ZP = 0
_IDX_TO_LABEL = {}
_FRAMES = 24


def _init_model():
    global _FRAMES
    global _INPUT_DETAILS, _OUTPUT_DETAILS, _IDX_TO_LABEL

    if _INTERPRETERS:
        return

    if Interpreter is None:
        raise RuntimeError(
            "No TFLite interpreter found. Please install ai-edge-litert: "
            "pip install ai-edge-litert"
        )

    base_dir = Path(__file__).resolve().parent.parent
    model_dir = base_dir / "models"
    label_path = model_dir / "landmark_labels.json"
    meta_path = model_dir / "landmark_model_meta.json"

    if not label_path.exists():
        raise FileNotFoundError(f"Label file not found at {label_path}")

    with open(label_path, "r", encoding="utf-8") as f:
        labels_map = json.load(f)
    _IDX_TO_LABEL = {int(v): k.upper() for k, v in labels_map.items()}

    # Members come from the meta sidecar so adding or dropping one never
    # requires touching this file.
    members = []
    if meta_path.exists():
        try:
            members = list(json.load(open(meta_path, encoding="utf-8")).get("members", []))
        except Exception:
            members = []
    if not members:
        # Fall back to the single-model artifact if the ensemble is absent.
        single = model_dir / "landmark_classifier_fp16.tflite"
        if not single.exists():
            raise FileNotFoundError(f"No model found in {model_dir}")
        members = [single.name]

    for name in members:
        p = model_dir / name
        if not p.exists():
            raise FileNotFoundError(f"Ensemble member missing: {p}")
        interp = Interpreter(model_path=str(p))
        interp.allocate_tensors()
        _INTERPRETERS.append(interp)

    # Tensor indices are per-interpreter and differ between architectures
    # (the conv member and the tcn members do not agree). Resolving them once
    # per model here avoids indexing one interpreter with another's handles.
    for interp in _INTERPRETERS:
        _MEMBER_IO.append((interp.get_input_details()[0]["index"],
                           interp.get_output_details()[0]["index"]))

    _INPUT_DETAILS = _INTERPRETERS[0].get_input_details()[0]
    _OUTPUT_DETAILS = _INTERPRETERS[0].get_output_details()[0]

    # The export wrapper takes float32 IO around an INT8 graph: the input is
    # the 255-float feature row, not a quantised image. Frame count ships in
    # the label sidecar because .scratch is gitignored and will not exist on
    # the evaluator's machine.
    _meta = base_dir / "models" / "landmark_model_meta.json"
    if _meta.exists():
        try:
            _FRAMES = int(json.load(open(_meta, encoding="utf-8"))["frames"])
        except Exception:
            pass


def _preprocess_video(video_path: str, target_frames: int = 24):
    """Video -> the model's (T, FEATURE_DIM) landmark feature matrix.

    MediaPipe extracts two hands per frame into canonical left/right slots,
    then features.py normalises them. The same functions run in training, so
    the two cannot drift.

    Raises ValueError when no hand is visible in any sampled frame: predicting
    a gloss from an empty clip would be a guess dressed up as a measurement.
    """
    lm, present = clip_to_landmarks(video_path, max_frames=max(32, target_frames))
    if lm.shape[0] == 0 or float(present.max()) < 0.5:
        raise ValueError(f"No hand detected in any frame of: {video_path}")

    feats, _mask = build_features(lm, present, expected_frames=target_frames)
    return feats


def predict(video_path: str) -> dict:
    """
    Return the predicted gloss/word for one video clip. Must run fully on-device.

    Return format:
    {
        "clip_id": "test_0412",
        "prediction": "HAPPY",
        "top5": ["HAPPY", "HAND", "HOUSE", "HELP", "HOME"],
        "confidence": 0.81,
        "latency_ms": 23.4
    }
    """
    start_time = time.perf_counter()

    _init_model()

    # Extract clip_id from video filename without extension
    clip_id = Path(video_path).stem

    # Landmarks -> FEATURE_DIM-wide feature matrix
    feats = _preprocess_video(video_path, target_frames=_FRAMES)

    # The export wrapper takes (1, T, F) float32 and returns logits.
    tensor = feats[None, ...].astype(np.float32)

    # Average logits across ensemble members, then softmax once. Averaging
    # probabilities instead would let one over-confident member dominate;
    # averaging logits keeps each member's vote proportional to its evidence.
    member_logits = []
    for interp, (in_idx, out_idx) in zip(_INTERPRETERS, _MEMBER_IO):
        interp.set_tensor(in_idx, tensor)
        interp.invoke()
        member_logits.append(interp.get_tensor(out_idx)[0].astype(np.float32))
    logits = np.mean(member_logits, axis=0)

    # Softmax for normalized confidence scores
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)

    # Sort descending
    top_indices = np.argsort(-probs)[:5]
    top5 = [_IDX_TO_LABEL.get(int(idx), f"UNKNOWN_{idx}") for idx in top_indices]
    prediction = top5[0]
    confidence = float(probs[top_indices[0]])

    latency_ms = (time.perf_counter() - start_time) * 1000.0

    return {
        "clip_id": clip_id,
        "prediction": prediction,
        "top5": top5,
        "confidence": round(confidence, 4),
        "latency_ms": round(latency_ms, 2)
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_video.mp4>")
        sys.exit(1)

    video_arg = sys.argv[1]
    res = predict(video_arg)
    print(json.dumps(res, indent=2))
