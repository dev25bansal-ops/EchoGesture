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
import cv2

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

# Global model state (singleton pattern to avoid reloading per clip)
_INTERPRETER = None
_INPUT_DETAILS = None
_OUTPUT_DETAILS = None
_INPUT_SCALE = 1.0
_INPUT_ZP = 0
_OUTPUT_SCALE = 1.0
_OUTPUT_ZP = 0
_IDX_TO_LABEL = {}


def _init_model():
    global _INTERPRETER, _INPUT_DETAILS, _OUTPUT_DETAILS
    global _INPUT_SCALE, _INPUT_ZP, _OUTPUT_SCALE, _OUTPUT_ZP, _IDX_TO_LABEL

    if _INTERPRETER is not None:
        return

    if Interpreter is None:
        raise RuntimeError(
            "No TFLite interpreter found. Please install ai-edge-litert: "
            "pip install ai-edge-litert"
        )

    # Locate models relative to this file
    base_dir = Path(__file__).resolve().parent.parent
    model_path = base_dir / "models" / "s3d_conv1d_qat_int8.tflite"
    if not model_path.exists():
        model_path = base_dir / "models" / "ptq_100_int8.tflite"

    label_path = base_dir / "models" / "label100.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"Label file not found at {label_path}")

    with open(label_path, "r", encoding="utf-8") as f:
        labels_map = json.load(f)
    _IDX_TO_LABEL = {int(v): k.upper() for k, v in labels_map.items()}

    _INTERPRETER = Interpreter(model_path=str(model_path))
    _INTERPRETER.allocate_tensors()

    _INPUT_DETAILS = _INTERPRETER.get_input_details()[0]
    _OUTPUT_DETAILS = _INTERPRETER.get_output_details()[0]

    _INPUT_SCALE, _INPUT_ZP = _INPUT_DETAILS["quantization"]
    _OUTPUT_SCALE, _OUTPUT_ZP = _OUTPUT_DETAILS["quantization"]


def _preprocess_video(video_path: str, target_frames: int = 24, target_size: int = 64):
    """
    Decodes video, converts to grayscale, resizes to target_size x target_size,
    normalizes to [0, 1], and samples target_frames using linspace.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        resized = cv2.resize(frame, (target_size, target_size))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        norm = np.expand_dims(gray, -1).astype(np.float32) / 255.0
        frames.append(norm)
    cap.release()

    n = len(frames)
    if n == 0:
        raise ValueError(f"No valid frames decoded from: {video_path}")

    if n >= target_frames:
        indices = np.linspace(0, n - 1, target_frames).astype(int)
        sampled = np.stack([frames[i] for i in indices])
    else:
        # Pad with the last frame
        sampled = np.stack(frames + [frames[-1]] * (target_frames - n))

    # Add batch dimension: [1, 24, 64, 64, 1]
    return np.expand_dims(sampled, 0)


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

    # Preprocess video frames
    tensor = _preprocess_video(video_path, target_frames=24, target_size=64)

    # Quantize input to int8
    if _INPUT_SCALE > 0:
        q_tensor = np.clip(np.round(tensor / _INPUT_SCALE) + _INPUT_ZP, -128, 127).astype(np.int8)
    else:
        q_tensor = tensor.astype(np.float32)

    # Invoke TFLite model
    _INTERPRETER.set_tensor(_INPUT_DETAILS["index"], q_tensor)
    _INTERPRETER.invoke()

    # Dequantize output logits / probabilities
    raw_output = _INTERPRETER.get_tensor(_OUTPUT_DETAILS["index"])[0].astype(np.float32)
    if _OUTPUT_SCALE > 0:
        logits = (raw_output - _OUTPUT_ZP) * _OUTPUT_SCALE
    else:
        logits = raw_output

    # Softmax for normalized confidence scores
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)

    # Sort descending
    top_indices = np.argsort(-probs)[:5]
    top5 = [_IDX_TO_LABEL.get(idx, f"UNKNOWN_{idx}") for idx in top_indices]
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
