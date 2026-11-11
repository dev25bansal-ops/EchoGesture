"""
EchoGesture: Sign Language Prediction on Edge Device
Track 2 — Samsung XR Hackathon 2026 (Samsung R&D Institute India, Bangalore)

Fully on-device, exocentric sign language translation pipeline.
Evaluator contract:
    def predict(video_path: str) -> dict
"""

import atexit
import importlib.util
import os
import sys
import threading
import time
import json
from pathlib import Path
import numpy as np
import cv2

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

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
_LAST_PROFILE = None
_ACTIVE_MODEL = None
_FRAMES = 24
_MODEL_BACKEND = "s3d"

# Opt-in parallel landmark tracking (default off; see ECHOGESTURE_TRACKER below).
_TRACKER = None
_TRACKER_KEY = None
_TRACKER_LOCK = threading.Lock()
_TRACKER_ATEXIT_REGISTERED = False
_DEFAULT_WORKERS = 4
_TRACKER_MODULE_PATH = (
    Path(__file__).resolve().parent.parent
    / "evaluation/device/performance_final/tracker.py"
)


def _model_paths():
    backend = os.environ.get("ECHOGESTURE_MODEL", "landmark")
    if backend not in ("s3d", "landmark"):
        raise ValueError("ECHOGESTURE_MODEL must be 's3d' or 'landmark'")
    base_dir = Path(__file__).resolve().parent.parent
    if backend == "landmark":
        return backend, base_dir / "models/landmark_classifier_fp16.tflite", base_dir / "models/landmark_labels.json"
    model_path = base_dir / "models/s3d_conv1d_qat_int8.tflite"
    if not model_path.exists():
        model_path = base_dir / "models/ptq_100_int8.tflite"
    return backend, model_path, base_dir / "models/label100.json"


def _record_stage(name, start_ns, **metadata):
    if _LAST_PROFILE is not None:
        _LAST_PROFILE.setdefault("stages", {})[name] = {
            "milliseconds": (time.perf_counter_ns() - start_ns) / 1e6,
            **metadata,
        }


def get_last_profile():
    return _LAST_PROFILE


def _tracker_config():
    """Read (mode, workers) for the opt-in parallel tracker from the env.

    Default (unset) is the singleton IMAGE path.  Any value other than the
    documented opt-in failures loudly rather than silently ignoring a typo.
    """
    mode = os.environ.get("ECHOGESTURE_TRACKER", "").strip().lower()
    if mode in ("", "image", "singleton"):
        return "singleton", 0
    if mode != "parallel":
        raise ValueError(
            "ECHOGESTURE_TRACKER must be unset, 'image' or 'parallel' "
            f"(got {mode!r})")
    raw = os.environ.get("ECHOGESTURE_WORKERS", "").strip()
    if not raw:
        workers = _DEFAULT_WORKERS
    else:
        try:
            workers = int(raw)
        except ValueError:
            raise ValueError(
                f"ECHOGESTURE_WORKERS must be an integer (got {raw!r})") from None
    if workers < 1:
        raise ValueError(f"ECHOGESTURE_WORKERS must be >= 1 (got {workers})")
    return "parallel", workers


def _load_tracker_class():
    """Import ParallelImageTracker from the evaluation package by path.

    It lives outside `src/`, so it is not an importable module on the
    evaluator's path by default; load it explicitly without touching sys.path.
    """
    if not _TRACKER_MODULE_PATH.is_file():
        raise RuntimeError(
            f"Parallel tracker not available at {_TRACKER_MODULE_PATH}; "
            "unset ECHOGESTURE_TRACKER or restore the tracker module")
    if "echogesture_parallel_tracker" in sys.modules:
        return sys.modules["echogesture_parallel_tracker"].ParallelImageTracker
    spec = importlib.util.spec_from_file_location(
        "echogesture_parallel_tracker", _TRACKER_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load parallel tracker from {_TRACKER_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    return module.ParallelImageTracker


def _close_tracker():
    """Close and drop the persistent pool; idempotent and atexit-safe."""
    global _TRACKER, _TRACKER_KEY
    with _TRACKER_LOCK:
        tracker, _TRACKER = _TRACKER, None
        _TRACKER_KEY = None
    if tracker is not None:
        tracker.close()


def close_tracker():
    """Public hook so callers can release worker threads without exiting."""
    _close_tracker()


def _get_tracker(workers: int):
    """Return the process-wide pool for `workers`, (re)building if needed.

    Reusing one pool across predict() calls avoids spawning a fresh pool (and
    `workers` landmarkers) per clip.  A worker-count change replaces it.
    """
    global _TRACKER, _TRACKER_KEY, _TRACKER_ATEXIT_REGISTERED
    key = int(workers)
    if _TRACKER is not None and _TRACKER_KEY == key:
        return _TRACKER
    with _TRACKER_LOCK:
        if _TRACKER is not None and _TRACKER_KEY == key:
            return _TRACKER
        if _TRACKER is not None:
            _TRACKER.close()
            _TRACKER = None
            _TRACKER_KEY = None
        tracker = _load_tracker_class()(workers=key)
        _TRACKER, _TRACKER_KEY = tracker, key
    if not _TRACKER_ATEXIT_REGISTERED:
        atexit.register(_close_tracker)
        _TRACKER_ATEXIT_REGISTERED = True
    return tracker


def _set_interpreter_threads(interpreter):
    return interpreter


def _init_model():
    global _INTERPRETER, _INPUT_DETAILS, _OUTPUT_DETAILS
    global _INPUT_SCALE, _INPUT_ZP, _OUTPUT_SCALE, _OUTPUT_ZP, _IDX_TO_LABEL
    global _ACTIVE_MODEL, _MODEL_BACKEND, _FRAMES

    backend, model_path, label_path = _model_paths()
    if _INTERPRETER is not None and _ACTIVE_MODEL == model_path:
        return

    if Interpreter is None:
        raise RuntimeError(
            "No TFLite interpreter found. Please install ai-edge-litert: "
            "pip install ai-edge-litert"
        )
    if not model_path.exists():
        raise FileNotFoundError(f"Selected model file not found: {model_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"Selected vocabulary file not found: {label_path}")
    if backend == "landmark":
        for asset in ("hand_landmarker.task", "landmark_model_meta.json"):
            if not (model_path.parent / asset).is_file():
                raise FileNotFoundError(f"Missing landmark runtime asset: {model_path.parent / asset}")

    with open(label_path, "r", encoding="utf-8") as f:
        labels_map = json.load(f)
    if not isinstance(labels_map, dict) or not labels_map:
        raise ValueError("Vocabulary must be a non-empty label-to-index object")
    _IDX_TO_LABEL = {int(v): k.upper() for k, v in labels_map.items()}
    if sorted(_IDX_TO_LABEL) != list(range(len(_IDX_TO_LABEL))):
        raise ValueError("Vocabulary indices must be contiguous from zero")

    _INTERPRETER = Interpreter(model_path=str(model_path))
    _INTERPRETER.allocate_tensors()
    _INPUT_DETAILS = _INTERPRETER.get_input_details()[0]
    _OUTPUT_DETAILS = _INTERPRETER.get_output_details()[0]
    if _OUTPUT_DETAILS["shape"][-1] != len(_IDX_TO_LABEL):
        raise ValueError("Model output classes do not match the selected vocabulary")
    _INPUT_SCALE, _INPUT_ZP = _INPUT_DETAILS["quantization"]
    _OUTPUT_SCALE, _OUTPUT_ZP = _OUTPUT_DETAILS["quantization"]
    _MODEL_BACKEND = backend
    if backend == "landmark":
        meta_path = model_path.parent / "landmark_model_meta.json"
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        _FRAMES = int(metadata["frames"])
        if list(_INPUT_DETAILS["shape"]) != [1, _FRAMES, int(metadata["feature_dim"])]:
            raise ValueError("Model input shape does not match landmark metadata")
        if _INPUT_DETAILS["dtype"] != np.float32 or _OUTPUT_DETAILS["dtype"] != np.float32:
            raise ValueError("Dev FP16 classifier requires float32 input/output tensors")
        from features import FEATURE_DIM
        if FEATURE_DIM != int(metadata["feature_dim"]):
            raise ValueError("Feature implementation does not match landmark metadata")
    _ACTIVE_MODEL = model_path


def _preprocess_video(video_path: str, target_frames: int = 24, target_size: int = 64):
    started_ns = time.perf_counter_ns()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    source = {
        "declared_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    _record_stage("video_open_decode", started_ns, **source, decoded_frames=len(frames))

    started_ns = time.perf_counter_ns()
    processed = []
    for frame in frames:
        resized = cv2.resize(frame, (target_size, target_size))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        processed.append(np.expand_dims(gray, -1).astype(np.float32) / 255.0)
    _record_stage("frame_preprocess", started_ns)

    n = len(processed)
    if n == 0:
        raise ValueError(f"No valid frames decoded from: {video_path}")

    started_ns = time.perf_counter_ns()
    if n >= target_frames:
        indices = np.linspace(0, n - 1, target_frames).astype(int)
        sampled = np.stack([processed[i] for i in indices])
    else:
        sampled = np.stack(processed + [processed[-1]] * (target_frames - n))
    _record_stage("frame_selection", started_ns, processed_frames=target_frames)
    return np.expand_dims(sampled, 0)


def predict(video_path: str, *, clip_id: str | None = None, profile: bool = False) -> dict:
    """
    Return the predicted gloss/word for one video clip. Must run fully on-device.

    clip_id only overrides the filename-derived ID for callers that already have
    an organizer manifest ID; the one-argument evaluator contract is unchanged.

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
    global _LAST_PROFILE
    _LAST_PROFILE = {"stages": {}} if profile else None

    started_ns = time.perf_counter_ns()
    _init_model()
    _record_stage("model_initialization", started_ns)

    clip_id = clip_id if clip_id is not None else Path(video_path).stem

    if _MODEL_BACKEND == "landmark":
        from landmark_frontend import read_frames, detect_hands
        from features import build_features

        started_ns = time.perf_counter_ns()
        frames = read_frames(video_path, max_frames=max(32, _FRAMES))
        _record_stage("video_open_decode", started_ns, sampled_frames=len(frames))
        if not frames:
            raise ValueError(f"No valid frames decoded from: {video_path}")
        started_ns = time.perf_counter_ns()
        # Opt-in parallel tracking; unset env keeps the singleton path byte-for-byte.
        # A malformed config is a caller error and must surface, not be silently
        # ignored; only a runtime pool failure falls back to the singleton path.
        tracker_mode, tracker_workers = _tracker_config()
        if tracker_mode == "parallel":
            try:
                landmarks, present = _get_tracker(tracker_workers).detect(frames)
                tracking_engine = "parallel"
            except Exception:
                # Never let an opt-in speed path turn a good clip into a failure.
                landmarks, present = detect_hands(frames)
                tracking_engine = "singleton_fallback"
        else:
            landmarks, present = detect_hands(frames)
            tracking_engine = "singleton"
        _record_stage("landmark_tracking", started_ns,
                      frames_with_hands=int((present.max(axis=1) > 0.5).sum()),
                      engine=tracking_engine)
        if landmarks.shape[0] == 0 or float(present.max()) < 0.5:
            raise ValueError(f"No hand detected in any frame of: {video_path}")
        started_ns = time.perf_counter_ns()
        tensor, _ = build_features(landmarks, present, expected_frames=_FRAMES)
        tensor = tensor[None, ...].astype(np.float32)
        _record_stage("feature_preparation", started_ns, processed_frames=_FRAMES)
    else:
        tensor = _preprocess_video(video_path, target_frames=24, target_size=64)

    # Quantize input to int8
    started_ns = time.perf_counter_ns()
    if _INPUT_SCALE > 0:
        q_tensor = np.clip(np.round(tensor / _INPUT_SCALE) + _INPUT_ZP, -128, 127).astype(np.int8)
    else:
        q_tensor = tensor.astype(np.float32)
    _record_stage("input_quantization", started_ns)

    started_ns = time.perf_counter_ns()
    _INTERPRETER.set_tensor(_INPUT_DETAILS["index"], q_tensor)
    _INTERPRETER.invoke()
    _record_stage("model_inference", started_ns)

    started_ns = time.perf_counter_ns()
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
    _record_stage("output_postprocessing", started_ns)

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
