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

# Global model state (singleton pattern to avoid reloading per clip).
# The landmark backend may be a single model or an ensemble: a list of
# independently trained members whose logits are averaged before one softmax.
# Landing the ensemble keeps the same public contract; the s3d backend still
# uses exactly one interpreter.
_INTERPRETERS = []          # one TFLite interpreter per landmark member
_MEMBER_IO = []             # per-member (input_index, output_index, io tuple)
_MEMBER_NAMES = []          # per-member basename for profiling/errors
_INTERPRETER = None         # single-model interpreter (s3d, or 1-member fallback)
_INPUT_DETAILS = None       # s3d only: single-model input details
_OUTPUT_DETAILS = None      # s3d only: single-model output details
_INPUT_SCALE = 1.0          # s3d single-model quantization
_INPUT_ZP = 0
_OUTPUT_SCALE = 1.0
_OUTPUT_ZP = 0
_IDX_TO_LABEL = {}
_LAST_PROFILE = None
_ACTIVE_MODEL = None
_FRAMES = 24
_FEATURE_DIM = 255
_MODEL_BACKEND = "s3d"

# Opt-in parallel landmark tracking (default off; see ECHOGESTURE_TRACKER below).
_TRACKER = None
_TRACKER_KEY = None
_TRACKER_LOCK = threading.Lock()
_TRACKER_ATEXIT_REGISTERED = False
_DEFAULT_WORKERS = 4
_TRACKER_MODULE_PATH = (
    Path(__file__).resolve().parent
    / "parallel_tracker.py"
)
# Research-tree fallback: before the module was vendored into src/, it lived
# under evaluation/. Keeping the fallback means an older checkout still runs
# the opt-in path; the shipped location wins when both exist.
_TRACKER_FALLBACK_PATH = (
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


def _landmark_config(model_dir: Path, frames_hint: int, feature_dim_hint: int):
    """Return (members, frames, feature_dim) for the landmark backend.

    Members come from the meta sidecar so adding or dropping one never needs a
    code change.  The single-model artifact is used ONLY when the meta has no
    "members" key at all; a listed member that is missing raises rather than
    being silently dropped.
    """
    meta_path = model_dir / "landmark_model_meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing landmark runtime asset: {meta_path}")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"{meta_path.name} must contain a JSON object")
    frames = int(metadata.get("frames", frames_hint))
    feature_dim = int(metadata.get("feature_dim", feature_dim_hint))

    if "members" in metadata:
        members = metadata["members"]
        if not isinstance(members, list) or not members:
            raise ValueError(
                f"{meta_path.name} 'members' must be a non-empty list")
        for name in members:
            if not isinstance(name, str) or not name:
                raise ValueError(
                    f"{meta_path.name} 'members' entries must be non-empty strings "
                    f"(got {name!r})")
            if name != Path(name).name or "/" in name or "\\" in name or ".." in name:
                raise ValueError(
                    f"Unsafe ensemble member name in {meta_path.name}: {name!r} "
                    "(must be a plain basename)")
        return list(members), frames, feature_dim

    # Explicit fallback: meta predates the ensemble and lists no members.
    single = model_dir / "landmark_classifier_fp16.tflite"
    if not single.is_file():
        raise FileNotFoundError(
            f"No landmark model found in {model_dir}: meta has no 'members' "
            f"and {single.name} is missing")
    return [single.name], frames, feature_dim


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
    """Import ParallelImageTracker, preferring the module shipped in src/.

    The tracker now lives in `src/parallel_tracker.py`, so a release bundle
    carries it and the opt-in fast path works after extraction. Older checkouts
    kept it under evaluation/, which is why the fallback still exists: without
    it, `ECHOGESTURE_TRACKER=parallel` would fail on a bundle that never
    contained the research tree.
    """
    path = None
    for candidate in (_TRACKER_MODULE_PATH, _TRACKER_FALLBACK_PATH):
        if candidate.is_file():
            path = candidate
            break
    if path is None:
        raise RuntimeError(
            "Parallel tracker not available at "
            f"{_TRACKER_MODULE_PATH} or {_TRACKER_FALLBACK_PATH}; "
            "unset ECHOGESTURE_TRACKER or restore the tracker module")
    if "echogesture_parallel_tracker" in sys.modules:
        return sys.modules["echogesture_parallel_tracker"].ParallelImageTracker
    spec = importlib.util.spec_from_file_location(
        "echogesture_parallel_tracker", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load parallel tracker from {path}")
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
    global _ACTIVE_MODEL, _MODEL_BACKEND, _FRAMES, _FEATURE_DIM
    global _INTERPRETERS, _MEMBER_IO, _MEMBER_NAMES

    backend, model_path, label_path = _model_paths()
    if _ACTIVE_MODEL == model_path:
        return

    if Interpreter is None:
        raise RuntimeError(
            "No TFLite interpreter found. Please install ai-edge-litert: "
            "pip install ai-edge-litert"
        )
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
    labels = {int(v): k.upper() for k, v in labels_map.items()}
    if sorted(labels) != list(range(len(labels))):
        raise ValueError("Vocabulary indices must be contiguous from zero")

    # Build everything locally, then commit to the module globals only once
    # the whole configuration validated. A partial load must never leave the
    # process with half an ensemble.
    new_interpreters = []
    new_member_io = []
    new_member_names = []
    new_single = None
    new_input_details = None
    new_output_details = None
    input_scale = input_zp = output_scale = output_zp = None
    frames = _FRAMES
    feature_dim = _FEATURE_DIM

    if backend == "landmark":
        from features import FEATURE_DIM
        meta_path = model_path.parent / "landmark_model_meta.json"
        names, frames, feature_dim = _landmark_config(
            meta_path.parent, _FRAMES, int(FEATURE_DIM))
        if feature_dim != int(FEATURE_DIM):
            raise ValueError(
                f"Feature implementation does not match landmark metadata: "
                f"meta feature_dim={feature_dim}, implementation={int(FEATURE_DIM)}")

        expected_shape = [1, frames, feature_dim]
        expected_out = [1, len(labels)]
        for name in names:
            member_path = model_path.parent / name
            if not member_path.is_file():
                raise FileNotFoundError(
                    f"Ensemble member listed in landmark_model_meta.json is missing: {name}")
            interp = Interpreter(model_path=str(member_path))
            interp.allocate_tensors()
            in_details = interp.get_input_details()[0]
            out_details = interp.get_output_details()[0]
            if list(in_details["shape"]) != expected_shape:
                raise ValueError(
                    f"Ensemble member {name!r} input shape "
                    f"{list(in_details['shape'])} != {expected_shape}")
            if list(out_details["shape"]) != expected_out:
                raise ValueError(
                    f"Ensemble member {name!r} output shape "
                    f"{list(out_details['shape'])} != {expected_out}")
            in_scale, in_zp = in_details["quantization"]
            out_scale, out_zp = out_details["quantization"]
            for label, details in (("input", in_details), ("output", out_details)):
                if details["dtype"] not in (np.float32, np.int8):
                    raise ValueError(
                        f"Ensemble member {name!r} {label} dtype "
                        f"{details['dtype'].__name__} is not supported "
                        "(expected float32 or int8)")
            if in_details["dtype"] == np.int8 and in_scale == 0:
                raise ValueError(
                    f"Ensemble member {name!r} has int8 input with zero scale")
            new_interpreters.append(interp)
            new_member_names.append(name)
            # Tensor indices are per-interpreter and differ between
            # architectures; resolve and keep each member's own handles.
            new_member_io.append((
                in_details["index"], out_details["index"],
                in_scale, in_zp, out_scale, out_zp,
            ))
            if new_single is None:
                new_single = interp
                new_input_details = in_details
                new_output_details = out_details
                input_scale, input_zp = in_scale, in_zp
                output_scale, output_zp = out_scale, out_zp
        if len(new_interpreters) != len(names):
            raise ValueError("Ensemble member set was not fully loaded")
    else:
        if not model_path.exists():
            raise FileNotFoundError(f"Selected model file not found: {model_path}")
        new_single = Interpreter(model_path=str(model_path))
        new_single.allocate_tensors()
        new_input_details = new_single.get_input_details()[0]
        new_output_details = new_single.get_output_details()[0]
        if new_output_details["shape"][-1] != len(labels):
            raise ValueError("Model output classes do not match the selected vocabulary")
        input_scale, input_zp = new_input_details["quantization"]
        output_scale, output_zp = new_output_details["quantization"]

    # Commit.
    _INTERPRETERS = new_interpreters
    _MEMBER_IO = new_member_io
    _MEMBER_NAMES = new_member_names
    _INTERPRETER = new_single
    _INPUT_DETAILS = new_input_details
    _OUTPUT_DETAILS = new_output_details
    _INPUT_SCALE = float(input_scale if input_scale is not None else 0.0)
    _INPUT_ZP = int(input_zp or 0)
    _OUTPUT_SCALE = float(output_scale if output_scale is not None else 0.0)
    _OUTPUT_ZP = int(output_zp or 0)
    _IDX_TO_LABEL = labels
    _MODEL_BACKEND = backend
    _FRAMES = frames
    _FEATURE_DIM = feature_dim
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


def _run_ensemble(tensor):
    """Invoke every member on `tensor` and average their logits.

    Returns (averaged_logits, {member_name: milliseconds}). Each member's own
    input/output indices and quantization are used; averaging happens on
    logits, before any softmax.
    """
    member_logits = []
    member_ms = {}
    for name, interp, io in zip(_MEMBER_NAMES, _INTERPRETERS, _MEMBER_IO):
        in_idx, out_idx, in_scale, in_zp, out_scale, out_zp = io
        if in_scale > 0:
            q_tensor = np.clip(
                np.round(tensor / in_scale) + in_zp, -128, 127).astype(np.int8)
        else:
            q_tensor = tensor.astype(np.float32)
        member_start_ns = time.perf_counter_ns()
        interp.set_tensor(in_idx, q_tensor)
        interp.invoke()
        raw = interp.get_tensor(out_idx)[0].astype(np.float32)
        member_ms[name] = (time.perf_counter_ns() - member_start_ns) / 1e6
        member_logits.append((raw - out_zp) * out_scale if out_scale > 0 else raw)
    return np.mean(member_logits, axis=0), member_ms


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

    started_ns = time.perf_counter_ns()
    if _MEMBER_IO:
        # Landmark ensemble: invoke every member, average logits, softmax once.
        logits, member_ms = _run_ensemble(tensor)
        _record_stage("model_inference", started_ns,
                      members=len(_MEMBER_IO), member_ms=member_ms)
    else:
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
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        # The organizer contract (deck p.11) is a single function:
        # predict(video_path) -> dict. There is no subcommand surface, so this
        # prints the contract rather than pretending to be a CLI parser.
        print(
            "EchoGesture — on-device isolated sign recognition (100 glosses)\n"
            "\n"
            "Usage:\n"
            "  python -m src.main <path_to_video.mp4> [clip_id]\n"
            "\n"
            "Prints the organizer prediction object:\n"
            '  {"clip_id", "prediction", "top5", "confidence", "latency_ms"}\n'
            "\n"
            "Requires model weights under models/ (see models/README.md). They are\n"
            "not redistributable, so a source-only checkout raises FileNotFoundError\n"
            "naming the missing member rather than silently running fewer model.\n"
            "\n"
            "Environment:\n"
            "  ECHOGESTURE_TRACKER=parallel   opt-in multi-worker hand tracking\n"
            "  ECHOGESTURE_WORKERS=<n>        worker count (default 4)"
        )
        sys.exit(0 if len(sys.argv) > 1 else 1)

    video_arg = sys.argv[1]
    clip_id = sys.argv[2] if len(sys.argv) > 2 else None
    res = predict(video_arg, clip_id=clip_id)
    print(json.dumps(res, indent=2))
