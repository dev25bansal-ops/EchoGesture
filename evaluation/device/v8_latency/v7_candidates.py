"""V7 latency lab: does anything actually beat IMAGE singleton without changing predictions?

Owned files: evaluation/device/v8_latency/ only (isolated copy of v7_latency
with worker shutdown/lifecycle fixes). Dev modules loaded read-only
from src/. No production code is modified.

Strategies
----------
baseline            IMAGE singleton (Dev Landmarker), 256px, 32 frames.
                    Production path. Current default.
parallel_2          IMAGE mode split across 2 persistent trackers in threads.
                    IMAGE mode is stateless per frame, so interleaving frames
                    across independent landmarker instances is exact-parity by
                    construction. Trackers are built once at pool creation
                    (construction cost amortized), frames distributed round-robin.
parallel_4          Same with 4 workers.
video_cached        Singleton VIDEO, monotonic timestamps across clips.
                    Reproduced here as a control — prior agents measured
                    48/60 top-1 parity. Not a recommendation.
video_fresh         VIDEO fresh per clip, timestamps restart at 0.
                    Reproduced as control.

All strategies: same 32-frame sampling, 256x256, thresholds 0.3/0.3/0.3,
num_hands=2, same build_features, same FP16 classifier.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import atexit
import threading

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

MAX_FRAMES = 32
FRAME_SIZE = 256
MODEL_FRAMES = 24
MODEL_WIDTH = 255
STRATEGIES = ("baseline", "parallel_2", "parallel_4", "video_cached", "video_fresh")

_INTERNALS: dict = {}
# Guards _initialise/_singleton/_get_parallel_pool/shutdown against concurrent
# predict() calls and makes lazy pool creation safe (v8 lifecycle fix).
_LOCK = threading.RLock()
_ATEXIT_REGISTERED = False
_ATEXIT_LOCK = threading.Lock()


def _load(name: str, path: Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _initialise():
    if _INTERNALS.get("state") is not None:
        return
    with _LOCK:
        if _INTERNALS.get("state") is not None:
            return
        src = ROOT / "src"
        frontend = _load("v7_landmark_frontend", src / "landmark_frontend.py")
        features = _load("v7_features", src / "features.py")
        main = _load("v7_main", src / "main.py")
        _INTERNALS.update(
            root=ROOT, frontend=frontend, features=features, main=main,
            state={"image": None, "video_cached": None, "video_ts": 0,
                   "parallel_2": None, "parallel_4": None},
        )
        _register_atexit()


def _register_atexit():
    global _ATEXIT_REGISTERED
    with _ATEXIT_LOCK:
        if not _ATEXIT_REGISTERED:
            atexit.register(shutdown)
            _ATEXIT_REGISTERED = True


def _close_landmarker(lm) -> None:
    """Close one landmarker, swallowing any close-time error.

    MediaPipe close() can raise if called twice or during interpreter
    shutdown; cleanup must never mask the original error.
    """
    if lm is None:
        return
    try:
        lm.close()
    except Exception:
        pass


def shutdown() -> None:
    """Close all singleton/pool landmarkers. Idempotent and thread-safe.

    After shutdown, the next predict() lazily recreates the singletons and
    pools (soft drain, not permanent poison), so tests may call shutdown()
    for isolation and keep using the module. The classifier interpreter
    (owned by src/main.py) is intentionally left alone.
    """
    if not _INTERNALS:
        return
    with _LOCK:
        st = _INTERNALS.get("state") or {}
        for key in ("image", "video_cached", "parallel_2", "parallel_4"):
            val = st.get(key)
            if isinstance(val, list):
                for lm in val:
                    _close_landmarker(lm)
            else:
                _close_landmarker(val)
            st[key] = None
        st["video_ts"] = 0


def _ensure_classifier():
    _initialise()
    _INTERNALS["main"]._init_model()


def _make_landmarker(mode: str, cpu: bool = False):
    import mediapipe as mp  # noqa: F401
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    task = _INTERNALS["root"] / "models/hand_landmarker.task"
    opts = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=str(task),
            delegate=mp_python.BaseOptions.Delegate.CPU if cpu else None,
        ),
        running_mode=vision.RunningMode.VIDEO if mode == "video" else vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    return vision.HandLandmarker.create_from_options(opts)


def _singleton(kind: str):
    _initialise()
    with _LOCK:
        st = _INTERNALS["state"]
        if st[kind] is None:
            st[kind] = _make_landmarker("video" if kind == "video_cached" else "image")
        return st[kind]


def _detect_image(frames, lm):
    import mediapipe as mp
    import numpy as np
    seq, present = [], []
    for f in frames:
        try:
            res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=f))
        except Exception:
            res = None
        slots = {"Left": np.zeros((21, 3), np.float32), "Right": np.zeros((21, 3), np.float32)}
        seen = {"Left": 0.0, "Right": 0.0}
        if res is not None and res.hand_landmarks:
            for hl, hd in zip(res.hand_landmarks, res.handedness):
                cat = hd[0].category_name if hd else "Right"
                if cat not in slots:
                    continue
                slots[cat] = np.array([[p.x, p.y, p.z] for p in hl], np.float32)
                seen[cat] = 1.0
        seq.append(np.stack([slots["Left"], slots["Right"]]))
        present.append([seen["Left"], seen["Right"]])
    import numpy as np
    if not seq:
        return (np.zeros((0, 2, 21, 3), np.float32), np.zeros((0, 2), np.float32))
    return (np.stack(seq), np.array(present, np.float32))


def _detect_video(frames, lm, stamp_from):
    import mediapipe as mp
    import numpy as np
    seq, present = [], []
    ts = stamp_from
    for f in frames:
        ts += 33
        try:
            res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=f), ts)
        except Exception:
            res = None
        slots = {"Left": np.zeros((21, 3), np.float32), "Right": np.zeros((21, 3), np.float32)}
        seen = {"Left": 0.0, "Right": 0.0}
        if res is not None and res.hand_landmarks:
            for hl, hd in zip(res.hand_landmarks, res.handedness):
                cat = hd[0].category_name if hd else "Right"
                if cat not in slots:
                    continue
                slots[cat] = np.array([[p.x, p.y, p.z] for p in hl], np.float32)
                seen[cat] = 1.0
        seq.append(np.stack([slots["Left"], slots["Right"]]))
        present.append([seen["Left"], seen["Right"]])
    if not seq:
        return (np.zeros((0, 2, 21, 3), np.float32), np.zeros((0, 2), np.float32)), ts
    return (np.stack(seq), np.array(present, np.float32)), ts


def _get_parallel_pool(workers: int):
    _initialise()
    with _LOCK:
        st = _INTERNALS["state"]
        key = f"parallel_{workers}"
        if st[key] is None:
            # Build incrementally so a mid-pool failure closes the
            # already-built trackers instead of leaking native handles.
            pool = []
            try:
                for _ in range(workers):
                    pool.append(_make_landmarker("image"))
            except Exception:
                for lm in pool:
                    _close_landmarker(lm)
                raise
            st[key] = pool
        return st[key]


def _detect_parallel(frames, workers: int, meta: dict):
    """Split frames round-robin across persistent IMAGE trackers in threads.

    IMAGE mode is stateless: each detect() call is independent. Splitting
    frames across independent landmarker instances with thread parallelism
    must produce bit-identical landmarks to the singleton baseline.
    """
    import numpy as np
    pool = _get_parallel_pool(workers)
    # Build per-worker frame lists
    worker_frames = [frames[i::workers] for i in range(workers)]
    # Run in thread pool
    results = [None] * workers

    def run(idx):
        results[idx] = _detect_image(worker_frames[idx], pool[idx])

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(run, i) for i in range(workers)]
        for f in futures:
            f.result()

    # Reassemble in original order
    n = len(frames)
    seq = [None] * n
    present = [None] * n
    for worker_idx, (lm_seq, pres) in enumerate(results):
        for j, (l, p) in enumerate(zip(lm_seq, pres)):
            seq[worker_idx + j * workers] = l
            present[worker_idx + j * workers] = p
    meta["parallel_workers"] = workers
    return np.stack(seq), np.array(present, np.float32)


def _detect(strategy: str, frames, meta: dict):
    import time
    started = time.perf_counter_ns()
    if strategy == "baseline":
        result = _detect_image(frames, _singleton("image"))
    elif strategy == "parallel_2":
        result = _detect_parallel(frames, 2, meta)
    elif strategy == "parallel_4":
        result = _detect_parallel(frames, 4, meta)
    elif strategy == "video_fresh":
        lm = _make_landmarker("video")
        try:
            result, _ = _detect_video(frames, lm, 0)
        finally:
            lm.close()
    elif strategy == "video_cached":
        with _LOCK:
            st = _INTERNALS["state"]
            result, st["video_ts"] = _detect_video(frames, _singleton("video_cached"), st["video_ts"])
    else:
        raise ValueError(f"unsupported strategy: {strategy}")
    return result, (time.perf_counter_ns() - started) / 1e6


def evaluate_clip(video_path, strategy: str):
    if strategy not in STRATEGIES:
        raise ValueError(f"unsupported strategy: {strategy}")
    init_started = time.perf_counter_ns()
    _ensure_classifier()
    init_ms = (time.perf_counter_ns() - init_started) / 1e6
    frontend, features, main = _INTERNALS["frontend"], _INTERNALS["features"], _INTERNALS["main"]
    timings: dict = {"runtime_initialization_ms": init_ms}
    meta: dict = {}
    total_started = time.perf_counter_ns()

    stage = time.perf_counter_ns()
    frames = frontend.read_frames(video_path, max_frames=MAX_FRAMES, size=FRAME_SIZE)
    timings["decode_ms"] = (time.perf_counter_ns() - stage) / 1e6
    timings["decoded_frames"] = len(frames)
    if not frames:
        raise ValueError(f"No valid frames decoded from: {video_path}")

    (landmarks, present), tracking_ms = _detect(strategy, frames, meta)
    timings["tracking_ms"] = tracking_ms
    timings["landmarks_shape"] = [int(x) for x in landmarks.shape]
    timings["frames_with_hands"] = int((present.max(axis=1) > 0.5).sum()) if len(present) else 0
    timings["meta"] = {k: (v if not isinstance(v, list) or len(v) < 5 else f"list[{len(v)}]") for k, v in meta.items()}
    if landmarks.shape[0] == 0 or float(present.max()) < 0.5:
        raise ValueError(f"No hand detected in any frame of: {video_path}")
    if landmarks.shape[0] != 32:
        raise ValueError(f"Expected 32 sampled frames, got {landmarks.shape[0]}")

    stage = time.perf_counter_ns()
    tensor, _ = features.build_features(landmarks, present, expected_frames=MODEL_FRAMES)
    tensor = tensor[None, ...].astype("float32")
    timings["features_ms"] = (time.perf_counter_ns() - stage) / 1e6
    if tuple(tensor.shape) != (1, MODEL_FRAMES, MODEL_WIDTH):
        raise ValueError(f"Feature tensor shape changed: {tensor.shape}")

    stage = time.perf_counter_ns()
    main._INTERPRETER.set_tensor(main._INPUT_DETAILS["index"], tensor)
    main._INTERPRETER.invoke()
    timings["classifier_ms"] = (time.perf_counter_ns() - stage) / 1e6

    stage = time.perf_counter_ns()
    raw_output = main._INTERPRETER.get_tensor(main._OUTPUT_DETAILS["index"])[0].astype("float32")
    logits = raw_output
    import numpy as np
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)
    top_indices = np.argsort(-probs)[:5]
    top5 = [main._IDX_TO_LABEL.get(int(idx), f"UNKNOWN_{int(idx)}") for idx in top_indices]
    prediction = top5[0]
    confidence = float(probs[top_indices[0]])
    timings["postprocess_ms"] = (time.perf_counter_ns() - stage) / 1e6
    timings["total_ms"] = (time.perf_counter_ns() - total_started) / 1e6
    result = {
        "clip_id": Path(video_path).stem,
        "prediction": prediction,
        "top5": top5,
        "confidence": round(confidence, 4),
        "latency_ms": round(timings["total_ms"], 2),
    }
    return result, timings


def predict(video_path: str, *, clip_id: str | None = None, profile: bool = False) -> dict:
    import os
    strategy = os.environ.get("V7_STRATEGY", "baseline").lower()
    if strategy not in STRATEGIES:
        raise ValueError("V7_STRATEGY must be one of " + ",".join(STRATEGIES))
    started = time.perf_counter()
    result, timings = evaluate_clip(str(video_path), strategy)
    if clip_id is not None:
        result = dict(result, clip_id=clip_id)
    result = dict(result, latency_ms=round((time.perf_counter() - started) * 1000.0, 2))
    predict.last_timings = timings
    return result


predict.last_timings = {}
