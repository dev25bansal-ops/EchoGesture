"""Offline tap-to-capture demo using the repository's real predictor.

SPACE: capture a sign; R: reopen camera / reset after an error; Q/ESC: quit.
No footage is kept on disk after each capture and nothing is sent to a
network: each camera capture is encoded to one temporary clip inside a
worker-owned private temporary directory, predicted once, then deleted
before the result is surfaced. Frames are held in memory only during
inference.

The active model is read from ECHOGESTURE_MODEL (default: landmark candidate).
This integration is not a claim of meeting accuracy or latency targets.

Camera note (macOS): ``cv2.VideoCapture(0).isOpened()`` can be True while every
``read()`` fails when Terminal/Python lacks camera permission, so opening is
probed with real reads and failures report macOS TCC guidance. A live camera
preview is only claimed after that probe passes on the demo machine.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import queue
import sys
import tempfile
import threading
import time
from collections import deque
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]

PROVISIONAL_THRESHOLD = 0.25

MACOS_CAMERA_GUIDANCE = (
    "macOS camera permission is per-app: System Settings > Privacy & Security > "
    "Camera > enable the terminal/application that launched Python, then fully "
    "quit and relaunch it. If it is already enabled, reset the prompt with "
    "`tccutil reset Camera` (re-grants only on the next launch) or try another "
    "--camera index. Fallback without a camera: --video <clip.mp4>."
)


class CameraUnavailableError(RuntimeError):
    """Raised when the camera cannot deliver frames (index, busy, or TCC)."""


def load_predictor(path):
    spec = importlib.util.spec_from_file_location("echogesture_demo_predictor", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load predictor: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.predict


def camera_failure_message(camera_index, detail=""):
    suffix = f" ({detail})" if detail else ""
    return (
        f"Camera {camera_index} unavailable{suffix}. {MACOS_CAMERA_GUIDANCE} "
        "Pass --camera <index> or --video <clip.mp4> to continue without a camera."
    )


def probe_camera(capture, attempts=5):
    """Read until one real frame arrives; return total frames read.

    ``isOpened()`` alone is not evidence the camera works under macOS TCC.
    """
    frames_read = 0
    for _ in range(attempts):
        ok, frame = capture.read()
        if ok and frame is not None:
            return 1
        frames_read += 1
    return 0


def open_camera(camera_index, probe_attempts=5):
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        capture.release()
        raise CameraUnavailableError(camera_failure_message(camera_index, "not opened"))
    if probe_camera(capture, probe_attempts) == 0:
        capture.release()
        raise CameraUnavailableError(
            camera_failure_message(camera_index, "open succeeded but no frames arrived")
        )
    return capture


def model_identity(root=ROOT):
    """Describe the selected model without uploading or copying any artifact.

    Mirrors the backend selection in ``src/main.py`` (read-only) so the HUD
    names what will actually run. The landmark backend is a 4-member FP16
    ensemble listed in ``landmark_model_meta.json``; this reports the ordered
    members and a combined identity so the HUD never names a single classifier
    that is no longer loaded. SHA-256 is computed once per launch.

    Display-only: this does not load or validate tensors. The predictor
    (``src/main.py``) is the authority on what actually executes.
    """
    backend = os.environ.get("ECHOGESTURE_MODEL", "landmark")
    if backend not in ("s3d", "landmark"):
        backend = "landmark"

    def _describe(path):
        try:
            data = path.read_bytes()
        except OSError:
            return {"name": path.name, "bytes": None, "sha256_prefix": None}
        return {
            "name": path.name,
            "bytes": len(data),
            "sha256_prefix": hashlib.sha256(data).hexdigest()[:12],
        }

    label_path = root / ("models/landmark_labels.json" if backend == "landmark"
                         else "models/label100.json")

    if backend == "landmark":
        member_names = _landmark_member_names(root)
        members = [_describe(root / "models" / name) for name in member_names]
        combined = _describe_combined(root, member_names)
        return {
            "backend": backend,
            "model": combined,
            "members": members,
            "labels": _describe(label_path),
            "label_count": _label_count(label_path),
            "threshold": PROVISIONAL_THRESHOLD,
            "threshold_status": "provisional, not validation-calibrated",
        }

    model_path = root / "models/s3d_conv1d_qat_int8.tflite"
    if not model_path.exists():
        model_path = root / "models/ptq_100_int8.tflite"
    return {
        "backend": backend,
        "model": _describe(model_path),
        "labels": _describe(label_path),
        "label_count": _label_count(label_path),
        "threshold": PROVISIONAL_THRESHOLD,
        "threshold_status": "provisional, not validation-calibrated",
    }


def _landmark_member_names(root):
    """Ordered member basenames from the meta sidecar (single-model fallback)."""
    try:
        meta = json.loads((root / "models/landmark_model_meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        meta = None
    if isinstance(meta, dict) and isinstance(meta.get("members"), list):
        names = [m for m in meta["members"] if isinstance(m, str) and m]
        if names:
            return names
    return ["landmark_classifier_fp16.tflite"]


def _describe_combined(root, member_names):
    """One identity for the whole ensemble, hashed over members in meta order."""
    digest = hashlib.sha256()
    total = 0
    readable = 0
    try:
        for name in member_names:
            data = (root / "models" / name).read_bytes()
            digest.update(data)
            total += len(data)
            readable += 1
    except OSError:
        pass
    complete = readable == len(member_names)
    return {
        "name": f"{len(member_names)}-model ensemble" + ("" if complete else " (incomplete)"),
        "members": list(member_names),
        "bytes": total if complete else None,
        "sha256_prefix": digest.hexdigest()[:12] if complete else None,
    }


def _label_count(labels_path):
    try:
        parsed = json.loads(Path(labels_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return len(parsed) if isinstance(parsed, dict) and parsed else None


def model_label_from_identity(identity):
    model = identity.get("model") or {}
    sha = model.get("sha256_prefix") or "missing"
    size = model.get("bytes")
    size_text = f" {size} B" if size is not None else ""
    return f"{identity.get('backend')}:{model.get('name')} sha {sha}{size_text}"


def sanitize_result(result, clip_id=None):
    """Validate/normalize one predictor result; raise ValueError if unusable."""
    if not isinstance(result, dict):
        raise ValueError(f"predictor returned {type(result).__name__}, expected dict")
    prediction = result.get("prediction")
    if not isinstance(prediction, str) or not prediction.strip():
        raise ValueError("predictor result missing non-empty 'prediction'")
    try:
        confidence = float(result.get("confidence"))
    except (TypeError, ValueError) as exc:
        raise ValueError("predictor result missing numeric 'confidence'") from exc
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError(f"predictor confidence out of range: {confidence!r}")
    try:
        latency_ms = float(result.get("latency_ms", 0.0))
    except (TypeError, ValueError):
        latency_ms = 0.0
    if not math.isfinite(latency_ms) or latency_ms < 0:
        latency_ms = 0.0
    top5 = result.get("top5") or [prediction]
    if isinstance(top5, str):
        top5 = [top5]
    top5 = [str(item) for item in list(top5)[:5]] or [prediction]
    cleaned = {
        "clip_id": str(result.get("clip_id") or clip_id or "demo"),
        "prediction": prediction.strip(),
        "top5": top5,
        "confidence": round(confidence, 4),
        "latency_ms": round(latency_ms, 2),
    }
    for key in ("capture_to_result_ms", "queue_wait_ms", "encode_ms"):
        if key in result:
            try:
                value = float(result[key])
            except (TypeError, ValueError):
                continue
            if math.isfinite(value) and value >= 0:
                cleaned[key] = round(value, 2)
    if "capture_to_result_ms" not in cleaned:
        cleaned["capture_to_result_ms"] = cleaned["latency_ms"]
    return cleaned


def write_temp_clip(frames, fps, tmpdir, clip_id):
    """Encode captured frames to one temporary clip; caller deletes it."""
    if not frames:
        raise ValueError("no captured frames")
    first = frames[0]
    if first is None or len(getattr(first, "shape", ())) < 2:
        raise ValueError("captured frame has no image data")
    if not math.isfinite(fps) or not 1 <= fps <= 240:
        raise ValueError(f"invalid capture fps: {fps!r}")
    height, width = first.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("captured frame has zero area")
    safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(clip_id))[:40] or "demo"
    Path(tmpdir).chmod(0o700) if Path(tmpdir).exists() else None
    _fd, _name = tempfile.mkstemp(prefix=f"echogesture-capture-{safe_id}-", suffix=".mp4", dir=str(tmpdir))
    os.close(_fd)
    path = Path(_name)
    try:
        path.chmod(0o600)
    except OSError:
        pass
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        raise RuntimeError("Temporary clip encoder failed")
    try:
        for frame in frames:
            if frame is None or frame.shape[:2] != (height, width):
                raise ValueError("captured frames have inconsistent shape")
            writer.write(frame)
    finally:
        writer.release()
    try:
        path.chmod(0o600)
    except OSError:
        pass
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError("Temporary clip encoder produced no data")
    return path


class PredictionWorker:
    """Run real predictions off the UI thread; delete every temp clip."""

    def __init__(self, predict, tmpdir=None):
        self.predict = predict
        self.requests = queue.Queue(maxsize=1)
        self.results = queue.Queue()
        self.closed = threading.Event()
        self._owns_tmpdir = True
        # A caller-supplied tmpdir is a parent hint, never the directory
        # clips land in directly: captures stay in a fresh private child.
        self._tmpdir = tempfile.TemporaryDirectory(
            prefix="echogesture-capture-", dir=str(tmpdir) if tmpdir is not None else None
        )
        self.tmpdir = Path(self._tmpdir.name)
        try:
            self.tmpdir.chmod(0o700)
        except OSError:
            pass
        self._close_timeout = 5.0
        self._close_lock = threading.Lock()
        self._closed_once = False
        self.thread = threading.Thread(target=self._run, name="offline-inference", daemon=True)
        self.thread.start()

    @property
    def tmpdir_path(self):
        return self.tmpdir

    def submit(self, frames, fps, clip_id):
        if self.closed.is_set():
            return False
        if not frames:
            return False
        try:
            self.requests.put_nowait((list(frames), fps, clip_id, time.perf_counter_ns()))
            return True
        except queue.Full:
            return False

    def _run(self):
        while not self.closed.is_set():
            try:
                frames, fps, clip_id, queued_ns = self.requests.get(timeout=0.1)
            except queue.Empty:
                continue
            dequeued_ns = time.perf_counter_ns()
            path = None
            try:
                encode_started = time.perf_counter_ns()
                path = write_temp_clip(frames, fps, self.tmpdir, clip_id)
                encode_ms = (time.perf_counter_ns() - encode_started) / 1e6
                raw = self.predict(str(path), clip_id=clip_id)
                result = sanitize_result(raw, clip_id=clip_id)
                result["queue_wait_ms"] = (dequeued_ns - queued_ns) / 1e6
                result["encode_ms"] = encode_ms
                result["capture_to_result_ms"] = (time.perf_counter_ns() - queued_ns) / 1e6
                self.results.put(("result", result))
            except Exception as exc:
                self.results.put(("error", f"{type(exc).__name__}: {exc}"))
            finally:
                if path is not None:
                    try:
                        path.unlink(missing_ok=True)
                    except OSError:
                        pass
                try:
                    for _f in frames:
                        try:
                            _f.fill(0)
                        except Exception:
                            pass
                    frames.clear()
                except Exception:
                    pass

    def close(self):
        with self._close_lock:
            if self._closed_once:
                return
            self._closed_once = True
        self.closed.set()
        _deadline = time.perf_counter() + getattr(self, "_close_timeout", 5.0)
        while time.perf_counter() < _deadline:
            if not self.thread.is_alive():
                break
            self.thread.join(timeout=0.05)
        self.drained = not self.thread.is_alive()
        try:
            for _leftover in list(self.tmpdir.iterdir()):
                try:
                    _leftover.unlink(missing_ok=True)
                except OSError:
                    pass
        except OSError:
            pass
        if self._tmpdir is not None:
            try:
                self._tmpdir.cleanup()
            except OSError:
                pass
        try:
            for _leftover in list(self.tmpdir.iterdir()):
                try:
                    _leftover.unlink(missing_ok=True)
                except OSError:
                    pass
        except OSError:
            pass


class CaptionState:
    """Thread-safe HUD state; render() reads a snapshot, never torn fields."""

    def __init__(self, threshold=PROVISIONAL_THRESHOLD):
        if not isinstance(threshold, (int, float)) or isinstance(threshold, bool) or not 0 <= threshold <= 1:
            raise ValueError("threshold must be in [0, 1]")
        self.threshold = float(threshold)
        self._lock = threading.Lock()
        self._state = "WAITING"
        self._message = "Press SPACE to capture an isolated sign"
        self._result = None

    @property
    def state(self):
        with self._lock:
            return self._state

    @property
    def message(self):
        with self._lock:
            return self._message

    @property
    def result(self):
        with self._lock:
            return dict(self._result) if self._result is not None else None

    def snapshot(self):
        with self._lock:
            return {
                "state": self._state,
                "message": self._message,
                "result": dict(self._result) if self._result is not None else None,
                "threshold": self.threshold,
            }

    def processing(self):
        with self._lock:
            self._state = "PROCESSING"
            self._message = "Running local model..."
            self._result = None

    def reset(self, message=None):
        with self._lock:
            self._state = "WAITING"
            self._message = message or "Press SPACE to capture an isolated sign"
            self._result = None

    def update(self, result):
        try:
            cleaned = sanitize_result(result)
        except ValueError as exc:
            return self.error(f"Invalid model result: {exc}")
        with self._lock:
            if cleaned["confidence"] < self.threshold:
                self._state = "RETRY"
                self._message = "Low confidence - try again"
            else:
                self._state = "CAPTION"
                self._message = cleaned["prediction"]
            self._result = cleaned
            return self._state

    def error(self, message):
        with self._lock:
            self._state = "ERROR"
            self._message = str(message)[:120] or "Error"
            self._result = None
            return self._state


def render(frame, state, model_label, threshold, buffer_count=0, buffer_target=0):
    image = frame.copy()
    snap = state.snapshot() if hasattr(state, "snapshot") else {
        "state": getattr(state, "state", "WAITING"),
        "message": getattr(state, "message", ""),
        "result": getattr(state, "result", None),
    }
    state_name = snap.get("state", "WAITING")
    colors = {"WAITING": (200, 200, 200), "PROCESSING": (50, 200, 255),
              "RETRY": (40, 150, 255), "CAPTION": (100, 230, 100), "ERROR": (50, 50, 230)}
    color = colors.get(state_name, (255, 255, 255))
    cv2.rectangle(image, (0, 0), (image.shape[1], 135), (15, 20, 25), -1)
    cv2.putText(image, "EchoGesture | OFFLINE | integration not release-certified", (18, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, str(snap.get("message") or "")[:65], (18, 75), cv2.FONT_HERSHEY_SIMPLEX,
                1.1 if state_name == "CAPTION" else 0.75, color, 2, cv2.LINE_AA)
    detail = f"{state_name} | {model_label} | provisional retry threshold {threshold:.2f}"
    if len(detail) > 86:
        detail = detail[:85] + "..."
    cv2.putText(image, detail, (18, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)
    result = snap.get("result")
    if result:
        confidence = float(result.get("confidence", 0.0))
        latency = float(result.get("latency_ms", 0.0))
        capture_ms = float(result.get("capture_to_result_ms", latency))
        text = f"confidence {confidence:.3f} | clip {latency:.1f} ms | capture-result {capture_ms:.1f} ms"
        if result.get("queue_wait_ms") is not None:
            text += f" | queued {float(result['queue_wait_ms']):.1f} ms"
        cv2.putText(image, text, (18, image.shape[0] - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image, "Top 5: " + ", ".join(result.get("top5", []))[:70], (18, image.shape[0] - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    elif buffer_target:
        cv2.putText(image, f"Capturing {buffer_count}/{buffer_target} frames | SPACE to cancel",
                    (18, image.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
    else:
        cv2.putText(image, "SPACE capture | R reopen/reset | Q quit | no footage kept on disk after each capture", (18, image.shape[0] - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def render_demo_frame(frame, result, model="landmark"):
    state = CaptionState()
    result = dict(result)
    result.setdefault("capture_to_result_ms", result.get("latency_ms", 0.0))
    state.update(result)
    return render(frame, state, model, PROVISIONAL_THRESHOLD)


def sample_file_frames(video_path, max_frames=32):
    """Read frames spread across a user-supplied clip (no copy is retained)."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frames = []
        if total > max_frames:
            targets = sorted(set(int(i * (total - 1) / (max_frames - 1)) for i in range(max_frames)))
            current = 0
            for target in targets:
                while current < target:
                    if not capture.grab():
                        break
                    current += 1
                ok, frame = capture.read()
                if not ok:
                    break
                current += 1
                frames.append(frame)
        else:
            while len(frames) < max_frames:
                ok, frame = capture.read()
                if not ok:
                    break
                frames.append(frame)
        return frames
    finally:
        capture.release()


def run_file(video_path, predictor_path, threshold, identity=None):
    """Predict one real file and return (state, result, model_identity).

    The input file is read in place; nothing is copied and nothing is
    retained. Used by the camera-less smoke path (``--video`` / demo lab).
    """
    identity = identity or model_identity()
    model_label = model_label_from_identity(identity)
    predict = load_predictor(predictor_path)
    frames = sample_file_frames(video_path)
    if not frames:
        raise ValueError(f"No valid frames decoded from: {video_path}")
    clip_id = Path(video_path).stem
    started = time.perf_counter_ns()
    result = sanitize_result(predict(str(video_path), clip_id=clip_id), clip_id=clip_id)
    result.setdefault("queue_wait_ms", 0.0)
    result["capture_to_result_ms"] = (time.perf_counter_ns() - started) / 1e6
    state = CaptionState(threshold)
    state.update(result)
    return state, result, identity


def render_file_result(video_path, predictor_path, threshold, render_output=None):
    state, result, identity = run_file(video_path, predictor_path, threshold)
    frames = sample_file_frames(video_path)
    middle = frames[len(frames) // 2] if frames else None
    if middle is None:
        raise ValueError(f"No valid frames decoded from: {video_path}")
    image = render(middle, state, model_label_from_identity(identity), threshold)
    if render_output:
        output = Path(render_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output), image)
    return image, result, identity


def run(camera_index, seconds, predictor_path, threshold, headless=False, max_frames=0):
    capture = open_camera(camera_index)
    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and 1 <= fps <= 240 else 30.0
    target = max(1, int(seconds * fps))
    identity = model_identity()
    model_label = model_label_from_identity(identity)
    state = CaptionState(threshold)
    worker = PredictionWorker(load_predictor(predictor_path))
    captured = deque(maxlen=target)
    capturing = False
    frame_count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                state.error("Camera read failed - press R to reopen the camera, or Q to quit")
                if headless:
                    break
                cv2.waitKey(500)
                continue
            frame_count += 1
            if capturing:
                captured.append(frame)
                if len(captured) >= target:
                    capturing = False
                    if worker.submit(list(captured), fps, f"demo-{time.time_ns()}"):
                        state.processing()
                    else:
                        state.error("Inference busy - try again")
                    captured.clear()
            try:
                kind, value = worker.results.get_nowait()
                if kind == "result":
                    state.update(value)
                else:
                    state.error(value)
            except queue.Empty:
                pass
            if headless and (state.result or state.state == "ERROR" or (max_frames and frame_count >= max_frames)):
                return state.state, state.result
            if not headless:
                cv2.imshow("EchoGesture - Offline", render(frame, state, model_label, threshold,
                                                          len(captured) if capturing else 0, target if capturing else 0))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord(" "):
                    if capturing:
                        capturing = False
                        captured.clear()
                        state = CaptionState(threshold)
                    elif state.state != "PROCESSING":
                        capturing = True
                        captured.clear()
                if key in (ord("r"), ord("R")):
                    state.reset("Reopening camera...")
                    try:
                        capture.release()
                        capture = open_camera(camera_index)
                        state.reset()
                    except CameraUnavailableError as exc:
                        state.error(str(exc))
            elif frame_count == 1:
                capturing = True
    finally:
        capture.release()
        worker.close()
        cv2.destroyAllWindows()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--threshold", type=float, default=PROVISIONAL_THRESHOLD,
                        help="Provisional UI threshold; not validation-calibrated")
    parser.add_argument("--predictor", type=Path, default=ROOT / "src/main.py")
    parser.add_argument("--video", type=Path, default=None,
                        help="Camera-less fallback: run the real predictor on one clip file and render the HUD without a window")
    parser.add_argument("--render-output", type=Path, default=None,
                        help="With --video, write the rendered HUD frame here")
    parser.add_argument("--headless", action="store_true", help="Exercise one camera capture without a window")
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args(argv)
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be in [0, 1]")
    try:
        if args.video:
            image, result, identity = render_file_result(args.video, args.predictor, args.threshold, args.render_output)
            print(json.dumps({"state": CaptionState(args.threshold).update(result), "result": result, "model": identity, "rendered_frames": int(image.shape[0])}, indent=2))
            return
        result = run(args.camera, args.seconds, args.predictor, args.threshold, args.headless, args.max_frames)
    except CameraUnavailableError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    if result is not None:
        print(result)


if __name__ == "__main__":
    main()
