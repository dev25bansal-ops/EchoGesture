"""Bounded, timestamp-aware streaming for isolated-sign landmarks.

A timestamp gap pauses the current sign in SIGN_HOLD and returns
``timestamp_gap``. The caller must flush that earlier window (or reset it),
start a new sign, and submit the post-gap frame again. Frames from disjoint
capture intervals are never silently concatenated.

File mode is sequential VIDEO-mode tracking, not live camera evidence. Timing
fields use only compatible clocks; landmark replay does not claim live latency.
Dev-owned features and the landmarker are imported, never modified.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("ECHOGESTURE_ROOT", _HERE.parents[2]))
_SRC = PROJECT_ROOT / "src"
for _path in (_HERE, _SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from capture_states import (  # noqa: E402
    CAPTURING,
    COMMITTED,
    IDLE,
    SIGN_HOLD,
    STOPPED,
    CaptureSession,
    CaptureStateError,
)
from ring_buffer import PushResult, RingBuffer  # noqa: E402

FRAME_SIZE = 256
DEFAULT_MAX_CAPTURE_GAP_MS = 350.0
DEFAULT_CAPTION_TTL_SECONDS = 3.0


def _now() -> int:
    return time.monotonic_ns()


def _ms(start_ns: int, end_ns: int) -> float:
    return (end_ns - start_ns) / 1e6


def _timestamp_ns(value) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError("timestamp must be an integer number of nanoseconds")
    value = int(value)
    if value < 0:
        raise ValueError("timestamp must be non-negative")
    return value


def _positive_seconds(value, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


class StreamingClassifier:
    """One FP16 LiteRT interpreter, with serialized invocations."""

    def __init__(self, model_path=None, labels_path=None, meta_path=None):
        from ai_edge_litert.interpreter import Interpreter

        self.model_path = Path(model_path) if model_path else PROJECT_ROOT / "models/landmark_classifier_fp16.tflite"
        self.labels_path = Path(labels_path) if labels_path else PROJECT_ROOT / "models/landmark_labels.json"
        self.meta_path = Path(meta_path) if meta_path else PROJECT_ROOT / "models/landmark_model_meta.json"
        labels = json.loads(self.labels_path.read_text(encoding="utf-8"))
        self.idx_to_label = {int(v): k.upper() for k, v in labels.items()}
        self.frames = int(json.loads(self.meta_path.read_text(encoding="utf-8"))["frames"])
        self.model_sha256 = hashlib.sha256(self.model_path.read_bytes()).hexdigest()
        self._interp = Interpreter(model_path=str(self.model_path))
        self._interp.allocate_tensors()
        self._in = self._interp.get_input_details()[0]
        self._out = self._interp.get_output_details()[0]
        self.warmup_ms = None
        self.infer_count = 0
        self._infer_lock = threading.Lock()

    def warmup(self):
        start = _now()
        zeros = np.zeros(tuple(self._in["shape"]), dtype=np.float32)
        with self._infer_lock:
            self._interp.set_tensor(self._in["index"], zeros)
            self._interp.invoke()
            self._interp.get_tensor(self._out["index"])
        self.warmup_ms = _ms(start, _now())
        return self.warmup_ms

    def classify_window(self, landmarks, present):
        """Return label, confidence, top5, interpreter ms, total classify ms."""
        from features import build_features

        start = _now()
        tensor, _mask = build_features(landmarks, present, expected_frames=self.frames)
        tensor = tensor[None, ...].astype(np.float32)
        invoke_start = _now()
        with self._infer_lock:
            self._interp.set_tensor(self._in["index"], tensor)
            self._interp.invoke()
            self.infer_count += 1
            raw = self._interp.get_tensor(self._out["index"])[0].astype(np.float32)
        invoke_end = _now()
        scale, zero_point = self._out["quantization"]
        logits = (raw - zero_point) * scale if scale > 0 else raw
        exp = np.exp(logits - np.max(logits))
        probs = exp / np.sum(exp)
        top = np.argsort(-probs)[:5]
        top5 = [self.idx_to_label.get(int(i), f"UNKNOWN_{i}") for i in top]
        return top5[0], float(probs[top[0]]), top5, _ms(invoke_start, invoke_end), _ms(start, invoke_end)


class StreamingSession:
    """Serialize sign transitions while bounding and auditing frame storage.

    ``timestamp_clock`` is ``monotonic_ns`` for live/file capture using this
    process's monotonic clock, or ``external_ns`` for replayed timestamps.
    Cross-clock caption timings are ``None`` in external mode, never clamped.
    """

    def __init__(
        self,
        capacity: int = 96,
        classifier: StreamingClassifier | None = None,
        max_capture_gap_ms: float = DEFAULT_MAX_CAPTURE_GAP_MS,
        caption_ttl_seconds: float | None = DEFAULT_CAPTION_TTL_SECONDS,
        timestamp_clock: str = "monotonic_ns",
    ):
        if timestamp_clock not in ("monotonic_ns", "external_ns"):
            raise ValueError("timestamp_clock must be monotonic_ns or external_ns")
        self.session = CaptureSession()
        self.buffer = RingBuffer(capacity=capacity)
        self.classifier = classifier or StreamingClassifier()
        self.max_capture_gap_ms = _positive_seconds(max_capture_gap_ms, "max_capture_gap_ms")
        self.caption_ttl_seconds = (
            None if caption_ttl_seconds is None
            else _positive_seconds(caption_ttl_seconds, "caption_ttl_seconds")
        )
        self.timestamp_clock = timestamp_clock
        self._caption = None
        self._caption_expires_at_ns = None
        self.frame_processing_ms = []
        self.caption_events = []
        self.drop_events = []
        self.dropped_out_of_order = 0
        self.dropped_duplicate = 0
        self.dropped_timestamp_gaps = 0
        self.rejected_frames = 0
        self.discarded_stale_results = 0
        self.last_capture_ns = None
        self.last_error = None
        self._lock = threading.RLock()

    @property
    def state(self) -> str:
        return self.session.state

    @property
    def sign_seq(self) -> int:
        return self.session.sign_seq

    def _clear_caption(self) -> None:
        self._caption = None
        self._caption_expires_at_ns = None

    def begin_sign(self) -> int:
        """Begin the next sign and invalidate all previous-sign state."""
        with self._lock:
            if self.state not in (IDLE, COMMITTED):
                raise CaptureStateError(f"cannot begin a new sign from {self.state}")
            self._clear_caption()
            self.buffer.discard()
            self.last_capture_ns = None
            self.last_error = None
            return self.session.begin_sign()

    def current_caption(self) -> dict | None:
        with self._lock:
            if self._caption is None:
                return None
            if self._caption_expires_at_ns is not None and _now() >= self._caption_expires_at_ns:
                self._clear_caption()
            return dict(self._caption) if self._caption is not None else None

    def ingest_landmark_frame(self, frame_index: int, t_capture_ns: int, landmarks, present) -> str:
        """Return ok, ok_dropped_oldest, or a rejected/dropped reason.

        On ``timestamp_gap`` the new frame is not buffered. Re-submit it after
        flushing/resetting the old sign and calling ``begin_sign``.
        """
        capture_ns = _timestamp_ns(t_capture_ns)
        landmarks = np.asarray(landmarks, dtype=np.float32)
        present = np.asarray(present, dtype=np.float32)
        if landmarks.shape != (2, 21, 3) or present.shape != (2,):
            raise ValueError("landmarks must be (2,21,3) and present must be (2,)")
        if not np.isfinite(landmarks).all() or not np.isfinite(present).all():
            raise ValueError("landmarks and present must contain finite values")
        with self._lock:
            if self.state == IDLE:
                self.begin_sign()
            if self.state != CAPTURING:
                self.rejected_frames += 1
                return "rejected_state"
            if self.last_capture_ns is not None:
                delta_ns = capture_ns - self.last_capture_ns
                if delta_ns < 0:
                    self.dropped_out_of_order += 1
                    return "dropped_stale"
                if delta_ns == 0:
                    self.dropped_duplicate += 1
                    return "dropped_duplicate"
                if delta_ns > self.max_capture_gap_ms * 1e6:
                    self.session.transition(SIGN_HOLD)
                    self._clear_caption()
                    self.dropped_timestamp_gaps += 1
                    return "timestamp_gap"
            self.last_capture_ns = capture_ns
            result = self.buffer.push({
                "frame_index": int(frame_index),
                "t_capture_ns": capture_ns,
                "landmarks": landmarks,
                "present": present,
            })
        self._record_overflow(result)
        return "ok_dropped_oldest" if result.dropped else "ok"

    def _record_overflow(self, result: PushResult) -> None:
        if not result.dropped:
            return
        # Bounded diagnostic history; counters remain complete for long runs.
        if len(self.drop_events) < 100:
            self.drop_events.append({
                "reason": "capacity_overflow",
                "dropped_frame_index": result.dropped_frame_index,
            })

    def flush_sign(self, t_sign_end_ns: int | None = None) -> dict | None:
        """Classify the current window, excluding every subsequent sign.

        Returns ``None`` if reset/stop retired the sign during inference.
        Failures clear the window and caption, record the error, and return
        the session to ``IDLE`` for an explicit retry; they raise and never
        increment ``sign_seq``.
        """
        with self._lock:
            if self.state not in (CAPTURING, SIGN_HOLD):
                raise CaptureStateError(f"cannot flush from {self.state}")
            sign_id = self.sign_seq
            if self.state == CAPTURING:
                self.session.transition(SIGN_HOLD)
            items = self.buffer.take_all()
            if not items:
                self._fail_locked(sign_id, "flush requested with an empty sign window")
            if t_sign_end_ns is None:
                if self.timestamp_clock != "monotonic_ns":
                    self._fail_locked(sign_id, "external timestamps require an explicit sign-end timestamp")
                sign_end_ns = _now()
            else:
                sign_end_ns = _timestamp_ns(t_sign_end_ns)
            if sign_end_ns < items[-1]["t_capture_ns"]:
                self._fail_locked(sign_id, "sign-end timestamp precedes the last accepted frame")
            first_capture_ns = items[0]["t_capture_ns"]
            window_duration_ms = _ms(first_capture_ns, items[-1]["t_capture_ns"])
            landmarks = np.stack([item["landmarks"] for item in items])
            present = np.stack([item["present"] for item in items])
            if float(present.max()) < 0.5:
                self._fail_locked(sign_id, "no hand detected in the sign window")
            self._clear_caption()
            flush_started_ns = _now()

        try:
            label, confidence, top5, invoke_ms, classifier_ms = self.classifier.classify_window(
                landmarks, present
            )
        except Exception as exc:
            with self._lock:
                if self.state == SIGN_HOLD and self.sign_seq == sign_id:
                    self._fail_locked(sign_id, f"classifier failed: {exc}")
            raise

        emit_ns = _now()
        with self._lock:
            if self.state != SIGN_HOLD or self.sign_seq != sign_id:
                self.discarded_stale_results += 1
                return None
            self.session.transition(COMMITTED)
            compatible_clock = self.timestamp_clock == "monotonic_ns"
            caption = {
                "prediction": label,
                "confidence": confidence,
                "top5": top5,
                "sign_seq": sign_id,
                "window_frames": len(items),
                "window_duration_ms": window_duration_ms,
                "timestamp_clock": self.timestamp_clock,
                "t_emit_monotonic_ns": emit_ns,
                "first_frame_to_caption_ms": _ms(first_capture_ns, emit_ns) if compatible_clock else None,
                "sign_end_to_caption_ms": _ms(sign_end_ns, emit_ns) if compatible_clock else None,
                "flush_processing_ms": _ms(flush_started_ns, emit_ns),
                "classifier_invoke_ms": invoke_ms,
                "classifier_total_ms": classifier_ms,
            }
            self._caption = caption
            self._caption_expires_at_ns = (
                None if self.caption_ttl_seconds is None
                else emit_ns + int(self.caption_ttl_seconds * 1e9)
            )
            self.caption_events.append(dict(caption))
            return dict(caption)

    def _fail_locked(self, sign_id: int, message: str) -> None:
        if self.state == SIGN_HOLD and self.sign_seq == sign_id:
            self.session.transition("ERROR")
        self.session.reset()
        self.buffer.discard()
        self.last_capture_ns = None
        self._clear_caption()
        self.last_error = message
        raise CaptureStateError(message)

    def resume(self) -> int:
        """Explicitly begin the next sign after COMMITTED."""
        if self.state != COMMITTED:
            raise CaptureStateError(f"cannot resume from {self.state}")
        return self.begin_sign()

    def reset(self) -> str:
        with self._lock:
            self.buffer.discard()
            self.last_capture_ns = None
            self.last_error = None
            self._clear_caption()
            return self.session.reset()

    def stop(self) -> str:
        with self._lock:
            self.buffer.discard()
            self.last_capture_ns = None
            self._clear_caption()
            if self.state != STOPPED:
                self.session.transition(STOPPED)
            return self.state

    def stats(self) -> dict:
        with self._lock:
            return {
                "state": self.state,
                "sign_seq": self.sign_seq,
                "buffer": self.buffer.stats(),
                "dropped_out_of_order": self.dropped_out_of_order,
                "dropped_duplicate": self.dropped_duplicate,
                "dropped_timestamp_gaps": self.dropped_timestamp_gaps,
                "rejected_frames": self.rejected_frames,
                "discarded_stale_results": self.discarded_stale_results,
                "drop_events": list(self.drop_events),
                "last_error": self.last_error,
                "frame_processing_ms": stats_list(self.frame_processing_ms),
                "captions": len(self.caption_events),
                "caption_events": list(self.caption_events),
            }


def run_video_file(
    video_path,
    *,
    capacity: int = 96,
    do_warmup: bool = True,
    max_frames: int | None = None,
    max_capture_gap_ms: float = DEFAULT_MAX_CAPTURE_GAP_MS,
):
    """Decode every frame through VIDEO-mode tracking and classify at EOF.

    This is a file benchmark, not a live camera run. ``clip_to_result_ms``
    includes model setup and optional warm-up but ends before JSON serialization.
    """
    import cv2
    from landmark_frontend import NUM_LANDMARKS, get_landmarker
    import mediapipe as mp

    started_ns = _now()
    classifier = StreamingClassifier()
    if do_warmup:
        classifier.warmup()
    session = StreamingSession(
        capacity=capacity,
        classifier=classifier,
        max_capture_gap_ms=max_capture_gap_ms,
    )
    landmarker = get_landmarker(mode="video")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"could not open video: {video_path}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_index = 0
    last_detector_timestamp_ms = int(getattr(get_landmarker, "_last_ts", 0))
    try:
        while max_frames is None or frame_index < max_frames:
            capture_ns = _now()
            ok, frame = capture.read()
            if not ok:
                break
            rgb = cv2.cvtColor(cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE)), cv2.COLOR_BGR2RGB)
            # Preserve the process-wide VIDEO-mode timestamp invariant.
            detector_timestamp_ms = max(int(capture_ns // 1e6), int(last_detector_timestamp_ms) + 1)
            last_detector_timestamp_ms = detector_timestamp_ms
            get_landmarker._last_ts = detector_timestamp_ms
            detection = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), detector_timestamp_ms
            )
            slots = {
                side: np.zeros((NUM_LANDMARKS, 3), np.float32)
                for side in ("Left", "Right")
            }
            seen = {"Left": 0.0, "Right": 0.0}
            for hand, handedness in zip(detection.hand_landmarks, detection.handedness):
                side = handedness[0].category_name if handedness else "Right"
                if side in slots:
                    slots[side] = np.array([[p.x, p.y, p.z] for p in hand], np.float32)
                    seen[side] = 1.0
            landmarks = np.stack([slots["Left"], slots["Right"]])
            present = np.array([seen["Left"], seen["Right"]], np.float32)
            session.frame_processing_ms.append(_ms(capture_ns, _now()))
            status = session.ingest_landmark_frame(frame_index, capture_ns, landmarks, present)
            if status not in ("ok", "ok_dropped_oldest"):
                raise CaptureStateError(f"unexpected sequential-file ingest status: {status}")
            frame_index += 1
    finally:
        capture.release()
    caption = session.flush_sign(t_sign_end_ns=_now())
    if caption is None:
        raise CaptureStateError("file sign was retired before its caption committed")
    return {
        "mode": "file_sequential_video",
        "live_capture": False,
        "video": str(video_path),
        "model_sha256": classifier.model_sha256,
        "total_frames_declared": total,
        "fps_declared": fps,
        "frames_streamed": frame_index,
        "prediction": caption["prediction"],
        "confidence": caption["confidence"],
        "top5": caption["top5"],
        "warmup_ms": classifier.warmup_ms,
        "frame_processing_ms": stats_list(session.frame_processing_ms),
        "first_frame_to_caption_ms": caption["first_frame_to_caption_ms"],
        "sign_end_to_caption_ms": caption["sign_end_to_caption_ms"],
        "classifier_invoke_ms": caption["classifier_invoke_ms"],
        "classifier_total_ms": caption["classifier_total_ms"],
        "clip_to_result_ms": _ms(started_ns, _now()),
        "buffer": session.buffer.stats(),
        "dropped_out_of_order": session.dropped_out_of_order,
        "dropped_duplicate": session.dropped_duplicate,
        "dropped_timestamp_gaps": session.dropped_timestamp_gaps,
        "caption": caption,
        "timing_note": (
            "File-to-result and file-frame-to-caption are measured on this host. "
            "They are not live-camera latency and are not clip p95 from run_manifest.py."
        ),
    }, session


def run_landmark_file(npz_path, *, capacity: int = 96, do_warmup: bool = True):
    """Replay timestamped landmarks without inventing frame or camera latency."""
    with np.load(npz_path, allow_pickle=False) as data:
        timestamps = np.asarray(data["timestamps_ns"])
        landmarks = np.asarray(data["landmarks"])
        present = np.asarray(data["present"])
    if len(timestamps) != len(landmarks) or len(timestamps) != len(present):
        raise ValueError("timestamp, landmark, and presence counts must match")
    classifier = StreamingClassifier()
    if do_warmup:
        classifier.warmup()
    session = StreamingSession(
        capacity=capacity, classifier=classifier, timestamp_clock="external_ns"
    )
    for index in range(len(timestamps)):
        status = session.ingest_landmark_frame(
            index, timestamps[index], landmarks[index], present[index]
        )
        if status not in ("ok", "ok_dropped_oldest"):
            raise CaptureStateError(f"landmark replay rejected frame {index}: {status}")
    if not len(timestamps):
        raise ValueError("landmark replay requires at least one frame")
    caption = session.flush_sign(t_sign_end_ns=int(timestamps[-1]))
    if caption is None:
        raise CaptureStateError("landmark sign was retired before commit")
    return {
        "mode": "landmark_replay",
        "live_capture": False,
        "source_npz": str(npz_path),
        "model_sha256": classifier.model_sha256,
        "frames": len(timestamps),
        "prediction": caption["prediction"],
        "confidence": caption["confidence"],
        "top5": caption["top5"],
        "warmup_ms": classifier.warmup_ms,
        "frame_processing_ms": {
            "n": 0, "p50": None, "p95": None, "max": None,
            "note": "not measured: landmark replay does not execute a frame front end",
        },
        "first_frame_to_caption_ms": None,
        "sign_end_to_caption_ms": None,
        "cross_clock_timing_note": "file timestamps cannot be subtracted from the runtime monotonic clock",
        "classifier_invoke_ms": caption["classifier_invoke_ms"],
        "classifier_total_ms": caption["classifier_total_ms"],
        "buffer": session.buffer.stats(),
        "dropped_out_of_order": session.dropped_out_of_order,
        "dropped_duplicate": session.dropped_duplicate,
        "dropped_timestamp_gaps": session.dropped_timestamp_gaps,
        "caption": caption,
        "timing_note": "Landmark replay is not a live-stream or clip-p95 measurement.",
    }, session


def stats_list(values):
    if not values:
        return {"n": 0, "p50": None, "p95": None, "max": None}
    array = np.array(values, dtype=float)
    return {
        "n": int(len(array)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }
