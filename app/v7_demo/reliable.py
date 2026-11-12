"""A10 V7 reliability wrappers over the shared offline demo primitives.

Fixes concrete demo defects without editing ``app/offline_demo.py``:

1. **Stale-result race.** The shared ``run()`` applies any queued worker
   result to whatever state is current, so a result from capture N can
   overwrite capture N+1's ``PROCESSING``/``ERROR`` state. Here every
   submission carries a monotonically increasing ``seq``; only the newest
   sequence is accepted, older results are counted and discarded as
   ``stale``.

2. **``R`` reopen clobber.** Reopening/resetting bumps the sequence epoch,
   so any in-flight prediction from before the reopen is discarded.

3. **Worker close racing a busy inference.** The shared ``close()``
   joins 2s then calls ``TemporaryDirectory.cleanup()``, which raises
   ``FileNotFoundError``/leaks when the worker is still encoding or the
   predictor is mid-run. ``ReliableWorker.close()`` waits a configurable
   bounded time, reports whether it drained, and explicitly purges any
   leftover temp clips.

4. **HUD text clipped mid-word.** The shared ``render()`` slices to
   65/86 chars regardless of frame width. ``fit_text()`` measures with
   ``cv2.getTextSize`` and shrinks the scale until the text fits.

5. **Double decode / exit-0-on-error.** ``render_file_result`` decodes
   the clip twice and ``main()`` exits 0 even for an error result.
   ``run_file()`` decodes once and returns an explicit exit code.

6. **Headless early exit.** The shared headless loop exits on the first
   camera read error or when ``max_frames`` is hit, even with no
   prediction ever submitted. ``run_session()`` waits (bounded) for a
   real submitted prediction to complete.

Camera remains unverified on this machine (macOS TCC denied). Every
report carries ``live_camera_verified: false``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import queue
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.offline_demo import (  # noqa: E402
    PROVISIONAL_THRESHOLD,
    CaptionState,
    PredictionWorker,
    load_predictor,
    model_identity,
    model_label_from_identity,
    open_camera,
    sample_file_frames,
    sanitize_result,
    write_temp_clip,
)

EXPECTED_FP16_SHA256 = "a012f2b5f7a1110961079f19eb0e28c9abacb95e4d0787352e4157ebca7044a4"


# ---------------------------------------------------------------------------
# Text fitting: measure instead of slicing a fixed character count.
# ---------------------------------------------------------------------------
def fit_text(text: str, max_width: int, base_scale: float,
             thickness: int = 1, font=cv2.FONT_HERSHEY_SIMPLEX,
             min_scale: float = 0.30) -> tuple[str, float]:
    """Shrink ``text`` until ``cv2.getTextSize`` fits ``max_width`` px.

    Returns ``(display_text, scale)``. When even ``min_scale`` overflows,
    ellipsises at character level so nothing is silently lost mid-word.
    """
    text = "" if text is None else str(text)
    max_width = max(1, int(max_width))
    scale = float(base_scale)
    while scale > min_scale:
        (w, _), _ = cv2.getTextSize(text, font, scale, thickness)
        if w <= max_width:
            return text, scale
        scale -= 0.05
    # At minimum scale: truncate by measured width, not by char count.
    candidate = text
    while candidate:
        (w, _), _ = cv2.getTextSize(candidate + "…", font, min_scale, thickness)
        if w <= max_width:
            return candidate + "…", min_scale
        candidate = candidate[:-1]
    return "", min_scale


class ReliableHUD:
    """Width-aware HUD rendering on top of ``CaptionState`` snapshots."""

    def __init__(self, caption_threshold: float = PROVISIONAL_THRESHOLD):
        self.caption_threshold = float(caption_threshold)

    def render(self, frame: np.ndarray, state: CaptionState,
               model_label: str, threshold: float | None = None) -> np.ndarray:
        threshold = self.caption_threshold if threshold is None else threshold
        image = frame.copy()
        snap = state.snapshot()
        name = snap["state"]
        message = snap["message"] or ""
        result = snap["result"]
        colors = {
            "WAITING": (200, 200, 200),
            "PROCESSING": (50, 200, 255),
            "RETRY": (40, 150, 255),
            "CAPTION": (100, 230, 100),
            "ERROR": (50, 50, 230),
        }
        color = colors.get(name, (255, 255, 255))
        width = image.shape[1]
        margin = 18
        usable = width - 2 * margin
        header_h = 135
        cv2.rectangle(image, (0, 0), (width, header_h), (15, 20, 25), -1)
        line, scale = fit_text(
            "EchoGesture | OFFLINE | integration not release-certified",
            usable, 0.58, 1)
        cv2.putText(image, line, (margin, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (255, 255, 255), 1, cv2.LINE_AA)
        base = 1.1 if name == "CAPTION" else 0.75
        line, scale = fit_text(message, usable, base, 2 if name == "CAPTION" else 1)
        cv2.putText(image, line, (margin, 75), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, 2 if name == "CAPTION" else 1, cv2.LINE_AA)
        detail = f"{name} | {model_label} | provisional retry threshold {threshold:.2f}"
        line, scale = fit_text(detail, usable, 0.48, 1)
        cv2.putText(image, line, (margin, 108), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (200, 200, 200), 1, cv2.LINE_AA)
        footer_note = "SPACE capture | R reopen/reset | Q quit | no footage retained"
        if result:
            confidence = float(result.get("confidence", 0.0))
            latency = float(result.get("latency_ms", 0.0))
            capture_ms = float(result.get("capture_to_result_ms", latency))
            text = (f"confidence {confidence:.3f} | clip {latency:.1f} ms | "
                    f"capture-result {capture_ms:.1f} ms")
            if result.get("queue_wait_ms") is not None:
                text += f" | queued {float(result['queue_wait_ms']):.1f} ms"
            line, scale = fit_text(text, usable, 0.55, 1)
            cv2.putText(image, line, (margin, image.shape[0] - 42),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
            top5 = ", ".join(result.get("top5", []) or [])
            line, scale = fit_text("Top 5: " + top5, usable, 0.5, 1)
            cv2.putText(image, line, (margin, image.shape[0] - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            line, scale = fit_text(footer_note, usable, 0.55, 1)
            cv2.putText(image, line, (margin, image.shape[0] - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
        return image


class ReliableCaptionState:
    """CaptionState + sequence guard against stale worker results.

    A result is applied only when its ``seq`` equals the current epoch's
    latest submitted sequence. ``reset()``/``reopen()`` advances the epoch,
    discarding anything submitted before the reset.
    """

    def __init__(self, threshold: float = PROVISIONAL_THRESHOLD):
        self._state = CaptionState(threshold)
        self._lock = threading.Lock()
        self._seq = 0
        self._epoch = 0
        self.stale_count = 0

    @property
    def state(self) -> str:
        return self._state.state

    @property
    def result(self):
        return self._state.result

    def snapshot(self) -> dict:
        snap = self._state.snapshot()
        with self._lock:
            snap["seq"] = self._seq
            snap["epoch"] = self._epoch
            snap["stale_count"] = self.stale_count
        return snap

    def submit(self) -> int:
        with self._lock:
            self._seq += 1
            seq = self._seq
        self._state.processing()
        return seq

    def apply(self, seq: int, epoch: int, kind: str, payload,
              threshold: float | None = None) -> str:
        with self._lock:
            current = (self._seq == seq and self._epoch == epoch)
            if not current:
                self.stale_count += 1
                return "STALE"
        if kind == "result":
            return self._state.update(payload)
        return self._state.error(payload)

    def reopen(self, message: str = "Reopening camera...") -> None:
        with self._lock:
            self._epoch += 1
        self._state.reset(message)

    def reset(self, message: str = None) -> None:
        self._state.reset(message)

    def error(self, message: str) -> str:
        return self._state.error(message)


class ReliableWorker:
    """PredictionWorker + epoch/sequence tagging + race-free close.

    Delegates temp-clip encode/predict/delete to ``PredictionWorker`` but
    tags every delivered result with the sequence and epoch it belongs to,
    and makes ``close()`` safe when called mid-inference: it waits a
    bounded time and purges any temp leftovers instead of racing
    ``TemporaryDirectory.cleanup()``.
    """

    def __init__(self, predict, state: ReliableCaptionState,
                 tmpdir=None, close_timeout: float = 5.0):
        self._state = state
        self._predict = predict
        self._close_timeout = float(close_timeout)
        self._worker = PredictionWorker(predict, tmpdir=tmpdir)
        self._pending: dict[tuple[int, int], Any] = {}
        self._pending_lock = threading.Lock()
        self.closed = False
        self.drained = False

    @property
    def tmpdir_path(self) -> Path:
        return self._worker.tmpdir_path

    @property
    def queue_depth(self) -> int:
        return self._worker.requests.qsize()

    def submit(self, frames, fps, clip_id) -> int | None:
        """Submit and return the sequence tag, or ``None`` if busy/closed."""
        if self.closed or not frames:
            return None
        seq = self._state.submit()
        epoch = self._state.snapshot()["epoch"]
        if not self._worker.submit(list(frames), fps, clip_id):
            # Worker busy/closed: surface as a real error for this sequence.
            self._state.apply(seq, epoch, "error", "Inference busy - try again")
            return None
        with self._pending_lock:
            self._pending[(seq, epoch)] = clip_id
        return seq

    def drain(self, timeout: float = 0.0):
        """Pull finished predictions and apply the non-stale ones."""
        deadline = time.perf_counter() + max(0.0, timeout)
        applied = []
        while True:
            remaining = deadline - time.perf_counter()
            if remaining < 0:
                break
            try:
                kind, payload = self._worker.results.get(
                    timeout=min(0.05, remaining) if timeout else 0)
            except queue.Empty:
                if timeout:
                    if time.perf_counter() >= deadline:
                        break
                    continue
                break
            with self._pending_lock:
                key = next(iter(self._pending), None)
                # Match by FIFO order: worker completes in submission order.
                if key is not None:
                    self._pending.pop(key)
            if key is None:
                continue
            seq, epoch = key
            applied.append(self._state.apply(seq, epoch, kind, payload))
            if not timeout:
                break
        return applied

    def close(self):
        if self.closed:
            return
        self.closed = True
        # Give an in-flight predict() a bounded chance to finish and let
        # the worker delete its temp clip itself.
        deadline = time.perf_counter() + self._close_timeout
        while time.perf_counter() < deadline:
            if self._worker.requests.empty() and not self._worker.thread.is_alive():
                break
            # If the worker thread is the only thing left, wait for it.
            if self._worker.thread.is_alive():
                self._worker.thread.join(timeout=0.05)
            else:
                break
        self.drained = not self._worker.thread.is_alive()
        # Purge any leftover temp clips so none are retained on disk,
        # even when close raced a still-running prediction.
        try:
            leftovers = list(self._worker.tmpdir_path.iterdir())
        except OSError:
            leftovers = []
        for leftover in leftovers:
            try:
                leftover.unlink(missing_ok=True)
            except OSError:
                pass
        self._worker.close()
        # Final sweep: after worker close, remove the directory contents.
        try:
            for leftover in list(self._worker.tmpdir_path.iterdir()):
                try:
                    leftover.unlink(missing_ok=True)
                except OSError:
                    pass
        except OSError:
            pass
        self.drained = not self._worker.thread.is_alive()


@dataclass
class CaptureResult:
    seq: int
    state: str
    result: dict | None
    stale_count: int


@dataclass
class SessionReport:
    mode: str
    captures: list[CaptureResult] = field(default_factory=list)
    stale_count: int = 0
    exit_code: int = 0
    live_camera_verified: bool = False
    model: dict | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "captures": [
                {"seq": c.seq, "state": c.state, "result": c.result,
                 "stale_count": c.stale_count}
                for c in self.captures
            ],
            "stale_count": self.stale_count,
            "exit_code": self.exit_code,
            "live_camera_verified": self.live_camera_verified,
            "model": self.model,
            "notes": self.notes,
        }


def run_session(camera_index: int, seconds: float, predictor_path,
                threshold: float = PROVISIONAL_THRESHOLD,
                headless: bool = True, max_captures: int = 1,
                max_frames: int = 0,
                result_timeout_s: float = 30.0,
                hud: ReliableHUD | None = None) -> SessionReport:
    """Run a bounded headless/interactive session with stale-result safety.

    Camera must open (real frames probed by the shared ``open_camera``).
    A ``CameraUnavailableError`` propagates to the caller — camera success
    is never inferred.
    """
    identity = model_identity()
    model_label = model_label_from_identity(identity)
    hud = hud or ReliableHUD(threshold)
    state = ReliableCaptionState(threshold)
    report = SessionReport(mode="camera", model=identity)

    capture = open_camera(camera_index)
    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and 1 <= fps <= 240 else 30.0
    target = max(1, int(seconds * fps))
    worker = ReliableWorker(load_predictor(predictor_path), state)
    captured = deque(maxlen=target)
    capturing = False
    frame_count = 0
    captures_done = 0
    last_seq: int | None = None
    try:
        while captures_done < max_captures:
            ok, frame = capture.read()
            if not ok:
                state.error("Camera read failed - press R to reopen the camera, or Q to quit")
                if headless:
                    report.notes.append("camera read failed before a capture completed")
                    break
                cv2.waitKey(500)
                continue
            frame_count += 1
            if capturing:
                captured.append(frame)
                if len(captured) >= target:
                    capturing = False
                    seq = worker.submit(list(captured), fps, f"demo-{time.time_ns()}")
                    if seq is not None:
                        last_seq = seq
                    captured.clear()
            if headless and frame_count == 1 and captures_done == 0 and not capturing:
                capturing = True
            if not headless:
                cv2.imshow("EchoGesture - Offline",
                           hud.render(frame, state._state, model_label, threshold))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord(" "):
                    if capturing:
                        capturing = False
                        captured.clear()
                        state.reset()
                    elif state.state != "PROCESSING":
                        capturing = True
                        captured.clear()
                if key in (ord("r"), ord("R")):
                    state.reopen()
                    try:
                        capture.release()
                        capture = open_camera(camera_index)
                        state.reset()
                    except Exception as exc:  # CameraUnavailableError etc.
                        state.error(str(exc))
            worker.drain()
            if max_frames and frame_count >= max_frames:
                break
            # Headless completion: wait (bounded) for the submitted result.
            if headless and last_seq is not None and state.state != "PROCESSING":
                outcome = state.state
                if outcome in ("CAPTION", "RETRY", "ERROR"):
                    report.captures.append(CaptureResult(
                        seq=last_seq, state=outcome,
                        result=state.result,
                        stale_count=state.snapshot()["stale_count"]))
                    captures_done += 1
                    last_seq = None
                    if captures_done >= max_captures:
                        break
                    capturing = True  # auto-capture next sign
                    captured.clear()
    finally:
        capture.release()
        worker.close()
        if not headless:
            cv2.destroyAllWindows()
    if headless and last_seq is not None:
        # Bounded wait for the final in-flight prediction.
        deadline = time.perf_counter() + result_timeout_s
        while time.perf_counter() < deadline and state.state == "PROCESSING":
            worker.drain(timeout=0.1)
        if state.state in ("CAPTION", "RETRY", "ERROR"):
            report.captures.append(CaptureResult(
                seq=last_seq, state=state.state, result=state.result,
                stale_count=state.snapshot()["stale_count"]))
            captures_done += 1
        else:
            report.notes.append("headless result timeout; no result surfaced")
    report.stale_count = state.snapshot()["stale_count"]
    report.exit_code = 0 if report.captures and report.captures[-1].state in ("CAPTION", "RETRY") else 1
    if not report.captures:
        report.notes.append("no prediction completed — camera/capture did not succeed")
    return report


def run_file(video_path, predictor_path, threshold: float = PROVISIONAL_THRESHOLD,
             render_output=None, hud: ReliableHUD | None = None) -> SessionReport:
    """Camera-less real prediction on one file (single decode, real HUD).

    Exit code: 0 when the model produced a sanitized result (CAPTION or
    RETRY), 1 on any load/decode/predict failure. The input file is read
    in place and never copied.
    """
    video_path = Path(video_path)
    identity = model_identity()
    report = SessionReport(mode="file", model=identity)
    report.notes.append(
        "File-input path. NOT a live-camera claim; macOS camera access is "
        "denied on this machine.")
    if not video_path.exists():
        report.notes.append(f"video not found: {video_path}")
        report.exit_code = 1
        return report
    hud = hud or ReliableHUD(threshold)
    state = CaptionState(threshold)
    started = time.perf_counter_ns()
    try:
        frames = sample_file_frames(video_path)
    except Exception as exc:  # noqa: BLE001
        report.notes.append(f"decode failed: {type(exc).__name__}: {exc}")
        report.exit_code = 1
        return report
    if not frames:
        report.notes.append(f"no valid frames decoded from: {video_path}")
        report.exit_code = 1
        return report
    try:
        predict = load_predictor(predictor_path)
    except Exception as exc:  # noqa: BLE001
        report.notes.append(f"predictor load failed: {type(exc).__name__}: {exc}")
        report.exit_code = 1
        return report
    clip_id = video_path.stem
    # Write one temp clip from the sampled frames, predict once, delete.
    tmpdir = None
    tmp_path = None
    try:
        import tempfile
        with tempfile.TemporaryDirectory(prefix="echogesture-v7-") as td:
            tmp_path = write_temp_clip(frames, 30.0, td, clip_id)
            raw = predict(str(tmp_path), clip_id=clip_id)
            result = sanitize_result(raw, clip_id=clip_id)
            result.setdefault("queue_wait_ms", 0.0)
            result["capture_to_result_ms"] = (time.perf_counter_ns() - started) / 1e6
            outcome = state.update(result)
            report.captures.append(CaptureResult(
                seq=1, state=outcome, result=result, stale_count=0))
    except Exception as exc:  # noqa: BLE001
        report.notes.append(f"prediction failed: {type(exc).__name__}: {exc}")
        report.exit_code = 1
        return report
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
    # Render the middle frame with the fitted HUD.
    model_label = model_label_from_identity(identity)
    middle = frames[len(frames) // 2]
    image = hud.render(middle, state, model_label, threshold)
    if render_output:
        output = Path(render_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        report.notes.append(f"HUD written: {output}")
        if not cv2.imwrite(str(output), image):
            report.notes.append(f"HUD write failed: {output}")
            report.exit_code = 1
    report.exit_code = 0 if report.captures and report.captures[-1].state in ("CAPTION", "RETRY") else 1
    if report.exit_code == 0:
        report.notes.append("real predict() result surfaced; see captures[0].result")
    return report
