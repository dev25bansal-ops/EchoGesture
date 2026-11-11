"""Persistent parallel IMAGE hand tracker (A01 performance final).

Single reusable production-quality tracker that runs Dev's image hand
detection across a fixed pool of worker threads, each owning one independent
MediaPipe IMAGE landmarker, and reassembles the per-frame results in their
original order.

Why this is the correct shape (and not a rewrite):

* Per-frame detection, handedness slots, presence flags, thresholds and
  coordinate conversion are delegated to Dev's `src/landmark_frontend.py`
  `detect_hands`, so the output is bit-identical to the production singleton
  path by construction.  Nothing in Dev's feature/model contract is touched.
* MediaPipe IMAGE mode is stateless between `detect()` calls, which means
  frames can be split across independent landmarker instances without any
  cross-clip or cross-worker tracker state.  VIDEO-mode temporal state is
  deliberately not used (prior agents showed it changes the landmark stream).
* Workers are constructed once and reused across clips (persistent).  A worker
  that raises or exceeds `timeout` is replaced exactly once per failing call
  (bounded recovery); if the replacement also fails, the error propagates.

Resource bound: exactly `workers` landmarkers plus one `ThreadPoolExecutor`
with `workers` threads.  `close()` is idempotent, thread-safe and releases the
executor without waiting on a possibly-hung detector thread.

This module is owned by A01 and lives only under
`evaluation/device/performance_final/`.  It does not edit `src/main.py`,
`src/features.py`, `src/landmark_frontend.py` or any model file.
"""

from __future__ import annotations

import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE                        # this module ships inside src/
ROOT = HERE.parent
MODELS = ROOT / "models"

# The module lives in src/ in the release bundle and in
# evaluation/device/performance_final/ in the research tree. Resolve the
# frontend relative to this file so both layouts import the same module
# rather than depending on cwd or sys.path order.
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:  # package import (release bundle: python -m src.main)
    from . import landmark_frontend as _frontend
except ImportError:  # flat import (research tree)
    import landmark_frontend as _frontend  # noqa: E402

NUM_LANDMARKS = _frontend.NUM_LANDMARKS      # 21
HANDS = _frontend.HANDS                      # 2
FRAME_SIZE = _frontend.FRAME_SIZE            # 256

DEFAULT_MAX_FRAMES = 32
DEFAULT_WORKERS = 4

# These three thresholds and `running_mode=IMAGE` must match
# src/landmark_frontend.get_landmarker() exactly to preserve parity.
_MIN_HAND_DETECTION_CONFIDENCE = 0.3
_MIN_HAND_PRESENCE_CONFIDENCE = 0.3
_MIN_TRACKING_CONFIDENCE = 0.3

__all__ = ["ParallelImageTracker", "TrackerError", "detect_parallel"]


class TrackerError(RuntimeError):
    """Raised when a worker fails and its one-shot replacement also fails."""


class _TaskFailure(Exception):
    """Internal: carries which worker index produced the first failure."""

    def __init__(self, worker_index: int, cause: BaseException):
        super().__init__(f"worker {worker_index} failed")
        self.worker_index = worker_index
        self.cause = cause


def _create_image_landmarker(task_path: Path):
    """Build one independent IMAGE HandLandmarker with Dev's exact options.

    Kept module-level so tests can inject fakes; production code never calls it
    indirectly.
    """
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(task_path)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=HANDS,
        min_hand_detection_confidence=_MIN_HAND_DETECTION_CONFIDENCE,
        min_hand_presence_confidence=_MIN_HAND_PRESENCE_CONFIDENCE,
        min_tracking_confidence=_MIN_TRACKING_CONFIDENCE,
    )
    return vision.HandLandmarker.create_from_options(options)


class _Worker:
    """One independent IMAGE landmarker guarded by its own lock.

    The lock makes a single worker safe under repeated/concurrent `detect()`
    calls even though MediaPipe itself is not thread-safe.
    """

    def __init__(self, index: int, task_path: Path):
        self.index = index
        self.task_path = task_path
        self._lock = threading.Lock()
        self._landmarker = _create_image_landmarker(task_path)

    def detect(self, frames: list[np.ndarray]):
        with self._lock:
            landmarks, present = _frontend.detect_hands(
                frames, landmarker=self._landmarker, mode="image")
        return landmarks, present

    def close(self) -> None:
        with self._lock:
            landmarker, self._landmarker = self._landmarker, None
            if landmarker is not None:
                try:
                    landmarker.close()
                except Exception:
                    pass  # never let cleanup mask the caller's error


class ParallelImageTracker:
    """A persistent pool of worker threads, each owning one IMAGE landmarker.

    Public API:
        tracker = ParallelImageTracker(workers=4)
        landmarks, present = tracker.detect(frames, timeout=None)
        tracker.close()            # idempotent; also available as a context
                                   # manager (``with ParallelImageTracker(...)``)

    ``frames`` must be the exact RGB uint8 256x256x3 arrays returned by
    `src/landmark_frontend.read_frames`.  ``detect`` returns fresh
    ``(T, 2, 21, 3)`` landmarks and ``(T, 2)`` presence arrays in input order,
    identical to `src/landmark_frontend.detect_hands` (see the parity tests).
    """

    def __init__(self, workers: int = DEFAULT_WORKERS, *,
                 task_path: str | Path | None = None,
                 max_frames: int = DEFAULT_MAX_FRAMES):
        if workers < 1:
            raise ValueError("workers must be >= 1")
        max_frames = int(max_frames)
        if max_frames < 1:
            raise ValueError("max_frames must be >= 1")

        self.workers = int(workers)
        self.max_frames = max_frames
        self.task_path = (
            Path(task_path).resolve()
            if task_path is not None else (MODELS / "hand_landmarker.task").resolve()
        )
        if not self.task_path.is_file():
            raise FileNotFoundError(self.task_path)

        self._pool_lock = threading.Lock()
        self._workers = [_Worker(i, self.task_path) for i in range(self.workers)]
        self._executor = ThreadPoolExecutor(
            max_workers=self.workers,
            thread_name_prefix=f"echogesture-img-{self.workers}")
        self._closed = False
        self.frames_processed = 0

    # -- input validation ---------------------------------------------------

    def _validate_frames(self, frames):
        if frames is None:
            raise ValueError("frames must be a sequence of RGB frames, not None")
        frames = list(frames)
        if len(frames) > self.max_frames:
            raise ValueError(f"expected at most {self.max_frames} frames, got {len(frames)}")
        for i, frame in enumerate(frames):
            if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
                    or frame.shape != (FRAME_SIZE, FRAME_SIZE, 3)):
                raise ValueError(
                    f"frame {i} must be uint8 {FRAME_SIZE}x{FRAME_SIZE}x3, "
                    f"got shape={getattr(frame, 'shape', None)} "
                    f"dtype={getattr(frame, 'dtype', None)}")
        return frames

    # -- worker scheduling --------------------------------------------------

    def _chunks(self, total: int):
        """Round-robin frame index lists, one per worker."""
        return [list(range(i, total, self.workers)) for i in range(self.workers)]

    def _run_worker(self, index: int, frames: list[np.ndarray]):
        return self._workers[index].detect(frames)

    def _detect_once(self, frames: list[np.ndarray], timeout: float | None):
        chunks = self._chunks(len(frames))
        results: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        futures: dict[int, object] = {}
        for index, chunk_indices in enumerate(chunks):
            if not chunk_indices:
                continue
            worker_frames = [frames[j] for j in chunk_indices]
            futures[index] = self._executor.submit(self._run_worker, index, worker_frames)

        for index, future in futures.items():
            try:
                results[index] = future.result(timeout=timeout)
            except FutureTimeout:
                raise _TaskFailure(
                    index, TimeoutError(f"worker {index} exceeded {timeout}s")) from None
            except Exception as exc:  # noqa: BLE001  -> carry the cause up
                raise _TaskFailure(index, exc) from exc

        total = len(frames)
        landmarks = np.zeros((total, HANDS, NUM_LANDMARKS, 3), np.float32)
        present = np.zeros((total, HANDS), np.float32)
        for index, chunk_indices in enumerate(chunks):
            if not chunk_indices:
                continue
            worker_landmarks, worker_present = results[index]
            if worker_landmarks.shape[0] != len(chunk_indices):
                raise RuntimeError(
                    f"worker {index} returned {worker_landmarks.shape[0]} frames "
                    f"for a {len(chunk_indices)}-frame chunk")
            for position, global_index in enumerate(chunk_indices):
                landmarks[global_index] = worker_landmarks[position]
                present[global_index] = worker_present[position]
        return landmarks, present

    def _replace_worker(self, index: int) -> None:
        with self._pool_lock:
            self._workers[index].close()
            self._workers[index] = _Worker(index, self.task_path)

    # -- public API ---------------------------------------------------------

    def detect(self, frames, *, timeout: float | None = None):
        """Detect landmarks on ``frames``, restoring original order.

        On a worker exception or timeout the failing worker is replaced once and
        the clip is retried; a second failure propagates (bounded recovery).
        """
        if self._closed:
            raise RuntimeError("tracker is closed")
        frames = self._validate_frames(frames)
        if not frames:
            return (np.zeros((0, HANDS, NUM_LANDMARKS, 3), np.float32),
                    np.zeros((0, HANDS), np.float32))
        try:
            landmarks, present = self._detect_once(frames, timeout)
        except _TaskFailure as failure:
            self._replace_worker(failure.worker_index)
            try:
                landmarks, present = self._detect_once(frames, timeout)
            except _TaskFailure as failure:
                raise TrackerError(
                    f"worker {failure.worker_index} failed again after replacement "
                    f"({failure.cause})") from failure.cause
        self.frames_processed += len(frames)
        return landmarks, present

    def close(self) -> None:
        """Idempotent, thread-safe teardown; never waits on a hung worker."""
        if self._closed:
            return
        self._closed = True
        with self._pool_lock:
            for worker in self._workers:
                worker.close()
            self._workers = []
        self._executor.shutdown(wait=False, cancel_futures=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def detect_parallel(frames, *, workers: int = DEFAULT_WORKERS,
                    timeout: float | None = None):
    """One-shot convenience: build a pool, detect once, tear it down.

    Use the class directly when the pool must persist across clips; this helper
    exists for the common single-clip case and for drop-in parity checks.
    """
    with ParallelImageTracker(workers=workers) as tracker:
        return tracker.detect(frames, timeout=timeout)