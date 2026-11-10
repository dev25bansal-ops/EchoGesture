"""Hardened demo media handling (D4 + retention wording fixes).

The shipped ``app/offline_demo.py`` has three release-blocking retention
gaps (see A15 findings P4/P4b/D4 and ``app/v7_demo/reliable.py`` notes):

1. **Shared-dir landing.**  When ``tmpdir=`` is passed, temp clips are
   written directly into that (possibly shared) directory with
   predictable names and ``0644`` permissions.  Even the default path
   creates ``0644`` files under the system temp dir (protected only by
   the parent's ``0700``).
2. **Close race.**  ``close()`` joins 2 s and then calls
   ``TemporaryDirectory.cleanup()`` while the worker may still be
   mid-inference; the resulting ``OSError`` is swallowed, silently
   orphaning temp clips.  "No footage is retained" is then false.
3. **RAM retention.**  Frames live in a ``deque`` plus ``list(captured)``
   copies and are never scrubbed.  The honest claim is on-disk
   retention only.

This module is a drop-in hardened replacement that preserves the
public interface (``PredictionWorker`` / ``write_temp_clip`` /
``tmpdir_path`` / ``submit`` / ``close`` with the same semantics)
while fixing all three:

* Every clip lives in a private ``0700`` temporary directory owned by
  the worker (an explicit ``tmpdir=`` is still honoured as the *parent*
  of a fresh private child, never written to directly).
* Files are created ``O_EXCL`` + unguessable (``mkstemp``) with
  ``0600`` permissions and per-file ``0600`` chmod (``VideoWriter``
  applies the process umask, so an explicit chmod is required).
* ``close()`` waits a bounded time for in-flight work, purges leftover
  clips, then closes the worker, then sweeps the private directory
  again — matching ``ReliableWorker``'s documented contract.
* Frame buffers are cleared / zeroed after encode so RAM hold time is
  minimal and stated honestly.
* :data:`RETENTION_WORDING` is the accurate replacement copy for the
  demo header, HUD footer, and docs (use verbatim).

Python 3.11 compatible.  Requires ``cv2`` and ``numpy`` only in the
same way the original does (imported lazily so the path helpers stay
testable without a camera stack).
"""

from __future__ import annotations

import math
import os
import queue
import sys
import tempfile
import threading
import time
from pathlib import Path

# Accurate retention claim.  Replace every "no footage is retained"
# / "No footage is retained" string in shipped copy with this wording
# (module docstring, app/README.md, HUD footer).  It claims only what
# was verified: no file persists after each capture completes.
RETENTION_WORDING = (
    "No footage is kept on disk after each capture; frames are held in "
    "memory only during inference."
)

# Backwards-compatible shorter form for the HUD footer (single line).
RETENTION_WORDING_SHORT = "No footage kept on disk after each capture"

_SECURE_DIR_MODE = 0o700
_SECURE_FILE_MODE = 0o600


def _ensure_private_dir(path: Path) -> Path:
    path = Path(path)
    try:
        path.chmod(_SECURE_DIR_MODE)
    except OSError:
        pass
    return path


def _unique_clip_path(tmpdir: Path, safe_id: str) -> Path:
    """Return an unguessable, non-colliding clip path inside ``tmpdir``.

    Uses ``mkstemp`` (``O_EXCL`` + random suffix) so the name cannot be
    guessed or pre-planted, reserves the path at ``0600``, then hands
    it to ``cv2.VideoWriter``.  The writer truncates the placeholder —
    safe because the directory is private and the file is already ours.
    """
    tmpdir = Path(tmpdir)
    fd, name = tempfile.mkstemp(
        prefix=f"echogesture-capture-{safe_id}-", suffix=".mp4", dir=str(tmpdir)
    )
    os.close(fd)
    path = Path(name)
    try:
        path.chmod(_SECURE_FILE_MODE)
    except OSError:
        pass
    return path


def write_temp_clip(frames, fps, tmpdir, clip_id):
    """Encode captured frames to one temporary clip; caller deletes it.

    Mirrors the original contract but creates the file inside a
    private directory with ``0600`` permissions.
    """
    import cv2  # local import: keeps module importable without OpenCV

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
    _ensure_private_dir(Path(tmpdir))
    path = _unique_clip_path(Path(tmpdir), safe_id)
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
    except Exception:
        writer.release()
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    writer.release()
    try:
        path.chmod(_SECURE_FILE_MODE)
    except OSError:
        pass
    if not path.exists() or path.stat().st_size == 0:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        raise RuntimeError("Temporary clip encoder produced no data")
    return path


def _zero_frames(frames) -> None:
    """Best-effort scrub of in-memory frames (numpy buffers)."""
    if not frames:
        return
    for frame in frames:
        try:
            frame.fill(0)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - non-array frames pass through
            pass


def _purge_dir(path: Path) -> int:
    """Remove every entry under ``path``; return how many were removed."""
    try:
        entries = list(Path(path).iterdir())
    except OSError:
        return 0
    removed = 0
    for entry in entries:
        try:
            if entry.is_dir() and not entry.is_symlink():
                for child in list(entry.iterdir()):
                    try:
                        child.unlink(missing_ok=True)
                        removed += 1
                    except OSError:
                        pass
                try:
                    entry.rmdir()
                    removed += 1
                except OSError:
                    pass
            else:
                entry.unlink(missing_ok=True)
                removed += 1
        except OSError:
            pass
    return removed


class SecurePredictionWorker:
    """Hardened ``PredictionWorker``: private temp dir, 0600 clips, race-free close."""

    def __init__(self, predict, tmpdir=None, close_timeout: float = 5.0):
        self.predict = predict
        self.requests = queue.Queue(maxsize=1)
        self.results = queue.Queue()
        self.closed = threading.Event()
        self._close_timeout = float(close_timeout)
        parent = Path(tmpdir) if tmpdir is not None else None
        # Always create a private child; a caller-supplied tmpdir is a
        # parent hint, never the directory clips land in directly.
        self._tmpdir = tempfile.TemporaryDirectory(
            prefix="echogesture-capture-", dir=str(parent) if parent else None
        )
        self.tmpdir = _ensure_private_dir(Path(self._tmpdir.name))
        self._owns_tmpdir = True
        self._close_lock = threading.Lock()
        self._closed_once = False
        self.drained = False
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
        # Imported here so the module remains importable in manifests-only
        # environments; the worker itself needs the cv2 path via write_temp_clip.
        try:
            from app.offline_demo import sanitize_result  # local, avoids top-level cv2 dep
        except Exception:
            import importlib.util
            offline_path = Path(__file__).resolve().parents[3] / "app" / "offline_demo.py"
            spec = importlib.util.spec_from_file_location("echogesture_offline_demo", offline_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            sanitize_result = module.sanitize_result

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
            except Exception as exc:  # noqa: BLE001 - surfaced as worker error
                self.results.put(("error", f"{type(exc).__name__}: {exc}"))
            finally:
                if path is not None:
                    try:
                        path.unlink(missing_ok=True)
                    except OSError:
                        pass
                _zero_frames(frames)
                try:
                    frames.clear()  # type: ignore[attr-defined]
                except Exception:  # noqa: BLE001
                    pass

    def close(self):
        """Bounded drain, leftover purge, worker close, directory sweep."""
        with self._close_lock:
            if self._closed_once:
                return
            self._closed_once = True
        self.closed.set()
        deadline = time.perf_counter() + self._close_timeout
        while time.perf_counter() < deadline:
            if not self.thread.is_alive():
                break
            self.thread.join(timeout=0.05)
        self.drained = not self.thread.is_alive()
        # Purge leftovers (e.g. a clip whose writer was mid-close when
        # the join timed out), then close the worker's TemporaryDirectory.
        _purge_dir(self.tmpdir)
        if self._tmpdir is not None:
            try:
                self._tmpdir.cleanup()
            except OSError:
                pass
        # Final sweep: anything written after the close was signalled.
        _purge_dir(self.tmpdir)
        self.drained = not self.thread.is_alive()


# Backwards-compatible alias: patched call sites may keep the old name.
PredictionWorker = SecurePredictionWorker
