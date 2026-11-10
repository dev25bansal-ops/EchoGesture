"""A10 V7 offline demo reliability wrappers (isolated).

These modules do NOT modify ``app/offline_demo.py`` or the shared demo.
They wrap the shared primitives to fix concrete demo defects:

* stale inference results overwriting a newer capture's state (sequence
  tags now discard out-of-date worker results),
* ``R`` reopen leaving the previous in-flight prediction able to clobber
  the freshly reset state (sequence tags + worker drain),
* ``PredictionWorker.close()`` racing a busy inference and raising or
  leaking the temporary clip (bounded join + explicit leftover purge),
* HUD text clipped mid-word / overflowing narrow frames (measured
  ``cv2.getTextSize`` fitting instead of fixed character slicing),
* ``--video`` exiting 0 even when the result was an error (wrapper
  returns explicit exit codes) and decoding the clip twice (one decode
  reused for render),
* headless runs exiting on the first transient camera error before any
  real capture happened (explicit ``--captures`` / ``--frames`` budget
  with a bounded wait for results).

Camera status on this machine is still DENIED (macOS TCC). Nothing here
claims live-camera success; the file path is the supported fallback and
``live_camera_verified`` stays ``false`` in every report.
"""

from .reliable import (  # noqa: F401
    EXPECTED_FP16_SHA256,
    CaptureResult,
    ReliableCaptionState,
    ReliableHUD,
    ReliableWorker,
    SessionReport,
    fit_text,
    run_file,
    run_session,
)
from .cli import main  # noqa: F401

__all__ = [
    "EXPECTED_FP16_SHA256",
    "CaptureResult",
    "ReliableCaptionState",
    "ReliableHUD",
    "ReliableWorker",
    "SessionReport",
    "fit_text",
    "run_file",
    "run_session",
    "main",
]
