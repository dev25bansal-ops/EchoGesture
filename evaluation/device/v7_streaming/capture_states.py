"""Explicit capture/sign states for the streaming session.

IDLE -> CAPTURING -> SIGN_HOLD -> COMMITTED is one sign. A new sign must be
started explicitly with begin_sign() (or resume()); a caption is not accepted
as evidence for a later sign. COMMITTED captions have a display deadline and
are invalidated on reset, stop, or the next sign. Failed windows are cleared
without incrementing sign_seq, then return to IDLE for a recoverable retry.
"""

from __future__ import annotations

IDLE = "IDLE"
CAPTURING = "CAPTURING"
SIGN_HOLD = "SIGN_HOLD"
COMMITTED = "COMMITTED"
STOPPED = "STOPPED"
ERROR = "ERROR"

_ALLOWED = {
    IDLE: {CAPTURING, STOPPED, ERROR},
    CAPTURING: {SIGN_HOLD, STOPPED, ERROR, IDLE},
    SIGN_HOLD: {COMMITTED, STOPPED, ERROR, IDLE},
    COMMITTED: {CAPTURING, STOPPED, IDLE},
    STOPPED: {IDLE},
    ERROR: {IDLE, STOPPED},
}


class CaptureStateError(RuntimeError):
    pass


class CaptureSession:
    def __init__(self):
        self._state = IDLE
        self._sign_seq = 0

    @property
    def state(self) -> str:
        return self._state

    @property
    def sign_seq(self) -> int:
        return self._sign_seq

    def transition(self, target: str) -> str:
        if target not in _ALLOWED.get(self._state, set()):
            raise CaptureStateError(f"illegal transition {self._state} -> {target}")
        self._state = target
        return self._state

    def begin_sign(self) -> int:
        """Allocate and start a new sign, or return the active sign's ID."""
        if self._state in (IDLE, COMMITTED):
            self._state = CAPTURING
            self._sign_seq += 1
        elif self._state not in (CAPTURING, SIGN_HOLD):
            raise CaptureStateError(f"cannot start a sign from {self._state}")
        return self._sign_seq

    def reset(self) -> str:
        if self._state not in (STOPPED, ERROR, COMMITTED, CAPTURING, SIGN_HOLD):
            raise CaptureStateError(f"cannot reset from {self._state}")
        self._state = IDLE
        return self._state
