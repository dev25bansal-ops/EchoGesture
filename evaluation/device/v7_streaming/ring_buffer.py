"""Bounded ring buffer for timestamped landmark frames.

The buffer owns its storage and its counters. Push returns an explicit result
so a slow producer never silently loses data. Consumers can atomically take
one sign window; frames arriving concurrently cannot overwrite an in-flight
window or enter a new sign without a commit.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading


@dataclass(frozen=True)
class PushResult:
    accepted: bool
    dropped: int
    dropped_frame_index: int | None = None


class RingBuffer:
    """Thread-safe fixed-capacity FIFO with drop-oldest backpressure."""

    def __init__(self, capacity: int = 96):
        if isinstance(capacity, bool) or int(capacity) != capacity or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        self._capacity = int(capacity)
        self._buf = deque()
        self._lock = threading.Lock()
        self.pushed = 0
        self.dropped_overflow = 0
        self.dropped_stale = 0
        self.cleared = 0

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)

    def push(self, item) -> PushResult:
        with self._lock:
            dropped = 0
            dropped_frame_index = None
            if len(self._buf) >= self._capacity:
                dropped_frame_index = self._buf.popleft()["frame_index"]
                self.dropped_overflow += 1
                dropped = 1
            self._buf.append(item)
            self.pushed += 1
            return PushResult(True, dropped, dropped_frame_index)

    def snapshot(self):
        with self._lock:
            return list(self._buf)

    def take_all(self):
        """Atomically remove the current window; do not classify held storage."""
        with self._lock:
            items = list(self._buf)
            self._buf.clear()
            return items

    def discard(self) -> int:
        with self._lock:
            n = len(self._buf)
            self._buf.clear()
            self.dropped_stale += n
            self.cleared += n
            return n

    def clear(self) -> int:
        """Backward-compatible explicit discard."""
        return self.discard()

    def clear_before(self, frame_index: int) -> int:
        with self._lock:
            before = len(self._buf)
            self._buf = deque(x for x in self._buf if x["frame_index"] >= frame_index)
            n = before - len(self._buf)
            self.dropped_stale += n
            self.cleared += n
            return n

    def stats(self):
        with self._lock:
            return {
                "capacity": self._capacity,
                "size": len(self._buf),
                "pushed": self.pushed,
                "dropped_overflow": self.dropped_overflow,
                "dropped_stale": self.dropped_stale,
                "cleared": self.cleared,
            }
