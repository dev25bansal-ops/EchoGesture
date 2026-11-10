# A01 performance_final — integration note

Owner: A01 (Inference Performance Engineer). Everything here lives only under
`evaluation/device/performance_final/`.  `src/`, `models/` and the existing
`evaluation/device/*` labs are read-only inputs.

## What was built

A production-quality, persistent, parallel IMAGE hand tracker plus a benchmark
and a 19-test suite, none of which rewrite the disproven VIDEO/canonicalization
work.  The tracker delegates per-frame detection, handedness slots, presence
flags, thresholds and coordinate conversion to Dev's `src/landmark_frontend.py`
`detect_hands`, so its output is bit-identical to the singleton production path
by construction.  MediaPipe IMAGE mode is stateless per frame, so frames can be
split across independent landmarker instances with no cross-clip tracker state.

## Exact API

```python
from evaluation.device.performance_final.tracker import ParallelImageTracker

tracker = ParallelImageTracker(workers=4)        # workers >= 1, default 4
landmarks, present = tracker.detect(frames, timeout=None)
#   landmarks: (T, 2, 21, 3) float32, canonical [Left, Right] slots
#   present:   (T, 2) float32 presence flags
#   frames: exactly Dev read_frames() output (uint8 256x256x3 RGB)
tracker.close()          # idempotent, thread-safe, does not wait on hung workers
# or:
with ParallelImageTracker(workers=4) as tracker:
    ...
```

`detect_parallel(frames, workers=4, timeout=None)` is a one-shot convenience
that builds, uses and tears down a pool for a single clip.

Behaviour guarantees (all covered by tests):

- **Startup/reuse** — constructs exactly `workers` landmarkers, once; reuses them.
- **Per-clip isolation** — returns fresh arrays; no buffer leakage across clips.
- **Ordered results** — round-robin split is reassembled to input order.
- **Handedness/presence parity** — `np.array_equal` vs `detect_hands`.
- **Recovery** — a worker that raises or exceeds `timeout` is replaced once; a
  second failure promotes to `TrackerError` (bounded, never a hang).
- **Malformed input** — rejects None/oversized/frame-shaped/typed frames and a
  missing task file; empty frame lists return `(0, ...)` empty arrays.
- **Cleanup** — `close()` closes all landmarkers and shuts the executor down
  (`wait=False`), so a hung detector thread cannot block teardown.

## Selection recommendation

Use `ParallelImageTracker(workers=4)` as an **opt-in** faster path, not the
default, matching the existing v8 finding.  On 30 identical permitted
validation clips (the same `v5_accuracy/data/val_30` corpus) the measured
full `predict()` p50/p95 were:

| workers | total p50 (ms) | total p95 (ms) | tracking p50 | peak RSS |
|---|---:|---:|---:|---:|
| baseline singleton | 210.72 | 262.73 | 210.24 | 330 MiB |
| 1 (this pool) | 211.20 | 258.57 | 210.24 | 330 MiB |
| 2 | 115.55 | 141.81 | 115.17 | 381 MiB |
| 4 | 63.16 | 79.66 | 63.59 | 473 MiB |

Parity: **60/60 bit-identical landmarks, 60/60 bit-identical presence, 60/60
identical predictions**.  Speedup ~3.3x at 4 workers, matching the v8 claim
(2.90x) within run variance.  Cold vs warm is negligible (construction is
amortised; ~1 ms delta).  The 50 ms clip-p95 target is still not met; tracking
is the bottleneck regardless of worker count.

**Gate recommendation:** adopt only when the target device has >=4 CPU threads
and ~140 MiB headroom over the baseline (4 workers cost ~+143 MiB).  Do not
promote to default without target-device (phone/XR) evidence, and do not exceed
4 workers: the 3.3x ceiling is bounded by the 32-frame split granularity.

## Reproduction

```bash
.venv/bin/python -m unittest discover -s evaluation/device/performance_final -p 'test_*.py' -v
.venv/bin/python -I evaluation/device/performance_final/benchmark.py \
  --clips evaluation/device/v5_accuracy/data/val_30 \
  --output evaluation/device/performance_final/results \
  --workers 1 2 4 --repeats 2
```

## Files

- `tracker.py` — `ParallelImageTracker` + `detect_parallel` + `TrackerError`.
- `test_tracker.py` — 19 tests (startup/reuse, isolation, ordering, parity,
  recovery, malformed input, repeated calls, cleanup).
- `benchmark.py` — lock-guarded p50/p95/cold/warm/memory/parity benchmark.
- `results/benchmark.json` — the 30-clip measured report above.

## Blockers / not measured

- Target-device (phone/XR) numbers, 8-worker point, parallel decode, GPU delegate.
- The 50 ms clip-p95 target is not met by laptop measurements; no fabrication.