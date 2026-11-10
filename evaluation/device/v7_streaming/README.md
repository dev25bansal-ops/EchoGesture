# EchoGesture V7 streaming (Agent A04)

This isolated lab preserves the legacy streaming contract while making its
reliability behavior explicit: bounded backpressure, timestamp continuity,
sign-scoped captions, and separate file-mode timing boundaries.

## Files

| File | Purpose |
|---|---|
| `ring_buffer.py` | Thread-safe bounded FIFO with drop-oldest backpressure. `push` returns `PushResult(accepted, dropped, dropped_frame_index)`; `take_all()` atomically removes one sign window. |
| `capture_states.py` | Explicit `IDLE → CAPTURING → SIGN_HOLD → COMMITTED` transitions and sign-ID allocation. |
| `streaming_runtime.py` | `StreamingSession`, `StreamingClassifier`, sequential video-file runner, and landmark replay runner. |
| `run_streaming.py` | CLI for one real file execution. |
| `test_streaming_lab.py` | Deterministic unit/regression tests plus real-model smoke coverage. |
| `result_real.json` | Copied legacy measured output; interpretation in `RESULTS_LEGACY.md`. |
| `HANDOFF.md` | Final verified commands, host/model details, and measured result. |
| `RESULTS_LEGACY.md` | Measured legacy file-mode output and its interpretation. |

## Reliability semantics

- **Backpressure:** the ring buffer is bounded. When full, `push` drops the oldest retained frame, increments `dropped_overflow`, and returns `PushResult.dropped=1`. `StreamingSession.ingest_landmark_frame` exposes that as `ok_dropped_oldest`, not silent data loss.
- **Timestamp continuity:** timestamps must be non-negative integers and strictly increasing for a single sign. Out-of-order timestamps return `dropped_stale`; repeated timestamps return `dropped_duplicate`. A gap larger than the configured `max_capture_gap_ms` (default 350 ms) moves the session to `SIGN_HOLD`, returns `timestamp_gap`, and does **not** buffer the post-gap frame. Flush or reset and explicitly begin a new sign before resubmitting it. This prevents two disjoint captures from being concatenated as one temporal window.
- **Sign transitions:** the first accepted frame in `IDLE` starts sign 1. `resume()` after `COMMITTED` always allocates a new sign ID and clears buffer, caption, deadline, and last-capture timestamp. Illegal transitions raise.
- **No stale captions:** a caption carries its sign ID, expires after a configurable TTL (default 3 seconds), and is cleared by `begin_sign`, `reset`, `stop`, failure, or expiry. Classification is performed on a taken snapshot; if the sign is reset or retired while the classifier runs, the result returns `None`, increments `discarded_stale_results`, and is not published. No historical landmark/embedding store is involved.
- **Failures:** empty or no-hand flushes clear the current window and caption, record `last_error`, return the session to `IDLE`, and raise. They do not increment `sign_seq`.
- **Model serialization:** even though state transitions are guarded, LiteRT invocations use one interpreter lock. A producer/consumer adapter can serialize frames through the same session, but no concurrent `invoke` is performed.

## Timing boundaries

For file-sequential VIDEO mode, these are all monotonic-clock measurements on the executing host:

- `frame_processing_ms`: per-frame decode/resize/MediaPipe tracking after its decode timestamp.
- `first_frame_to_caption_ms`: first accepted frame → committed caption. File mode only; not a live-camera claim.
- `sign_end_to_caption_ms`: explicit sign-end flush timestamp → committed caption.
- `clip_to_result_ms`: model setup/warm-up + full sequential file processing → commit; JSON serialization is not included.
- Warm-up, feature build, and interpreter invoke are reported separately from frame tracking.

For landmark replay, timestamps are external file metadata. `frame_processing_ms.n` is zero, and first-frame/sign-end caption timings are `None`. Replay never manufactures camera or frame-front-end latency, and timestamps are not subtracted from the runtime monotonic clock.

`clip_to_result_ms` is **not** clip p95 from `evaluation/device/run_manifest.py`: that path uses a different clip sampling contract. Streaming results must not be reported as a clip-latency pass.

## Validation

Run from the isolated directory with the project interpreter installed at `$REPO/.venv/bin/python`:

```bash
$REPO/.venv/bin/python -I -m unittest discover -s . -p 'test_*.py' -v
$REPO/.venv/bin/python -I run_streaming.py \
  "$TMP/echogesture-.../0009701491968876486-PULL CONVINCE.mp4" \
  --out results/file-result.json
```

## Blockers

- No live camera is present. File-mode frame-to-caption numbers are host measurements from decoded frames, not camera capture.
- The existing project interpreter is outside this isolated directory. Only read-only import/invocation of its installed runtime is used; no packages were installed.
- A concurrent producer/consumer queue is intentionally not enabled by default: it would need explicit abort/cancellation policy and a separate bounded event buffer. The session's thread-safe admission and stale-result guards are testable without pretending that the camera pipeline exists.
