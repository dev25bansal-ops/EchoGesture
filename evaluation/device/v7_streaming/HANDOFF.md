# V7 streaming verification (Agent A04)

## Verified scope

- Directory: `/tmp/echogesture-v7-agents/A04/evaluation/device/v7_streaming/`
- Host: macOS on arm64 (Apple Silicon)
- Runtime: Python 3.11.18, existing EchoGesture virtual environment
- Model: `landmark_classifier_fp16.tflite`
- Model SHA-256: `a012f2b5f7a1110961079f19eb0e28c9abacb95e4d0787352e4157ebca7044a4`
- Fixture: `/tmp/echogesture-real-fixture/0009701491968876486-PULL CONVINCE.mp4`

## Results

### Regression suite

18 tests passed, including real FP16 classifier and real landmark-replay tests.
The suite also verifies overflow accounting, duplicate/out-of-order timestamps,
gap-separated windows, independent next-sign commits, caption expiry,
stop/reset invalidation, failure recovery, protection of active windows, and
stale-result rejection after a reset during inference.

Command:

```bash
/Users/abhijeetmiskin/AppData/EchoGesture/.venv/bin/python -I -m unittest \
  discover -s /tmp/echogesture-v7-agents/A04/evaluation/device/v7_streaming \
  -p 'test_*.py' -v
```

Output: `results/tests.txt` (`Ran 18 tests ... OK`).

### Real sequential video file

Command:

```bash
/Users/abhijeetmiskin/AppData/EchoGesture/.venv/bin/python -I \
  /tmp/echogesture-v7-agents/A04/evaluation/device/v7_streaming/run_streaming.py \
  "/tmp/echogesture-real-fixture/0009701491968876486-PULL CONVINCE.mp4" \
  --out /tmp/echogesture-v7-agents/A04/evaluation/device/v7_streaming/results/file-result.json
```

Final command completed with exit code 0 and emitted valid JSON at
`results/file-result.json`. MediaPipe initialization warnings are in
`results/file-run.stderr`.

| Field | Measured value |
|---|---:|
| Frames streamed | 103 / 103 |
| Prediction | PULLCONVINCE |
| Confidence | 0.6142476201 |
| Warm-up | 0.126458 ms |
| Frame processing p50 / p95 / max | 4.954917 / 7.471271 / 10.201333 ms |
| First-frame-to-caption | 513.288917 ms |
| Sign-end-to-caption | 0.777792 ms |
| Classifier invoke | 0.200791 ms |
| Classifier total | 0.491083 ms |
| Clip-to-result | 596.998833 ms |
| Buffer capacity / retained at caption | 96 / 96 |
| Overflow drops | 7 |
| Out-of-order / duplicate / timestamp-gap drops | 0 / 0 / 0 |

These are sequential file-mode measurements on this host, not live-camera
measurements and not clip p95 from `run_manifest.py`. The first-frame-to-caption
boundary is based on decoded-frame timestamps; no physical camera capture clock
was available.

## Handoff notes

- `run_video_file` uses the existing MediaPipe VIDEO-mode detector and preserves
  its monotonic timestamp invariant with Python `int` timestamps. It accepts
  counted `ok_dropped_oldest` admissions rather than aborting after capacity is
  reached.
- A timestamp gap transitions to `SIGN_HOLD` and rejects the post-gap frame;
  callers must flush/reset and begin a new sign before resubmitting it.
- In-flight classification is tied to the sign ID and state. Reset or stop
  retires the result and it is not published. Caption deadlines are based on the
  runtime monotonic clock.
- Landmark replay deliberately reports unavailable live/frame timing as `None`;
  it never mixes file timestamps with the runtime clock.
- No live camera was present. No packages were installed, no nested agents were
  used, and no historical samples were stored.
- The copied original result is retained as `result_real.json`; its legacy
  interpretation is in `RESULTS_LEGACY.md`. New measurements are under
  `results/`.
- No Git commit was created in this isolated agent directory.
