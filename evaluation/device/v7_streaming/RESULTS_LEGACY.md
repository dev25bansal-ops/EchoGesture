# Legacy streaming measurement (Agent A04)

Source result: `result_real.json`, copied without modification from the legacy
`evaluation/device/streaming_lab/result_real.json` snapshot.

## Measured file

- Fixture: `/tmp/echogesture-real-fixture/0009701491968876486-PULL CONVINCE.mp4`
- Frames streamed: 103 of 103 declared frames at 30.0029 fps
- Prediction: `PULLCONVINCE`
- Confidence: 0.6142476201057434
- Warm-up: 0.129833 ms
- Frame processing: p50 5.070542 ms, p95 7.585779 ms, max 10.263125 ms
- First-frame-to-caption: 526.670125 ms
- Sign-end-to-caption: 0.669583 ms
- Classifier invoke: 0.101625 ms
- Legacy clip-to-JSON wall time: 838.833292 ms
- Backpressure: 103 pushed, 7 dropped on overflow, 96 retained and then cleared
- Out-of-order drops: 0

## Interpretation

The 838.833292 ms legacy value is a full-file wall-clock result, not clip p95
from `evaluation/device/run_manifest.py`. It includes model setup and processing
plus result emission on that legacy host and must not be used as a live-camera
or clip-latency claim. The new runtime reports `clip_to_result_ms` separately as
model setup/warm-up + sequential file processing to commit, before JSON
serialization.

No live camera was available. The measurement is sequential file-frame file-mode
evidence, not camera evidence.
