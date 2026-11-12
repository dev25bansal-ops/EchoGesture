# Offline demo

## Run on this Mac

From the repository root with the Python 3.11 environment:

```bash
.venv/bin/python app/offline_demo.py
```

Press **SPACE** to capture a two-second isolated sign, **R** to reopen the
camera / reset after an error, and **Q** or **Escape** to quit. The preview
remains responsive while inference runs in a worker thread: each capture is
encoded to one temporary clip inside a worker-owned directory, predicted
once, and deleted before the result is surfaced — no footage is retained.
The real model is selected by `ECHOGESTURE_MODEL`; the default is
`landmark` (`models/landmark_classifier_fp16.tflite`, float16 with float32
IO — not an INT8 release solution). `ECHOGESTURE_MODEL=s3d` selects the old
research baseline.

Camera opening is verified with real frame reads, not just `isOpened()`:
on macOS a camera can report opened while TCC blocks every frame. If the
probe fails the demo exits with the permission steps below.

## Camera permission (macOS)

Camera permission is per-app. If you see "Camera N unavailable":

1. Open System Settings > Privacy & Security > Camera and enable the
   terminal/application that launched Python.
2. Fully quit and relaunch that terminal (a running process keeps the old
   decision), then run the demo again.
3. If it is already enabled, reset the prompt with
   `tccutil reset Camera` — the prompt returns on the next launch — or try
   another `--camera <index>`.

## Without a camera: demo lab (real file-input path)

Camera access is denied on this machine, so the file-input path is the only
verified render. It runs the same real predictor and HUD headlessly:

```bash
.venv/bin/python -m app.demo_lab --video "/path/to/clip.mp4"
.venv/bin/python -m app.demo_lab --check   # model-free harness checks
.venv/bin/python app/offline_demo.py --video "/path/to/clip.mp4" --render-output /tmp/hud.png
```

See `app/demo_lab/README.md` for what the JSON report proves. Never present
file-input output as a live-camera claim.

## HUD and states

The HUD includes waiting, processing, caption, retry and error states,
confidence, top-5, clip latency, capture-to-result latency, queue-wait and
encode timings, plus the selected model name and SHA-256 prefix. Its 0.25
retry threshold is **provisional**, not validation calibrated. No claim of a
measured rejection rate is made. No footage is retained after each temporary
local capture is processed, and no cloud inference is used.

Current limitation: macOS denied camera access during the engineering check,
so a live camera preview has **not** been verified on this machine. The
worker, real clip encoding, cleanup, HUD states and failure handling have
tests. This is a desktop fallback, **not an Android implementation**. No
Android SDK/device was available.

## Test without camera permission

```bash
.venv/bin/python -m unittest evaluation.device.test_offline_demo -v
.venv/bin/python app/offline_demo.py --headless --camera 9 --max-frames 1
```

The second command should fail clearly with the macOS camera guidance above;
it is an error-path check, not a demo success measurement.
