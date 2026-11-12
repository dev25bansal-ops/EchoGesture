# Demo lab (camera-less file-input path)

The `app.offline_demo` tap-to-capture flow needs a live camera on macOS.
Camera access is currently denied on this Mac (TCC), so a live preview has
**not** been verified here. Do not present file-input runs as camera results.

The demo lab is the supported fallback: it runs the repository's **real**
`predict(video_path)`, renders the HUD headlessly from real frames, and writes
a PNG + JSON report. No captions are scripted and no model is mocked.

## Harness self-check (no model, no camera)

```bash
.venv/bin/python -m app.demo_lab --check
```

Verifies: worker encode → real prediction → temp-clip deletion, sanitization
rejecting out-of-range confidence, thread-safe `CaptionState` snapshots and
recovery, HUD shape, and that `model_identity()` names the selected artifact
with a provisional-threshold disclaimer.

## Real file-input render

```bash
.venv/bin/python -m app.demo_lab --video "/path/to/clip.mp4"
# or, with an explicit output directory:
.venv/bin/python -m app.demo_lab --video "/path/to/clip.mp4" --outdir /tmp/demo-lab-out
```

The input clip is read in place — never copied, never retained. Output is
`hud-<clip-stem>.png` and `hud-<clip-stem>.json` inside `--outdir`.

The same path is available without the lab wrapper:

```bash
.venv/bin/python app/offline_demo.py --video "/path/to/clip.mp4" --render-output /tmp/hud.png
```

## What the JSON proves

- `result.prediction` / `top5` / `confidence` / `latency_ms` come from
  `src/main.py::predict`.
- `result.capture_to_result_ms` is wall time from file start to JSON.
- `model.backend`, `model.model.name`, `sha256_prefix`, `bytes`,
  `labels` and `label_count` identify the exact artifact that ran
  (`ECHOGESTURE_MODEL` selects `landmark` by default).
- `live_camera_verified` is always `false` in this path — keep it that way
  until a real camera probe passes on the demo machine.

## Camera path (unverified here)

```bash
.venv/bin/python app/offline_demo.py            # interactive window
.venv/bin/python app/offline_demo.py --headless # one auto-capture, no window
```

Keyboard: **SPACE** capture/cancel, **R** reopen camera / reset after an
error, **Q**/**ESC** quit. If the camera is still denied, follow the
on-screen macOS guidance (System Settings > Privacy & Security > Camera,
then fully relaunch the terminal) or use `--video` above.
