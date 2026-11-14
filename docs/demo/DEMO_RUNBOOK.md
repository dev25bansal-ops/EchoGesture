# Demo runbook — real runs, exact commands, HUD reference

Status: **offline clip-path only.** There is no camera on this machine and
macOS TCC camera access is denied, so the live-camera path is documented
below but is **unverified here**. Nothing in this file claims a live-camera
demo.

All figures and paths in this runbook come from real runs recorded in
`evaluation/device/demo_final/` on 2026-11-11 with
`ai-edge-litert 2.3.0, XNNPACK CPU` on a macOS arm64 laptop (**not** the
target hardware).

---

## 1. Environment

| Item | Value |
| --- | --- |
| Repo root | `$REPO` |
| Interpreter | `.venv/bin/python -I` (Python 3.11, isolated mode) |
| Model backend | `ECHOGESTURE_MODEL=landmark` (default; 4-model FP16 ensemble) |
| Entry point | `app/v7_demo/cli.py` (run **by path**) |
| Reports | `evaluation/device/demo_final/*_run.json` (stdout), `*_run.err` (stderr) |
| HUD images | `evaluation/device/demo_final/*_hud.png` (640×480) |

### Run by path, not `-m`

`python -I -m app.v7_demo` **fails**: `-I` drops the current directory from
`sys.path`, so `app` is not importable.

```
Error while finding module specification for 'app.v7_demo'
(ModuleNotFoundError: No module named 'app')
```

`app/v7_demo/README.md` still shows `-m` with `/tmp/...` paths; those
instructions are stale. Use the by-path command below.

---

## 2. File demo (the supported path)

```bash
cd $REPO

.venv/bin/python -I app/v7_demo/cli.py \
  --video "evaluation/device/v5_accuracy/data/val_30/5784034175717614-EASY.mp4" \
  --render-output evaluation/device/demo_final/easy_hud.png
```

Optional flags (all default to safe values):

| Flag | Default | Effect |
| --- | --- | --- |
| `--video PATH` | none | File mode. Without it the CLI runs camera mode. |
| `--render-output PATH` | none | Write the rendered HUD frame to this PNG. |
| `--threshold F` | `0.25` | Provisional retry threshold. **Not** validation-calibrated. |
| `--predictor PATH` | `src/main.py` | Predictor module loaded by path. |
| `--camera N` | `0` | Camera mode only. |
| `--seconds F` | `2.0` | Camera capture length. |
| `--headless` | off | Camera mode without a window. |
| `--captures N` | `1` | Camera mode captures before exit. |
| `--result-timeout F` | `30.0` | Camera mode bounded wait for the final prediction. |

### What exit codes mean

| Exit | Meaning |
| --- | --- |
| `0` | The real predictor returned a sanitized `CAPTION` or `RETRY` result. |
| `1` | Load/decode/predict failure, or the HUD PNG could not be written. |
| `2` | Camera mode only: camera unavailable (index wrong, busy, or macOS TCC denied). |

Exit `0` means **a prediction was produced**, not that the prediction is
correct and not that any accuracy target was met.

### Expected output

stdout is one JSON `SessionReport`. Shape (abridged from the real EASY run):

```json
{
  "mode": "file",
  "captures": [{
    "seq": 1,
    "state": "CAPTION",
    "result": {
      "clip_id": "5784034175717614-EASY",
      "prediction": "EASY",
      "top5": ["EASY", "DARK1", "HOW1", "KNIGHT1", "CALENDAR1"],
      "confidence": 0.8976,
      "latency_ms": 614.26,
      "capture_to_result_ms": 663.203542,
      "queue_wait_ms": 0.0
    },
    "stale_count": 0
  }],
  "stale_count": 0,
  "exit_code": 0,
  "live_camera_verified": false,
  "model": { "...": "identity block, see below" },
  "notes": [
    "File-input path. NOT a live-camera claim; macOS camera access is denied on this machine.",
    "HUD written: evaluation/device/demo_final/easy_hud.png",
    "real predict() result surfaced; see captures[0].result"
  ]
}
```

`live_camera_verified` is `false` in every report; never present it
otherwise.

### Real runs on this machine (2026-11-11)

| Clip | Prediction | Confidence | Latency (clip) | Capture→result | Exit | HUD |
| --- | --- | --- | --- | --- | --- | --- |
| `3738312466703404-TEACH.mp4` | `TEACH1` | **0.4221** | 537.24 ms | 583.0 ms | 0 | `teach_hud.png` (301,451 B) |
| `5784034175717614-EASY.mp4` | `EASY` | **0.8976** | 614.26 ms | 663.2 ms | 0 | `easy_hud.png` (362,879 B) |

For reference, calling `src/main.py::predict` directly on the same two files
(no re-encode) gives `TEACH1` @ 0.4715 and `EASY` @ 0.9028. **The demo
lowered confidence slightly and did not change the top-1 label.**

### Why the demo confidence differs from direct `predict()`

`run_file()` samples 32 frames spread across the clip, re-encodes them with
OpenCV `mp4v` into one temporary clip at 30 fps, and predicts once on that
temp clip (the app's capture path), then deletes it. Re-encoding changes
pixel data and frame timing, so confidence shifts. Observed deltas here:
TEACH −0.049, EASY −0.005. **Quote demo-run confidences, not direct-predict
confidences, for anything described as "what the demo showed".**

### stderr is noisy and expected

MediaPipe prints `I0000`/`W0000` lines (fiber init, feedback-tensor note,
`NORM_RECT` note) to stderr on every launch. They are not errors. Redirect
stderr to a file when recording so it does not scroll over the JSON:

```bash
.venv/bin/python -I app/v7_demo/cli.py --video "<clip>" \
  --render-output evaluation/device/demo_final/out.png \
  > evaluation/device/demo_final/out_run.json \
  2> evaluation/device/demo_final/out_run.err
```

---

## 3. Camera / headless mode (unverified here)

```bash
# Headless camera probe
.venv/bin/python -I app/v7_demo/cli.py --camera 0 --headless

# Interactive camera demo (needs a display + camera permission)
.venv/bin/python -I app/v7_demo/cli.py --camera 0 --seconds 2.0
```

On this machine the probe **exits `2`** and prints the macOS guidance
because no camera is available. The camera is probed with real frame reads
(`probe_camera`), not just `isOpened()`, so an open-but-frame-less device
under TCC denial is reported as unavailable rather than as success.

### What a live-camera session would require (macOS, not verified here)

1. A camera (built-in or USB) attached and not in use by another app.
2. **System Settings → Privacy & Security → Camera** → enable the
   application that launches Python (Terminal, iTerm, VS Code, …).
3. Fully quit and relaunch that application — TCC grants apply on next
   launch.
4. If a stale denial persists: `tccutil reset Camera` and relaunch; the
   prompt reappears once.
5. Re-run `--camera 0 --headless`; exit `0` is the first evidence the
   camera path works. Only then try the interactive window.
6. Interactive controls: **SPACE** capture, **R** reopen/reset after an
   error, **Q/ESC** quit.

Do not record a live-camera demo until step 5 returns exit `0` on the
recording machine and the checklist in `DEMO_CHECKLIST.md` is complete.

---

## 4. Environment variables

| Variable | Read by | Values | Default | Notes |
| --- | --- | --- | --- | --- |
| `ECHOGESTURE_MODEL` | `src/main.py`, HUD identity | `landmark`, `s3d` | `landmark` | Any other value **raises** in `src/main.py` (no silent fallback). The HUD's display helper falls back to `landmark` instead — a display-only inconsistency. |
| `ECHOGESTURE_TRACKER` | `src/main.py` | unset/`image`/`singleton`, or `parallel` | `singleton` | `parallel` is **opt-in/experimental**; any other value raises. |
| `ECHOGESTURE_WORKERS` | `src/main.py` | integer ≥ 1 | `4` | Only meaningful with `ECHOGESTURE_TRACKER=parallel`. |

Record the demo on the **default singleton** path unless the shot list
explicitly says otherwise: `ECHOGESTURE_TRACKER=parallel` changes latency,
not the predicted label or the HUD.

---

## 5. HUD reference

The HUD is drawn by `app/v7_demo/reliable.py` `ReliableHUD.render()` (the
width-fitted variant). Actual rendered layout, 640×480:

```
┌────────────────────────────────────────────────────────────┐
│ EchoGesture | OFFLINE | integration not release-certified  │  header
│ TEACH1                                                     │  state/message (colour-coded)
│ CAPTION | landmark:4-model ensemble sha 0e4c392732e7 3544080 B | provisional retry threshold 0.25
│                                                            │
│                    (middle sampled frame)                  │
│                                                            │
│ confidence 0.422 | clip 537.2 ms | capture-result 583.0 ms | queued 0.0 ms
│ Top 5: TEACH1, CHEER, DECIDE1, HALLOWEEN1, PARTY1          │
└────────────────────────────────────────────────────────────┘
```

- **Header** is fixed text: `EchoGesture | OFFLINE | integration not
  release-certified`. Keep it visible on screen — it is the honest status.
- **State** is one of `WAITING` (grey), `PROCESSING` (amber),
  `RETRY` (orange), `CAPTION` (green), `ERROR` (red).
- **Detail line** shows state, backend, model identity (name + 12-hex SHA-256
  prefix + byte size) and the provisional threshold. On the current
  ensemble the identity reads `landmark:4-model ensemble sha 0e4c392732e7
  3544080 B` — i.e. all four members, not a single classifier.
- **Footer** appears only with a result: confidence, clip latency,
  capture-to-result, queue wait, and top-5.

Reading rules for the recording:

- `clip … ms` is the predictor's own clip latency for **one** clip. It is
  **not** a p95.
- `capture-result … ms` includes decode/encode overhead. It is **not** a
  p95 either.
- `confidence` is the softmax of the mean of four members' logits. It is
  **uncalibrated** and is not a probability of correctness.
- The `0.25` threshold is **provisional**, not validation-calibrated.

---

## 6. What the HUD supports vs. what is missing

Implemented in `app/offline_demo.py` + `app/v7_demo/reliable.py`:

- Five states: `WAITING`, `PROCESSING`, `CAPTION`, `RETRY`, `ERROR`.
- Stale-result rejection: `ReliableCaptionState` tags each submission with a
  sequence number and epoch; only the newest sequence is applied. Older
  results are counted in `stale_count` and discarded.
- Reopen/reset epoch bump, so an in-flight result cannot overwrite the
  state after `R`.
- Temp-clip cleanup: captures are encoded into a private `0700` temp dir,
  predicted once, deleted in a `finally`, and the worker purges leftovers on
  close (bounded 5 s wait).
- Model identity display: 4 ensemble members + combined SHA prefix + total
  bytes, per the identity block.
- Width-fitted text so the HUD does not clip mid-word on narrow frames.

Not implemented / caveats:

- **No live preview on this machine.** The interactive window
  (`cv2.imshow`) is never exercised here; only the file path is verified.
- **`RETRY` is only labelled provisional.** The `0.25` threshold has no
  validation calibration; a `RETRY` is a UI hint, not a measured error rate.
- **Identity is display-only.** The HUD hashes the files on disk but does not
  load them; `src/main.py` is the authority on what executes. If the meta
  sidecar and the loaded members diverge, the HUD would not detect it.
- **`ECHOGESTURE_MODEL` display fallback differs from the runtime.** See
  §4.
- **Camera mode is unverified**; every report sets `live_camera_verified:
  false`.

---

## 7. Reproducing the two recorded runs

```bash
cd $REPO

.venv/bin/python -I app/v7_demo/cli.py \
  --video "evaluation/device/v5_accuracy/data/val_30/3738312466703404-TEACH.mp4" \
  --render-output evaluation/device/demo_final/teach_hud.png \
  > evaluation/device/demo_final/teach_run.json \
  2> evaluation/device/demo_final/teach_run.err
echo "exit=$?"   # expect 0

.venv/bin/python -I app/v7_demo/cli.py \
  --video "evaluation/device/v5_accuracy/data/val_30/5784034175717614-EASY.mp4" \
  --render-output evaluation/device/demo_final/easy_hud.png \
  > evaluation/device/demo_final/easy_run.json \
  2> evaluation/device/demo_final/easy_run.err
echo "exit=$?"   # expect 0
```

Predictions are deterministic for a fixed clip and model SHA; confidence may
vary by a few thousandths across runs due to encoder timing.