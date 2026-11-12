# Demo checklist — pre-record and post-record verification

Supersedes the earlier integration-era checklist in this path: that version
pinned the single-model classifier
(`landmark_classifier_fp16.tflite`, `a012f2b5…`) and a `PULLCONVINCE` 0.7507
caption. The shipped runtime is now the **4-model FP16 ensemble**; the old
pins are stale. Its still-open camera/Android items are carried forward at
the end.

Any unchecked box means **do not record** that shot. Never tick a box for a
clip that was never run.

---

## A. Pre-record — environment

- [ ] Working directory is the repo root:
      `cd $REPO`
- [ ] Interpreter is Python 3.11:
      `.venv/bin/python -I -c "import sys; print(sys.version)"` → `3.11.x`
- [ ] Branch and commit recorded for the closing shot:
      `git rev-parse --short HEAD` (evidence files were cut at `efaa323`)
- [ ] No other agent is mid-run on `models/` (otherwise SHAs can shift
      under you).

## B. Pre-record — weights present and SHAs match

- [ ] All ensemble members, the detector, labels and meta exist:

```bash
ls -l models/landmark_ens_{tcn21,tcnm61,tcnm62,rob71}_fp16.tflite \
      models/hand_landmarker.task models/landmark_labels.json \
      models/landmark_model_meta.json
```

- [ ] SHAs match the shipped/evidence set:

```bash
shasum -a 256 \
  models/landmark_ens_tcn21_fp16.tflite \
  models/landmark_ens_tcnm61_fp16.tflite \
  models/landmark_ens_tcnm62_fp16.tflite \
  models/landmark_ens_rob71_fp16.tflite \
  models/hand_landmarker.task \
  models/landmark_labels.json
```

  Expected prefixes:

  | File | SHA-256 prefix |
  | --- | --- |
  | `landmark_ens_tcn21_fp16.tflite` | `112279bc8609` |
  | `landmark_ens_tcnm61_fp16.tflite` | `01a261d8eb6c` |
  | `landmark_ens_tcnm62_fp16.tflite` | `27058a85e032` |
  | `landmark_ens_rob71_fp16.tflite` | `08417e72da59` |
  | `hand_landmarker.task` | `fbc2a30080c3` |
  | `landmark_labels.json` | `e78f5028e97e` |

- [ ] Meta lists exactly four members and matches the loaded set:
      `models/landmark_model_meta.json` → `members` = the four files above.
- [ ] `ECHOGESTURE_MODEL` is unset or `landmark`; record the value used.
- [ ] `ECHOGESTURE_TRACKER` is unset (default singleton) unless the shot
      explicitly uses `parallel`; `ECHOGESTURE_WORKERS` only with `parallel`.

## C. Pre-record — demo runs and exit code

- [ ] Run the two reference clips and confirm **exit 0** each:

```bash
.venv/bin/python -I app/v7_demo/cli.py \
  --video "evaluation/device/v5_accuracy/data/val_30/5784034175717614-EASY.mp4" \
  --render-output evaluation/device/demo_final/easy_hud.png \
  > evaluation/device/demo_final/easy_run.json \
  2> evaluation/device/demo_final/easy_run.err; echo "exit=$?"

.venv/bin/python -I app/v7_demo/cli.py \
  --video "evaluation/device/v5_accuracy/data/val_30/3738312466703404-TEACH.mp4" \
  --render-output evaluation/device/demo_final/teach_hud.png \
  > evaluation/device/demo_final/teach_run.json \
  2> evaluation/device/demo_final/teach_run.err; echo "exit=$?"
```

- [ ] EASY run reports `EASY` @ 0.8976, `CAPTION`, `exit_code: 0`.
- [ ] TEACH run reports `TEACH1` @ 0.4221, `CAPTION`, `exit_code: 0`.
- [ ] No `ERROR:` line on stderr. The `I0000`/`W0000` MediaPipe lines are
      normal and are **not** errors.
- [ ] `live_camera_verified` is `false` in both reports. Keep it that way.

## D. Pre-record — HUD legibility

- [ ] Open the rendered PNGs at recording resolution and read all four
      lines without zooming:

```bash
.venv/bin/python -I -c "import cv2; \
print(cv2.imread('evaluation/device/demo_final/easy_hud.png').shape)"
# expect (480, 640, 3)
```

- [ ] Header `EchoGesture | OFFLINE | integration not release-certified` is
      on screen and not cropped.
- [ ] Identity line reads `landmark:4-model ensemble sha 0e4c392732e7
      3544080 B` — all four members, **not** a single classifier.
- [ ] The state word and message are high-contrast (green `CAPTION`).
- [ ] Footer shows `confidence`, `clip … ms`, `capture-result … ms`,
      `queued … ms`, and `Top 5: …` without mid-word clipping.
- [ ] Capture the actual on-screen recording area, not a screenshot of a
      terminal over the HUD.

## E. Pre-record — offline confirmed

- [ ] No network stack is used by the demo. Run it with all network denied
      and require exit 0 and the same label:

```bash
sandbox-exec -p '(version 1)(allow default)(deny network*)' \
  .venv/bin/python -I app/v7_demo/cli.py \
  --video "evaluation/device/v5_accuracy/data/val_30/5784034175717614-EASY.mp4" \
  > /tmp/offline.out 2> /tmp/offline.err; echo "exit=$?"
```

  Verified on 2026-11-11: `exit=0`, same `EASY` @ 0.8976.

- [ ] No network client libraries are even installed in the venv
      (recorded: none of `requests`, `urllib3`, `aiohttp`, `httpx`).
- [ ] Wi-Fi/Ethernet is off during recording (defence in depth), or the
      sandboxed run above is shown on camera.
- [ ] Show that no capture is retained: after a run, the temp capture dir
      is gone. `PredictionWorker` deletes the temp clip in `finally` and
      purges leftovers on close.

## F. Pre-record — claims hygiene

- [ ] No live-camera shot is planned. Camera is unavailable; TCC denied.
- [ ] No Android shot is planned. It is not implemented/tested on a device.
- [ ] Every number spoken matches `SHOT_LIST_FINAL.md` and its source row.
- [ ] The 50 ms target is described as **not met** (274.2 ms singleton,
      99.8 ms parallel-4).
- [ ] Crop robustness is described as **failing** (−10.0 pp vs ≤5 pp).
- [ ] Test accuracy is described with its caveat: not a pristine blind
      test.
- [ ] Confidence is described as **uncalibrated**.
- [ ] The 0.25 retry threshold is described as **provisional**.
- [ ] The demo re-encode delta is disclosed if confidences are compared to
      direct `predict()`.

## G. Post-record — verify what was actually captured

- [ ] Video duration is within the strictest limit: **≤ 3:00** (target
      2:50). The slides also mention 2–3 min and 5 min; 2:50 satisfies all.
- [ ] Every on-screen prediction is traceable to a `demo_final/*_run.json`
      from the same clip and model SHA.
- [ ] The commit SHA shown on the closing card equals `git rev-parse HEAD`
      at record time.
- [ ] Model SHA shown in the video matches §B.
- [ ] No frame shows a live camera feed or an Android device.
- [ ] No caption was edited over a clip it does not belong to.
- [ ] The `OFFLINE | integration not release-certified` header is visible
      in the recognition shots.
- [ ] Audio claims match the shot list; no unmeasured value slipped in
      (power, thermal, INT8, target-device latency, continuous translation).
- [ ] The recording and its JSON/HUD artifacts are archived together.

---

## Blocked — do not tick until genuinely verified

- [ ] A consenting signer who was **not** in training.
- [ ] Camera permission works in the actual launch environment
      (`--camera 0 --headless` exits `0`).
- [ ] Real camera preview and SPACE capture are visible.
- [ ] `RETRY` and `ERROR` camera states are exercised on camera.
- [ ] No raw footage is retained unintentionally.
- [ ] Demo runs on a physical Android device before any Android claim.

---

## Known findings (document, do not fix here)

Reported from `app/offline_demo.py` / `app/v7_demo/reliable.py`; `app/` is
owned by another agent.

1. `app/v7_demo/README.md` documents `python -m app.v7_demo`, which **fails
   under `-I`** (`ModuleNotFoundError: No module named 'app'`), and shows
   stale `/tmp/echogesture-v7-agents/...` paths. Use the by-path command.
2. `ECHOGESTURE_MODEL` display handling diverges from the runtime: the HUD
   helper (`model_identity`) silently falls back to `landmark` for any unknown
   value, while `src/main.py` raises `ValueError`. A typo would show a
   landmark identity while failing at runtime.
3. HUD model identity is display-only: it hashes files on disk but does not
   load them, so it cannot detect a mismatch between the meta sidecar and
   the members `src/main.py` actually executes.
4. `app/v7_demo/reliable.py` defines
   `EXPECTED_FP16_SHA256 = "a012f2b5…"`, the **single-model**
   `landmark_classifier_fp16.tflite` SHA. It is re-exported by
   `app/v7_demo/__init__.py` but **never compared anywhere**, so it is a
   dead pin and a footgun for anyone who assumes it guards the ensemble.
   The real integrity check is the per-member SHA list in §B.
5. Camera mode is unverified on this machine; every report sets
   `live_camera_verified: false`.