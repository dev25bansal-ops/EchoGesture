# A10 V7 offline demo — isolated wrappers

All code lives only under:

- `/tmp/echogesture-v7-agents/A10/app/v7_demo/`
- `/tmp/echogesture-v7-agents/A10/evaluation/device/v7_demo/`

Nothing in the shared demo (`app/offline_demo.py`, `app/demo_lab/`) or the
original checkout was modified. No GUI was launched; no packages installed;
no nested agents were used.

## What was fixed (in the wrappers, wrapping shared primitives)

1. **Stale-result race** (`ReliableCaptionState` + `ReliableWorker`).
   Every submission gets a monotonically increasing sequence; only the
   newest sequence for the current epoch is applied. Older results are
   counted as `stale` and discarded.
2. **Reopen clobber.** `reopen()` bumps the epoch, so any in-flight
   prediction submitted before the reopen cannot overwrite the fresh
   state.
3. **Worker close racing a busy inference.** `ReliableWorker.close()`
   waits a bounded time (`close_timeout`, default 5s), then explicitly
   purges any leftover temp clips before closing the underlying worker.
   Second close is idempotent.
4. **HUD text clipped mid-word / overflow.** `ReliableHUD.render()`
   measures each line with `cv2.getTextSize` (`fit_text()`) and shrinks
   the font scale until it fits the frame width, instead of fixed
   65/86-char slicing.
5. **Double decode + exit-0-on-error.** `run_file()` decodes the clip
   once, writes one temp clip, predicts once, deletes it, renders the
   width-fitted HUD from the already-decoded frames, and returns exit
   0 only when the model produced a sanitized `CAPTION`/`RETRY` result.
6. **Headless early exit.** `run_session()` waits (bounded,
   `--result-timeout`, default 30s) for a real submitted prediction to
   complete instead of exiting on the first transient read error.

## How to run

```bash
# Camera-less real prediction (supported path on this machine)
PYTHONPATH=/tmp/echogesture-v7-agents/A10 .venv/bin/python -m app.v7_demo \
  --video "/tmp/echogesture-validation-30/0283886564670357-WHAT FOR.mp4" \
  --render-output /tmp/v7-hud.png

# Headless camera probe (will print TCC guidance + exit 2 while denied)
PYTHONPATH=/tmp/echogesture-v7-agents/A10 .venv/bin/python -m app.v7_demo \
  --camera 0 --headless

# Tests
$REPO/.venv/bin/python -I \
  evaluation/device/v7_demo/test_v7_demo.py -v
```

## Verified results (this machine)

| Check | Result |
| --- | --- |
| V7 test suite | **16/16 pass** |
| Shared `test_offline_demo.py` (unmodified) | **17/17 pass** |
| Real file prediction (FP16, SHA pinned) | `CAPTION`, exit 0 |
| Camera probe | TCC denied, exit 2, guidance printed |

See `RESULTS.md` for the exact real-file output.
`live_camera_verified` is `false` in every report. Nothing here claims
live-camera success.
