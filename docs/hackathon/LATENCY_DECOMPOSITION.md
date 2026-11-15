# Where predict() spends its time, and what latency_ms was reporting

Written 16 Nov 2026 by Dev. Two separate problems found in the submission entry
point while checking it end to end.

## 1. `latency_ms` included cold start. Fixed.

`predict()` started its timer *before* `_init_model()`, which loads nine TFLite
interpreters and a MediaPipe graph. Measured on this machine:

| | |
| --- | --: |
| cold call (timer as it was) | **124 s** |
| warm call | ~5 s |
| `_init_model()` alone | 98 ms once warm, far more on a true first import |

So the number a judge would read from `latency_ms` on a fresh process was 124
seconds for work that happens once per process. The timer now starts after
`_init_model()`, and cold start is a separate, one-time cost.

This is a reporting fix, not a speed fix — but it matters because the JSON
contract is what gets scored.

## 2. MediaPipe dominates the remaining time

Per-clip breakdown, warm process, 32 sampled frames:

| stage | time |
| --- | --: |
| video decode + resize | 45–312 ms |
| **MediaPipe hand detection** | **4,300–4,400 ms** |
| feature build (both widths) | 6–8 ms |
| classifier, nine `tflite` members | ~7 ms |
| **total** | **~4,700 ms** |

The classifier — the part this branch spent days improving — is **0.15%** of
end-to-end latency. Everything else is the detector.

That is 135 ms per frame across 32 frames, against the ~23 ms/frame this repo
has been quoting. **These measurements are contaminated**: a `distllm` model
server was running on the same machine and competing for CPU. The figures are
useful as a *relative* breakdown but should not be quoted as absolute latency.

## What this means for the submission

The rubric's latency criterion (≤50 ms per frame) is about the detector, not
the classifier. Optimising the ensemble — the thing all the accuracy work
focused on — cannot move it meaningfully. The levers that would are:

- **fewer frames**: 32 sampled frames at 135 ms each. Dropping to 16 halves the
  detector cost, but changes the model input and would need re-validation.
- **a smaller/faster landmark model**: MediaPipe offers `full` and `lite`
  variants; this repo uses whichever `models/hand_landmarker.task` is.
- **the phone's own detector**: on-device `detectForVideo` in VIDEO mode reuses
  its tracker across frames, which is why the Android app may well be much
  faster than this Python path. That has never been measured.

## Not concluded

I have **not** established on-device latency. These numbers are desktop x86
under CPU contention from another process. The emulator cannot run MediaPipe at
all (no x86_64 native library in Google's AAR), and the phone disconnected
mid-install, so this remains `NOT_MEASURED` in `results.csv`.
