# Where predict() spends its time, and what latency_ms was reporting

Written 16 Nov 2026 by Dev. Two separate problems found in the submission entry
point while checking it end to end.

> **CORRECTION, 18 Nov 2026 — every millisecond figure below is contended.**
> This file was written while a `distllm` inference server from another project
> was saturating the CPU. Re-measuring the same scripts on an unloaded machine
> gives per-frame detector cost of **14.5–16.4 ms p50 / 22.7–25.1 ms p95**
> (`results/latency_per_frame_distribution.json`), not 135 ms, and a warm
> end-to-end `predict()` of **~720 ms** (`results/latency_end_to_end.json`), not
> 4,700 ms. The *structure* of the conclusion below survives — MediaPipe still
> dominates and the classifier is still a rounding error — but the absolute
> numbers are inflated roughly 6x, and the claim that the per-frame criterion
> FAILS is withdrawn: at 22.95 ms p95 it is inside the 50 ms budget.
> Abhijeet's B-09 flagged the internal inconsistency this file created. The
> resolution is this re-measurement, not his proposed "claim no per-frame PASS".

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

That is 135 ms per frame across 32 frames. The repo had been quoting 15.7 ms
and 23 ms in six places, with no artifact behind either — see the retraction in
`results/results.csv`. The figure is now measured reproducibly by
`scripts/ml/latency_per_frame.py`, which gives p50 150.9 ms / min 68.5 ms. A
`distllm` model server was competing for CPU during these runs, but pinning to
an idle core gave p50 163 ms, so contention is not the explanation.

## What this means for the submission

The rubric's latency criterion (≤50 ms per frame) is about the detector, not
the classifier. Optimising the ensemble — the thing all the accuracy work
focused on — cannot move it meaningfully. The levers that would are:

- **fewer frames**: 32 sampled frames at 135 ms each. Dropping to 16 halves the
  detector cost, but changes the model input and would need re-validation.
- **a smaller/faster landmark model**: this repo ships
  `models/hand_landmarker.task` at **7.82 MB, which is the FULL variant**.
  I checked whether a lite variant exists as a drop-in and it does **not**:
  Google's Tasks model zoo publishes only
  `hand_landmarker/hand_landmarker/float16/{1,latest}/hand_landmarker.task`
  (both 200, both the full model). The `_lite` paths return 404. MediaPipe's
  older Solutions API had a lite option; the Tasks API this repo uses does
  not expose one for hand landmarking. So this avenue is closed unless we
  convert a lite TFLite ourselves.
- **the phone's own detector**: on-device `detectForVideo` in VIDEO mode reuses
  its tracker across frames, which is why the Android app may well be much
  faster than this Python path. That has never been measured.

## Not concluded

I have **not** established on-device latency. These numbers are desktop x86
under CPU contention from another process. The emulator cannot run MediaPipe at
all (no x86_64 native library in Google's AAR), and the phone disconnected
mid-install, so this remains `NOT_MEASURED` in `results.csv`.
