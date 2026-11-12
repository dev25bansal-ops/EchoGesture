# V7 latency lab — persistent-pool parallel IMAGE tracking

Owner: A01 (Latency Architect), working in `$TMP/echogesture-.../A01`.
Only files under `evaluation/device/v7_latency/` were created.
Original checkout, Dev feature code, `src/main.py`, and all `models/` are untouched.
**No production default is changed.** No strategy is merged.

Date measured: 8 Nov 2026.

## Bottom line

A **real, exact-parity speedup was found**. Splitting the 32 IMAGE-mode frames
round-robin across a persistent pool of independent MediaPipe `HandLandmarker`
instances in threads gives a **2.7× tracking speedup (4 workers)** and a
**1.7× total clip speedup**, with **bit-identical landmarks on all 30 clips**
and **60/60 top-1 / 60/60 top-5 prediction parity** across 2 repeats.

Prior agents (V4) measured the same mechanism in a microprofile (1.81× median
speedup, bit-exact landmarks) but **rejected** it because the full-clip wrapper
created trackers *per clip* (construction cost swamped the parallel gain).
The fix here is a **persistent pool**: trackers built once and reused across
clips. Construction cost is amortized to zero across the 30-clip run.
V4 explicitly flagged this as "a persistent pool could change that result,
but feeding that through persistent MediaPipe IMAGE instances correctly was
not proven in this session." This lab proves it.

| Strategy | n | Total p50 / p95 (ms) | Mean total | Paired Δ vs baseline (mean) | Same top-1 | Same top-5 | Correct |
|---|---:|---:|---:|---:|---:|---:|---:|
| **baseline** (IMAGE singleton) | 60 | 227.2 / 278.5 | 231.9 | — | — | — | 44/60 |
| baseline, isolated (3-repeat) | 90 | **225.88 / 277.30** | 230.34 | — | — | — | 66/90 |
| `parallel_2` (persistent pool, 2 threads) | 60 | **132.4 / 162.1** | 135.1 | −96.8 | 60/60 | 60/60 | 44/60 |
| `parallel_4` (persistent pool, 4 threads) | 60 | **79.6 / 96.4** | 81.6 | −150.4 | 60/60 | 60/60 | 44/60 |
| `video_cached` (control) | 60 | 216.9 / 256.9 | 220.2 | −11.7 | 48/60 | 8/60 | 44/60 |
| `video_fresh` (control) | 60 | 503.8 / 601.6 | 515.8 | +283.9 | 48/60 | 8/60 | 44/60 |

Isolated single-strategy runs (cleanest headline numbers):

| Strategy | n | Total p50 / p95 | Mean total | Tracking p50 / p95 | Correct |
|---|---:|---:|---:|---:|---:|
| baseline (90) | 90 | 225.88 / 277.30 | 230.34 | 210.60 / 262.78 | 66/90 (73.3%) |
| parallel_2 (90) | 90 | 133.19 / 162.39 | 136.62 | 118.39 / 147.03 | 66/90 (73.3%) |
| parallel_4 (90) | 90 | 83.58 / 99.98 | 85.39 | 68.44 / 84.16 | 66/90 (73.3%) |

## Why this works

IMAGE mode is **stateless per frame**: `lm.detect(frame)` does not carry
temporal state between calls. Distributing frames round-robin across N
independent landmarker instances is therefore exact-parity by construction.
The GIL is released inside MediaPipe's native `detect()` call, so threads
run genuinely in parallel on Apple M5 Pro's 18 CPUs (19 hardware threads).

Speedup scaling (persistent pool, 90-clip isolated):

- 1 worker (baseline): tracking p50 210.60 ms
- 2 workers: tracking p50 118.39 ms → **1.78×**
- 4 workers: tracking p50 68.44 ms → **3.08×**

Not fully linear (4 workers gets 3.08× not 4×) due to thread/contention
overhead and decode being serial, but still a real 3× tracking gain.

## Prediction parity — three levels, all proven

1. **Landmark level (bit-exact):** `np.array_equal` on the full
   `(32,2,21,3)` landmark array + `(32,2)` presence array for **all 30 clips**
   (parallel_4 vs baseline singleton). Result in
   `results/landmark_parity_parallel4.json`: 30/30 identical, max_abs_diff 0.0.
   Because this operates on the raw MediaPipe output before `build_features`,
   it proves the downstream feature tensor and classifier input are identical.

2. **Prediction level (60 paired observations):**
   - parallel_2 vs baseline: 60/60 same top-1, 60/60 same exact top-5.
   - parallel_4 vs baseline: 60/60 same top-1, 60/60 same exact top-5.

3. **Accuracy level:** All strategies score identically (44/60 = 73.3% in
   paired run, 66/90 = 73.3% in isolated runs). parallel_4 is not a
   different model; it is the same model with the same inputs.

## Stage breakdown (isolated 3-repeat baseline vs parallel_4)

| Stage | Baseline mean | parallel_4 mean | Δ |
|---|---:|---:|---:|
| video_open_decode (32 frames) | 15.98 ms | — | serial in both |
| landmark_tracking | 214.02 ms | 68.23 ms | **−145.79 ms** |
| feature_preparation | ~0.17 ms | ~0.17 ms | unchanged |
| model_inference (FP16) | ~0.10 ms | ~0.10 ms | unchanged |
| output_postprocessing | ~0.03 ms | ~0.03 ms | unchanged |
| **Total** | **230.34 ms** | **85.39 ms** | **−144.95 ms (−1.7×)** |

Tracker pool construction cost is amortized: 4 trackers built once at import,
not per clip. V4's per-clip-construction wrapper was 4.8× slower; the
persistent pool eliminates that cost entirely.

## Production-path confirmation

`src/main.py` via `run_manifest.py --profile` (authoritative path):
- 30/30 successful, 0 failures, accuracy 22/30 = 73.3%.
- Warm calls: p50 **241.46 ms**, p95 **294.48 ms**.
- These match the in-process baseline (225.88 / 277.30) within measurement
  overhead, confirming fidelity.

## What did NOT help (controls)

- `video_cached`: −11.7 ms mean paired delta, but 48/60 top-1 parity
  (reproduces prior result: 12/60 top-1 flips). Rejected as a drop-in.
- `video_fresh`: +283.9 ms mean (fresh construction per clip). Rejected.
- Neither video strategy is integrated. Both are included so the dominance
  claim is re-derivable on THIS hardware.

## Why prior agents missed this

V4 measured 1.81× speedup in the microprofile (`profile_parallel.py`) with
bit-exact landmarks, but the full-clip wrapper created trackers per clip by
architecture, making it 4.8× *slower* end-to-end. The persistent-pool
lifecycle (build once, reuse across clips) was explicitly listed as
unproven future work in the V4 README. This lab is the proof.

## Constraints respected

- **Streaming is separate** from clip evaluation. This lab measures full-clip
  p50/p95 only. No streaming numbers are claimed.
- **Production default unchanged.** `src/main.py` SHA remains
  `c695d78b1b1980e3b95b5357e0549385638eda3274a76eba479c6f3dcddf8141`.
  `parallel_4` is opt-in only via `V7_STRATEGY=parallel_4`.
- **No frame budget change.** 32 sampled frames, 256×256, identical sampling.
- **No installs.** No nested agents.
- **Model hashes verified:** FP16 `a012f2b5…ca7044a4`, INT8 `9d24640a…3700b`,
  hand task `fbc2a300…cde1` — all match assignment values exactly.

## Hardware and environment

Recorded in every `benchmark.json` (`environment`, `hardware` blocks):

- Platform: macOS 26.3 (`Darwin 25.3.0`), machine `arm64`, Apple **M5 Pro**
- CPUs: 18 (hw.ncpu), RAM: 25,769,803,776 bytes (~24 GiB)
- Python: 3.11.18 (Clang 21.0.0)
- Runtime: `ai-edge-litert` 2.3.0 (XNNPACK CPU, runtime default threads)
- MediaPipe: 1.2.2; OpenCV: 5.0.0; NumPy: 2.4.6

Laptop numbers only. **Not target-device evidence.**

## The 50 ms budget

parallel_4 total p50 = 83.58 ms, p95 = 99.98 ms. Still above 50 ms p95 on
this hardware. The remaining serial cost is decode (~16 ms) + the parallel
tracking floor (68 ms at p50). A 50 ms budget would require either:

- Parallelizing decode as well (streaming decode across threads), **or**
- A fundamentally faster tracker (not available in this experiment scope),
- **Or** inference on a faster device (unmeasured here).

This lab's contribution: the tracker is no longer 82% of total time
(parallel_4 tracking is 80% of 85 ms total, but the absolute floor is
lower — 68 ms vs 214 ms). The dominant remaining cost is now split between
tracking (68 ms) and decode (16 ms). Further speedup would need to attack
decode or use more workers (8 was not tested — see "not tested" below).

## Not tested

- **8 workers:** Not benchmarked. Would likely give diminishing returns given
  4-worker efficiency is 3.08/4.0 = 77%. Worth trying if the target has many
  cores, but left untested to avoid resource pressure during measurement.
- **Parallel decode:** Decode is serial in all strategies.
- **GPU delegate:** Known to fail (V5 measured `Delegate.GPU` construction
  failure on this build). Not retested.
- **FP16→INT8 classifier swap:** V5 measured 0.022 ms difference. Irrelevant.

## Reproduction commands

All from `$TMP/echogesture-.../A01`. Venv interpreter lives in the
original checkout; nothing else in the original is touched.

### Contract tests

```bash
$REPO/.venv/bin/python -I \
  -m unittest discover -s evaluation/device/v7_latency -p 'test_*.py' -v
# Ran 4 tests ... OK
```

### Main paired sweep (60 pairs per comparison, 5 strategies)

```bash
PYTHONDONTWRITEBYTECODE=1 \
  $REPO/.venv/bin/python -I \
  evaluation/device/v7_latency/benchmark.py \
  $TMP/echogesture-.../manifest.csv \
  --output-dir evaluation/device/v7_latency/results/full_30x2 \
  --repeats 2 --warmup 1 \
  --strategies baseline,parallel_2,parallel_4,video_cached,video_fresh
```

### Isolated runs (90 clips each, single-strategy process)

```bash
PYTHONDONTWRITEBYTECODE=1 \
  $REPO/.venv/bin/python -I \
  evaluation/device/v7_latency/benchmark.py \
  $TMP/echogesture-.../manifest.csv \
  --output-dir evaluation/device/v7_latency/results/baseline_only \
  --repeats 3 --warmup 1 --strategies baseline

PYTHONDONTWRITEBYTECODE=1 \
  $REPO/.venv/bin/python -I \
  evaluation/device/v7_latency/benchmark.py \
  $TMP/echogesture-.../manifest.csv \
  --output-dir evaluation/device/v7_latency/results/parallel4_only \
  --repeats 3 --warmup 1 --strategies parallel_4

PYTHONDONTWRITEBYTECODE=1 \
  $REPO/.venv/bin/python -I \
  evaluation/device/v7_latency/benchmark.py \
  $TMP/echogesture-.../manifest.csv \
  --output-dir evaluation/device/v7_latency/results/parallel2_only \
  --repeats 3 --warmup 1 --strategies parallel_2
```

### Landmark bit-exact parity verification

```bash
PYTHONDONTWRITEBYTECODE=1 \
  $REPO/.venv/bin/python -I \
  evaluation/device/v7_latency/verify_landmark_parity.py
# 30/30 clips bit-identical landmarks + present flags
```

### Authoritative production-path profile (`src/main.py`)

```bash
ECHOGESTURE_MODEL=landmark \
  $REPO/.venv/bin/python -I \
  evaluation/device/run_manifest.py \
  $TMP/echogesture-.../manifest.csv \
  --output-dir evaluation/device/v7_latency/results/run_manifest_profiled \
  --profile \
  --labels models/landmark_labels.json \
  --model models/landmark_classifier_fp16.tflite \
  --predictor src/main.py \
  --feature-config models/landmark_model_meta.json
```

### Using the wrapper as a drop-in (NOT enabled by default)

```bash
V7_STRATEGY=parallel_4 \
  PYTHONPATH=evaluation/device/v7_latency \
  $REPO/.venv/bin/python -I \
  evaluation/device/run_manifest.py \
  $TMP/echogesture-.../manifest.csv \
  --output-dir /tmp/v7-out \
  --predictor evaluation/device/v7_latency/v7_candidates.py \
  --labels models/landmark_labels.json \
  --model models/landmark_classifier_fp16.tflite
```

`V7_STRATEGY` defaults to `baseline`. `parallel_4` is the recommended opt-in.

## Files delivered

- `evaluation/device/v7_latency/v7_candidates.py` — 5-strategy wrapper with persistent pool.
- `evaluation/device/v7_latency/benchmark.py` — paired serial benchmark.
- `evaluation/device/v7_latency/test_v7_latency.py` — 4 contract tests (pass).
- `evaluation/device/v7_latency/verify_landmark_parity.py` — bit-exact proof script.
- `evaluation/device/v7_latency/README.md` — this file.
- `evaluation/device/v7_latency/results/full_30x2/benchmark.json` + `predictions.jsonl`
  — main 60-pair sweep.
- `evaluation/device/v7_latency/results/baseline_only/benchmark.json` + `predictions.jsonl`
  — clean isolated baseline (90 clips).
- `evaluation/device/v7_latency/results/parallel4_only/benchmark.json` + `predictions.jsonl`
  — clean isolated parallel_4 (90 clips).
- `evaluation/device/v7_latency/results/parallel2_only/benchmark.json` + `predictions.jsonl`
  — clean isolated parallel_2 (90 clips).
- `evaluation/device/v7_latency/results/landmark_parity_parallel4.json`
  — bit-exact landmark proof (30/30).
- `evaluation/device/v7_latency/results/run_manifest_profiled/benchmark.json` + `predictions.jsonl`
  — authoritative production-path run of `src/main.py`.

Nothing under `src/`, `models/`, or any `evaluation/device/` directory outside
`v7_latency/` was touched. No nested agents were launched. No packages were
installed.
