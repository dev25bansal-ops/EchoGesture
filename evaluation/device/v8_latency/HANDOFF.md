# V8 latency handoff (isolated copy, lifecycle fix + identical-input benchmark)

Scope: `evaluation/device/v8_latency/` only. No edits outside this directory,
no installs, no nested agents. Production `src/` and `models/` untouched.
`V7_STRATEGY` default is still `baseline`; `parallel_4` remains opt-in.

## Lifecycle defects found in v7_candidates.py and fixed here

1. No shutdown path - singletons + parallel pools (up to 7 landmarkers) were
   never closed; native handles leaked until process exit.
2. No atexit hook - cleanup relied on GC at interpreter exit.
3. Race in lazy creation - `_singleton` / `_get_parallel_pool` / `_initialise`
   had no lock; concurrent `predict()` could double-create trackers.
4. Partial-pool leak - if tracker k of N failed to build, trackers 0..k-1 were
   leaked (not closed, not stored).
5. `video_ts` monotonic counter never reset on teardown.

Fix (prediction-neutral, no strategy/constant change):
- `shutdown()` closes all singleton/pool landmarkers, clears state, resets
  `video_ts`; idempotent, thread-safe, swallows close-time errors. Next
  `predict()` lazily recreates (soft drain, for test isolation).
- `threading.RLock` guards `_initialise`, `_singleton`, `_get_parallel_pool`,
  `video_cached` timestamp update, and `shutdown()`.
- `atexit.register(shutdown)` once at first initialise.
- Pool build is incremental with cleanup of already-built trackers on failure.
- `benchmark.py` / `verify_landmark_parity.py` call `shutdown()` in `finally`.
- Tests: 4 original contract tests preserved + 6 new lifecycle tests
  (default-is-baseline, idempotent shutdown, uninitialised no-op, close-error
  swallow, partial-pool cleanup, concurrent singleton = 1 instance). 10/10 pass.

## Identical-input measurement (30 permitted real clips, x3 repeats = 90 runs/strategy)

Clips: `/tmp/echogesture-validation-30/manifest.csv` (sha256 004e5eff...).
Same 32-frame sampling, 256px, same FP16 classifier, same process for paired run.
Hardware: Apple M5 Pro, 18 CPUs, macOS 26.3, MediaPipe 1.2.2, litert 2.3.0.

Isolated single-strategy processes (headline, clip-to-JSON = full `predict()` dict):

| Strategy | n | total p50 (ms) | total p95 (ms) | total mean | tracking p50 | correct |
|---|---:|---:|---:|---:|---:|---:|
| baseline (IMAGE singleton) | 90/90 | 231.56 | 281.67 | 234.08 | 215.66 | 66/90 |
| parallel_4 (persistent pool) | 90/90 | 79.78 | 96.09 | 81.85 | 65.00 | 66/90 |

Speedup (identical inputs): **2.90x total p50, 2.93x total p95, 3.32x tracking p50**
from isolated runs. Paired same-process run (90 pairs): baseline total p50
231.83 / p95 285.17 vs parallel_4 79.38 / 97.56; mean paired delta -152.73 ms.
Decode p50 unchanged (~14.4 ms both); features/classifier unchanged.

Peak memory (ru_maxrss, process-wide, bytes): baseline-only 269 MB vs
parallel_4-only 407 MB (+138 MB for 3 extra landmarkers); paired process with
both pools alive 452 MB. Tradeoff for the 2.9x speedup.

## Parity (all on the same 30 identical clips)

- Landmarks: `np.array_equal` full `(32,2,21,3)` + `(32,2)` present arrays:
  **30/30 bit-identical, max_abs_diff 0.0** (`results/landmark_parity_parallel4.json`).
- Predictions (paired, 90 pairs): **90/90 same top-1, 90/90 same exact top-5**.
- Cross-process: paired-vs-isolated prediction+top5 agree **90/90 both strategies**.
- Accuracy identical: 66/90 both (73.3%); same model, same inputs.
- Consistent with prior v7 numbers on same manifest/model hashes
  (v7 baseline p50 225.88 / parallel_4 83.58); v8 p50 231.56 / 79.78 is the
  same effect within run-to-run variance.

## Failure handling verified

- Missing / empty / garbage / truncated video: both strategies raise
  `ValueError` ("No valid frames decoded"); corrupt inputs never return a
  fabricated prediction.
- Invalid strategy arg and bogus `V7_STRATEGY` env raise `ValueError`.
- `predict()` default (no env) runs baseline; `V7_STRATEGY=parallel_4` runs
  parallel_4 with same top-1 on the smoke clip.
- `shutdown()` -> `predict()` recreates singleton and succeeds; double
  `shutdown()` is a no-op.

## Integration decision

**Keep default = baseline. Offer parallel_4 as opt-in only** (e.g. via
`V7_STRATEGY=parallel_4`), gated on the target device having enough RAM for
~140 MB extra tracker footprint and >=4 CPU threads. Do not promote to default
without target-device evidence: laptop p95 (96 ms) is still well above a 50 ms
budget, and memory headroom + thread availability are device-specific.

Blockers / not measured here: target-device (phone/XR) numbers, 8-worker point,
parallel decode, GPU delegate. Streaming is out of scope (clip p50/p95 only).

## Files

- `v7_candidates.py` - v7 logic + lifecycle fix (default unchanged).
- `benchmark.py` - + `shutdown()` in finally, + `peak_memory` (ru_maxrss).
- `test_v7_latency.py` - 10 tests (4 preserved + 6 lifecycle).
- `verify_landmark_parity.py` - + `shutdown()` in finally.
- `results/paired_30x3/`, `results/baseline_only/`, `results/parallel4_only/`
  (`benchmark.json` + `predictions.jsonl` each), `results/landmark_parity_parallel4.json`,
  `results/summary.json`, `results/failure_fixtures/`.
