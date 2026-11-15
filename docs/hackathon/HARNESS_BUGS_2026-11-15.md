# Harness bugs found 15 Nov, and what they invalidated

Written 15 Nov 2026 by Dev. Both are in `scripts/ml/robustness.py` and both are
fixed. They are recorded because each one produced a confident, wrong number.

## 1. `--limit 0` silently meant 120

```python
res = run_pixel(..., args.limit or 120, ...)
```

`--limit` defaults to `0`, documented as "every clip". Python treats `0` as
falsy, so `0 or 120` evaluated to `120`. Every "full split" pixel run therefore
scored a 120-clip sample and still printed a PASS/FAIL verdict.

Fixed in two places:

- the call site now passes `args.limit` through unchanged;
- `run_pixel` no longer does `items[:limit]`, which with `limit=0` would have
  scored *nothing*:

```python
items = load_split(split, vocab_idx)
items = items[:limit] if limit else items
```

Verified: `--limit 5` scores 5, `--limit 0` scores all 1,249.

**What it invalidated:** a run I reported as the full-split confirmation was a
120-clip sample. It has been renamed to
`results/pixel_matrix/512letterbox_120clips_meanlogit.json`; the filename it
carried (`robustness_pixel_full.json`) asserted something untrue.

## 2. Every matrix cell was scored with the wrong aggregator

Covered in detail in [AGGREGATOR_FINDING.md](AGGREGATOR_FINDING.md). The short
version: `ensemble_probs` averaged per-member softmax probabilities, while
`src/main.py` and `LandmarkClassifier.kt` both average logits and softmax once.

All four cells of the detector-input matrix were computed with the rejected
aggregator. They are now named for it — `*_meanprob.json` in
`results/pixel_matrix/` — so the filename states the operating point.

## Why both matter together

The matrix was built to answer "does detector resolution fix pixel robustness?"
It answered yes — `downscale_35pct` moves from +28.33 to −0.83 in the 512-squash
cell, a 29-point swing — but that answer rests on a **120-clip sample** scored
with an **aggregator we do not ship**. Two independent errors, either of which
alone would weaken the comparison.

The direction survived both corrections: the genuine full-split run with the
shipped aggregator gives worst **+4.48 PASS**. So the conclusion held, but it
was not established when this file was first written, and `results.csv` carried
the affected rows as `PENDING` until the full run landed. They now carry measured
values (landmark 5.36 FAIL, pixel 4.48 pass).

## 3. The pixel harness was too slow to finish (not a bug, a cost)

`run_pixel` is serial. At 512px, 1,249 clips × 5 corruptions × 32 frames measures
**5.55 s/clip** on this machine, which projects to **~9.6 hours**. A serial run
was 2h20m in and roughly a fifth done when I stopped it.

MediaPipe releases the GIL, so processes scale nearly linearly. I added
`scripts/ml/robustness_pixel_parallel.py`, which is a scheduling change only —
corruptions, model set and aggregator are all imported from `robustness.py`, so
it cannot drift from the serial harness.

**Verified identical**: on 40 clips, 4 workers, parallel and serial agree on
every corruption to the decimal (clean 90.00, blur 87.50, low-light 82.50,
downscale 90.00, jpeg 90.00; worst +7.50 in both). Only then did I use it.

At 6 workers on 28 logical cores the projection drops to roughly 1.6 hours.

## Why the cost is inherent

Each corruption is applied to **pixels**, so MediaPipe must re-detect every
frame for every corruption. 1,249 × 5 × 32 = 199,840 frame detections. There is
no way to cache that, which is why the parallel scheduling is the only real
lever.

## What is still running

Genuine full-split pixel run with the corrected aggregator
(`results/robustness_pixel_full.json`), using the parallel harness.