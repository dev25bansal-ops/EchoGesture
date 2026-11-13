# Edge latency: measured, and what the 50 ms target actually requires

The submission's weakest measured metric. Every number here is measured on
**macOS arm64 laptop CPU** (AI-Edge-LiteRT 2.3.0, XNNPACK). None of it is
target-device evidence.

## The bottleneck is not the model

| Stage | Time |
| --- | ---: |
| Classifier inference (all 4 ensemble members) | **0.36 ms** |
| MediaPipe hand tracking (rest of `predict()`) | **~195 ms** at 1 worker |

Quantizing or shrinking the classifier cannot move the total, because the
classifier is 0.2% of it. The target-device claim that matters is tracking
throughput, not model size.

## Worker scaling closes most of the gap

`predict()` runs MediaPipe IMAGE mode per frame over 32 sampled frames. The
`ECHOGESTURE_TRACKER=parallel` path fans those frames across independent
landmarkers. Measured on one clip, 5 warm calls per configuration:

| Workers | p50 | p95-equivalent |
| --: | --: | --: |
| 1 (default) | 196.4 ms | 275.8 ms (recorded elsewhere, 12 warm clips) |
| 2 | 119.6 ms | 152.5 ms (recorded) |
| 4 | 75.3 ms | 93.9 ms (recorded) |
| 8 | 54.7 ms | — |
| 12 | **51.4 ms** | — |

An 18-core machine reaches ~51 ms median at 12 workers. The ≤50 ms target is
**still not met**, but the gap is now a factor of ~1.03 rather than ~5.5.

## Parallelism does not change predictions

Same 7 clips through 1 worker and 12 workers:

```
10557248501909733-MICROSCOPE  MICROSCOPE1  0.3246   (both)
3049387689711547-GUESS        GUESS1       0.5382   (both)
6523145816470133-WHAT FOR     WHATFOR1     0.6248   (both)
744008724339543-EAT           EAT1         0.5294   (both)
799770954333842-DOG           DOG1         0.1950   (both)
8423017864044668-BASKETBALL   BASKETBALL1  0.3103   (both)
```

Identical top-1 and identical confidence **to four decimal places**. The frames
are independent and each is tracked once, so the result is order-independent by
construction — confirmed empirically rather than assumed.

## Why `results.csv` still says 275.8 ms

The submission row reports the **default singleton configuration**, which is what
an organizer running `predict()` with no environment variables will get. Quoting
the parallel number would describe a configuration they will not use unless they
set it. Both numbers are in `results.csv`'s notes.

## What is not claimed

- **Not** "under 50 ms". The measured best is 51.4 ms p50 at 12 workers on an
  18-core laptop, and the target is a *p95 on the organizer's device*.
- **Not** a phone measurement. A Galaxy-class phone has fewer, slower cores and
  a different thermal envelope. The scaling shape should hold; the constants
  will not.
- **Not** a reason to quantize the classifier. It is already 0.36 ms.

## The honest options, in order

1. **Measure on the target device.** Everything above may be irrelevant if a
   phone's tracking throughput differs enough. This is the only way to know.
2. **Report both configurations** in `results.csv` (already done) and let the
   judge see the parallel path as engineering, not a claim.
3. **Reduce tracked frames.** 32 samples is a training contract; reducing it
   would change the feature time base and invalidate the accuracy evidence. Not
   worth it.
4. **Quantize for its own sake** — no. It would cost 5.2 pt accuracy to save
   time in a stage that takes 0.36 ms.