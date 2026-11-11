# Ensemble video-level robustness — measured

Real 4-model FP16 ensemble, `ECHOGESTURE_TRACKER=parallel` (4 workers), 30 permitted
val_30 clips, 5 signers, seed 42, corruption implementation v2. 210 real `predict()`
calls total. Model SHA-256 per condition is recorded in each JSON
(`model_sha256` maps all four members).

| Condition | Severity | Scored | Correct | No-hand | Accuracy | vs clean | vs re-encode |
| --- | --- | --: | --: | --: | --: | --: | --: |
| clean | — | 30 | 23 | 0 | 76.67% | — | — |
| **re-encode control** | — | 30 | 23 | 0 | 76.67% | 0.00 | — |
| low_light | 0.5 | 30 | 24 | 0 | 80.00% | +3.33 | +3.33 |
| motion_blur | 0.5 | 30 | 23 | 0 | 76.67% | 0.00 | 0.00 |
| crop | 0.3 | 30 | 20 | 0 | 66.67% | **−10.00** | **−10.00** |
| occlusion | 0.5 | 29 (of 30) | 21 | 1 | 72.41% | **−6.67** | **−6.67** |
| frame_drop | 0.5 | 30 | 23 | 0 | 76.67% | 0.00 | 0.00 |

## The re-encode control, and why it matters

Every corruption rewrites the clip, so a naive comparison could attribute
codec damage to the corruption. The control re-encodes with no corruption at
all: **accuracy is identical (23/30, 76.67 %)**, so the drops above are real
corruption effects, not encoder artifacts.

Caveat stated honestly: re-encoding **flipped 2 of 30 individual predictions**
(`NOON`, `FLIP`) — they cancelled in the aggregate, so net accuracy is
unchanged but the pipeline is not per-clip invariant to a codec round-trip.
On a 30-clip sample one clip is 3.3 pp, so deltas of that size are noise.
Only **crop (−10 pp)** and **occlusion (−6.7 pp)** are clear effects.

## Interpretation

- **Crop is the real failure mode** (−10 pp, 3 clips) and would fail the ≤5 pt
  budget. Cropping pushes hands toward the frame edge, which degrades MediaPipe
  detection before the classifier is involved.
- **Low light is slightly *better*** (+3.33 pp). Not a real improvement — it is
  one clip on a 30-clip sample; treat as noise.
- **Occlusion** is **−6.67 pp** when its single no-hand failure is counted
  wrong (the stricter, submission-safe denominator). Counting only the 29
  clips that produced a prediction would understate it at −4.25 pp.
- Motion blur and frame drop had no measurable effect at these severities.

## What this is not

- Not Dev's landmark-level numbers and not comparable to them.
- Not a substitute for the 1,249-clip accuracy figure.
- 30 clips is a small sample; per-condition deltas of ±3.3 pp are one clip.
