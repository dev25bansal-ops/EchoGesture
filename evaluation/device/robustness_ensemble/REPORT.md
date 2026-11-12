# Ensemble robustness on real video (val_30)

Measured with the shipped 4-model FP16 landmark ensemble through the real `src/main.py` `predict()` entry point on the labelled `val_30` clips. Corruptions reuse `evaluation/device/corruptions.py` (version 2) at the lab default severities and a fixed seed.

## Run configuration

- Clips: 30 (`val_30`, sorted by clip_id, deterministic prefix)
- Signers: 5 (P12, P21, P26, P39, P5)
- Seed: 42; per-clip seeds derived via `derive_clip_seed(seed, index, clip_id)`
- Tracker: parallel (4 workers), parity-verified on 2 clips
- Runtime: 54.86 s for 210 predict calls (7 conditions x 30 clips)
- Manifest SHA-256: `004e5effb7e6c19b515fdd00b048459fff236d8b35c6b7f7b49e91a1efa5d618`

## Model SHAs (measured artifacts)

- `landmark_ens_rob71_fp16.tflite`: `08417e72da59150ac9adf749e6324c30fb001d4a401394e32f125d3276d1ab21`
- `landmark_ens_tcn21_fp16.tflite`: `112279bc86090470b82b20b510746a946220322c25ec84cb3706a538c73dc22a`
- `landmark_ens_tcnm61_fp16.tflite`: `01a261d8eb6c2c4d8b501f0ce1a36db338365d17fddabc63528a7941b1390f63`
- `landmark_ens_tcnm62_fp16.tflite`: `27058a85e032bee3af444fb1538ff0bf32fd38fa2b01ce01a2f25c3a428d20b3`

## Per-condition results

`corrupted %` (eligible) counts every clip, with a no-hand failure counted wrong. `scored %` excludes clips where the pipeline produced no prediction. The pp drop uses the eligible denominator, so it is the deployment risk. The re-encode control decodes and re-encodes each clip with no pixel change; the `vs re-encode` column therefore removes the mp4 round-trip from the corruption effect.

| Condition | Severity | N | Clean % (eligible) | Corrupted % (eligible) | Corrupted % (scored) | pp drop vs clean | pp drop vs re-encode | no-hand | wrong | other errors | Budget |
|---|---|---|---|---|---|---|---|---|---|---|---|
| clean | - | 30 | 76.7 | - | 76.7 | - | - | 0 | 7 | 0 | - |
| re-encode control | - | 30 | 76.7 | 76.7 | 76.7 | 0.0 | 0.0 | 0 | 7 | 0 | - |
| low_light | 0.5 | 30 | 76.7 | 80.0 | 80.0 | -3.3 | -3.3 | 0 | 6 | 0 | PASS (5.0 pt) |
| motion_blur | 0.5 | 30 | 76.7 | 76.7 | 76.7 | 0.0 | 0.0 | 0 | 7 | 0 | PASS (5.0 pt) |
| occlusion | 0.5 | 30 | 76.7 | 70.0 | 72.4 | 6.7 | 6.7 | 1 | 8 | 0 | FAIL (5.0 pt) |
| crop | 0.3 | 30 | 76.7 | 66.7 | 66.7 | 10.0 | 10.0 | 0 | 10 | 0 | FAIL (5.0 pt) |
| frame_drop | 0.5 | 30 | 76.7 | 76.7 | 76.7 | 0.0 | 0.0 | 0 | 7 | 0 | PASS (5.0 pt) |

## Failure-stage split

For each corrupted clip whose result is not correct, the pipeline either produced no hand at all (detection failure) or produced a wrong label. The clean condition is shown for reference.

| Condition | non-correct clips | no-hand (detection) | wrong prediction | flips correct->wrong | flips wrong->correct |
|---|---|---|---|---|---|
| clean | 7 | 0 | 7 | - | - |
| low_light | 6 | 0 | 6 | 1 | 2 |
| motion_blur | 7 | 0 | 7 | 1 | 1 |
| occlusion | 9 | 1 | 8 | 3 | 1 |
| crop | 10 | 0 | 10 | 4 | 1 |
| frame_drop | 7 | 0 | 7 | 1 | 1 |

## Budget verdict

Budget: at most 5.0 percentage points of top-1 loss versus clean, with failures counted wrong (eligible denominator).

- Within budget: low_light, motion_blur, frame_drop.
- Exceeds budget: occlusion, crop.

## Limits

- N=30 with 5 signers; per-condition differences of 1 clip are 3.3 pt. Treat single-clip differences as noise.
- One severity per corruption (the lab default), one seed. This is a point measurement of the shipped ensemble on real video, not a severity sweep.
- Corrupted clips are re-encoded to mp4 before prediction while the clean clip is read from source. The `re-encode control` row (decode + re-encode, no pixel change) measures that path difference directly, and `pp drop vs re-encode` attributes any remaining loss to the corruption itself.
- A negative pp drop means the corrupted variant scored above clean on this sample; at N=30 that is one or two clips and is within sampling noise.
