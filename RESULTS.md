# EchoGesture — measured results

Every number here was produced by a script in this repository on 7 Nov 2026,
or is explicitly marked as unmeasured. Nothing is copied from a paper without
saying so.

## Accuracy

| Measurement | n | Result | Chance | Script |
|---|---|---|---|---|
| In-distribution positive control (WLASL100, training signers) | 17 | **100.0%** | — | `evaluation/eval_wlasl_signers.py` |
| Signer-disjoint, PTQ INT8 (ASL Citizen official test) | 1,705 | **1.47%** | 1.02% | `evaluation/eval_asl_citizen.py` |
| Signer-disjoint, QAT INT8 (ASL Citizen official test) | 1,705 | **1.00%** | 1.02% | `evaluation/eval_asl_citizen.py` |
| Signer-disjoint, top-5 (PTQ, 341-clip sample) | 341 | **5.87%** | 5.10% | inline, see README |

The positive control is the load-bearing check: it proves the harness is not
broken. Without it, 1.47% would be indistinguishable from a preprocessing bug.

## Latency (dev laptop, Windows, x86 CPU, 4 threads)

| Stage | p50 | p95 | p99 | n |
|---|---|---|---|---|
| TensorFlow Lite invoke only | 0.75 ms | 1.01 ms | 1.15 ms | 200 |
| Quantise + invoke | 2.1 ms | 3.2 ms | — | 40 |
| Resize + gray + normalise | 4.1 ms | 7.1 ms | — | 40 |
| **Video decode (full file)** | **83.4 ms** | **158.6 ms** | — | 40 |
| **End-to-end `predict()`** | **54.1 ms** | **117.8 ms** | **146.3 ms** | 60 |

Decode dominates. The model is under 1% of the budget.

### Decode optimisations tested

| Configuration | p50 | p95 |
|---|---|---|
| Full-res, all frames (current) | 57.8 ms | 132.5 ms |
| Capped at 240p | 41.8 ms | 112.5 ms |
| Capped at 120p | 39.5 ms | 109.0 ms |
| 240p + 12 sampled frames | 38.9 ms | 92.1 ms |
| 240p + 8 sampled frames | 40.4 ms | 94.1 ms |
| Seek to 24 frames (`CAP_PROP_POS_FRAMES`) | 549.8 ms | 1376.6 ms |

Seeking is dramatically worse than sequential decode — do not use it.

## Size

| Artefact | Bytes | MB |
|---|---|---|
| `models/s3d_conv1d_qat_int8.tflite` | 901,200 | 0.86 |
| `models/ptq_100_int8.tflite` | 904,256 | 0.86 |

Budget is ≤20 MB. Using 0.9 MB.

## Not measured

These appear in the hackathon's `results.csv` schema and must be filled before
submission. They are **not** currently measured, and earlier drafts of this
repository contained invented values for them:

| Metric | Status | What it needs |
|---|---|---|
| `accuracy_drop_robustness` | **not measured** | Script applying seeded occlusion / low-light / blur / crop / frame-drop to a val set, then a paired comparison. The harness does not exist yet. |
| `avg_power` | **not measured** | A physical device plus either a Monsoon meter or Android `dumpsys batterystats` over a fixed workload. |
| `thermal_drift_10min` | **not measured** | A 10-minute sustained inference loop on the target device, sampling p95 latency or FPS every 30 s. |

Previous drafts claimed 2.56 pt / 285 mW / 4.2% for these. The 2.56 figure was
the paper's SemLex cross-dataset delta, mislabelled as an occlusion test; the
power and thermal figures had no source at all. They have been removed.

## Reproducing

```bash
pip install -r src/requirements.txt

# in-distribution control
python evaluation/eval_wlasl_signers.py

# signer-disjoint evaluation (needs ASL Citizen, ~46 GB, range-extracted)
python evaluation/eval_asl_citizen.py
```
