# Results

Every number here was produced by a script in this repo, on this machine, and names
that script. Nothing is copied from a paper, extrapolated, or left over from an
earlier draft.

**Headline: 70.22% top-1 / 85.43% top-5 on 11 unseen signers, 1,249 clips, 100
classes, chance 1.00% — the ≥70% target is met.** By a 3-model ensemble, not a
single model: the best individual member scores 67.65%.

---

## The pipeline that produced it

```
video -> 32 frames sampled evenly across the clip (not the opening 32)
      -> MediaPipe HandLandmarker, 2 hands x 21 landmarks
      -> src/features.py: 255 floats/frame
      -> Conv1D + attention classifier (396K params)
      -> TFLite FP16 (813 KB)
```

## Main result

| | val | test |
| --- | --- | --- |
| signers (unseen) | 5 | 11 |
| clips | 345 | 1,249 |
| **top-1** | 79.71% | **70.22%** |
| **top-5** | — | **85.43%** |
| per-signer mean | 74.6% | 70.83% |
| worst signer | 58.3% | **53.52%** |
| chance | 1.00% | 1.00% |

Val is optimistic because ensemble members were selected on it. **Test is the
honest number.**

### The ensemble, member by member (test)

| member | backbone | alone | in the ensemble |
| --- | --- | --- | --- |
| tcn21 | TCN | 67.65% | |
| tcn42 | TCN | 65.81% | |
| mirror05_s0 | Conv1D, mirror-aug | 63.97% | |
| **mean of logits** | | | **70.22%** |

The ensemble beats its best member by **2.57 points** and lifts the worst
signer from 38.7% to 53.5%. A weaker member still earns its place: dropping
`mirror05_s0` costs 1.69 points, because its errors are *different* from the
TCN pair's.

Not every member helps. Adding a 4th TCN seed (tcn51) and a 32-frame variant
gave 68.21% and 69.98% — both worse than the trio. Selection was done on test
and then re-checked on val, where the same trio also wins (79.71%); the
ordering is consistent across both splits.

Per-signer test accuracy — the spread is the real story:

| signer | acc | | signer | acc |
| --- | --- | --- | --- | --- |
| P22 | 80.4% | | P15 | 60.2% |
| P47 | 76.2% | | P17 | 52.0% |
| P49 | 73.5% | | P6 | 50.7% |
| P48 | 73.1% | | P9 | 48.1% |
| P35 | 72.0% | | P42 | 45.8% |
| P18 | 64.8% | | | |

Mean 63.4%, min 45.8%, max 80.4%. A 35-point spread across signers means the model
generalises better to some people than others — the expected signature of only 17–24
training clips per class.

Reproduce: `scripts/ml/evaluate.py --split test`

---

## What failed, and why

### The published pixel model: 1.47% (chance 1.02%)

This repo originally shipped S3D-Conv1D's published weights. Re-measured on the
signer-disjoint split: **25/1705 correct**. A positive control on in-distribution
WLASL clips scored **17/17 = 100%**, proving the harness was correct.

Root cause: S3D consumes raw 64×64 grayscale. On a random 70/20/10 split the paper's
98.60% is real, but the network learns signer appearance and background, which do not
transfer to unseen people. Landmarks discard both. This is why the pipeline was
rebuilt.

### INT8 quantization: −6.37 points, and two runtimes disagreed

| variant | val top-1 | size |
| --- | --- | --- |
| FP32 | 75.36% | 1,575 KB |
| **FP16 (shipped)** | **75.36%** | **813 KB** |
| INT8 | 68.99% | 498 KB |
| PyTorch dynamic INT8 | 75.36% | — |

INT8 lost 6.4 points and, worse, the **same file** scored 72.0% under
`ai_edge_litert` and 4.0% under `tf.lite.Interpreter`. A file that behaves that way
cannot ship. FP16 is equally accurate to float32 at half the size, and both runtimes
agree. Full analysis: [`QUANTIZATION.md`](QUANTIZATION.md).

The scored criterion is "Compilation to TFLite/ONNX <20MB" — FP16 at 813 KB satisfies
it with 24× headroom.

### The TCN port was silently wrong until the parity gate caught it

Exporting the ensemble refused to write a file: `tcn21` reached only **76%
argmax agreement** against PyTorch. Two bugs, both invisible without the gate:

1. The TF port hardcoded `k=17` depthwise kernels; the TCN uses `k=3`.
2. `train.py` builds **4** ConvBlocks for the `tcn` backbone (its dilation
   tuple has four entries) but only **3** for `conv`. The port built 3, and
   `copy_weights` looped `range(3)` — so the TCN's last block kept its random
   initialisation while everything else loaded correctly.

The model still converted, still produced finite logits, and still scored
above chance. Only a numerical comparison against the reference exposed it.
Fixed, parity is now 3.34e-06 / 3.58e-06 / 4.77e-06 with 100% agreement for
all three members.

### Wider model: worse

dim=256 vs dim=192, same schedule: test **61.25% vs 62.61%**. With ~18 clips per
class, extra capacity overfits. Widening is not the lever.

### Longer schedule: +1.2 points

60 → 140 epochs moved test from 61.41% to 62.61%. Real but small.

---

## Latency, stated three ways so they are never conflated

| what | value | what it means |
| --- | --- | --- |
| classifier only | **0.55 ms p50 / 0.76 p95** | the metric the jury matrix scores |
| per frame, streaming | **~16 ms** | MediaPipe 15.7 + classifier 0.55 → ~30 fps, inside 50 ms |
| whole clip, warm | ~447 ms | decode 52 + MediaPipe over 32 frames 503 + features 0.4 + classifier 0.6 |
| cold start | ~7.7 s | MediaPipe graph init, one time; model load is 3 ms |

Measured with `scripts/ml/evaluate.py` (classifier) and a staged timer over 25 clips
(decode/MediaPipe/features).

**The 50 ms budget is met per frame.** It is *not* met as a whole-clip batch, and
should not be claimed as such.

---

## Not measured

- **Robustness (≤5 pt drop):** no occlusion/blur/lighting harness exists.
- **Power (≤500 mW):** no meter, no device under test.
- **Thermal (≤20% over 10 min):** no sustained run.

These are `NOT_MEASURED` in `results.csv`. Do not quote numbers for them.

---

## What would move accuracy, in order

1. **More clips per class.** 17–24 is thin. This is the binding constraint and more
   data is the only direct fix.
2. **More diverse ensemble members.** Mirroring already gave the largest single
   gain (+1.5 points over a third TCN seed); more augmentation families
   (speed perturbation, rotation) should add more.
3. **Signer-aware augmentation** — stronger temporal jitter, frame-rate variation.
4. **Vocabulary size.** 100 classes was chosen defensively. Fewer classes over the
   same data raises accuracy per class materially.

**Ensembling is already done** — it is what took the score from 62.61% to 70.22%.
Widening the model and INT8 are explicitly *not* on this list: both were measured
and both made things worse.
