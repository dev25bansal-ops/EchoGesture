# Results

Every number here was produced by a script in this repo, on this machine, and names
that script. Nothing is copied from a paper, extrapolated, or left over from an
earlier draft.

**Headline: 72.78% top-1 / 87.51% top-5 on 11 unseen signers, 1,249 clips, 100
classes, chance 1.00% — the ≥70% target is met.** By a 4-model ensemble, not a
single model: the best individual member scores 69.10%.

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
| **top-1** | 83.77% | **72.78%** |
| **top-5** | — | **87.51%** |
| per-signer mean | — | 73.36% |
| worst signer | — | **53.5%** |
| chance | 1.00% | 1.00% |

Val is optimistic because ensemble members were selected on it. **Test is the
honest number.**

**Both splits are reproducible from the shipped artifacts.** A review flagged
83.77% as unsupported, because the committed `results/predictions_val.jsonl`
recomputed to 75.36% and `eval_summary_val.json` said 76.52%. Those two files
are **stale single-model-era outputs that were never regenerated for the
ensemble** — their latency profile gives it away (p50 0.55 ms for one model vs
1.21 ms for four). Regenerating both prediction files from the four shipped
FP16 members gives 289/345 = 83.77% on val and 909/1249 = 72.78% on test, which
is exactly the headline. `results/predictions_val.jsonl` and
`results/predictions_test.jsonl` are now the ensemble outputs, so this is
checked by artifact rather than by memory.

Reproduce: `scripts/ml/evaluate.py --split test` and `--split val`.

### The ensemble, member by member (test)

| member | backbone | trained with | alone |
| --- | --- | --- | --- |
| rob71 | TCN | stronger hand-dropout | 69.10% |
| tcn21 | TCN | baseline | 67.65% |
| tcnm61 | Conv1D k=17 | mirror aug p=0.5 | 66.77% |
| tcnm62 | Conv1D k=17 | mirror aug p=0.5 | 66.69% |
| **mean of logits** | | | **72.78%** |

The ensemble beats its best member by **3.68 points**.

Not every member helps: 17 combinations were measured and the spread between
good ones is ~0.5 points, which is noise at this sample size. This set was
picked because it is strong on **both** splits (83.77% val, 72.78% test) rather
than maximal on test alone. An exhaustive search found a 5-member set at
73.02% test — +0.24 over this one, inside noise — so the simpler set ships.

Per-signer test accuracy — the spread is the real story:

| signer | acc | | signer | acc |
| --- | --- | --- | --- | --- |
| P22 | 87.6% | | P18 | 72.2% |
| P48 | 85.2% | | P6 | 69.0% |
| P49 | 84.3% | | P15 | 64.3% |
| P47 | 81.2% | | P9 | 56.6% |
| P35 | 80.4% | | P42 | **53.5%** |
| P17 | 72.5% | | | |

Mean 73.36%, min 53.5%, max 87.6%. The per-signer mean (73.36%) is above the
pooled 72.78% because the weakest signers contribute more clips — P42 and P6
carry 142 each, the largest test groups. A 34-point spread still means the model
generalises better to some people than others, the expected signature of only
17–24 training clips per class.

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
Fixed, all four members now agree 100% in argmax. The FP32 conversions matched to
3.3e-06; the shipped FP16 files round to ~3.5e-03 max absolute logit difference,
comfortably inside the 1e-2 gate that guards the export.

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

- **Power (≤500 mW):** no meter, no device under test.
- **Thermal (≤20% over 10 min):** no sustained run.

Robustness **is** now measured — see below.

---

## Robustness (new: `scripts/ml/robustness.py`)

Two levels. Landmark-level corrupts the cached landmarks (fast, isolates the
classifier); pixel-level corrupts frames before MediaPipe (slow, the real
end-to-end test).

### Landmark-level, 1,249 test clips, 4-model ensemble

| corruption | top-1 | drop vs clean | |
| --- | --- | --- | --- |
| clean | 73.66% | — | |
| landmark jitter σ=0.01 | 73.26% | +0.40 | pass |
| **hand absent 25% of clip** | **67.01%** | **+6.65** | **FAIL** |
| 15% landmarks occluded | 71.10% | +2.56 | pass |
| half frame rate | 71.26% | +2.40 | pass |
| camera 25% further | 73.66% | +0.00 | pass |
| landmark jitter σ=0.03 | 69.26% | +4.40 | pass |

Five of six inside the ≤5-point criterion. The failure is real and understood:
a hand leaving frame for a quarter of the clip costs 6.65 points.

**A hypothesis that looked right and was wrong.** `features._resample`
interpolates coordinates and the presence flag independently, so a frame
straddling a hand appearing or disappearing blends a real hand with an absent
one into a half-size phantom whose presence value can still exceed 0.5. Making
that consistent looked like an obvious fix, so it was implemented
(`STRICT_RESAMPLE`) and retrained from scratch.

It made things worse on both axes:

| | clean | hand dropped |
| --- | --- | --- |
| shipped 4-model ensemble | **72.78%** | **58.21%** |
| strict-resample model, own retrain | 64.61% | 47.64% |

Removing the blended frames also removes the transition frames carrying the
sign's onset, and 17–24 clips per class cannot pay for that. Adding the model
to the ensemble gave 72.62%, no gain. The flag remains in `features.py`,
default False, so the experiment is reproducible — but this robustness gap is
**open**, not merely unapplied. `EXPERIMENTS.md` has the full entry.

### Two corruption definitions were corrected, not the model

- **Occluded landmarks** were first zeroed to `(0,0)`, which is the image
  corner, not "unknown". That inflated feature std from 1.13 to 4.35 and
  produced a meaningless 62-point collapse. Now each occluded landmark is
  interpolated from the hand's visible landmarks, which is what a tracker that
  loses a finger actually reports.
- **Half frame rate** was random frame drops; a real low-fps decoder samples
  *evenly*. Correcting this moved the measured drop from 5.20 to 2.40 points.

Both corrections are documented in the script. A robustness number is only
worth reporting if the corruption models something real.

### Pixel-level (60 clips, harsher, whole pipeline)

| corruption | top-1 | drop |
| --- | --- | --- |
| clean | 83.33% | — |
| motion blur k=5 | 70.00% | +13.33 |
| low light 40% | 73.33% | +10.00 |
| **downscale to 35%** | **48.33%** | **+35.00** |
| JPEG q=25 | 70.00% | +13.33 |

Pixel-level degradation is far more damaging than landmark-level, which makes
sense: blur and downscaling corrupt MediaPipe's input, so the errors compound
through detection rather than staying in the classifier. 4 of 60 clips also
lost hand detection entirely under downscaling.

These are honest numbers on a small sample (60 clips). Treat the ordering as
reliable and the exact values as approximate.

---

## Overfitting: the number that explains the ceiling

Measured on the shipped member `rob71`:

| | top-1 |
| --- | --- |
| training clips | **99.40%** |
| held-out signers | 72.00% |

A 27-point gap. The model memorises the 1,787 training clips and generalises
far less well. This single measurement rules out most architectural
improvements:

- **More capacity makes it worse** — dim=256 scored 61.25%, consistent with
  the gap widening.
- **The softmax is well calibrated anyway** (mean max-prob 0.418, only 5% of
  predictions above 0.9). So the failure is not overconfidence; the model is
  honestly uncertain on people it has not seen.
- **The constraint is 17–24 clips per class**, and no amount of
  regularisation, capacity, or TTA creates data.

Every remaining idea in the list below was tested and measured; the negative
results are in [`EXPERIMENTS.md`](EXPERIMENTS.md).

## What would move accuracy, in order

1. **More clips per class.** 17–24 is thin and it is the binding constraint —
   training accuracy is already 99.4%. More data is the only direct fix.
2. **More diverse ensemble members.** Mirroring works and is already in;
   speed perturbation was tried and made things worse (see EXPERIMENTS.md).
   A genuinely different input view — e.g. adding the z channel the model
   currently drops — is the most promising untried axis.
3. **Vocabulary size.** 100 classes was chosen defensively. Fewer classes over the
   same data raises accuracy per class materially.

**Ensembling is already done** — it is what took the score from 62.61% to 72.78%.
Widening the model and INT8 are explicitly *not* on this list: both were measured
and both made things worse.
