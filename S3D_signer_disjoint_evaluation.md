# S3D-Conv1D — independent evaluation (measured 2026-11-07)

## What was tested
The shipped INT8 weights from `Samuel-KIMPINDE/S3D-Conv1D-for-SLR-for-Microcontrollers` (MIT),
run unmodified on a **signer-disjoint** held-out set, using the paper's exact preprocessing.

- **Model:** `ptq_100_int8.tflite` (904,256 B) and `s3d_conv1d_qat_int8.tflite` (901,200 B)
- **Runtime:** `ai-edge-litert==2.3.0`, XNNPACK CPU delegate (no TensorFlow needed)
- **Preprocess:** OpenCV decode → BGR2GRAY → 64×64 → /255 → linspace resample to T=24 (pad by repeat)
- **Input int8[1,24,64,64,1]**, scale 0.00392156886, zp −128; output int8[1,100]

## Result 1 — off-distribution (the honest number)

**ASL Citizen** official participant-disjoint test split, restricted to glosses in the model's
100-class vocabulary: **n = 1,705 videos, 11 participants, 98 of 100 classes present.**

| Model | Top-1 | vs chance |
|---|---|---|
| PTQ INT8 | **1.47%** (25/1705) | 1.02% chance → 1.44×, P≈0.049 |
| QAT INT8 | **1.00%** (17/1705) | 1.02% chance → indistinguishable |

Per-participant spread is flat: 0.0%–2.5%, no signer is meaningfully better.
Top-5 on a 341-video sample: **5.87% vs 5.10% chance** — no usable ranking signal either.

## Result 2 — positive control (proves the pipeline is not broken)

Same script, same weights, same preprocessing, **WLASL100 clips from signers present in training**:

| Set | Result |
|---|---|
| 12 largest training signers, readable clips | **17/17 = 100.0%** top-1 |
| Per-signer breakdown | 5/5, 4/4, 2/2, then 1/1 ×5 — no partial failures |

48 of 65 WLASL files were dead YouTube downloads ("moov atom not found"), a known WLASL issue,
not a pipeline fault. Every decodable in-distribution clip was classified correctly.

**Conclusion:** the 1.47% is genuine cross-signer domain shift, not a preprocessing bug.
The model is near-perfect on its own distribution and at chance on a new signing population.

## Why this happens (mechanism, not speculation)
- Input is **raw grayscale 64×64 pixels** — no pose abstraction, no signer normalization.
  Every nuance of background, clothing, skin tone, camera, and framing is in the feature.
- Training split was **70/20/10 stratified random, not signer-disjoint** (paper states this
  explicitly). The network learned signer identity as much as sign shape.
- Cross-dataset evidence inside the paper agrees: WLASL100 98.96% → SemLex100 82.48% (−16.5 pp).
  Our measurement shows the same failure continuing to a full collapse on a third population.
- The paper's own worst-class F1 on SemLex is ≈0.20 — near-total failure on some classes even
  in its own within-distribution claim.

## What this means for EchoGesture
1. **The shipped weights are not a usable baseline for the evaluator's held-out set.** If the
   organizer's test clips come from unseen signers (highly likely — WLASL/How2Sign are
   multi-signer), 1.0–1.5% is what to expect, not 98.6%.
2. **The architecture + QAT recipe remain valuable.** 883 KB, INT8-only, 26 ms on a Celeron,
   MIT-licensed, with reproducible training scripts. The *method* transfers; the *weights* don't.
3. **Any honest submission must train signer-diversely.** The reference paper's headline is a
   split artifact; do not repeat it.
4. **Where a landmark pipeline wins:** MediaPipe produces signer-normalized joint coordinates,
   which is exactly the invariance S3D's pixel front-end lacks. This is the strongest evidence
   yet that our original pose-first instinct was right — it was rejected earlier for *latency*
   and *accuracy ceiling*, both of which remain true, but it has the invariance property that
   matters for unseen signers.

## Recommended path (revised)
- **Train our own classifier on a genuinely signer-disjoint split.** Use S3D-Conv1D's
  architecture and QAT recipe (MIT, reproducible, INT8-only, tiny) but with signer-aware
  splitting and signer-diverse augmentation.
- **Consider a hybrid input:** grayscale pixels for shape + MediaPipe landmarks for
  signer-invariant pose, fused at the classifier. Both branches are cheap; the union addresses
  the exact failure measured here.
- **Report the honest number.** A measured 45–65% signer-disjoint beats an unreproducible 98.6%
  and matches what the rubric's Top-1 line actually rewards.
- **In the deck, cite this evaluation.** "We independently evaluated the published SOTA tiny
  model on a signer-disjoint split and found it collapses to chance; we then built X" is a far
  stronger story than adopting an unvalidated number.

## Artifacts
- `E:\samsung\.scratch\eval_research\aslcit_run\full\s3d_aslcitizen_predictions.csv` (1,705 rows)
- `E:\samsung\.scratch\eval_research\aslcit_run\full\s3d_aslcitizen_result.json`
- `E:\samsung\.scratch\eval_research\aslcit_run\qat\` (same for QAT)
- `E:\samsung\.scratch\eval_research\wlasl_signer_split\eval_wlasl_signers.py` (positive control)
- Scripts: `aslcit_run\scripts\eval_s3d_aslcitizen.py`, `..._v2.py`