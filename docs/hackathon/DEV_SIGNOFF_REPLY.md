# Dev → Abhijeet · response to the sign-off request

Answers [DEV_SIGNOFF_REQUEST.md](../release/DEV_SIGNOFF_REQUEST.md). Every claim
below was re-measured on this machine against the shipped artifacts, not
recalled. Where the audit was right, it says so; where it was wrong, it gives the
measurement that settles it.

**Headline: his branch is shippable.** It builds, it runs, and it reproduces the
headline from a clean extraction.

---

## Item 6 — the 83.77% discrepancy. **The audit found a real defect, and its diagnosis was wrong.**

The audit is correct that **no artifact supported 83.77%**. That is my fault, and
the cause is worse than a typo:

`results/predictions_val.jsonl` and `results/eval_summary_val.json` were
**stale single-model-era files that I never regenerated when the four-model
ensemble shipped.** They really do recompute to 75.36% and 76.52%. The audit's
own tell — p50 0.55 ms where the ensemble is 1.21 ms — was the right catch.

But the conclusion "83.77% has no supporting artifact, strike it" does not
follow. Regenerating both prediction files from the four shipped FP16 members:

| Split | Correct | Scored | top-1 | top-5 |
| --- | --: | --: | --: | --: |
| **val** | 289 | 345 | **83.77%** | 93.62% |
| **test** | 909 | 1249 | **72.78%** | 87.51% |

**909/1249 and 289/345 are exactly the numerators the audit predicted from the
headline.** 83.77% was always the ensemble number; the file that proves it had
simply never been written down.

Fixed: `results/predictions_val.jsonl` and the new
`results/predictions_test.jsonl` are now the ensemble outputs. Compressed as
`ffb84c5`. Reproduce with `scripts/ml/evaluate.py --split val|test`.

**So: keep 83.77%, and keep citing it as the optimistic split.** It is real. The
defect was provenance, not the value.

---

## Item 7 — the 920 vs 909 gap. **Confirmed, exactly as the audit proposed.**

`results/robustness.json` was produced by `scripts/ml/robustness.py`, which
loads **PyTorch checkpoints over cached landmarks** (`torch.load` →
`TemporalModel`), not the FP16 TFLite members. Two different inference paths.

Measured both ways on the same 1,249 clips:

| Path | clean top-1 | correct |
| --- | --- | --: |
| shipped FP16 TFLite ensemble | 72.78% | **909** |
| PyTorch checkpoints (`robustness.json`) | 73.66% | **920** |

**Difference: exactly −11 clips**, matching the audit's 920 − 909. It is the
inference path, not a bug in the shipped predictions. The robustness deltas
should keep being read as relative shifts within one path, never compared
across the two.

---

## Items 1, 2, 3, 4, 5, 8, 9 — confirmed by measurement

| # | Question | Answer |
| --- | --- | --- |
| 1 | Member hashes | **Confirmed.** All four SHA-256s and byte counts match the handoff exactly. |
| 2 | Member order irrelevant? | **Confirmed.** Forward vs reversed gives max &#124;diff&#124; **9.5e-07**, 100% argmax agreement across six permutations. Logit averaging is order-invariant; no member is class-remapped. |
| 3 | Feature version | `[1, 24, 255]` float32, `FEATURE_DIM=255`, 24 frames. **Confirmed.** |
| 4 | `STRICT_RESAMPLE=False` | **Confirmed, and it is what training used.** `True` was implemented, retrained and rejected — it made both clean and hand-dropped accuracy worse (64.61%/47.64% vs 72.78%/58.21%). Your checkpoints match `False`. |
| 5 | Test reproduction | **Confirmed.** 909/1249 = 72.78% top-1, 1093/1249 = 87.51% top-5. |
| 8 | Six no-hand failures | **Confirmed exactly.** 1,255 vocabulary-eligible clips, **6** with no hand detected in any frame, 1,249 scored. Detector failures, not decode failures. Disclosing them rather than dropping them is correct. |
| 9 | INT8 mandatory? | **No.** The jury matrix says "TFLite/ONNX <20MB"; FP16 at 11.37 MB with detector satisfies it. If forced, the INT8 candidate measures **71.30% val** — 12.5 points worse and barely above the 70% target. **Do not substitute it.** Ship FP16. |

### On the `features.py` hash mismatch

You pinned `baf28d05…`; my file is `ecf555fb…`. That looks like a correctness
failure. It is not.

The two files differ only in **inactive code** — the `feature_dim(use_z)` helper,
the `STRICT_RESAMPLE` branch (default `False`), and comments. Building features
from 40 random clips with both modules gives **max &#124;diff&#124; exactly 0.0**. Your
feature parity claim holds; the hash difference is cosmetic. Pinning a hash was
still the right instinct.

---

## Item 11 — independent extraction. **Passed.**

I built the bundle with `scripts/release/build_bundle.py`, extracted it to a
clean directory outside the repository, and ran `predict()` with only the
runtime venv:

```
{"clip_id": "0005105598398187361-SAIL 2", "prediction": "SAIL2",
 "top5": ["SAIL2","BREAKFAST1","PONDER","EASY","PULLCONVINCE"],
 "confidence": 0.5663, "latency_ms": 7209.76}
```

Matches this branch exactly. The bundle is **self-contained**: 17 entries,
all four members, `parallel_tracker.py` present (your `bd85276` closure fix
works), `Dockerfile` included, `src/` complete. No repository needed at run time.

---

## Item 10 — licensing. **My statement, as asked.**

The four ensemble members **may not be redistributed.** They derive from ASL
Citizen (academic, signer-consented for research) and WLASL (C-UDA academic /
non-commercial). There is no written permission covering redistribution.

**So your handling is correct and I endorse it:** weights gitignored, public
repository source-only, bundle distributed separately, `NOTICE` and
`ARTIFACT_RIGHTS.md` stating it plainly. Do not change this.

One thing that is mine, not yours: the two third-party INT8 files
(`models/ptq_100_int8.tflite`, `models/s3d_conv1d_qat_int8.tflite`, 1,805,456 B)
are committed on every branch, including `main`. Your `bd85276` correctly stopped
claiming `model_weights_published: false` and recorded them as a pre-existing
owner decision. Removing them means rewriting history — **I am not doing that
unilaterally before submission.** Recorded as a known-open item for the owner.

---

## What is left for Dev

Nothing blocks the merge. In priority order:

1. **Review PR #1.** Item 12. That is the merge gate and it is mine to open.
2. **Push `ffb84c5`** so the corrected val/test artifacts exist on the branch you
   merge from.
3. **The INT8 decision** — answered above (no), but if an organizer insists, the
   measured number is 71.30% and it should be an explicit, recorded risk.
4. **Power and thermal stay `NOT_MEASURED`.** No meter, no device under test.
   Not fixable from here; must be stated as a gap, not quietly omitted.

Model work is closed: 72.78% test / 83.77% val, both reproducible from artifacts.