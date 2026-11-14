# Ensemble held-out accuracy — independent recomputation

**Verdict: the 72.78% top-1 / 87.51% top-5 claim reproduces exactly from frozen
predictions.** The predictions file is the ensemble's, not the older 62.61%
single model. Two caveats are recorded below: the headline uses a
*scored-clip* denominator (72.43% over all eligible clips), and Dev's
robustness "clean" baseline does not reconcile with this prediction file.

No model was run and nothing was tuned. All inputs are frozen artifacts.

---

## 1. Which artifact the numbers belong to

| | |
| --- | --- |
| Predictions | `origin/hackathon/dev-model:results/predictions.jsonl` |
| SHA-256 | `1aacbacbc5da9c3ddc2bd7f796f8a73f7199ebaa020de430848ca4035e48ad4a` |
| Lines | 1,249 |
| Fields | `clip_id`, `prediction`, `top5`, `confidence`, `latency_ms` |
| Local copy | `evaluation/device/ensemble_evidence/predictions_ensemble.jsonl` (byte-identical) |
| Last written by commit | `4594bfa` "feat(DEV-05): 72.78% with a 4-model ensemble" |
| Unchanged since | `4594bfa` == `ace8a15` (byte-identical) |

### It is the 4-model ensemble file — three independent signals

1. **Line count alone does not decide it.** Every era's `predictions.jsonl` has
   1,249 lines (DEV-03 single model, DEV-04 3-model, DEV-05 4-model), because
   the scored clip set is fixed at 1,249. Provenance cannot rest on the count.
2. **It is not the 62.61% single-model file.** The prior file
   (`226bd41`) has the same 1,249 clip IDs but predicts a different gloss for
   306 of them (943/1,249 agree) and recomputes to **62.61%**. The shipped file
   recomputes to **72.78%**. The in-repo frozen copy under
   `evaluation/device/v5_accuracy/results/test_split/predictions.jsonl`
   (SHA-256 `8872e91e…`) is byte-identical to that *old* single-model file and
   also scores 62.61% — it is **not** the ensemble evidence.
3. **The latency fingerprint is single-model-impossible.** The file's classifier
   latency is p50 **1.205 ms / p95 2.519 ms**. A single member measures
   ~0.55 ms p50; the 3-model file measured 0.639 ms p50 (3.417→1.489 sample
   latencies scale up from the single-model file). 1.205 ms matches the
   four-invocation cost Dev reports, and `results/eval_summary_test.json`
   declares `"ensemble": "tcn21+tcnm61+tcnm62+rob71 (4x fp16 tflite)"` with
   `latency_p50_ms: 1.205`.

Ensemble member SHA-256s in this checkout match Dev's handoff exactly
(`08417e72…`, `112279bc…`, `01a261d8…`, `27058a85…`), and
`models/landmark_labels.json` is byte-identical (`e78f5028…`).

### Provenance gap (stated, not hidden)

`results/predictions.jsonl` has **no ground_truth/correct field and no
`_truth`/`_signer` field**. Dev's committed `scripts/ml/evaluate.py` would have
written both (`_truth`, `_signer`, stripped on write) *and* names its output
`predictions_{split}.jsonl` — that path has never been committed. So **no
committed script is the exact writer of `results/predictions.jsonl`**. The file
is consistent with the shipped ensemble (member set, latency, class vocabulary,
file size), but its production path is not proven end-to-end. Treat the tie to
the 4 artifacts as *strongly consistent*, not *script-verified*.

---

## 2. Recomputed figures

Run:

```bash
.venv/bin/python -I evaluation/device/ensemble_evidence/recompute.py \
    --predictions evaluation/device/ensemble_evidence/predictions_ensemble.jsonl \
    --split-csv evaluation/device/v5_accuracy/splits/test.csv
```

| metric | value |
| --- | --- |
| eligible clips (100-class vocab) | 1,255 |
| scored clips | 1,249 |
| unmatched predictions | 0 |
| eligible but unscored | 6 |
| correct (top-1) | 909 |
| top-5 correct | 1,093 |
| **top-1, scored denominator** | **72.78%** (909/1,249) |
| **top-5, scored denominator** | **87.51%** (1,093/1,249) |
| top-1, all-eligible denominator (unscored = wrong) | **72.43%** (909/1,255) |
| top-5, all-eligible denominator | **87.09%** (1,093/1,255) |
| signers | 11 |
| per-signer mean / min / max | 73.36% / 53.52% / 87.63% |

Per-signer (scored clips). Signers are labelled S1–S11 here rather than by their
raw participant identifiers; the raw labels are held locally with the source
evidence.

| signer | acc | n | | signer | acc | n |
| --- | --- | --- | --- | --- | --- | --- |
| S1 | 64.29% | 98 | | S7 | 53.52% | 142 |
| S2 | 72.55% | 102 | | S8 | 81.19% | 101 |
| S3 | 72.22% | 108 | | S9 | 85.19% | 108 |
| S4 | 87.63% | 97 | | S10 | 84.31% | 102 |
| S5 | 80.42% | 143 | | S11 | 69.01% | 142 |
| | | | | S6 | 56.60% | 106 |

The 6 unscored eligible clips: `COMB` (S7), `FOREIGNER` (S5), `DECIDE`
(S4), `DEMAND` (S7), `ANYONE` (S4), `TAIL` (S7). They are absent from the
predictions file. Dev attributes the 1,255→1,249 gap to clips with no hand
detections; the count and the identity of the six are consistent with that,
but the cause cannot be confirmed from the artifacts alone (no detector log).

---

## 3. Denominator integrity

- **(a) Over scored clips:** 72.78% top-1 / 87.51% top-5. This is what Dev
  reports and what `eval_summary_test.json` records (`n: 1249`).
- **(b) Treating the 6 unscored eligible clips as incorrect:** 72.43% top-1 /
  87.09% top-5.

**Dev's 72.78% uses denominator (a), the 1,249 scored clips.** This is stated
plainly in `RESULTS.md` ("1,249 clips") and `DEV_HANDOFF_MODEL.md` ("1,249
scored clips"), and `eval_summary_test.json` carries `n: 1249`. The 6 skipped
clips are disclosed in the handoff's "Known failure cases". Denominator (b) is
not claimed and is 0.35 points lower; a submission that must count a
failed clip as wrong should use 72.43%, not 72.78%.

---

## 4. Discrepancies (explicit)

1. **Robustness "clean" does not reconcile.** `results/robustness.json` reports
   `clean.top1 = 73.6589…` over the test split, which is exactly **920/1,249**
   (and `top5 = 87.4299…` = 1,092/1,249). The shipped predictions yield
   **909/1,249 = 72.78%** and 1,093/1,249. An 11-clip difference on the same
   split and nominally the same "clean" input. The robustness harness
   (`scripts/ml/robustness.py`) runs PyTorch checkpoints over a landmark cache
   rather than the FP16 TFLite files, so it is a different inference path and
   is not expected to be bit-identical — but the 11-clip gap is larger than a
   rounding difference and is **unexplained** by the evidence in this repo.
   This does not move the 72.78% headline; it means the clean baseline in
   `robustness.json` should not be quoted as if it were the same measurement.
2. **Val handoff figure is stale.** `DEV_HANDOFF_MODEL.md`/`RESULTS.md` claim
   val 83.77%, but the committed `results/predictions_val.jsonl` (DEV-03 era,
   `226bd41`) recomputes to **75.36%** top-1 / 87.25% top-5 over 345 scored of
   346 eligible, and `eval_summary_val.json` also says 76.52%. No committed val
   prediction file corresponds to 83.77%. The test number is unaffected; the
   val number in the docs is not backed by a shipped artifact.
3. No discrepancy in the headline test figures themselves: 909/1,249 and
   1,093/1,249 reproduce 72.78% / 87.51% exactly, as do 73.36% / 53.52% /
   87.63% per-signer mean/min/max.

---

## 5. Reproducing the guard

```bash
.venv/bin/python -I evaluation/device/ensemble_evidence/test_recompute.py -v
```

Pins: 1,249 scored, 909 correct, 72.78%/87.51% scored, 72.43%/87.09%
all-eligible, 11 signers, per-signer 73.36/53.52/87.63, and 6 unscored. Swapping
in the old single-model file fails the top-1 pin (62.61%), so the test
distinguishes the two artifacts rather than merely matching a count.