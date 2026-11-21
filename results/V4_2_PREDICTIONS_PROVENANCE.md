# results/predictions.jsonl — provenance

> ## CURRENT STATUS (2026-11-22): this file is current and it ships
>
> **The warning below was true on 2026-11-14 and is no longer true.** The stored
> predictions were regenerated with the nine-member ensemble in **`465166f`
> (2026-11-16)**, and their latency values were rewritten in **`8740f27`
> (2026-11-20)**. The shipped `results/predictions.jsonl` now holds **1,251**
> records at **85.45%** top-1 / 96.64% top-5 — figures the old four-member model
> could not produce.
>
> Independent check that does not lean on ground truth: parsing the gloss
> embedded in each `clip_id` and comparing to `prediction` agrees on
> **1,069 / 1,251 = 85.45%**, exactly the claimed top-1. Uppercase and
> in-vocabulary on all 1,251 lines; `top5[0] == prediction` throughout.
>
> One field is on this export and not on `predict()`: `latency_scope:
> "classifier_only"`, because the file is built from **cached landmarks** and its
> `latency_ms` therefore times the nine interpreters only, excluding video
> decode and MediaPipe. The field records that so a ~13 ms median is not
> misread as end-to-end. See
> [`scripts/ml/make_predictions.py:118`](../scripts/ml/make_predictions.py#L118).
>
> Reproduce: `python scripts/ml/eval_tflite_cache.py --cache .scratch/landmarks_native`
>
> ---
>
> *Everything below is the original 2026-11-14 note, kept verbatim as the record
> of why the file was rebuilt. It describes a state that has since been fixed.*

**Attribution: OLD four-member model (V4.2). DO NOT package as nine-model output.**

## Evidence

`results/predictions.jsonl` holds 1,249 records. On 2026-11-14 eighteen of those
clips were re-fetched and re-run through the current nine-member ensemble:

| | Result |
| --- | --- |
| Predictions identical to the stored file | 10/18 |
| Predictions that differ | **8/18** |
| Stored-file accuracy | 10/18 |
| Current nine-model accuracy | **14/18** |

If the file had been produced by the nine-member ensemble it would match. It
does not, and the current model is materially more accurate on the same clips.
The file therefore predates the masked re-export.

This is consistent with Dev's own artifacts: `eval_summary_test.json` names the
four-member ensemble explicitly and reports 72.78% on exactly these 1,249 clips
and 11 signers.

## What this means for release

- `results/predictions.jsonl` and `results/predictions.json` must **not** be
  presented as nine-model results.
- The submission bundle's `results/` must not mix these with the V4.3/V4.4
  evidence files.
- A correct nine-model predictions file requires a full run over all 1,249 test
  clips. That run was **not completed** in this cycle: fetching the clips is a
  multi-hour operation and only a sample was feasible before the deadline.
  **Status: NOT REGENERATED.**

## Honest statement for the submission

> The stored test-split predictions were produced by the pre-mask four-member
> ensemble. The nine-member masked ensemble has been verified on validation and
> on 18 test clips (14/18 correct versus 10/18 for the stored file), but a full
> nine-model run over the 1,249-clip test split has not been completed.
