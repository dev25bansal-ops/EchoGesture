# results/predictions.jsonl — provenance

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
