# Ensemble validation accuracy — independent recomputation

Recomputed from Dev's frozen `results/predictions_val.jsonl` joined to the
official signer-disjoint **validation** split, vocabulary-restricted.

| Population | Correct | Scored | Top-1 | Top-5 |
| --- | --: | --: | --: | --: |
| Validation, scored clips | 260 | 345 | **75.36%** | 87.25% |
| Validation, all eligible | 260 | 346 | 75.14% | — |

5 signers. One eligible clip produced no prediction
(`5989539775823847-INTUITIVE`, the same no-hand clip seen on test).

## Resolves one Dev discrepancy

Dev's `RESULTS.md` quotes **83.77%** validation for this candidate. No shipped
artifact supports that number: the committed `predictions_val.jsonl` recomputes
to **75.36%**, and `eval_summary_val.json` states `top1_pct: 76.52`
(which counts the run before the 4-model ensemble). 83.77% should be treated as
unverified until Dev identifies the artifact and script that produced it.

Validation is optimistic relative to test (75.36% vs 72.78%) because the
ensemble members were selected on it.
