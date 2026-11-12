# PR: four-model FP16 ensemble integration (inference, evaluation, Android, release)

**Base:** `hackathon/two-person-sprint` · **Head:** `hackathon/abhijeet-public`
**State:** DRAFT — not submission-ready (see Remaining gates).

## Summary

The default landmark runtime is now **Dev's four-model FP16 TFLite ensemble**.
Every member runs on the same `[1,24,255]` float32 feature tensor, logits are
averaged element-wise, and one softmax runs over the mean. Public `predict()`
schema is unchanged.

| | |
| --- | --- |
| Members | `landmark_ens_{tcn21,tcnm61,tcnm62,rob71}_fp16.tflite` (3,543,080 B) |
| Input / output | `[1,24,255]` float32 / `[1,100]` logits |
| Aggregation | `mean(member_logits)`, then a single softmax |
| Per-member tensor indices | **differ** (out_idx 224 for TCN, 191 for Conv1D) |
| Quantization | float16 weights, float32 IO |
| Deployed size | 11.37 MB incl. MediaPipe detector |

**Weights are not in this branch.** They are ASL Citizen / WLASL derived with no
confirmed redistribution permission; `.gitignore` blocks them and
`models/README.md` documents the contract, all SHA-256 values and local setup.
A clone without weights starts, validates the metadata, then raises
`FileNotFoundError` naming the missing member — it never silently runs fewer.

## Accuracy — independently recomputed

Recomputed from Dev's frozen predictions joined to the official signer-disjoint
test split (`evaluation/device/ensemble_evidence/recompute.py`):

| Population | Previous single model | Ensemble |
| --- | --- | --- |
| Held-out test, scored clips | 782/1249 = 62.61% | **909/1249 = 72.78%** top-1 |
| Top-5 | — | 1093/1249 = 87.51% |
| All eligible (failures counted wrong) | — | 72.43% (1,255 denominator) |

11 unseen signers; per-signer mean 73.36% (min 53.52%, max 87.63%).
**Six of 1,255 eligible clips produced no prediction** (no hand detected) and are
disclosed rather than dropped. The test split was inspected during model
selection, so it is **not** a pristine blind test, and 62.61% / 72.78% are
**separate models**, not one evolving run.

## Performance — full clip-to-JSON, 12 permitted clips

| Runtime | p50 | p95 | Peak RSS | Parity |
| --- | --- | --- | --- | --- |
| Singleton IMAGE | 231.8 ms | 274.2 ms | 255 MiB | — |
| Parallel-2 | 132.0 ms | 154.8 ms | 300 MiB | 12/12 |
| Parallel-4 (`ECHOGESTURE_TRACKER=parallel`) | **79.3 ms** | **99.8 ms** | 394 MiB | 12/12 |

Ensemble inference is ~0.36 ms; **landmark tracking dominates**. The **50 ms
full-clip target is NOT met.** Model-only and per-frame latencies are different
measurements and are not substituted for this one. Parallel tracking stays
opt-in; the default singleton path is unchanged.

## Android

Kotlin classifier now loads all four members with per-member indices, averages
logits and runs one softmax. The aggregation is extracted into a pure object and
pinned against logits captured from `src/main.py`: mean logits agree to 1.3e-7,
top-1 probability to 1.9e-10 (`0.8821105`). The model contract pins each member
SHA-256 plus a combined ensemble hash and refuses a partial bundle.
**No APK and no on-device run** — Gradle, SDK dependency trees and adb are all
absent on this host. Device parity, latency, power and thermal are unverified.

## Release

`build_bundle.py` discovers members from metadata, stages each through the
existing path-guard helpers, records per-member SHA-256 in `release.json`, and
verifies them on clean extraction. Single-model bundles remain byte-compatible.
A locally built private package (weights included) clean-extracts and predicts
`WHATFOR1` at 0.8821; **it is gitignored and not part of this branch.**

## Verification

- **Clean clone: 281 tests, 0 failures, 29 weight-dependent skips.**
- Python 3.11 import closure verified; release builder imports.
- `recompute.py` exits 2 with a clear message when the private split CSV is absent.
- Privacy scan of all 30 new files: no participant IDs, no absolute home paths,
  no hostnames, no secrets, no model binaries, no ZIPs.

## Dev review requested

1. Four member SHA-256 values and the ordered `members` list.
2. Ordered label map (`e78f5028…`) and feature SHA (`baf28d05…`).
3. `STRICT_RESAMPLE=False` is the trained-under setting — confirm.
4. Reproduce 72.78% on the 1,249 scored clips.
5. **Unresolved:** the **83.77%** validation figure is unsupported by any artifact
   on your branch (committed `predictions_val.jsonl` recomputes to 75.36%,
   `eval_summary_val.json` says 76.52%).
6. **Unresolved:** the robustness "clean" baseline is 920/1,249 correct while the
   shipped predictions give 909/1,249 — an unexplained 11-clip gap.
7. The six no-hand clips: confirm they are genuine detection failures.
8. Quantization: whether FP16-only satisfies organizer INT8 expectations.
9. Redistribution permission for the weights (currently none).

## Remaining gates

50 ms full-clip target unmet · no Android APK/device · no power/thermal ·
organizer manifest/vocabulary absent · model redistribution unconfirmed ·
demo/video and final submission approval outstanding.

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)