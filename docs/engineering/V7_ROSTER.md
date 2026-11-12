# V7 bounded parallel engineering roster

Protected source snapshot: `$TMP/echogesture-...`.
Integration checkout: `hackathon/abhijeet-integration` (uncommitted preserved).
Worker copies: `$TMP/echogesture-.../A01` through `A16`.

| Agent | Scope | Owned output |
| --- | --- | --- |
| A01 | Latency | `evaluation/device/v7_latency/` |
| A02 | Decoder | `evaluation/device/v7_decoder/` |
| A03 | Tracking | `evaluation/device/v7_tracking/` |
| A04 | Streaming | `evaluation/device/v7_streaming/` |
| A05 | Model contract | `evaluation/device/v7_model_contract/` |
| A06 | INT8 | `evaluation/device/v7_int8/` |
| A07 | Accuracy | `evaluation/device/v7_accuracy/` |
| A08 | Robustness | `evaluation/device/v7_robustness/` |
| A09 | Android | `app/v7_android/`, `evaluation/device/v7_android/` |
| A10 | HUD | `app/v7_demo/`, `evaluation/device/v7_demo/` |
| A11 | Benchmark | `evaluation/device/v7_benchmark/` |
| A12 | Evidence | `evaluation/device/v7_evidence/` |
| A13 | Release | `evaluation/device/v7_release/`, `scripts/release/v7_release/` |
| A14 | QA | `evaluation/device/v7_qa/` |
| A15 | Privacy/licensing | `evaluation/device/v7_compliance/` |
| A16 | Submission material | `docs/release/v7_submission/` |

Acceptance: real code and tests, exact hashes and commands, no fabricated
metrics, no production default change without measured prediction parity and
lead review. Android/camera/organizer inputs remain external blockers unless
actually available. No commits, pushes, merges or release PRs.
