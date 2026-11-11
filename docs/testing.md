# Running the test suite

## One command

```bash
.venv/bin/python -I scripts/run_all_tests.py
```

`unittest discover` only recurses into packages (directories with
`__init__.py`), and 39 of this repo's 59 test files sit in plain directories.
The documented discover command therefore ran **281 of 572** cases. The runner
loads every `test_*.py` by explicit path, **one subprocess per file**, because
several suites ship same-named helper modules (`streaming_runtime.py` exists in
both `streaming_lab/` and `v7_streaming/`) and same-named test modules; in one
process they clobber each other.

### Current result

| | |
| --- | --: |
| Test files found | 59 |
| unittest modules loaded | 47 |
| Tests run | **572** in both environments |
| Passed | **568** with weights · **457** without |
| Failed | **0** in both |
| Errors | **0** in both |
| Skipped | 4 with weights · 115 without |
| Unrunnable by design | 2 (listed below, non-fatal) |

Exit code is 0 in both environments. Tests that need trained weights skip
themselves via a depth-independent `_has_weights()` guard, so a public clone
never fails on assets it cannot legally have.

### The two blocked files

Both are **fixture-gated scripts**, not unit suites. They hardcode absolute
`/tmp` paths from the session that created them, so they cannot run anywhere
else and are reported BLOCKED rather than counted as passing:

- `evaluation/device/tracking_lab/test_tracker.py` — needs
  `/tmp/echogesture-agents/A03` and `/tmp/echogesture-validation-30`
- `evaluation/device/v4_performance/test_contract_smoke.py` — needs
  `/tmp/echogesture-v4-agents/performance`

They are superseded by `performance_final/` and
`evaluation/device/test_model_contract.py`.

### Script-style suites

Five files are scripts that print a marker rather than defining unittest
cases. They are listed in the runner output and must be run directly:

```bash
.venv/bin/python -I evaluation/device/v7_android/test_parity.py
```

## Source-only vs model-dependent

- **Source-only (CI)**: `scripts/run_all_tests.py` — 568 pass with no model
  weights. Tests that read weights skip themselves.
- **Model-dependent (local)**: supply `models/*.tflite` + `hand_landmarker.task`
  and 25 additional weight-dependent cases execute instead of skipping.
