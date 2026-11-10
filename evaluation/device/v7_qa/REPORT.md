# A14 V7 QA — regression expansion report

## Scope & constraints
- Own only `$TMP/echogesture-.../A14/evaluation/device/v7_qa/`.
- No edits to original checkout or production code. No installs, no nested agents.
- Existing suite: 161 tests (all passing via `.venv/bin/python -m unittest discover -s evaluation/device -p "test_*.py"`).

## Delivered
5 new test files (43 tests) + 1 reproduction script + this report in `v7_qa/`.

Total after delivery: 204 tests, all passing in 0.8s.

### Coverage map (all asserted against verified production behavior)

| Area | File | Tests | Notes |
|---|---|---|---|
| Repeated model/predictor load (D-1 family) | test_repeated_load.py | 6 | fixed-name replacement verified; first-function globals survive; import failure cleans registry; missing/misbehaving predictors rejected |
| Malformed config / model / labels | test_config_rejection.py | 11 | empty & malformed vocabulary, SHA/size mismatch, release path escape, empty results, INT8 misdeclared as float16 rejected by tensor census |
| No-hand profiled failures | test_no_hand_failures.py | 5 | typed failure recorded, top1 nulled, non-representative, work/ cleaned; paired decode_error counted |
| Unicode clip IDs & label permutations | test_unicode_labels.py | 9 | emoji/CJK/combining/comma/BOM roundtrip; whitespace & NFC/NFD variants distinct; label permutation rejected |
| Atomic evidence & temp cleanup & ZIP safety | test_atomic_zip_safety.py | 12 | NaN publishes nothing, no temp residue, work/ removed on fault; duplicate/traversal/absolute/backslash/symlink/NUL/drive-letter rejected; empty predictions.jsonl rejected |

### No-silent-INT8-default (explicit)
- `ECHOGESTURE_MODEL` default resolves to `landmark` + `landmark_classifier_fp16.tflite` (not the INT8 artifact).
- `ECHOGESTURE_MODEL=INT8|int8|landmark2|s3D` all raise `ValueError` (no silent fallback).
- An INT8 tensor-census artifact declared `quantization=float16` raises `CompatibilityError: quantization mismatch`.

### Prior defect status (verified, not fabricated)
- D-1 module replacement: current production replaces the fixed-name `sys.modules["echogesture_predictor"]` entry but a retained `predict` keeps its own globals. 6 passing tests pin the safe side.
- D-2 workdir cleanup: current production cleans `output_dir/work` via `try/finally` even on decode failure. Covered by passing tests in test_no_hand_failures.py and test_atomic_zip_safety.py.
- Label permutations: covered (LabelPermutationTest).
- Empty artifacts: covered (release empty results + empty predictions.jsonl).

## Reproducible open interaction (reproduction script, not a failing test)
`v7_qa/repro_d1_profile_crosstalk.py` — exits 0 (reproduces) as of 2026-11-09:

`run_manifest.evaluate_profiled()` resolves `get_last_profile()` through
`sys.modules[predict.__module__]` at call time. Because `load_predictor()`
registers every predictor under the fixed name `echogesture_predictor`,
loading an unprofiled predictor B after loading profiled predictor A makes
`evaluate_profiled(A)` raise `EvaluationError: Predictor does not expose
get_last_profile()` even though A does expose it. The lookup should be
bound to A's own module, not re-resolved through the replaced fixed-name
entry. No production fix is fabricated here; this is reported for the
owner of `run_manifest.py`.

## Verification command
```
cd /tmp && $REPO/.venv/bin/python -I -m unittest discover -s $TMP/echogesture-.../A14/evaluation/device -p "test_*.py"
# Ran 204 tests in 0.812s — OK
```
