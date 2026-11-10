# Model contract lab — A05 (Model Contract Engineer), 8 Nov 2026

Scope: strict versioned candidate integration for the landmark FP16
candidate. Owned files only: `evaluation/device/model_contract.py`,
`evaluation/device/test_model_contract.py`, and this lab directory.
Dev-owned `src/main.py`, `src/features.py`, `src/landmark_frontend.py`,
and training/export code were not modified.

## Candidate under test (pinned, verified in this copy)

| Asset | SHA-256 (verified) | Size |
|---|---|---|
| `models/landmark_classifier_fp16.tflite` | `a012f2b5…7044a4` | 832,888 B |
| `models/landmark_labels.json` | `e78f5028…9a4a` | 1,492 B |
| `models/landmark_model_meta.json` | `7f189a16…947e` | 123 B |
| `models/hand_landmarker.task` | `fbc2a300…7cde1` | 7,819,105 B |

Source branch inspected read-only: `origin/hackathon/dev-model`
(`docs/hackathon/DEV_HANDOFF_MODEL.md`, `results/QUANTIZATION.md`).
Dev candidate worktree `/private$TMP/echogesture-.../models`
matches byte-for-byte (same SHAs). No Dev metadata was overwritten.

## What changed

### `evaluation/device/model_contract.py`

- Added `CANDIDATE_SPEC` / `CANDIDATE_VERSION = "landmark-fp16-v1"` pinning
  SHA, size, `[1,24,255]` float32 in, `[1,100]` float32 out, feature dim
  255, 24 frames, 100 classes, `quantization: "float16"` /
  `fp16_float32_io`.
- Added `quantization_census()` — counts all internal tensor dtypes via
  LiteRT `get_tensor_details()` and how many int8 tensors are truly
  quantized (scale != 0). This is the only way to distinguish FP16 from
  INT8 when both expose float32 IO.
- Added `identify_quantization()` — classifies an artifact from its actual
  tensors: `fp16_float32_io`, `int8_mixed_float_io`, `int8_io`,
  `float32`, or `mixed_unknown`. Never reports FP16 as INT8.
- Added `probe_output_semantics()` — invokes the real interpreter on
  deterministic inputs (zeros + seeded random) and classifies the output
  as logits vs probabilities from finiteness, negativity, and unit-interval
  behavior. No fabricated parity: FP16↔INT8 logits were measured to differ
  (max abs diff 0.94 on random input; different argmax on zero input).
- Added `validate_candidate_bundle()` — strict, no-fallback validation of
  all four assets: existence/non-emptiness, detector ZIP magic, exact
  model/labels/detector SHAs, meta parity (frames/feature_dim/classes/
  quantization), labels ordering vs class count, actual tensor shapes,
  quantization census, and probed logits semantics. A missing detector
  raises `CompatibilityError` instead of falling back to S3D.
- Hardened `validate_bundle()`: optional `required_assets` and
  `labels_file` checks, optional `quantization` /
  `quantization_classification` census cross-check, and probed output
  semantics when an interpreter is supplied.

### `evaluation/device/test_model_contract.py`

Preserved all 7 original tests (including legacy `test_real_dev_fp16_contract`
back-compat) and added 14 `RealArtifactIdentificationTest` tests against the
real artifacts:

- FP16 actual tensor contract `[1,24,255]`/`[1,100]` float32.
- FP16 identified as `fp16_float32_io` (38 float16 tensors, 0 int8-quantized).
- INT8 identified as `int8_mixed_float_io` (82 int8-quantized tensors, float32 IO).
- FP16 and INT8 are distinct artifacts (different census + classification).
- S3D baseline identified as `int8_io`.
- Both FP16 and INT8 probed as logits.
- `validate_candidate_bundle` accepts the FP16 candidate.
- Rejects INT8 bytes under the FP16 filename (SHA mismatch).
- Rejects INT8 even when its own SHA is pinned (census `quantization mismatch`).
- Missing detector raises (no silent fallback).
- Labels parity (100 contiguous indices, output width matches).
- Meta parity (frames/feature_dim/classes/quantization match tensors).
- FP16↔INT8 logit drift is real (max abs diff > 0.1 on seeded input).

### Findings (measured, not claimed)

- FP16 census: 192 tensors = 117 float32 + 38 float16 + 37 int32; 0 int8-quantized.
- INT8 census: 158 tensors = 100 int8 (82 quantized) + 54 int32 + 4 float32; 0 float16.
  Both expose float32 IO — IO alone cannot distinguish them.
- S3D baseline: int8 IO `[1,24,64,64,1]`/`[1,100]` (different frontend).
- Labels: `landmark_labels.json` (100, `DOG1`…`LONGLINE`) is a disjoint
  vocabulary from `label100.json` (100, `book`…`thursday`; 1 case-insensitive
  overlap) — labels must not be swapped between backends.
- Detector `hand_landmarker.task` is a ZIP archive (PK header at offset 2);
  size 7,819,105 B; included in size accounting.
- Output semantics: both landmark artifacts produce logits (negatives present,
  finite, deterministic across repeated invokes).
- FP16≠INT8: same random input → max abs logit diff 0.94; zero input →
  different argmax (FP16 idx 3, INT8 idx 48). Not interchangeable.

## Exact run results

```
PYTHONPATH=$TMP/echogesture-.../A05 \
  .venv/bin/python -m unittest discover -s evaluation/device -p 'test_*.py' -v
Ran 55 tests in 0.204s — OK
```

Targeted: `test_model_contract.py` alone → 21 tests, OK (0.04s).
Full suite (excluding the `-I` isolation quirk that drops the repo root
from `sys.path` and breaks the pre-existing `test_offline_demo`
`app` import) → 55 tests, OK. The `test_offline_demo` import failure under
`-I` is pre-existing environment behavior, unrelated to these changes.

## Integration handoff

- `src/main.py` (untouched, Dev/Abhijeet-owned) already selects the landmark
  backend by default and raises on missing assets — the strict contract
  here complements it; no changes to `src/main.py` are proposed.
- To validate the candidate in a release/CI step:
  ```python
  from model_contract import validate_candidate_bundle
  from ai_edge_litert.interpreter import Interpreter
  interp = Interpreter(model_path="models/landmark_classifier_fp16.tflite")
  interp.allocate_tensors()
  validate_candidate_bundle("models", interp)  # raises on any mismatch
  ```
- The FP16 candidate satisfies the `<20MB` TFLite criterion but is
  explicitly **not** an INT8 solution; no INT8 compliance is claimed.
- Blockers / not measured: no representative accuracy or clip-p95 claim
  (see `CANDIDATE_INTEGRATION.md`); no physical-device run; organizer
  vocabulary/timing-boundary questions remain open.

## Commands used (in `$TMP/echogesture-.../A05`)

- `shasum -a 256 models/*` — verified all pinned SHAs.
- Interpreter probes via `.venv/bin/python -I -c …` for tensor census,
  IO contract, semantics, and FP16↔INT8 drift (measured values above).
- `.venv/bin/python -I evaluation/device/test_model_contract.py -v` — 21 OK.
- `PYTHONPATH=$TMP/echogesture-.../A05 .venv/bin/python -m unittest discover -s evaluation/device -p 'test_*.py' -v` — 55 OK.
