# Provisional release readiness (Track 2)

This workbook is the judge-runnable provisional route. It is intentionally
not an official submission: the organizer filename, ZIP details, manifest,
and INT8/device targets are still unresolved.

## Current candidate

- Default model: `models/landmark_classifier_fp16.tflite` (FP16, **not INT8**).
- Verified SHA-256: `a012f2b5f7a1110961079f19eb0e28c9abacb95e4d0787352e4157ebca7044a4`.
- Required local assets: `models/landmark_labels.json`,
  `models/landmark_model_meta.json`, `models/hand_landmarker.task`.
- Size note: the classifier is about 0.8 MB; total shipped model assets
  including the 8.6 MB-region detector must stay under 20 MB. The builder
  sums everything under `models/` and rejects bundles at or above 20,000,000 bytes.
- `models/landmark_classifier_int8.tflite` is an unselected artifact. Do not
  claim INT8 compliance until it is the selected, verified candidate.
- Conversion script: `scripts/ml/export_tflite_tf.py`. A provisional build
  requires it via `--extra` and records it in `release.json` as
  `conversion_script`.

## Provisional build (verified in this copy only)

A provisional bundle was built and verified in this copy (original checkout
untouched). The builder ran with `--source` set to the isolated-copy root,
the results input at
`scripts/release/release_lab/results_NOT_MEASURED.csv`, the output at
`scripts/release/release_lab/provisional.zip`, and `--extra` entries for
`models/landmark_classifier_fp16.tflite`, `models/landmark_labels.json`,
`models/landmark_model_meta.json`, `models/hand_landmarker.task`,
`scripts/ml/export_tflite_tf.py`, `src/features.py`, and
`src/landmark_frontend.py` (all paths relative to the isolated-copy root).

Result: `{"model_sha256": "a012f2b5...7044a4", "status": "structure_verified"}`.
`scripts/release/release_lab/results_NOT_MEASURED.csv` deliberately carries
`NOT_MEASURED` for every performance/robustness/power/thermal row; no
measured metrics are claimed. The ZIP holds 13 files, 8,690,956 bytes
uncompressed (6.3 MB on disk), including `Dockerfile`, `release.json`,
`results/results.csv`, `scripts/ml/export_tflite_tf.py`,
`models/landmark_classifier_fp16.tflite`, `models/landmark_labels.json`,
`models/landmark_model_meta.json`, `models/hand_landmarker.task`,
`src/main.py`, `src/requirements.txt`, `src/README.md`, `src/features.py`,
and `src/landmark_frontend.py`.

The same build with `--official` fails as designed:

```text
ERROR: Official bundle requires organizer-generated predictions.jsonl
```

Verify the portable form (paths relative to a checkout root):

```bash
python3 -I scripts/release/build_bundle.py build \
  --source . \
  --model models/landmark_classifier_fp16.tflite \
  --labels models/landmark_labels.json \
  --results scripts/release/release_lab/results_NOT_MEASURED.csv \
  --output scripts/release/release_lab/provisional.zip \
  --extra models/landmark_classifier_fp16.tflite \
  --extra models/landmark_labels.json \
  --extra models/landmark_model_meta.json \
  --extra models/hand_landmarker.task \
  --extra scripts/ml/export_tflite_tf.py \
  --extra src/features.py \
  --extra src/landmark_frontend.py
```

Verify an extracted bundle (or a ZIP) without running inference:

```bash
python3 -I scripts/release/build_bundle.py verify <provisional.zip> \
  --model-sha a012f2b5f7a1110961079f19eb0e28c9abacb95e4d0787352e4157ebca7044a4
```

With a real clip (only when a clip is available):

```bash
python3 -I scripts/release/build_bundle.py verify <provisional.zip> \
  --clip <clip.mp4>
```

`results/results.csv` must have columns
`metric,target,measured,unit,device,runtime,pass,notes` and statuses
`PASS|FAIL|NOT_MEASURED|NOT_APPLICABLE`. Generate it from measured evidence
only with `scripts/release/generate_results.py`; missing device
measurements stay `NOT_MEASURED`. Never hand-write measured numbers.

## What verification enforces

- Path safety: every `release.json` path, ZIP member, and `--extra`
  artifact must be a safe relative path inside the bundle; ZIP symlinks,
  duplicates, traversal, and absolute paths are rejected before extraction.
- Manifest ID validation: `release.json` model/labels/conversion paths are
  safe-relative and suffix-checked; `model_sha256` must be a 64-char
  lowercase hex digest matching the file bytes; the vocabulary must be a
  non-empty label-to-index map with contiguous indices from zero.
- Model evidence/hash checks: `--model-sha` (or the bundle's own
  `build()` hash chaining) must match the FP16 file; total `models/` bytes
  must stay under 20 MB; `results/results.csv` structure and statuses are
  validated when present.
- ZIP extraction checks: members are extracted manually into a temp
  directory, never with unchecked `extractall()`, and unknown member types
  (symlinks/devices) are rejected.
- Conversion script/Docker presence: a real `--extra` export script is
  required and recorded; a non-empty `Dockerfile` must ship.
- Predictions (when present): the file must live at
  `results/predictions.jsonl`; every row needs `clip_id`, `prediction`,
  numeric `confidence` and `latency_ms`; the prediction must be in the
  selected vocabulary; IDs must be unique and exactly match the ordered
  `clip_id` column of `manifest.json` at the bundle root. No absolute or
  private paths may appear in `release.json`, `results.csv`, or
  `predictions.jsonl`.
- Extracted predictor (when `--clip` is given): output is validated
  against the same vocabulary/range/clip-ID contract before being returned.

## Official path (blocked, by design)

`--official` is blocked in `build()`:

> `Official bundle requires organizer-generated predictions.jsonl`

and in `verify` it requires `results/predictions.jsonl` **and** a bundle-root
`manifest.json` whose `clip_id` rows exactly match the prediction IDs, plus
`release.json` with `"official_submission": true` and a matching FP16 SHA.
The builder never fabricates predictions, results, or an official ZIP.
Until organizer predictions and manifest details arrive, any ZIP produced
here is provisional only.

## Tests

```bash
python3 -I -m unittest discover -s evaluation/device -p 'test_release.py' -v
```

Covers the happy-path build/verify, stale-model rejection in
`generate_results.py`, path-traversal and absolute-path rejection, manifest
shape/SHA rules, total-size accounting, results validation, prediction
manifest matching, ZIP symlink/duplicate rejection, and the blocked
`--official` build.

## Blockers (do not close without real organizer inputs)

- Official filename and ZIP details (inner layout, compression, naming).
- Organizer manifest (`manifest.json`) and the real predictions file.
- INT8/device target confirmation: the selected model is FP16 today.
- Physical-device power/thermal and representative accuracy/p95 evidence.
- Submission form, registration, and receipt.
