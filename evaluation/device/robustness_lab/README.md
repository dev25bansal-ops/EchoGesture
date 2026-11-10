# EchoGesture robustness lab

Owner: A08 (Robustness Engineer). Branch state: isolated copy of
`hackathon/abhijeet-integration` at `$TMP/echogesture-.../A08`
(8 Nov 2026). Files here live under `evaluation/device/robustness_lab/`,
with the paired runner in `../robustness.py` and corruptions in
`../corruptions.py`.

## Layout

```
robustness_lab/
  __init__.py      public API (config, stats, reporting)
  config.py        deterministic RunConfig + derive_clip_seed + config_hash
  stats.py         aggregate_pairs, wilson_interval (pure; no model import)
  reporting.py     render_markdown, write_evidence (JSON/JSONL/CSV/report.md)
  runs/<name>/     evidence bundles produced by a real run
  README.md        this file
```

## Deterministic config

A run is `RunConfig(corruption, severity, seed, split, validation,
manifest_sha256, model_sha256)`.  `config_hash()` is the SHA-256 of its
canonical JSON; it is embedded in every `robustness.json`.  Per-clip
corruption seeds are derived as
`SHA256(f"{global_seed}:{index}:{clip_id}")[:8]`, so re-running the
same manifest + config reproduces identical corrupted inputs.

Default severities: occlusion 0.5, low_light 0.5, motion_blur 0.5,
crop 0.3, frame_drop 0.5.  All severities must be in (0, 1].

## Representative marking

Representative iff **all** hold:

1. `--validation` with `--split validation` (a documented labelled
   validation split, not locked test data),
2. >= 20 clips, every clip labelled,
3. >= 2 distinct signers (`signer_id` column),
4. zero clean/corrupted/decode failures.

Otherwise the run is a *transform check* and every output says so.
`generate_results.py` keys its PASS/NOT_MEASURED decision off
`representative_robustness_measured`, so this gate is load-bearing.

## Summary tables and diagnostics

Every run writes four files to `--output-dir`:

| File | Contents |
|---|---|
| `robustness.json` | summary (incl. config, config_hash, artifact SHAs, per-signer, CIs) |
| `pairs.jsonl` | one row per clip (clean/corrupted predictions, correctness, per-clip seed, latencies, errors) |
| `pairs.csv` | same rows, flat CSV for spreadsheets |
| `report.md` | human evidence report (accuracy table, per-signer table, limits) |

Failure accounting: every attempted pair stays in the denominator.
`failed_pair_count` + `failure_breakdown`
(`clean_prediction_failed`, `corrupted_prediction_failed`,
`decode_error`) make any dropped clip visible; `pairs.jsonl` carries
per-clip `clean_error` / `corrupted_error` / `decode_error` strings.
`work/` under the output dir is removed even on failure.

Statistics: Wilson 95% CIs for clean and corrupted top-1, paired flip
count/rate, clean→corrupted prediction agreement, and a per-signer
breakdown (N, clean/corrupted top-1, drop, failed).  Small samples are
reported with their intervals; point estimates are never promoted to
target claims.

## CLI

```bash
$REPO/.venv/bin/python \
  evaluation/device/robustness.py $TMP/echogesture-.../manifest.csv \
  --output-dir /tmp/<run> \
  --predictor src/main.py \
  --labels models/landmark_labels.json \
  --model models/landmark_classifier_fp16.tflite \
  --corruption low_light --severity 0.5 --seed 42 \
  --validation --split validation
```

The landmark model needs `ECHOGESTURE_MODEL=landmark` when `src/main.py`
is used via the predictor path; `--labels` must be the landmark label
map (`models/landmark_labels.json`), **not** `label100.json` — the
validation ground truths live in the landmark vocabulary, and the
runner rejects ground truth outside the supplied vocab.

## Measured results shipped in this copy

`runs/validation30_lowlight_s42/` — real paired run, 30-clip labelled
validation batch (`$TMP/echogesture-.../manifest.csv`, 5
signers, all ground truths in the landmark vocabulary), FP16 landmark
model `a012f2b5…7044a4`, `low_light` severity 0.5, seed 42,
`--validation --split validation`.  Representative by the gate above.

See `runs/validation30_lowlight_s42/report.md` and `robustness.json`
for the numbers.  Headline: clean 22/30 (73.3%), corrupted 21/30
(70.0%), drop +3.3 pt, 2 paired flips (6.7%), 0 failed pairs, 5/5
signers covered.  Wilson 95% CIs [55.6%, 85.8%] / [52.1%, 83.3%] —
wide; do not claim a ≤5 pt target is met from this run alone.

## Compatibility notes

- `robustness.json` keeps the original keys
  (`model_sha256`, `corruption_type`, `corruption_severity`,
  `sample_count`, `accuracy_drop_percentage_points`,
  `representative_robustness_measured`) so
  `scripts/release/generate_results.py` consumes it unchanged.
- `paired_evaluate` returns the summary dict (previously it returned
  the same shape minus diagnostics); `pairs` is now also embedded
  until `main()` pops it for file writing.
- `corruptions.CORRUPTION_VERSION = 2`; v2 fixed the occlusion
  y-range edge case and frame_drop rounding.  Outputs for inputs
  where v1 did not crash are unchanged.
