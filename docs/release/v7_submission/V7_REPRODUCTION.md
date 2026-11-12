# EchoGesture V7 — Reproduction Commands

All commands run from the isolated copy root
(`$TMP/echogesture-.../A16` — or your checkout root, same paths).
No installs are performed by these commands; the research video/download
steps are intentionally out of scope (clips and manifests stay outside Git).
Each block states what it reproduces and what the expected output is.
**Expected outputs are the measured values in [V7_FINAL_EVIDENCE.md](V7_FINAL_EVIDENCE.md);
a rerun that disagrees is a finding, not a rounding choice.**

Environment: Python 3.11 (tested 3.11.18), macOS 26.3 arm64 in the
measured runs; requirements in `src/requirements.txt`:
`numpy>=1.24.0`, `opencv-python>=4.8.0`, `ai-edge-litert>=2.3.0`,
`mediapipe>=1.2.2`.

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r src/requirements.txt
```

---

## 1. Single-clip prediction (smoke)

```bash
.venv/bin/python src/main.py /absolute/path/to/permitted-fixture.mp4
# {"clip_id": "...", "prediction": "UPPERCASE_GLOSS",
#  "top5": [...5 labels...], "confidence": 0.xxxx, "latency_ms": xx.x}
```

Contract: `predict(video_path) -> dict` in `src/main.py`, defaulting to the
FP16 landmark backend. Expected on the integration fixture:
`PULLCONVINCE`, confidence 0.7507 (N=1 smoke, not accuracy).

## 2. Test suite (161 tests)

```bash
.venv/bin/python -m unittest discover -s evaluation/device -p 'test_*.py' -v
# Expected: 161 / 161 pass (integration, evaluator, robustness,
# model-contract, release, demo, streaming, evidence)
```

Release-builder subset:

```bash
.venv/bin/python -m unittest discover -s evaluation/device -p 'test_release.py' -v
# 17 tests OK (per v6_release/commands.sh)
.venv/bin/python -m unittest discover -s evaluation/device/v6_release -p 'test_v6_*.py' -v
# 22 tests OK
```

## 3. Manifest evaluation (accuracy + timing through the real API)

```bash
.venv/bin/python evaluation/device/run_manifest.py \
  /absolute/path/to/manifest.csv \
  --output-dir /absolute/path/to/local-benchmark-output \
  --model models/landmark_classifier_fp16.tflite \
  --labels models/landmark_labels.json \
  --feature-config src/features.py
```

Manifest CSV: `clip_id,path` plus optional `ground_truth`,`signer_id`.
Ground truth must be present for every clip or none. Output:
`predictions.jsonl` + `benchmark.json` with artifact SHA-256 values,
cold/warm split, nearest-rank p50/p90/p95/p99. Laptop numbers only —
never target-device evidence.

Profiled variant (adds per-stage timings):

```bash
.venv/bin/python evaluation/device/run_manifest.py /absolute/path/to/manifest.csv \
  --output-dir /absolute/path/to/local-benchmark-output --profile
```

## 4. 346-clip paired FP16 vs INT8 accuracy

Inputs needed (outside Git): the fetched manifest
`evaluation/device/v5_accuracy/data/val_346/clips/fetched_manifest.csv`
(SHA-256 `3f06798a…f6c82`) with the 346 video files it references,
resolved via `evaluation/device/v5_accuracy/fetch_selected.py` /
`select_val_clips.py` from the official ASL Citizen val split.

```bash
.venv/bin/python -I evaluation/device/v5_accuracy/evaluate_paired.py \
  evaluation/device/v5_accuracy/data/val_346/clips/fetched_manifest.csv \
  --output-dir /absolute/path/to/val346-out --detector-mode image
# Writes predictions.json + predictions.csv + summary.json
# Expected: FP16 272/346 = 78.61% (Wilson [73.99%, 82.61%]),
#           INT8 254/346 = 73.41% (Wilson [68.52%, 77.79%]),
#           345 successful + 1 no-hand failure (signer_2f68a231 INTUITIVE)
```

The 30-clip subset run is the same command against
`$TMP/echogesture-.../manifest.csv` (SHA-256 `004e5eff…618`),
`--output-dir` a fresh directory. Expected: FP16 22/30 = 73.33%,
INT8 21/30 = 70.0%, 30/30 successful.

## 5. Split / leakage audit

```bash
.venv/bin/python -I evaluation/device/v5_accuracy/split_audit.py \
  --val-csv evaluation/device/v5_accuracy/splits/val.csv \
  --test-csv evaluation/device/v5_accuracy/splits/test.csv \
  --labels models/landmark_labels.json \
  --manifest30 $TMP/echogesture-.../manifest.csv \
  --output evaluation/device/v5_accuracy/results/split_audit.json \
  --train-manifest /absolute/path/to/dev-train.csv
# Expected: official val↔test signer overlap [],
#           vocab-eligible val 346 (5 signers) / test 1,255 (11 signers),
#           train∩val = train∩test = 0 signers, 0 clips, 0 duplicate IDs
```

## 6. Controlled latency baseline (V6)

The exact commands are recorded in
`evaluation/device/v6_release/commands.sh` (steps 1–11). In this copy:

```bash
PY=/path/to/.venv/bin/python
ROOT=$TMP/echogesture-.../A16
MANIFEST=$TMP/echogesture-.../manifest.csv
V6="$ROOT/evaluation/device/v6_release"

PYTHONDONTWRITEBYTECODE=1 "$PY" -I \
  "$ROOT/evaluation/device/v6_release/v6_baseline_latency.py" "$MANIFEST" \
  --output-dir "$V6/baseline"
# Expected: 30 calls, all p50 231.4835 ms, p95 306.554125 ms,
#           warm-29 p95 284.048625 ms, cold (first) 483.003458 ms,
#           landmark_tracking p95 274.336709 ms, model_inference p95 0.12025 ms
#           (thermal/contention variance on laptop is expected; order of
#           magnitude and bottleneck are the stable findings)

PYTHONDONTWRITEBYTECODE=1 "$PY" -I \
  "$ROOT/evaluation/device/v6_release/convert_v6_baseline.py" \
  "$V6/baseline/v6_baseline_latency.json" --output "$V6/baseline/benchmark.json"

PYTHONDONTWRITEBYTECODE=1 "$PY" -I \
  "$ROOT/evaluation/device/v6_release/generate_v6_results.py" \
  --output "$V6/baseline/results.csv"
```

## 7. Robustness (five conditions, FP16 selected)

One command per condition; INT8 runs add the predictor/model flags shown
in `evaluation/device/v5_release/commands.txt` (10 commands total).
FP16 (selected) example:

```bash
.venv/bin/python -I evaluation/device/robustness.py \
  $TMP/echogesture-.../manifest.csv \
  --output-dir /absolute/path/to/robustness-low-light \
  --predictor src/main.py --labels models/landmark_labels.json \
  --model models/landmark_classifier_fp16.tflite \
  --corruption low_light --severity 0.5 --seed 42 \
  --validation --split validation
# Repeat with --corruption {crop,occlusion,motion_blur,frame_drop}
# Expected FP16 drops: low_light 3.33 pt PASS; crop 6.67 pt FAIL;
# occlusion 13.33 pt FAIL; motion_blur 13.33 pt FAIL; frame_drop 13.33 pt FAIL
# (N=30, 5 signers, 0 failed pairs, clean top-1 0.7333 in every report)
```

INT8 variant: `--predictor scripts/release/v5_release/int8_predictor.py
--model models/landmark_classifier_int8.tflite`. Expected INT8 drops:
low_light 0.00, crop 0.00, occlusion 13.33, motion_blur 6.67,
frame_drop 13.33 pt (clean 0.700) — report with the N=30 caution in
V7 evidence §4.

## 8. Provisional bundle build + verify

```bash
.venv/bin/python -I scripts/release/build_bundle.py build \
  --source "$ROOT" --model models/landmark_classifier_fp16.tflite \
  --labels models/landmark_labels.json --results "$V6/baseline/results.csv" \
  --output "$V6/provisional.zip" \
  --extra models/landmark_classifier_fp16.tflite \
  --extra models/landmark_labels.json \
  --extra models/landmark_model_meta.json \
  --extra models/hand_landmarker.task \
  --extra scripts/ml/export_tflite_tf.py \
  --extra src/features.py --extra src/landmark_frontend.py

.venv/bin/python -I scripts/release/build_bundle.py verify "$V6/provisional.zip" \
  --model-sha a012f2b5f7a1110961079f19eb0e28c9abacb95e4d0787352e4157ebca7044a4

.venv/bin/python -I scripts/release/build_bundle.py verify "$V6/provisional.zip" \
  --clip '/absolute/path/to/6523145816470133-WHAT FOR.mp4' \
  --model-sha a012f2b5f7a1110961079f19eb0e28c9abacb95e4d0787352e4157ebca7044a4
# Expected: structure_verified; extracted WHATFOR1 (confidence 0.2138)
# ZIP SHA-256 41c5a938c3f59f60b44bb1ce267195ab8a6943c70fed823899ef39a2feb3ea95,
# 6,609,760 bytes, 13 members (of the recorded V6 build)
```

Official build (expected to **fail by design** without organizer input):

```bash
.venv/bin/python -I scripts/release/build_bundle.py build ... --official
# ERROR: Official bundle requires organizer-generated predictions.jsonl
```

Do not work around this refusal — there is no official ZIP to fabricate.

## 9. Safety audit reproduction

```bash
.venv/bin/python evaluation/device/privacy_lab/audit.py --root . \
  --out evaluation/device/privacy_lab/findings.json
.venv/bin/python evaluation/device/privacy_lab/reproduce_defects.py \
  --out evaluation/device/privacy_lab/reproductions.json
.venv/bin/python -m unittest evaluation.device.privacy_lab.test_privacy_lab -v
# 7 tests OK
```

## 10. Offline demo (blocked prerequisites)

```bash
.venv/bin/python app/offline_demo.py
# Requires granted camera permission. Locally blocked (macOS denied);
# Android not implemented. Record the offline clip-path demo instead
# (docs/demo/DEMO_SCRIPT.md) until the checklist is complete.
```
