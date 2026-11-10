# EchoGesture V7 benchmark (A11) — laptop harness + target-device protocol

Owner: A11 (Benchmark Engineer). Work only in this directory. Do not edit the original checkout.

## 1. Protocol (reproducible, laptop + target hardware)

Reuses `evaluation/device/run_manifest.py` (`evaluate`/`validate_prediction`,
nearest-rank percentiles, atomic JSONL/JSON writes) and
`evaluation/device/benchmark_lab/matrix.py` / `matrix_runner.py` (one heavy
process at a time, warmup discarded, cold/warm split, stage aggregates,
failures reported separately, power `NOT_MEASURED`). No installs, no nested agents.

### 1.1 INT8 frontend compatibility (checked first)

The INT8 candidate (`9d24640a…23700b`, 509,720 B) was inspected with LiteRT
before any benchmark:

- Input tensor: float32, shape `[1, 24, 255]`, quantization `(0.0, 0)` — identical to FP16.
- Output tensor: float32, shape `[1, 100]`, quantization `(0.0, 0)` — identical to FP16.
- Same `models/landmark_labels.json` vocabulary (100 classes) and same
  `landmark_model_meta.json` (`frames=24, feature_dim=255`).

`src/main.py` only forwards the float32 tensor and applies int8 quantization only
when `_INPUT_SCALE > 0` (false for both), so **the production predictor drives
both variants unchanged**. Only the file path inside `_model_paths()` differs.
`predict_model.py` (this directory) is a benchmark-only wrapper that pins that
path per variant; production defaults (`ECHOGESTURE_MODEL`, FP16 path) are untouched.

### 1.2 Paired FP16/INT8 comparison (identical inputs only)

`paired_runner.py` runs each variant as its own subprocess (isolated model load,
one heavy process at a time) over the **same manifest, same clips, same order**,
then joins them by `clip_id`. It aborts if:

- either candidate file's SHA-256 differs from the pinned candidate hash
  (FP16 `a012f2b5…7044a4`, INT8 `9d24640a…23700b`),
- the two cells' successful clip-id sets differ,
- a non-target run does not record `hardware_note` containing "NOT target device".

Each cell: 1 warmup (discarded) + each clip once, `profile=True`,
external wall time via `perf_counter_ns`, cold = first success, warm = rest.
Cells and the merged `paired.json` are written atomically (tempfile + os.replace).

```bash
<venv>/bin/python -I evaluation/device/v7_benchmark/paired_runner.py \
  /tmp/echogesture-validation-30/manifest.csv \
  --output-dir evaluation/device/v7_benchmark/results/laptop_run \
  --source-root /path/to/checkout          # contains src/main.py and models/
# -> results/laptop_run/{fp16_cell.json,int8_cell.json,paired.json}
```

### 1.3 Sustained / thermal workload

```bash
<venv>/bin/python -I evaluation/device/v7_benchmark/sustained_logger.py \
  /tmp/echogesture-validation-30/manifest.csv \
  --source-root /path/to/checkout --variant fp16 --duration 600 \
  --output results/sustained_fp16.csv --device-model "<exact device model>"
```

Loop the manifest ≥10 min, then analyze with the existing
`evaluation/device/thermal_report.py` (requires ≥600 s; first/last 30 s p95
drift; reports performance drift, not temperature or power). Cooldown ≥5 min
between runs; record ambient, charger, battery %, `vm.loadavg`. Power stays
`NOT_MEASURED` without an external meter or qualified platform method.

## 2. Measured laptop results (2026-11-08, Apple M5 Pro, macOS 26.3 arm64)

Hardware is the **laptop, not the target device** (`hardware_is_target_device: false`,
`target_device_status: blocked_not_target_hardware` in `paired.json`). These are
harness checks, **not** target-device evidence and **not** a ≤50 ms p95 result.

Manifest: `/tmp/echogesture-validation-30/manifest.csv` (30 real ASL Citizen
clips, all with ground truth). Predictor `src/main.py`, labels
`models/landmark_labels.json`, 1 warmup, 30 clips × 2 variants, zero failures.

| Variant | n | p50 (ms) | p95 (ms) | mean (ms) | top1 |
| --- | --- | --- | --- | --- | --- |
| FP16 | 30 | 254.0 | 321.0 | — | 22/30 = 0.7333 |
| INT8 | 30 | 247.7 | 308.7 | — | 21/30 = 0.7000 |

Paired (by clip_id, identical inputs): prediction agreement 0.8333 (25/30),
per-clip wall delta (INT8 − FP16) p50 −4.9 ms, mean −5.2 ms. Stage means
(FP16 → INT8): landmark_tracking 238.2 → 235.0 ms (dominant stage),
video_open_decode 19.8 → 17.8 ms, model_inference 0.10 → 0.08 ms,
feature_preparation ~0.21 ms both, input_quantization ~0.00 ms both.

FP16 and INT8 differ mainly in classifier bytes (832,888 vs 509,720) and a
~0.02 ms inference delta; the landmark frontend dominates wall time in both.

## 3. Blocking: target-device p95 / power / thermal

**No target-device (Samsung XR) hardware was available in this environment.**
The laptop numbers above are therefore never presentable as target-device p95 /
power / thermal evidence:

- Target-device clip p95: **NOT MEASURED** (requires the target device).
- Power (≤500 mW target): **NOT_MEASURED** (requires external meter or qualified method).
- Thermal drift (≤20% over 10 min): **NOT_MEASURED** (requires a ≥600 s sustained
  CSV on the target device analyzed by `thermal_report.py`).

### Target-device runbook (when the device is available)

1. Ensure no other heavy process runs; record ambient, charger, battery %, `vm.loadavg`.
2. Copy this directory (or the checkout) to the device host; **no install** — use the
   project's Python 3.11 venv (`<venv>/bin/python -I`).
3. Run the paired benchmark with the target flags (only on the target device):
   ```bash
   <venv>/bin/python -I evaluation/device/v7_benchmark/paired_runner.py \
     <manifest.csv> --output-dir results/target_run --source-root /path/to/checkout \
     --hardware-is-target --device-model "<exact device model>" \
     --hardware-note "Target device: <exact device model> (<runtime>)"
   ```
   This sets `hardware_is_target_device: true` and `target_device_status: measured`.
   The runner still verifies both candidate hashes and identical inputs before running.
4. Sustained run: `sustained_logger.py --duration 600 --device-model "<exact device model>"`,
   cooldown ≥5 min, then `thermal_report.py <sustained.csv> --model-sha <tflite sha> --window 30`.
5. Report p50/p95/p99 per variant with `count` and method ("nearest-rank, external wall
   time"), stage breakdown, and attach the output dirs outside Git. Never infer
   watts from CPU/temperature; mark unavailable measurements explicitly in
   `results/results.csv`.

## 4. Files

- `predict_model.py` — benchmark-only FP16/INT8 predictor wrapper (no production changes).
- `paired_runner.py` — paired runner + merged `paired.json` (reuses `run_manifest`, `matrix`).
- `sustained_logger.py` — sustained-workload CSV logger for the thermal protocol.
- `test_paired_runner.py` — unit tests for pairing/target-device gating (no heavy inference).
- `results/laptop_run/` — the measured laptop run above (cells + `paired.json`).
- `RESULTS.md` — auto-generated table of the laptop run (see its note on target-device status).
