# Thermal workload protocol (sustained workload)

Scope: performance drift under sustained load on the **actual** measurement
device. Laptop (macOS arm64) numbers are harness checks, not target-device
evidence. Power is `NOT_MEASURED` — never infer watts from CPU or temperature.

## Preconditions

- One heavy process at a time; no other benchmark running (check `ps aux`).
- Airflow/ambient: record room temp, device on desk, charger state, battery %.
- Close other CPU-heavy apps; note `vm.loadavg` and AC/battery.
- Fix thread count to runtime default (record it); pin ECHOGESTURE_MODEL.
- Never tune on the locked test set; use the validation batch.

## Workload definition

- Manifest: `/tmp/echogesture-validation-30/manifest.csv` (30 real ASL Citizen
  clips, SHA checked at run time — see `manifest_sha` in the summary).
- Command (single process, serialized):
  ```
  ECHOGESTURE_MODEL=landmark \
  <venv>/bin/python evaluation/device/benchmark_lab/matrix_runner.py \
    /tmp/echogesture-validation-30/manifest.csv \
    --output-dir <out> --repeats 1 --warmup 1 --profile
  ```
- Sustained phase: loop the 30-clip manifest continuously for >= 10 minutes,
  logging one row per clip call to CSV with columns:
  `timestamp_seconds,latency_ms,clip_id,device,temperature_c,thermal_status`
  (`temperature_c`/`thermal_status` only if the platform exposes them without
  sudo; otherwise leave blank — do not fabricate).
- Cooldown: >= 5 min idle between sustained runs; record cooldown duration.
- Cold/warm: first call after process start = cold; warmup calls discarded;
  report cold and warm separately (already split in the matrix summary).

## Sampling & analysis

- Use `evaluation/device/thermal_report.py`:
  ```
  <venv>/bin/python evaluation/device/thermal_report.py <sustained.csv> \
    --model-sha <sha256 of landmark_classifier_fp16.tflite> --window 30
  ```
  It requires >= 600 s elapsed and >= 2 samples; reports
  `thermal_drift_10min = 100*(late_p95 - early_p95)/early_p95` over the first
  and last 30 s windows. It does not estimate temperature change or power.

## Recording (per run)

- `timestamp_seconds` monotonic from run start; wall-clock date/time UTC.
- Manifest SHA-256, model SHA-256, predictor SHA-256, labels SHA-256.
- Environment: python, platform, machine, LiteRT version, delegate, threads.
- Hardware note: exact device model; `hardware_is_target_device` bool.
- Ambient, charger, battery, loadavg, cooldown duration.
- Failures listed separately with `clip_id` + `error_type` (never dropped).

## Reporting rules

- Report `p50/p90/p95/p99` with nearest-rank method and `count` per cell;
  do not call N=30 a target-device p95 unless hardware == actual target and
  the conditions above are recorded.
- Report stage breakdown (`video_open_decode`, `landmark_tracking`,
  `feature_preparation`, `model_inference`, `output_postprocessing`) with
  `count/mean/median/min/max/p90` per stage.
- Power field stays `NOT_MEASURED` without an external meter or qualified
  platform method.
- Results stay outside Git; only measured numbers are reported.
