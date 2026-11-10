# v7_benchmark results (auto-generated from paired.json)

recorded_at: 2026-11-08T18:57:59.944488+00:00
manifest: /private/tmp/echogesture-validation-30/manifest.csv sha256=004e5effb7e6c19b515fdd00b048459fff236d8b35c6b7f7b49e91a1efa5d618
hardware_note: Mac laptop (Apple M5 Pro, macOS 26.3 arm64) — NOT target device
hardware_is_target_device: False (target_device_status=blocked_not_target_hardware)

| variant | n | p50_ms | p95_ms | mean_ms | top1 |
| --- | --- | --- | --- | --- | --- |
| FP16 | 30 | 254.0 | 321.0 | 258.5 | 22/30 = 0.7333 |
| INT8 | 30 | 247.7 | 308.7 | 253.3 | 21/30 = 0.7000 |

paired: n=30 agreement=0.8333 wall_delta_p50_ms=-4.85 wall_delta_mean_ms=-5.24

Note: laptop harness check only; NOT target-device evidence. Power: NOT_MEASURED; requires a documented external meter or qualified platform method — do not infer from CPU
Thermal: NOT_MEASURED unless a sustained CSV (>=600s) analyzed by thermal_report.py is attached for this run
