"""Summarise experiments_a02.py output: per-variant stage split and parity.

Read-only helper; takes the results JSON as argv[1] and prints a table.

Usage:
    .venv/bin/python evaluation/device/performance_final/summarize_candidates.py \
        evaluation/device/performance_final/results_candidates/experiments.json
"""

from __future__ import annotations

import json
import statistics
import sys


def main(path: str) -> int:
    data = json.load(open(path, encoding="utf-8"))
    base = data["runs"].get(data.get("control", "mirror_4"))
    print(f"clips={data['clip_count']} head={data['git_head'][:12]}")
    print(f"{'variant':14s} {'p50':>7s} {'p95':>7s} {'min':>7s} {'rss':>7s} {'dp50':>7s} {'dp95':>7s}"
          f" {'pred':>5s} {'top5':>5s} {'tensor':>6s}")
    print(f"{'variant':14s} {'worst_round_p95':>15s}")
    for variant, run in data["runs"].items():
        print(f"{variant:14s} {max(run.get('round_p95', [float('nan')])):15.2f}")
    for variant, run in data["runs"].items():
        w = run["warm"]
        cmp = data["comparisons_vs_baseline"].get(variant)
        dp50 = f"{cmp['p50_delta_ms']:+7.2f}" if cmp else "      -"
        dp95 = f"{cmp['p95_delta_ms']:+7.2f}" if cmp else "      -"
        pred = len(cmp["prediction_mismatches"]) if cmp else 0
        top5 = len(cmp["top5_mismatches"]) if cmp else 0
        tensor = len(cmp["tensor_mismatches"]) if cmp else 0
        print(f"{variant:14s} {w['p50_ms']:7.2f} {w['p95_ms']:7.2f} {w['min_ms']:7.2f} "
              f"{run['peak_rss_mib']:7.1f} {dp50} {dp95} {pred:5d} {top5:5d} {tensor:6d}")

    print("\nstage means (ms) per variant")
    keys = ["video_open_decode", "read_ms", "decode_in_producer_ms", "landmark_tracking",
            "detect_ms", "pipelined_total_ms", "feature_preparation", "features_ms",
            "model_inference", "ensemble_ms"]
    print(f"{'variant':14s} " + " ".join(f"{k[:16]:>16s}" for k in keys))
    for variant, run in data["runs"].items():
        cells = []
        for key in keys:
            vals = [r["stages"][key] for r in run["records"] if key in r["stages"]]
            cells.append(f"{statistics.mean(vals):16.2f}" if vals else f"{'-':>16s}")
        print(f"{variant:14s} " + " ".join(cells))

    if base:
        print("\nper-clip baseline p50 drivers (sorted by time)")
        for rec in sorted(base["records"], key=lambda r: -r["ms"]):
            print(f"  {rec['clip'][:44]:44s} {rec['ms']:7.2f} ms  "
                  f"read={rec['stages'].get('video_open_decode', float('nan')):6.2f} "
                  f"track={rec['stages'].get('landmark_tracking', float('nan')):7.2f}")

    for variant, cmp in data["comparisons_vs_baseline"].items():
        if cmp["prediction_mismatches"]:
            print(f"\n{variant} prediction mismatches:")
            for m in cmp["prediction_mismatches"]:
                print(f"  {m['clip'][:44]:44s} {m['baseline']} -> {m['variant']}")
        if cmp["top5_mismatches"]:
            print(f"{variant} top5 mismatched clips: {len(cmp['top5_mismatches'])}")
            for clip in cmp["top5_mismatches"]:
                print(f"  {clip[:44]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))