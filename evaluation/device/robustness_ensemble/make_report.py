"""Render REPORT.md from the measured results written by the driver.

Every number in the report comes from ``results/{condition}.json`` and
``results/runtime.json``; nothing about accuracy is hardcoded here.  The
budget verdict is the only policy in this file: a condition passes when its
percentage-point drop (eligible denominator, failures counted wrong) is at
most ``BUDGET_PP``.

Usage:

    .venv/bin/python evaluation/device/robustness_ensemble/make_report.py
"""

from __future__ import annotations

import json
from pathlib import Path

from run_ensemble_robustness import percentage_point_drop

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
CONDITIONS = ("low_light", "motion_blur", "occlusion", "crop", "frame_drop")
BUDGET_PP = 5.0


def _fmt(value, digits=1):
    return "n/a" if value is None else f"{value:.{digits}f}"


def load(condition):
    return json.loads((RESULTS / f"{condition}.json").read_text())


def render():
    clean = load("clean")
    reencode = load("reencode")
    runtime = json.loads((RESULTS / "runtime.json").read_text())
    clean_summary = clean["summary"]
    reencode_summary = reencode["summary"]
    n = clean_summary["n_eligible"]
    rows = []
    for condition in CONDITIONS:
        payload = load(condition)
        summary = payload["summary"]
        drop = summary["pp_drop_eligible"]
        verdict = "PASS" if drop is not None and drop <= BUDGET_PP else "FAIL"
        rows.append((condition, payload, summary, verdict))

    lines = []
    lines.append("# Ensemble robustness on real video (val_30)")
    lines.append("")
    lines.append(
        "Measured with the shipped 4-model FP16 landmark ensemble through the real "
        "`src/main.py` `predict()` entry point on the labelled `val_30` clips. "
        "Corruptions reuse `evaluation/device/corruptions.py` (version "
        f"{clean['corruption_version']}) at the lab default severities and a fixed seed."
    )
    lines.append("")
    lines.append("## Run configuration")
    lines.append("")
    lines.append(f"- Clips: {n} (`val_30`, sorted by clip_id, deterministic prefix)")
    lines.append(f"- Signers: {len(clean['signers'])} ({', '.join(clean['signers'])})")
    lines.append(f"- Seed: {clean['seed']}; per-clip seeds derived via `derive_clip_seed(seed, index, clip_id)`")
    lines.append(f"- Tracker: {clean['tracker']} ({clean['workers']} workers), parity-verified on 2 clips")
    lines.append(f"- Runtime: {runtime['runtime_seconds']} s for {runtime['predict_calls']} predict calls "
                 f"({len(runtime['conditions'])} conditions x {n} clips)")
    lines.append(f"- Manifest SHA-256: `{clean['manifest_sha256']}`")
    lines.append("")
    lines.append("## Model SHAs (measured artifacts)")
    lines.append("")
    for name, digest in sorted(clean["model_sha256"].items()):
        lines.append(f"- `{name}`: `{digest}`")
    lines.append("")
    lines.append("## Per-condition results")
    lines.append("")
    lines.append(
        "`corrupted %` (eligible) counts every clip, with a no-hand failure counted wrong. "
        "`scored %` excludes clips where the pipeline produced no prediction. "
        "The pp drop uses the eligible denominator, so it is the deployment risk. "
        "The re-encode control decodes and re-encodes each clip with no pixel change; "
        "the `vs re-encode` column therefore removes the mp4 round-trip from the corruption effect."
    )
    lines.append("")
    lines.append("| Condition | Severity | N | Clean % (eligible) | Corrupted % (eligible) | "
                 "Corrupted % (scored) | pp drop vs clean | pp drop vs re-encode | "
                 "no-hand | wrong | other errors | Budget |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    lines.append(
        f"| clean | - | {clean_summary['n_eligible']} | "
        f"{_fmt(100 * clean_summary['accuracy_eligible'])} | - | "
        f"{_fmt(100 * clean_summary['accuracy_scored'])} | - | - | "
        f"{clean_summary['n_no_hand']} | {clean_summary['n_wrong']} | "
        f"{clean_summary['n_error']} | - |"
    )
    lines.append(
        f"| re-encode control | - | {reencode_summary['n_eligible']} | "
        f"{_fmt(100 * reencode_summary['accuracy_eligible'])} | "
        f"{_fmt(100 * reencode_summary['accuracy_eligible'])} | "
        f"{_fmt(100 * reencode_summary['accuracy_scored'])} | "
        f"{_fmt(percentage_point_drop(clean_summary['n_correct'], reencode_summary['n_correct'], clean_summary['n_eligible']))} | "
        f"{_fmt(0.0)} | "
        f"{reencode_summary['n_no_hand']} | {reencode_summary['n_wrong']} | "
        f"{reencode_summary['n_error']} | - |"
    )
    for condition, payload, summary, verdict in rows:
        lines.append(
            f"| {condition} | {payload['severity']} | {summary['n_eligible']} | "
            f"{_fmt(100 * summary['clean_accuracy_eligible'])} | "
            f"{_fmt(100 * summary['accuracy_eligible'])} | "
            f"{_fmt(100 * summary['accuracy_scored'])} | "
            f"{_fmt(summary['pp_drop_eligible'])} | "
            f"{_fmt(summary['pp_drop_vs_reencode'])} | "
            f"{summary['n_no_hand']} | {summary['n_wrong']} | {summary['n_error']} | "
            f"{verdict} ({_fmt(BUDGET_PP)} pt) |"
        )
    lines.append("")
    lines.append("## Failure-stage split")
    lines.append("")
    lines.append(
        "For each corrupted clip whose result is not correct, the pipeline either "
        "produced no hand at all (detection failure) or produced a wrong label. "
        "The clean condition is shown for reference."
    )
    lines.append("")
    lines.append("| Condition | non-correct clips | no-hand (detection) | wrong prediction | flips correct->wrong | flips wrong->correct |")
    lines.append("|---|---|---|---|---|---|")
    lines.append(
        f"| clean | {clean_summary['n_eligible'] - clean_summary['n_correct']} | "
        f"{clean_summary['n_no_hand']} | {clean_summary['n_wrong']} | - | - |"
    )
    for condition, payload, summary, _ in rows:
        lines.append(
            f"| {condition} | {summary['n_eligible'] - summary['n_correct']} | "
            f"{summary['n_no_hand']} | {summary['n_wrong']} | "
            f"{summary['flipped_correct_to_wrong']} | {summary['flipped_wrong_to_correct']} |"
        )
    lines.append("")
    failing = [c for c, _, s, v in rows if v == "FAIL"]
    passing = [c for c, _, s, v in rows if v == "PASS"]
    lines.append("## Budget verdict")
    lines.append("")
    lines.append(
        f"Budget: at most {_fmt(BUDGET_PP)} percentage points of top-1 loss versus clean, "
        "with failures counted wrong (eligible denominator)."
    )
    lines.append("")
    if passing:
        lines.append(f"- Within budget: {', '.join(passing)}.")
    if failing:
        lines.append(f"- Exceeds budget: {', '.join(failing)}.")
    else:
        lines.append("- No condition exceeds the budget on this sample.")
    lines.append("")
    lines.append("## Limits")
    lines.append("")
    lines.append(
        f"- N={n} with {len(clean['signers'])} signers; per-condition differences of 1 clip "
        f"are {_fmt(100 / n)} pt. Treat single-clip differences as noise."
    )
    lines.append(
        "- One severity per corruption (the lab default), one seed. This is a point "
        "measurement of the shipped ensemble on real video, not a severity sweep."
    )
    lines.append(
        "- Corrupted clips are re-encoded to mp4 before prediction while the clean clip is "
        "read from source. The `re-encode control` row (decode + re-encode, no pixel "
        "change) measures that path difference directly, and `pp drop vs re-encode` "
        "attributes any remaining loss to the corruption itself."
    )
    lines.append(
        "- A negative pp drop means the corrupted variant scored above clean on this "
        "sample; at N=30 that is one or two clips and is within sampling noise."
    )
    (HERE / "REPORT.md").write_text("\n".join(lines) + "\n")
    return passing, failing


if __name__ == "__main__":
    print(render())