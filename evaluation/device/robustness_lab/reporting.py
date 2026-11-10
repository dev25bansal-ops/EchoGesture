"""Evidence reporting: markdown tables + machine-readable JSON/CSV.

Reports are explicit about what was *not* measured.  A run that is not
representative is labelled as a transform check in every table, and
small-sample CIs are printed so a reader can see the uncertainty rather
than just a point estimate.
"""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone
from pathlib import Path

from . import stats


def _fmt_pct(value):
    return "NOT_MEASURED" if value is None else f"{100 * value:.1f}%"


def _fmt_pp(value):
    return "NOT_MEASURED" if value is None else f"{value:+.1f} pt"


def _fmt_ci(ci):
    lo, hi = ci
    if lo is None or hi is None:
        return "NOT_MEASURED"
    return f"[{100 * lo:.1f}%, {100 * hi:.1f}%]"


def render_markdown(summary, pairs=None):
    """Render a markdown evidence report from a robustness summary dict."""
    agg = stats.aggregate_pairs(pairs if pairs is not None else summary.get("_pairs_for_report", []))
    # Prefer aggregates already embedded in the summary when present so
    # that re-rendering a saved JSON gives identical numbers.
    if summary.get("sample_count") and not pairs:
        agg = {
            "sample_count": summary["sample_count"],
            "clean_correct": summary.get("clean_correct", 0),
            "corrupted_correct": summary.get("corrupted_correct", 0),
            "clean_top1": summary.get("clean_top1"),
            "corrupted_top1": summary.get("corrupted_top1"),
            "accuracy_drop_percentage_points": summary.get("accuracy_drop_percentage_points"),
            "paired_flip_count": summary.get("paired_flip_count", 0),
            "paired_flip_rate": summary.get("paired_flip_rate"),
            "failed_pair_count": summary.get("failed_pair_count", 0),
            "failure_breakdown": summary.get("failure_breakdown", {}),
            "clean_top1_ci95": summary.get("clean_top1_ci95", [None, None]),
            "corrupted_top1_ci95": summary.get("corrupted_top1_ci95", [None, None]),
            "per_signer": summary.get("per_signer", {}),
            "signer_count": summary.get("signer_count", 0),
            "prediction_agreement_rate": summary.get("prediction_agreement_rate"),
        }

    representative = bool(summary.get("representative_robustness_measured"))
    label = "REPRESENTATIVE VALIDATION" if representative else "TRANSFORM CHECK (not representative)"
    lines = [
        "# EchoGesture — paired robustness evidence",
        "",
        f"Recorded: {summary.get('recorded_at', datetime.now(timezone.utc).isoformat())}",
        f"Status: **{label}**",
        "",
        "## Run identity",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| corruption | `{summary.get('corruption_type', 'unknown')}` severity {summary.get('corruption_severity', '?')} (seed {summary.get('seed', '?')}) |",
        f"| split | `{summary.get('split', 'unspecified')}` |",
        f"| config hash | `{summary.get('config_hash', 'not recorded')}` |",
        f"| model SHA-256 | `{summary.get('model_sha256', 'not recorded')}` |",
        f"| manifest SHA-256 | `{summary.get('manifest_sha256', 'not recorded')}` |",
        f"| code commit | `{summary.get('code_commit') or 'not recorded'}` |",
        "",
        "## Paired accuracy (clean vs corrupted)",
        "",
        "| Arm | Correct | N | Top-1 | 95% Wilson CI |",
        "|---|---|---|---|---|",
        f"| clean | {agg['clean_correct']} | {agg['sample_count']} | "
        f"{_fmt_pct(agg['clean_top1'])} | {_fmt_ci(agg['clean_top1_ci95'])} |",
        f"| corrupted | {agg['corrupted_correct']} | {agg['sample_count']} | "
        f"{_fmt_pct(agg['corrupted_top1'])} | {_fmt_ci(agg['corrupted_top1_ci95'])} |",
        "",
        f"- **Accuracy drop (clean − corrupted): {_fmt_pp(agg['accuracy_drop_percentage_points'])}**",
        f"- Paired flips (clean correct → corrupted wrong): "
        f"{agg['paired_flip_count']} ({_fmt_pct(agg['paired_flip_rate'])})",
        f"- Prediction agreement clean vs corrupted: {_fmt_pct(agg['prediction_agreement_rate'])}",
        f"- Failed pairs: {agg['failed_pair_count']}"
        + (f" — {agg['failure_breakdown']}" if agg["failure_breakdown"] else ""),
        "",
        "## Per-signer coverage",
        "",
    ]
    per_signer = agg.get("per_signer") or {}
    if per_signer:
        lines += [
            "| Signer | N | Clean top-1 | Corrupted top-1 | Drop | Failed |",
            "|---|---|---|---|---|---|",
        ]
        for signer, slot in sorted(per_signer.items()):
            lines.append(
                f"| {signer} | {slot['sample_count']} | {_fmt_pct(slot['clean_top1'])} | "
                f"{_fmt_pct(slot['corrupted_top1'])} | {_fmt_pp(slot['accuracy_drop_percentage_points'])} | "
                f"{slot['failed']} |"
            )
    else:
        lines.append("No signer breakdown available (no signer_id in manifest).")
    lines.append("")
    lines += [
        "## Interpretation limits",
        "",
        "- Small samples: the 95% Wilson intervals above are the honest "
        "uncertainty; do not claim a target is met from a point estimate "
        "whose interval crosses it.",
        "- This report describes the corruption applied in this run only.",
        "- Representative status requires a documented labelled validation "
        "split, >= 20 clips, >= 2 signers, and zero failures.",
        "",
    ]
    return "\n".join(lines)


def pairs_to_csv(pairs):
    """Serialize pairs to CSV text (one row per clip)."""
    fields = [
        "clip_id", "signer_id", "ground_truth", "clean_prediction",
        "corrupted_prediction", "clean_correct", "corrupted_correct",
    ]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields)
    writer.writeheader()
    for pair in pairs:
        writer.writerow({k: pair.get(k, "") for k in fields})
    return buf.getvalue()


def write_evidence(output_dir, summary, pairs):
    """Write robustness.json-like summary, pairs.jsonl, pairs.csv and report.md."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = dict(summary)
    summary["recorded_at"] = summary.get("recorded_at") or datetime.now(timezone.utc).isoformat()
    # Strip any in-memory helper keys before writing JSON.
    summary.pop("_pairs_for_report", None)
    json_path = output_dir / "robustness.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "pairs.jsonl").write_text(
        "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pairs), encoding="utf-8"
    )
    (output_dir / "pairs.csv").write_text(pairs_to_csv(pairs), encoding="utf-8")
    (output_dir / "report.md").write_text(render_markdown(summary, pairs), encoding="utf-8")
    return {"json": str(json_path), "jsonl": str(output_dir / "pairs.jsonl"),
            "csv": str(output_dir / "pairs.csv"), "report": str(output_dir / "report.md")}
