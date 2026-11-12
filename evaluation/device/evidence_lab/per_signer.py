"""Per-signer accuracy, derived only when benchmark data supplies it.

run_manifest.py records signer_id per clip but its summary only carries
aggregate accuracy. The ASL Citizen research evaluation (evaluation/
asl_citizen_*_results.json and asl_citizen_predictions.csv) *does* supply
per_signer counts, so per-signer accuracy is computable there. For
run_manifest benchmark JSON, per-signer accuracy can only be derived if
predictions.jsonl rows carry signer_id alongside ground truth — when they do
not, the honest answer is "not supplied", never an estimate.

Formats accepted (all read-only):
  - {"per_signer": {"P15": [correct, total], ...}}  (research JSON)
  - predictions rows: [{"signer_id":..., "prediction":..., "ground_truth":...}]
    or manifest-aligned rows with participant/signer fields.
"""

from __future__ import annotations


def _coerce_pair(value):
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return int(value[0]), int(value[1])
        except (TypeError, ValueError):
            return None
    if isinstance(value, dict):
        correct = value.get("correct")
        total = value.get("total", value.get("n"))
        if correct is not None and total is not None:
            try:
                return int(correct), int(total)
            except (TypeError, ValueError):
                return None
    return None


def per_signer_accuracy(benchmark=None, predictions=None):
    """Return {signer: (correct, total)} or None if not derivable.

    Never raises on missing signer data; returns None so callers can emit
    NOT_MEASURED-style notes instead of fabricating a breakdown.
    """
    if isinstance(benchmark, dict):
        per = benchmark.get("per_signer")
        if isinstance(per, dict) and per:
            out = {}
            for signer, pair in per.items():
                coerced = _coerce_pair(pair)
                if coerced is not None and coerced[1] > 0:
                    out[str(signer)] = coerced
            return out or None
        # predictions embedded in the benchmark dict?
        embedded = benchmark.get("predictions")
        if isinstance(embedded, list) and embedded:
            predictions = embedded
    if predictions:
        out = {}
        seen_signer_field = False
        for row in predictions:
            if not isinstance(row, dict):
                continue
            signer = row.get("signer_id", row.get("participant", row.get("signer")))
            truth = row.get("ground_truth", row.get("true_gloss", row.get("truth")))
            pred = row.get("prediction", row.get("pred_gloss"))
            if signer is None or truth is None or pred is None:
                continue
            seen_signer_field = True
            signer = str(signer)
            correct, total = out.get(signer, (0, 0))
            total += 1
            if str(pred).upper() == str(truth).upper():
                correct += 1
            out[signer] = (correct, total)
        if seen_signer_field and out:
            return out
    return None


def per_signer_rows(benchmark=None, predictions=None, *, min_clips=1):
    """CSV-ready per-signer rows: metric, measured %, n, status.

    Returns [] when signer data is not supplied, so callers can omit the
    breakdown rather than emit zeros.
    """
    per = per_signer_accuracy(benchmark, predictions)
    if not per:
        return []
    rows = []
    for signer in sorted(per):
        correct, total = per[signer]
        if total < min_clips:
            continue
        rows.append({
            "metric": f"top1_accuracy_signer_{signer}",
            "measured": 100 * correct / total,
            "unit": "%",
            "correct": correct,
            "total": total,
        })
    return rows
