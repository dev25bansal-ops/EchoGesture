"""Paired statistics for robustness runs.

All functions here are pure: they take the list of pair dicts produced by
``robustness.paired_evaluate`` and return plain dicts/numbers.  Failure
accounting is explicit — every attempted pair is classified, and any
pair with a failed clean or corrupted prediction is counted in the
denominator *and* in the failure breakdown, never silently dropped.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

# Wilson score interval z for a 95% confidence interval.
_Z95 = 1.959963984540054


def wilson_interval(successes, trials, z=_Z95):
    """Wilson score CI for a binomial proportion.

    Returns ``(low, high)`` clipped to [0, 1].  Returns ``(None, None)``
    for zero trials so callers never print a fabricated interval next to
    an empty sample.
    """
    if trials == 0:
        return (None, None)
    n = float(trials)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(max(0.0, p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def _signer_of(pair):
    return pair.get("signer_id") or "unknown"


def _classify(pair):
    """Return one of: stable_correct, flip_correct_to_wrong, wrong_to_correct,
    stable_wrong, failed (any side failed)."""
    if pair.get("clean_prediction") is None or pair.get("corrupted_prediction") is None:
        return "failed"
    clean_ok = bool(pair.get("clean_correct"))
    corr_ok = bool(pair.get("corrupted_correct"))
    if clean_ok and corr_ok:
        return "stable_correct"
    if clean_ok and not corr_ok:
        return "flip_correct_to_wrong"
    if not clean_ok and corr_ok:
        return "wrong_to_correct"
    return "stable_wrong"


def aggregate_pairs(pairs):
    """Aggregate a list of pair dicts into summary statistics.

    Each pair must carry: clip_id, ground_truth, clean_prediction,
    corrupted_prediction, clean_correct, corrupted_correct.  Optional:
    signer_id, clean_latency_ms, corrupted_latency_ms, error fields.
    """
    pairs = list(pairs)
    n = len(pairs)
    if n == 0:
        return {
            "sample_count": 0,
            "clean_correct": 0,
            "corrupted_correct": 0,
            "clean_top1": None,
            "corrupted_top1": None,
            "accuracy_drop_percentage_points": None,
            "paired_flip_count": 0,
            "paired_flip_rate": None,
            "clean_correct_corrupted_wrong": 0,
            "clean_wrong_corrupted_correct": 0,
            "failed_pair_count": 0,
            "failure_breakdown": {},
            "clean_top1_ci95": [None, None],
            "corrupted_top1_ci95": [None, None],
            "per_signer": {},
            "signer_count": 0,
            "prediction_agreement_rate": None,
        }

    clean_correct = sum(1 for p in pairs if p.get("clean_correct"))
    corrupted_correct = sum(1 for p in pairs if p.get("corrupted_correct"))
    flips = sum(1 for p in pairs if _classify(p) == "flip_correct_to_wrong")
    recovered = sum(1 for p in pairs if _classify(p) == "wrong_to_correct")
    failed = sum(1 for p in pairs if _classify(p) == "failed")
    agreement = sum(
        1 for p in pairs
        if p.get("clean_prediction") is not None
        and p.get("clean_prediction") == p.get("corrupted_prediction")
    )
    failure_breakdown = Counter()
    for p in pairs:
        if p.get("clean_prediction") is None:
            failure_breakdown["clean_prediction_failed"] += 1
        if p.get("corrupted_prediction") is None:
            failure_breakdown["corrupted_prediction_failed"] += 1
        if p.get("decode_error"):
            failure_breakdown["decode_error"] += 1

    per_signer = defaultdict(
        lambda: {"sample_count": 0, "clean_correct": 0, "corrupted_correct": 0, "failed": 0}
    )
    for p in pairs:
        slot = per_signer[_signer_of(p)]
        slot["sample_count"] += 1
        slot["clean_correct"] += int(bool(p.get("clean_correct")))
        slot["corrupted_correct"] += int(bool(p.get("corrupted_correct")))
        slot["failed"] += int(_classify(p) == "failed")
    signer_summary = {}
    for signer, slot in sorted(per_signer.items()):
        count = slot["sample_count"]
        signer_summary[signer] = {
            "sample_count": count,
            "clean_correct": slot["clean_correct"],
            "corrupted_correct": slot["corrupted_correct"],
            "clean_top1": slot["clean_correct"] / count if count else None,
            "corrupted_top1": slot["corrupted_correct"] / count if count else None,
            "accuracy_drop_percentage_points": (
                100 * (slot["clean_correct"] - slot["corrupted_correct"]) / count if count else None
            ),
            "failed": slot["failed"],
        }

    clean_lo, clean_hi = wilson_interval(clean_correct, n)
    corr_lo, corr_hi = wilson_interval(corrupted_correct, n)

    return {
        "sample_count": n,
        "clean_correct": clean_correct,
        "corrupted_correct": corrupted_correct,
        "clean_top1": clean_correct / n,
        "corrupted_top1": corrupted_correct / n,
        "accuracy_drop_percentage_points": 100 * (clean_correct - corrupted_correct) / n,
        "paired_flip_count": flips,
        "paired_flip_rate": flips / n if n else None,
        "clean_correct_corrupted_wrong": flips,
        "clean_wrong_corrupted_correct": recovered,
        "failed_pair_count": failed,
        "failure_breakdown": dict(failure_breakdown),
        "clean_top1_ci95": [clean_lo, clean_hi],
        "corrupted_top1_ci95": [corr_lo, corr_hi],
        "per_signer": signer_summary,
        "signer_count": len(signer_summary),
        "signer_ids": sorted(signer_summary),
        "prediction_agreement_rate": agreement / n if n else None,
    }
