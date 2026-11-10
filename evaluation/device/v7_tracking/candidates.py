"""Explicit V7 landmark candidates; never changes Dev frontend or features."""
from __future__ import annotations

import numpy as np

HANDS = 2
LANDMARKS = 21


def _mean_distance(a, b):
    return float(np.mean(np.linalg.norm(np.asarray(a)[..., :2] - np.asarray(b)[..., :2], axis=-1)))


def _swap(lm, present, t):
    lm[t] = lm[t, ::-1].copy()
    present[t] = present[t, ::-1].copy()


def single_slot_continuity(landmarks, present, ratio=0.6):
    """Repair only adjacent one-hand slot changes supported by pose distance.

    `previous_slot` is the slot reported at t-1 and `other_slot` is the other
    hand. If the current pose is much closer to the previous slot's pose than
    to the current slot's pose, preserve the physical hand's slot at t. The
    missing previous slot contributes its measured zero sentinel only to the
    rejected alternative; no canonical majority mapping is used.
    """
    lm = np.asarray(landmarks, dtype=np.float32).copy()
    pr = np.asarray(present, dtype=np.float32).copy()
    ratio = float(ratio)
    if not 0 < ratio <= 1:
        raise ValueError("ratio must be in (0, 1]")
    repairs = []
    for t in range(1, len(lm)):
        if pr[t - 1].sum() != 1 or pr[t].sum() != 1:
            continue
        previous_slot = int(np.argmax(pr[t - 1] > 0.5))
        reported_slot = int(np.argmax(pr[t] > 0.5))
        if previous_slot == reported_slot:
            continue
        crossed = _mean_distance(lm[t, reported_slot], lm[t - 1, previous_slot])
        reported = _mean_distance(lm[t, reported_slot], lm[t - 1, reported_slot])
        if crossed < ratio * max(reported, 1e-9):
            repairs.append({"frame": t, "from": reported_slot, "to": previous_slot,
                            "crossed": crossed, "reported": reported})
            _swap(lm, pr, t)
    return lm, pr, repairs


def short_gap_interpolation(landmarks, present, max_gap=1):
    """Fill only bounded internal gaps by linear coordinate interpolation.

    Filled slot frames get presence=1 because Dev's feature builder otherwise
    correctly treats a zero/absent slot as absent. The returned `imputed` mask
    exposes every synthetic slot frame for evaluation and downstream policy.
    Prefixes, suffixes, and gaps longer than max_gap remain absent. No
    extrapolation, stale-pose hold, or one-hand canonicalization is used.
    """
    lm = np.asarray(landmarks, dtype=np.float32).copy()
    pr = np.asarray(present, dtype=np.float32).copy()
    max_gap = int(max_gap)
    if max_gap < 1:
        raise ValueError("max_gap must be >= 1")
    imputed = np.zeros_like(pr, dtype=bool)
    gaps = []
    for hand in range(HANDS):
        t = 0
        while t < len(pr):
            if pr[t, hand] > 0.5:
                t += 1
                continue
            start = t
            while t < len(pr) and pr[t, hand] <= 0.5:
                t += 1
            end = t - 1
            length = end - start + 1
            if start == 0 or t == len(pr) or length > max_gap:
                continue
            before = start - 1
            after = t
            if pr[before, hand] <= 0.5 or pr[after, hand] <= 0.5:
                continue
            for j, frame in enumerate(range(start, t), start=1):
                weight = j / (length + 1)
                lm[frame, hand] = (1 - weight) * lm[before, hand] + weight * lm[after, hand]
                pr[frame, hand] = 1.0
                imputed[frame, hand] = True
            gaps.append({"hand": hand, "start": start, "end": end, "length": length})
    return lm, pr, imputed, gaps


def apply_candidate(landmarks, present, candidate="baseline", *, max_gap=1, ratio=0.6):
    """Return baseline or one explicitly selected experimental candidate."""
    lm = np.asarray(landmarks, dtype=np.float32).copy()
    pr = np.asarray(present, dtype=np.float32).copy()
    info = {"candidate": candidate, "slot_repairs": [], "imputed_slot_frames": [], "gap_events": []}
    if candidate == "baseline":
        return lm, pr, info
    if candidate not in {"slot", "gap", "slot_gap"}:
        raise ValueError(f"Unknown V7 candidate {candidate!r}")
    if candidate in {"slot", "slot_gap"}:
        lm, pr, repairs = single_slot_continuity(lm, pr, ratio=ratio)
        info["slot_repairs"] = repairs
    if candidate in {"gap", "slot_gap"}:
        lm, pr, imputed, gaps = short_gap_interpolation(lm, pr, max_gap=max_gap)
        info["imputed_slot_frames"] = imputed.tolist()
        info["gap_events"] = gaps
    info["slot_repairs_count"] = len(info["slot_repairs"])
    info["imputed_count"] = int(sum(sum(x) for x in info["imputed_slot_frames"])) if info["imputed_slot_frames"] else 0
    return lm, pr, info


__all__ = ["apply_candidate", "single_slot_continuity", "short_gap_interpolation"]
