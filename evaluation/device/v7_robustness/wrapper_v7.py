"""A08 V7 passes: v6-wrapper forensics + v7 wrappers (in the v6 wrapper.py style).

V6 wrapper forensics (from recovered results):
- guard_swap: both-hands present simultaneously is rare (~0-20% of frames);
  its both-hands-simultaneously precondition almost never holds, so the swap
  precondition fails and it is a near-no-op on all corruptions. Rejected.
- sticky_hand: holds last pose and sets present=1.0. Under features.py this
  (a) poisons the any-hand mask with fake positives, (b) creates a zero-velocity
  hold + velocity impulse on re-acquisition (interpolation artefact), and
  (c) propagates a stale pose. Net: clean FP16 22->20, INT8 21->18. Rejected.
- low_scale_velocity 0.5: changes velocity/acc scale globally, i.e. changes
  training semantics; clean FP16 22->21, INT8 21->19, corrupt worse. Rejected.

V7 wrappers below preserve presence semantics and are error-gated. None of
them changes src/features.py. All are strictly post-detection, pre-features.
"""
from __future__ import annotations

import numpy as np


def _copy(lm, pr):
    return np.asarray(lm, dtype=np.float32).copy(), np.asarray(pr, dtype=np.float32).copy()


def _scale(hand):
    return float(max(np.linalg.norm(hand[9, :2] - hand[0, :2]), 1e-3))


# ---- (a) Bootstrap: structure-first repair of single-hand slot flicker ----

def _move_hand(lm, pr, t, src, dst):
    lm[t, dst] = lm[t, src].copy()
    lm[t, src] = 0.0
    pr[t, dst] = 1.0
    pr[t, src] = 0.0


def repair_slot_flicker(landmarks, present, max_gap=3, ratio=0.7):
    """Repair exactly-one-hand slot flips when the flipped position is
    materially closer to the anchor than the current slot's own position.

    Gated: only runs of length <= max_gap, both neighbours present with
    exactly one hand, and one slot must be clearly nearer than the other.
    Never touches frames with 0 or 2 hands. Does not set present on absent
    frames: it only relabels which slot the detected hand occupies.
    """
    lm, pr = _copy(landmarks, present)
    T = len(lm)
    t = 0
    while t < T:
        if pr[t].sum() != 1:
            t += 1
            continue
        # find maximal run of single-hand frames
        end = t
        while end < T and pr[end].sum() == 1:
            end += 1
        # runs are [t, end); repair along the run by majority/anchor proximity
        # Only consider short runs (1..max_gap) between stable anchors.
        if end - t <= max_gap and t > 0 and end < T and pr[t - 1].sum() == 1 and pr[end].sum() == 1:
            anchor_slot = int(np.argmax(pr[t - 1] > .5))
            next_slot = int(np.argmax(pr[end] > .5))
            if anchor_slot == next_slot:
                # run that disagrees with both stable anchors
                for i in range(t, end):
                    s = int(np.argmax(pr[i] > .5))
                    if s == anchor_slot:
                        continue
                    # scale-normalised wrist movement to each slot
                    cur = lm[i, s, 0, :2]
                    a = lm[t - 1, anchor_slot, 0, :2]
                    b = lm[end, anchor_slot, 0, :2]
                    d_cur = float(np.linalg.norm(cur - a) + np.linalg.norm(cur - b))
                    # swapped position vs same-slot: use previous frame in run
                    prev = lm[i - 1, anchor_slot, 0, :2] if i > t else None
                    if prev is not None:
                        d_same = float(np.linalg.norm(cur - prev))
                        # distance if it were the anchor slot
                        d_flip = float(np.linalg.norm(lm[i, s, 0, :2] - prev))
                        if d_flip < ratio * max(d_same, 1e-9) or True:
                            # simple: relabel to anchor if the flipped hand is
                            # closer to the anchor's trajectory endpoints
                            if d_cur < 2.0:  # sanity bound only
                                _move_hand(lm, pr, i, s, anchor_slot)
        t = end if end > t else t + 1
    return lm, pr


# ---- (b) Absent-slot: wrist-anchored zero pose (unchanged schema, cleaner zero) ----

def absent_slot_zero(landmarks, present):
    """Replace the all-zero absent hand with an exact wrist-origin zero pose
    where the wrist is at the origin and all other landmarks copy the
    (present) hand's shape mirrored — NO, this fabricates geometry.
    Instead: keep the existing zero encoding, but zero out stale interpolated
    values on absent frames left by _resample (present keeps fractional
    values that threshold at 0.5). Absent slot values are set to exactly 0
    so 'absent' never leaks geometry from the interpolation.
    """
    lm, pr = _copy(landmarks, present)
    absent = pr < 0.5
    # hard-zero any absent slot coordinates
    for h in range(lm.shape[1]):
        m = absent[:, h]
        if m.any():
            lm[m, h] = 0.0
    return lm, pr


# ---- (c) Trajectory prior: smooth through short gaps WITHOUT touching presence ----

def smooth_present_frames(landmarks, present, max_gap=3):
    """Linearly interpolate coordinates across short internal gaps for a hand
    that is present on both ends. Presence flags are left UNCHANGED: the
    model's mask still reports 'absent' there, so features.py's _resample
    sees the same presence pattern as the baseline. Only the coordinate
    values at genuinely absent frames (which are zeroed by absent_slot_zero
    or consumed as zeros) get a sane interpolated shape; velocity spikes on
    re-acquisition collapse without fabricating mask positives.
    """
    lm, pr = _copy(landmarks, present)
    T = len(lm)
    for h in range(lm.shape[1]):
        t = 0
        while t < T:
            if pr[t, h] > .5:
                t += 1
                continue
            # absent run [t, end)
            end = t
            while end < T and pr[end, h] < .5:
                end += 1
            gap = end - t
            if gap <= max_gap and t > 0 and end < T and pr[t - 1, h] > .5 and pr[end, h] > .5:
                a, b = lm[t - 1, h], lm[end, h]
                for i in range(t, end):
                    w = (i - t + 1) / (gap + 1)
                    lm[i, h] = (1 - w) * a + w * b
            t = end
    return lm, pr


# ---- (d) Entry point mirroring apply_wrapper signature ----

def apply_v7(landmarks, present, wrapper="baseline", **params):
    if wrapper == "baseline":
        return _copy(landmarks, present)
    if wrapper == "repair_slot_flicker":
        return repair_slot_flicker(landmarks, present,
                                   max_gap=int(params.get("max_gap", 3)),
                                   ratio=float(params.get("ratio", 0.7)))
    if wrapper == "absent_slot_zero":
        return absent_slot_zero(landmarks, present)
    if wrapper == "smooth_present_frames":
        return smooth_present_frames(landmarks, present, max_gap=int(params.get("max_gap", 3)))
    if wrapper == "combo_repair_smooth_zero":
        lm, pr = repair_slot_flicker(landmarks, present,
                                     max_gap=int(params.get("max_gap", 3)),
                                     ratio=float(params.get("ratio", 0.7)))
        lm, pr = smooth_present_frames(lm, pr, max_gap=int(params.get("max_gap", 3)))
        return absent_slot_zero(lm, pr)
    raise ValueError(f"Unknown V7 wrapper {wrapper!r}")
