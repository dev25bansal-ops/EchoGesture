#!/usr/bin/env python3
"""Summarize IMAGE-mode landmark presence and slot geometry in A03 evidence."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path("/tmp/echogesture-v7-agents/A03")
LAB = Path("/tmp/echogesture-agents/A03/evaluation/device/tracking_lab")
VAL = Path("/tmp/echogesture-validation-30/manifest.csv")


def gap_patterns(present):
    out = []
    for h in range(2):
        seq = (present[:, h] > 0.5).astype(int)
        t = 0
        while t < len(seq):
            if seq[t]:
                t += 1
                continue
            start = t
            while t < len(seq) and not seq[t]:
                t += 1
            end = t - 1
            internal = start > 0 and t < len(seq)
            out.append({
                "hand": h,
                "start": start,
                "end": end,
                "length": end - start + 1,
                "internal": internal,
                "prefix": start == 0,
                "suffix": t == len(seq),
            })
    return out


def slot_jump(landmarks, present, t, swapped=False):
    xy = landmarks[..., :2]
    prior = xy[t - 1, ::-1] if swapped else xy[t - 1]
    return float(np.mean(np.linalg.norm(xy[t] - prior, axis=1)))


def main():
    rows = list(csv.DictReader(VAL.open()))
    lm = np.load(LAB / "image_30.landmarks.npy")
    pr = np.load(LAB / "image_30.present.npy")
    result = {
        "clips": len(rows),
        "frames_per_clip": lm.shape[1],
        "slot_frames": pr.shape[1] * pr.shape[2],
        "present_by_hand": pr.sum(axis=(0, 1)).astype(int).tolist(),
        "frames_with_any": int((pr.max(axis=2) > 0.5).sum()),
        "frames_with_both": int((pr.sum(axis=2) == 2).sum()),
        "frames_with_one": int((pr.sum(axis=2) == 1).sum()),
        "frames_with_none": int((pr.sum(axis=2) == 0).sum()),
        "missing_internal_gaps": [],
        "missing_prefix_gaps": [],
        "missing_suffix_gaps": [],
        "strict_two_hand_swaps": [],
        "single_slot_changes": [],
        "clips_with_missing_internal_gap": [],
        "clips_with_single_slot_change": [],
        "clips_with_strict_two_hand_swap": [],
    }
    short_internal = 0
    total_internal = 0
    total_missing = 0
    for i, row in enumerate(rows):
        gaps = gap_patterns(pr[i])
        missing = [g for g in gaps if g["length"] > 0]
        internal = [g for g in missing if g["internal"]]
        total_missing += sum(g["length"] for g in missing)
        total_internal += sum(g["length"] for g in internal)
        short_internal += sum(g["length"] for g in internal if g["length"] <= 2)
        if internal:
            result["clips_with_missing_internal_gap"].append(row["clip_id"])
        result["missing_internal_gaps"].extend([{**g, "clip_id": row["clip_id"]} for g in internal])
        result["missing_prefix_gaps"].extend([{**g, "clip_id": row["clip_id"]} for g in missing if g["prefix"]])
        result["missing_suffix_gaps"].extend([{**g, "clip_id": row["clip_id"]} for g in missing if g["suffix"]])

        p = pr[i] > 0.5
        for t in range(1, len(p)):
            if p[t - 1].all() and p[t].all():
                direct = slot_jump(lm[i], pr[i], t)
                swapped = slot_jump(lm[i], pr[i], t, True)
                if swapped < 0.6 * direct and swapped < direct:
                    item = {
                        "clip_id": row["clip_id"],
                        "frame": t,
                        "direct": direct,
                        "swapped": swapped,
                        "ratio": swapped / max(direct, 1e-9),
                    }
                    result["strict_two_hand_swaps"].append(item)
        for t in range(1, len(p)):
            if p[t - 1].sum() == 1 and p[t].sum() == 1 and p[t - 1, 0] != p[t, 0]:
                left = int(np.argmax(p[t - 1]))
                right = int(np.argmax(p[t]))
                # Distance between observed pose at t and the pose assigned to
                # the previous slot versus the slot MediaPipe reported.
                same_slot = slot_jump(lm[i], pr[i], t)
                crossed = float(np.mean(np.linalg.norm(
                    lm[i, t, right, :2] - lm[i, t - 1, left, :2], axis=1)))
                result["single_slot_changes"].append({
                    "clip_id": row["clip_id"],
                    "frame": t,
                    "from": left,
                    "to": right,
                    "reported_slot_distance": same_slot,
                    "crossed_distance": crossed,
                    "ratio": same_slot / max(crossed, 1e-9),
                })
        if any(c["clip_id"] == row["clip_id"] for c in result["single_slot_changes"]):
            result["clips_with_single_slot_change"].append(row["clip_id"])
        if any(c["clip_id"] == row["clip_id"] for c in result["strict_two_hand_swaps"]):
            result["clips_with_strict_two_hand_swap"].append(row["clip_id"])

    result["missing_counts"] = {
        "total_missing_slot_frames": int(total_missing),
        "total_internal_missing_slot_frames": int(total_internal),
        "internal_gaps_length_le_2_slot_frames": int(short_internal),
        "internal_gap_events": len(result["missing_internal_gaps"]),
        "prefix_gap_events": len(result["missing_prefix_gaps"]),
        "suffix_gap_events": len(result["missing_suffix_gaps"]),
    }
    for name in ("missing_prefix_gaps", "missing_suffix_gaps"):
        result[name + "_slot_frames"] = int(sum(g["length"] for g in result[name]))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
