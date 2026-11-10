"""Tests for V7 landmark candidates (no detector or model assets required)."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from candidates import apply_candidate, short_gap_interpolation, single_slot_continuity

# A hand at x=0.3 in the left slot, then the same hand reported in the right
# slot at x=0.3: crossed distance 0.0 versus reported distance 0.3.
lm = np.zeros((4, 2, 21, 3), np.float32)
pr = np.zeros((4, 2), np.float32)
lm[0, 0, :, 0] = 0.3
lm[1:, 1, :, 0] = 0.3
pr[0] = [1, 0]
pr[1] = [0, 1]
pr[2] = [0, 1]
pr[3] = [0, 1]
repaired, repaired_pr, repairs = single_slot_continuity(lm, pr)
assert repairs[0]["frame"] == 1
assert repaired_pr[1].tolist() == [1.0, 0.0]
assert np.array_equal(repaired[1, 0], lm[1, 1])
assert np.array_equal(repaired[1, 1], np.zeros((21, 3), np.float32))

# A legitimate transition: the new slot's pose is at x=0 while the prior slot's
# pose was at x=1, so the reported-slot alternative is much closer.
lm = np.zeros((3, 2, 21, 3), np.float32)
pr = np.zeros((3, 2), np.float32)
lm[0, 0, :, 0] = 1.0
lm[1, 1, :, 0] = 0.0
lm[2, 1, :, 0] = 0.05
pr[:] = [[1, 0], [0, 1], [0, 1]]
output, output_pr, repairs = single_slot_continuity(lm, pr)
assert repairs == []
assert output_pr[1].tolist() == [0.0, 1.0]

# One-frame internal gaps are linearly interpolated and explicitly imputed.
lm = np.zeros((5, 2, 21, 3), np.float32)
pr = np.zeros((5, 2), np.float32)
lm[:, 0, :, 0] = np.array([0.0, 0.2, 0.0, 0.6, 0.8])[:, None]
pr[:, 0] = [1, 1, 0, 1, 1]
filled, filled_pr, imputed, gaps = short_gap_interpolation(lm, pr, max_gap=1)
assert filled[2, 0, 0, 0] == np.float32(0.4)
assert filled_pr[2, 0] == 1.0 and imputed[2, 0]
assert gaps == [{"hand": 0, "start": 2, "end": 2, "length": 1}]

# Prefix gaps are not extrapolated.
lm = np.zeros((3, 2, 21, 3), np.float32)
pr = np.zeros((3, 2), np.float32)
pr[1:, 0] = 1
_, pr, imputed, gaps = short_gap_interpolation(lm, pr, max_gap=2)
assert pr[0, 0] == 0.0 and not imputed.any() and gaps == []

# Baseline is an exact, detached copy.
lm = np.arange(24, dtype=np.float32).reshape(2, 2, 3, 2)
pr = np.zeros((2, 2), np.float32)
copy, copy_pr, info = apply_candidate(lm, pr)
assert np.array_equal(copy, lm) and np.array_equal(copy_pr, pr)
assert info == {"candidate": "baseline", "slot_repairs": [], "imputed_slot_frames": [], "gap_events": []}
copy[0, 0, 0, 0] = 999
assert lm[0, 0, 0, 0] != 999

print("v7_tracking candidate tests passed")
