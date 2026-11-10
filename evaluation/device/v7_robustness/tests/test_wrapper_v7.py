import numpy as np
import sys
from pathlib import Path

# Import wrapper_v7 from this checkout (the sibling package), not a stale
# per-run agent scratch path that does not survive the session.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wrapper_v7 import apply_v7

def make(T=8):
    lm = np.zeros((T, 2, 21, 3), np.float32)
    pr = np.zeros((T, 2), np.float32)
    lm[:5, 0, 0, :2] = [[0.1, 0.5], [0.11, 0.5], [0.12, 0.5], [0.13, 0.5], [0.14, 0.5]]
    lm[5:, 0, 0, :2] = [[0.3, 0.5], [0.31, 0.5], [0.32, 0.5]]
    pr[:8, 0] = 1.0
    return lm, pr

def test_baseline_exact_copy():
    lm, pr = make()
    lm2, pr2 = apply_v7(lm, pr, "baseline")
    assert np.array_equal(lm, lm2) and np.array_equal(pr, pr2)
    assert lm2 is not lm

def test_smooth_preserves_presence_and_gaps_are_interpolated():
    lm, pr = make()
    pr[3, 0] = 0.0  # single-frame dropout
    lm[3, 0] = np.nan
    lm2, pr2 = apply_v7(lm, pr, "smooth_present_frames", max_gap=3)
    assert pr2[3, 0] == 0.0, "presence must not be fabricated"
    assert np.isfinite(lm2[3, 0]).all()
    assert np.allclose(lm2[3, 0], 0.5 * (lm[2, 0] + lm[4, 0]))

def test_smooth_leaves_long_gaps_and_edges():
    lm, pr = make()
    pr[2:5, 0] = 0.0
    lm2, pr2 = apply_v7(lm, pr, "smooth_present_frames", max_gap=3)
    assert (pr2[2:5, 0] == 0).all()

def test_absent_zero_zeroes_stale_values():
    lm, pr = make()
    lm[3, 1] = 5.0
    pr2 = pr.copy(); pr2[3, 1] = 0.0
    lm2, pr2 = apply_v7(lm, pr2, "absent_slot_zero")
    assert (lm2[3, 1] == 0).all()

def test_unknown_wrapper_raises():
    import pytest
    with pytest.raises(ValueError):
        apply_v7(*make(), "nope")
