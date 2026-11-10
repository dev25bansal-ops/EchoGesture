"""Feature transform: raw hand landmarks -> model input.

Owner: Dev. This module is the contract between extraction and both training
and Python inference, so training and `predict()` cannot drift apart.

Input is TWO hands per frame in canonical left/right slots, because most ASL
signs are two-handed and handedness is semantically load-bearing.

Per hand: 21 landmarks x (x, y), root-centred on that hand's wrist, scaled by a
shared body-referenced measure, concatenated with lag-1 velocity and lag-2
acceleration. Hands are scaled by the SAME factor so their relative positions
survive - that is how signs like MOTHER (chin) vs FATHER (forehead) stay
distinguishable.

    2 hands x 21 landmarks x 2 coords x 3 (pos + vel + acc) = 252
  + 3 inter-hand features (distance, x offset, y offset)     = 255

Design decisions and their provenance:

  - **Shared scale across hands.** Scaling each hand independently would
    destroy the hand-to-hand and hand-to-face geometry that separates minimal
    pairs. The scale factor comes from whichever hand is present, preferring
    the larger wrist-to-middle-MCP distance.

  - **Inter-hand geometry is measured before centring.** `normalise_hands`
    roots each hand on its own wrist, which necessarily puts both wrists at
    the origin. So the hand-to-hand distances are computed from the raw
    landmarks first; computing them after centring yields a constant zero,
    which is a bug this module shipped with briefly.

  - **z is dropped.** A third-party repository reports dropping MediaPipe's z
    channel is worth +7.10pp on a transformer and +11.34pp on a TCN on
    WLASL-100 signer-disjoint, from 3 seeds on a 55.3%-complete corpus, untuned.
    We have NOT reproduced it. `use_z` exists so the ablation stays possible;
    the default is False because the prior is strong and the flip is cheap.

  - **Per-sequence normalisation** over the clip, not a global constant, so a
    new signer's framing and body size drop out.

  - **Missing hands are explicit.** `present` is a per-frame, per-hand flag
    carried to the model as a mask, so an all-zero slot is distinguishable
    from a real detection at the origin.
"""

import numpy as np

NUM_LANDMARKS = 21
HANDS = 2          # 0 = Left, 1 = Right, matching the extractor's slots
WRIST = 0
MIDDLE_MCP = 9

# 2 hands * 21 landmarks * 2 coords * 3 channels, plus 3 inter-hand features
# (wrist-to-wrist distance, and the signed x/y offset between the wrists).
FEATURE_DIM = HANDS * NUM_LANDMARKS * 2 * 3 + 3


def _hand_scale(hand_xy):
    """Wrist-to-middle-MCP distance for one 21x2 hand, or None if degenerate."""
    if hand_xy is None:
        return None
    d = np.linalg.norm(hand_xy[MIDDLE_MCP] - hand_xy[WRIST])
    return float(d) if np.isfinite(d) and d > 1e-6 else None


def normalise_hands(hands_xy, present):
    """Root-centre both hands and scale them by a shared factor.

    Args:
        hands_xy: (2, 21, 2) landmarks for [Left, Right].
        present:  (2,) float flags.

    Returns:
        (2, 21, 2) float32. Absent hands are zeros.

    Centring each hand on its own wrist is what makes the model signer
    invariant, but it also forces both wrists to the origin - so this
    function alone cannot express where the hands are relative to each
    other. Inter-hand geometry is computed by `inter_hand_geometry` BEFORE
    the centring, which is why that function takes the raw landmarks.
    """
    scales = [_hand_scale(hands_xy[i]) if present[i] > 0.5 else None for i in range(HANDS)]
    valid = [s for s in scales if s is not None]
    scale = float(np.mean(valid)) if valid else 1.0

    out = np.zeros((HANDS, NUM_LANDMARKS, 2), dtype=np.float32)
    for i in range(HANDS):
        if present[i] > 0.5 and scales[i] is not None:
            out[i] = (hands_xy[i] - hands_xy[i][WRIST]) / scale
    return out


def _shared_scale(hands_xy, present):
    """Mean wrist-to-middle-MCP distance over present hands, or 1.0."""
    scales = [_hand_scale(hands_xy[i]) if present[i] > 0.5 else None for i in range(HANDS)]
    valid = [s for s in scales if s is not None]
    return float(np.mean(valid)) if valid else 1.0


def inter_hand_geometry(hands_xy, present):
    """Hand-to-hand measurements that per-hand centring would destroy.

    Returns (3,) float32:
        [0] wrist-to-wrist distance / shared scale
        [1] x offset of right wrist from left wrist / shared scale
        [2] y offset of right wrist from left wrist / shared scale

    The x/y offsets are signed so the model can tell a sign made high on the
    body (FATHER, forehead) from one made low (MOTHER, chin) even when the
    hand shapes are similar. All three are zero when either hand is missing,
    so "absent" never masquerades as "hands exactly together".
    """
    if present[0] < 0.5 or present[1] < 0.5:
        return np.zeros(3, np.float32)
    scale = _shared_scale(hands_xy, present)
    d = (hands_xy[1][WRIST] - hands_xy[0][WRIST]) / scale
    return np.array([np.linalg.norm(d), d[0], d[1]], np.float32)


def build_features(landmarks, present, expected_frames=None, use_z=False):
    """Convert one clip's landmarks into the model's feature matrix.

    Args:
        landmarks: (T, 2, 21, 3) raw landmarks, or (T, 21, 3) single-hand.
        present:   (T, 2) flags, or (T,) single-hand.
        expected_frames: resample to this many frames if given.
        use_z:     include the z channel (ablation switch; default off).

    Returns:
        features: (T_out, FEATURE_DIM) float32
        mask:     (T_out,) float32, 1 where at least one hand was present
    """
    landmarks = np.asarray(landmarks, dtype=np.float32)
    present = np.asarray(present, dtype=np.float32)

    if landmarks.ndim == 3:                      # (T, 21, C) single-hand input
        # Place it in the right-hand slot and mark the left as absent, so the
        # feature vector keeps its 255-wide layout either way.
        T = landmarks.shape[0]
        two = np.zeros((T, HANDS) + landmarks.shape[1:], np.float32)
        two[:, 1] = landmarks
        landmarks = two
        p = np.zeros((T, HANDS), np.float32)
        p[:, 1] = np.asarray(present, np.float32).reshape(-1)
        present = p

    if landmarks.shape[0] != present.shape[0]:
        raise ValueError(f"landmarks {landmarks.shape[0]} != present {present.shape[0]}")
    if landmarks.shape[1] != HANDS:
        raise ValueError(f"expected {HANDS} hand slots, got {landmarks.shape[1]}")

    xy = landmarks[..., : 3 if use_z else 2]

    if expected_frames:
        xy, present = _resample(xy, present, expected_frames)

    T = len(xy)
    norm = np.stack([normalise_hands(xy[t], present[t]) for t in range(T)])  # (T,2,21,C)

    vel = np.zeros_like(norm)
    vel[1:] = norm[1:] - norm[:-1]
    acc = np.zeros_like(norm)
    acc[2:] = norm[2:] - norm[:-2]

    flat = np.concatenate(
        [norm.reshape(T, -1), vel.reshape(T, -1), acc.reshape(T, -1)], axis=1
    )

    # Inter-hand geometry must be measured on the RAW landmarks: normalise_hands
    # centres each hand on its own wrist, which forces both wrists to the
    # origin and makes every hand-to-hand distance identically zero. This was
    # a live bug - the wrist_d feature carried std 0.0 until it was moved here.
    geom = np.stack([inter_hand_geometry(xy[t], present[t]) for t in range(T)])

    feats = np.concatenate([flat, geom], axis=1).astype(np.float32)

    if feats.shape[1] != FEATURE_DIM:
        raise AssertionError(f"expected {FEATURE_DIM} features, got {feats.shape[1]}")

    mask = (present.max(axis=1) > 0.5).astype(np.float32)
    return feats, mask


def _resample(xy, present, target):
    """Linear resample along time to a fixed frame count."""
    t = len(xy)
    if t == target:
        return xy, present
    if t == 0:
        return (np.zeros((target, HANDS, NUM_LANDMARKS, xy.shape[-1]), np.float32),
                np.zeros((target, HANDS), np.float32))
    idx = np.linspace(0, t - 1, target)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, t - 1)
    w = (idx - lo).astype(np.float32)[:, None, None, None]
    out = xy[lo] * (1 - w) + xy[hi] * w
    pw = (idx - lo).astype(np.float32)[:, None]
    p = present[lo] * (1 - pw) + present[hi] * pw
    return out.astype(np.float32), p.astype(np.float32)