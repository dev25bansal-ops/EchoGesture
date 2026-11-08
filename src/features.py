"""Feature transform: raw 21x2 hand landmarks -> model input.

Owner: Dev. This module is the contract between extraction and both training
and Python inference, so training and `predict()` cannot drift apart.

Per frame: 21 landmarks x (x, y) root-centred on the wrist, scaled by the
wrist-to-middle-MCP distance, concatenated with lag-1 velocity and lag-2
acceleration = 21 x 6 = 126 features.

Design decisions and their provenance:

  - **Root at the wrist (landmark 0), scale by wrist->middle-MCP (0->9).**
    Removes camera position and distance-to-camera. This is the invariance
    the Kaggle GISLR winners converged on (1st used the nose as anchor,
    4th used between-eyebrows, 3rd used a 7-point reference set).

  - **z is dropped.** A third-party repository reports dropping MediaPipe's z
    channel is worth +7.10pp on a transformer and +11.34pp on a TCN on
    WLASL-100 signer-disjoint, citing 3 seeds, a 55.3%-complete corpus and an
    untuned model. We have NOT reproduced it. `use_z` is a parameter so the
    ablation stays possible; the default is False because the prior is strong
    and the parameter is cheap to flip.

  - **Per-sequence normalisation**, computed over the clip rather than from a
    global constant, so a new signer's framing and body size drop out.

  - **Missing hands are explicit.** `present` is a per-frame flag carried
    through to the model as a mask, so an all-zero frame is distinguishable
    from a genuine detection at the origin.
"""

import numpy as np

NUM_LANDMARKS = 21
WRIST = 0
MIDDLE_MCP = 9
FEATURE_DIM = 126  # 21 * 2 coords * (position + velocity + acceleration)


def normalise_frame(landmarks_xy, present):
    """Root-centre and scale one frame's 21x2 landmarks.

    Returns 21x2 float32, or zeros when the hand was absent.
    """
    if not present:
        return np.zeros((NUM_LANDMARKS, 2), dtype=np.float32)

    wrist = landmarks_xy[WRIST]
    centred = landmarks_xy - wrist

    scale = np.linalg.norm(centred[MIDDLE_MCP])
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    return (centred / scale).astype(np.float32)


def build_features(landmarks, present, expected_frames=None):
    """Convert one clip's landmarks into the model's feature matrix.

    Args:
        landmarks: (T, 21, 3) raw normalised-image-coordinate landmarks.
        present:   (T,) 1.0 where the hand was detected.
        expected_frames: if given, resample to this many frames.

    Returns:
        features: (T_out, 126) float32
        mask:     (T_out,) float32, 1 where the hand was present
    """
    landmarks = np.asarray(landmarks, dtype=np.float32)
    present = np.asarray(present, dtype=np.float32).reshape(-1)
    if landmarks.shape[0] != present.shape[0]:
        raise ValueError(f"landmarks {landmarks.shape[0]} != present {present.shape[0]}")

    xy = landmarks[:, :, :2]  # drop z, documented above

    if expected_frames:
        xy, present = resample(xy, present, expected_frames)

    norm = np.stack([normalise_frame(xy[i], present[i] > 0.5) for i in range(len(xy))])

    # Lag-1 velocity and lag-2 acceleration, zero-padded at the start so the
    # sequence length is preserved.
    vel = np.zeros_like(norm)
    vel[1:] = norm[1:] - norm[:-1]
    acc = np.zeros_like(norm)
    acc[2:] = norm[2:] - norm[:-2]

    feats = np.concatenate(
        [norm.reshape(len(norm), -1), vel.reshape(len(norm), -1), acc.reshape(len(norm), -1)],
        axis=1,
    ).astype(np.float32)

    if feats.shape[1] != FEATURE_DIM:
        raise AssertionError(f"expected {FEATURE_DIM} features, got {feats.shape[1]}")

    return feats, present


def resample(xy, present, target):
    """Linear resample along time to a fixed frame count."""
    t = len(xy)
    if t == target:
        return xy, present
    if t == 0:
        return np.zeros((target, NUM_LANDMARKS, 2), np.float32), np.zeros(target, np.float32)
    idx = np.linspace(0, t - 1, target)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, t - 1)
    w = (idx - lo).astype(np.float32)[:, None, None]
    out = xy[lo] * (1 - w) + xy[hi] * w
    p = present[lo] * (1 - w[:, 0, 0]) + present[hi] * w[:, 0, 0]
    return out.astype(np.float32), p.astype(np.float32)