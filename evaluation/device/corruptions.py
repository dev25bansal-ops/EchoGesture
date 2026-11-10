"""Deterministic visual/temporal corruptions for paired local evaluation.

Every corruption is a pure function of (frames, spec): the same inputs
and spec always produce byte-identical outputs.  Randomized effects
(occlusion position, crop window) draw from a single
``numpy.random.default_rng(spec.seed)`` stream, so the *order* in which
the RNG is consumed is part of the contract and must not change without
bumping the corruption version.  Callers that need per-clip variation
should derive a per-clip seed (see ``robustness_lab.config.derive_clip_seed``)
and pass it as ``spec.seed`` for that clip.

Corruption version: 2.  v1 -> v2 kept every kind/severity mapping
identical; v2 only fixed the occlusion y-range edge case (empty/inverted
range when box_height covers the lower 75% of the frame) and the
frame_drop linspace rounding, both of which changed outputs only for
inputs where v1 could crash or behaved discontinuously.
"""

from dataclasses import dataclass

import cv2
import numpy as np

CORRUPTION_VERSION = 2

VALID_KINDS = ("occlusion", "low_light", "motion_blur", "crop", "frame_drop")


@dataclass(frozen=True)
class CorruptionSpec:
    kind: str
    severity: float = 1.0
    seed: int = 0

    def __post_init__(self):
        if self.kind not in VALID_KINDS:
            raise ValueError(f"Unsupported corruption: {self.kind!r}; expected one of {VALID_KINDS}")
        if not 0 < self.severity <= 1:
            raise ValueError("severity must be in (0, 1]")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError(f"seed must be an int, got {self.seed!r}")


def corrupt_frames(frames, spec):
    """Apply ``spec`` to a list of BGR uint8 frames; returns a new list."""
    frames = list(frames)
    if not frames:
        raise ValueError("Cannot corrupt an empty clip")
    if not isinstance(spec, CorruptionSpec):
        raise ValueError(f"spec must be a CorruptionSpec, got {type(spec).__name__}")
    for frame in frames:
        if frame is None or getattr(frame, "size", 0) == 0:
            raise ValueError("Cannot corrupt a clip containing an empty frame")
    rng = np.random.default_rng(spec.seed)
    height, width = frames[0].shape[:2]
    if spec.kind == "occlusion":
        box_width = max(1, int(width * (0.12 + 0.28 * spec.severity)))
        box_height = max(1, int(height * (0.12 + 0.28 * spec.severity)))
        # Keep the occlusion in the lower 25%..100% band (hand region) but
        # never let the box run past the frame.
        y_low = int(height * 0.25)
        y_high = max(y_low, height - box_height)
        x = int(rng.integers(0, max(1, width - box_width + 1)))
        y = int(rng.integers(y_low, y_high + 1)) if y_high > y_low else y_low
        return [_fill(frame, x, y, box_width, box_height) for frame in frames]
    if spec.kind == "low_light":
        factor = 1 - 0.75 * spec.severity
        return [np.clip(frame.astype(np.float32) * factor, 0, 255).astype(np.uint8) for frame in frames]
    if spec.kind == "motion_blur":
        size = max(1, int(3 + 12 * spec.severity))
        kernel = np.zeros((size, size), np.float32)
        kernel[size // 2, :] = 1 / size
        return [cv2.filter2D(frame, -1, kernel) for frame in frames]
    if spec.kind == "crop":
        fraction = 1 - 0.3 * spec.severity
        crop_width, crop_height = max(2, int(width * fraction)), max(2, int(height * fraction))
        x = int(rng.integers(0, max(1, width - crop_width + 1)))
        y = int(rng.integers(0, max(1, height - crop_height + 1)))
        return [cv2.resize(frame[y:y + crop_height, x:x + crop_width], (width, height), interpolation=cv2.INTER_LINEAR)
                for frame in frames]
    if spec.kind == "frame_drop":
        keep_fraction = 1 - 0.6 * spec.severity
        keep = max(1, int(round(len(frames) * keep_fraction)))
        if keep >= len(frames):
            return [frame.copy() for frame in frames]
        targets = np.unique(np.linspace(0, len(frames) - 1, keep).round().astype(int))
        return [frames[index].copy() for index in targets]
    raise ValueError(f"Unsupported corruption: {spec.kind}")


def _fill(frame, x, y, width, height):
    result = frame.copy()
    result[y:y + height, x:x + width] = 0
    return result


def decode_video(path):
    """Decode every frame from a video file; returns (frames, fps)."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"No frames decoded: {path}")
    return frames, fps if fps > 0 else 30.0


def write_video(path, frames, fps):
    """Write frames to an mp4 file at ``fps``; raises if the writer cannot open."""
    frames = list(frames)
    if not frames:
        raise ValueError("Cannot write an empty clip")
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        writer.release()
        raise ValueError(f"Could not write video: {path}")
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()
