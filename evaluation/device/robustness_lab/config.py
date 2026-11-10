"""Deterministic configuration for paired robustness runs.

A run is fully described by (corruption kind, severity, seed, split,
validation flag, manifest SHA, model SHA).  Per-clip corruption seeds are
derived deterministically from the global seed and the ordered position of
the clip in the manifest, so re-running the same manifest + config
reproduces byte-identical corrupted inputs regardless of wall-clock,
threading, or dict/hash ordering.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

VALID_CORRUPTIONS = ("occlusion", "low_light", "motion_blur", "crop", "frame_drop")

# Default severity levels per corruption kind, all in (0, 1].
DEFAULT_SEVERITY = {
    "occlusion": 0.5,
    "low_light": 0.5,
    "motion_blur": 0.5,
    "crop": 0.3,
    "frame_drop": 0.5,
}

# Minimum batch size / signer diversity required before a result may be
# called representative.  Mirrors robustness.py's --validation gate.
MIN_REPRESENTATIVE_CLIPS = 20
MIN_REPRESENTATIVE_SIGNERS = 2


@dataclass(frozen=True)
class RunConfig:
    """Immutable description of one paired robustness run."""

    corruption: str
    severity: float
    seed: int
    split: str = "unspecified"
    validation: bool = False
    manifest_sha256: str = ""
    model_sha256: str = ""

    def validate(self):
        if self.corruption not in VALID_CORRUPTIONS:
            raise ValueError(
                f"Unsupported corruption {self.corruption!r}; "
                f"expected one of {VALID_CORRUPTIONS}"
            )
        if not 0 < self.severity <= 1:
            raise ValueError(f"severity must be in (0, 1], got {self.severity!r}")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError(f"seed must be an int, got {self.seed!r}")
        return self

    def to_dict(self):
        return asdict(self)

    def canonical_json(self):
        """Stable serialization used for the config hash."""
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )

    def hash(self):
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def build_config(corruption, severity=None, seed=0, split="unspecified",
                 validation=False, manifest_sha256="", model_sha256=""):
    """Build and validate a :class:`RunConfig`, applying default severity."""
    if severity is None:
        if corruption not in DEFAULT_SEVERITY:
            raise ValueError(f"Unsupported corruption: {corruption!r}")
        severity = DEFAULT_SEVERITY[corruption]
    cfg = RunConfig(
        corruption=corruption,
        severity=float(severity),
        seed=int(seed),
        split=split,
        validation=bool(validation),
        manifest_sha256=manifest_sha256 or "",
        model_sha256=model_sha256 or "",
    )
    return cfg.validate()


def config_hash(config):
    """SHA-256 of a RunConfig's canonical JSON ( accepts dict or RunConfig)."""
    if isinstance(config, dict):
        cfg = RunConfig(**{k: config[k] for k in RunConfig.__dataclass_fields__ if k in config})
    else:
        cfg = config
    return cfg.hash()


def derive_clip_seed(global_seed, clip_index, clip_id=""):
    """Derive a deterministic per-clip seed.

    Uses SHA-256 over ``f"{global_seed}:{clip_index}:{clip_id}"`` and takes
    the first 8 bytes as an unsigned 64-bit int.  Deterministic across
    platforms and Python builds; the clip_id is mixed in so reordering the
    manifest cannot silently reuse a seed for a different clip.
    """
    payload = f"{int(global_seed)}:{int(clip_index)}:{clip_id}"
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")
