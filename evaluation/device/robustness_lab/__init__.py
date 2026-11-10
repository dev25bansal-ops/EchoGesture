"""Robustness lab: deterministic config, paired statistics and evidence reporting.

This package sits beside ``robustness.py``/``corruptions.py`` and is the
source of truth for how paired results are summarised.  It never runs a
model itself; ``robustness.py`` drives inference and hands raw pairs to
:mod:`robustness_lab.stats` for aggregation.

Representative marking rule (enforced in ``robustness.py`` and mirrored
here for reporting): a run may be marked ``representative`` only when the
caller explicitly declares a documented labelled validation split
(``--validation --split validation``), the batch has >= 20 clips, every
clip has ground truth, at least 2 distinct signers are present, and there
are zero clean/corrupted/decode failures.  Anything else is a transform
check, however large the headline drop looks.
"""

from .config import RunConfig, build_config, config_hash, derive_clip_seed
from .stats import aggregate_pairs, wilson_interval
from .reporting import render_markdown, write_evidence

__all__ = [
    "RunConfig",
    "build_config",
    "config_hash",
    "derive_clip_seed",
    "aggregate_pairs",
    "wilson_interval",
    "render_markdown",
    "write_evidence",
]
