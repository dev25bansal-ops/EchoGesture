"""A10 V7 evaluation-device harness (isolated copy layer).

Thin wrappers so evaluation runs against the V7 app layer instead of the
shared demo, plus a model-identity checker pinned to the FP16 candidate
SHA-256 from the task brief.
"""

from .identity import (  # noqa: F401
    EXPECTED_FP16_SHA256,
    check_model_identity,
    model_identity_report,
)

__all__ = ["EXPECTED_FP16_SHA256", "check_model_identity", "model_identity_report"]
