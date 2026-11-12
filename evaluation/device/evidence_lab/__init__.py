"""Evidence hardening helpers: traceability, confidence semantics, size accounting.

This package only *reads* existing run_manifest / robustness / benchmark JSON
and the model files on disk. It never fabricates measurements: any quantity
that cannot be derived from supplied evidence is reported as NOT_MEASURED.
"""

from .confidence import describe_output_semantics, softmax_confidence
from .provenance import ProvenanceError, ProvenanceRecord, extract_provenance
from .sizes import SizeBreakdown, artifact_inventory, total_model_bytes
from .per_signer import per_signer_accuracy, per_signer_rows

__all__ = [
    "ProvenanceError",
    "ProvenanceRecord",
    "extract_provenance",
    "describe_output_semantics",
    "softmax_confidence",
    "SizeBreakdown",
    "artifact_inventory",
    "total_model_bytes",
    "per_signer_accuracy",
    "per_signer_rows",
]
