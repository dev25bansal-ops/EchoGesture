"""Provenance / staleness extraction from existing evidence JSON.

Every result row must trace to: candidate SHA, feature config, code revision,
dataset/split, clips/signers, device/runtime, timestamp and method.

This module extracts whatever provenance fields the *existing* run_manifest /
robustness / benchmark JSON actually contain, and raises ProvenanceError when
a required binding (candidate model SHA) is absent or mismatched, instead of
silently accepting stale evidence.
"""

from __future__ import annotations


class ProvenanceError(ValueError):
    """Raised when evidence is stale, unbound, or missing required provenance."""


# Keys checked, in order, for the candidate model SHA in a benchmark dict.
_MODEL_SHA_KEYS = ("model_sha256",)
_ARTIFACT_MODEL_KEYS = ("model",)


def _nested_model_sha(benchmark: dict):
    artifacts = benchmark.get("artifacts") or {}
    for key in _ARTIFACT_MODEL_KEYS:
        entry = artifacts.get(key)
        if isinstance(entry, dict) and entry.get("sha256"):
            return entry["sha256"]
    return None


def model_sha_of(benchmark: dict):
    """Return the model SHA bound by a benchmark/robustness dict, or None."""
    if not isinstance(benchmark, dict):
        return None
    for key in _MODEL_SHA_KEYS:
        value = benchmark.get(key)
        if isinstance(value, str) and value:
            return value
    return _nested_model_sha(benchmark)


class ProvenanceRecord:
    """Traceability fields extracted from one evidence file (read-only)."""

    def __init__(self, benchmark: dict, source_name: str = ""):
        if not isinstance(benchmark, dict):
            raise ProvenanceError(f"Evidence {source_name!r} is not a JSON object")
        self.source = source_name
        self.model_sha256 = model_sha_of(benchmark)
        artifacts = benchmark.get("artifacts") or {}
        self.artifacts = {
            name: (entry.get("sha256"), entry.get("bytes"))
            for name, entry in artifacts.items()
            if isinstance(entry, dict)
        } if isinstance(artifacts, dict) else {}
        self.feature_config_sha256 = self._artifact_sha(artifacts, "feature_config")
        self.code_commit = benchmark.get("code_commit")
        manifest = benchmark.get("manifest") or {}
        self.manifest_sha256 = manifest.get("sha256") if isinstance(manifest, dict) else None
        self.manifest_path = manifest.get("path") if isinstance(manifest, dict) else None
        self.dataset = benchmark.get("dataset") or manifest.get("dataset", "unspecified")
        self.split = benchmark.get("split") or manifest.get("split", "unspecified")
        self.fixture_type = benchmark.get("fixture_type", "unspecified")
        env = benchmark.get("environment") or {}
        self.device = env.get("device_model", "unknown") if isinstance(env, dict) else "unknown"
        self.runtime = env.get("inference_runtime", "unknown") if isinstance(env, dict) else "unknown"
        self.python = env.get("python") if isinstance(env, dict) else None
        self.platform = env.get("platform") if isinstance(env, dict) else None
        self.recorded_at = benchmark.get("recorded_at")
        self.clip_count = benchmark.get("clip_count")
        self.successful = benchmark.get("successful_predictions")
        self.status = benchmark.get("status")
        self.representative_performance = bool(
            benchmark.get("representative_performance_measured"))
        accuracy = benchmark.get("accuracy") or {}
        self.representative_accuracy = bool(accuracy.get(
            "representative_accuracy_measured", False)) if isinstance(accuracy, dict) else False
        self.raw = benchmark

    @staticmethod
    def _artifact_sha(artifacts, name):
        entry = artifacts.get(name) if isinstance(artifacts, dict) else None
        return entry.get("sha256") if isinstance(entry, dict) else None

    def require_bound_to(self, expected_model_sha: str, *, stale_label: str = "evidence"):
        """Reject stale/unbound evidence: candidate SHA must match exactly."""
        if not self.model_sha256:
            raise ProvenanceError(
                f"{stale_label} carries no model SHA; cannot bind it to candidate {expected_model_sha}")
        if self.model_sha256 != expected_model_sha:
            raise ProvenanceError(
                f"Stale {stale_label}: bound to model {self.model_sha256}, "
                f"candidate is {expected_model_sha}; regenerate before release")
        return self

    def trace_string(self) -> str:
        parts = [f"source={self.source or 'unknown'}"]
        parts.append(f"model_sha256={self.model_sha256 or 'NOT_RECORDED'}")
        parts.append(f"feature_config_sha256={self.feature_config_sha256 or 'NOT_RECORDED'}")
        parts.append(f"code_commit={self.code_commit or 'NOT_RECORDED'}")
        parts.append(f"manifest_sha256={self.manifest_sha256 or 'NOT_RECORDED'}")
        parts.append(f"dataset={self.dataset};split={self.split};fixture_type={self.fixture_type}")
        parts.append(f"device={self.device};runtime={self.runtime}")
        parts.append(f"recorded_at={self.recorded_at or 'NOT_RECORDED'}")
        if self.clip_count is not None:
            parts.append(f"clips={self.clip_count};successful={self.successful}")
        return "; ".join(parts)

    def missing_provenance(self):
        """List traceability fields the evidence file does not record."""
        missing = []
        if not self.model_sha256:
            missing.append("model_sha256")
        if not self.code_commit:
            missing.append("code_commit")
        if not self.manifest_sha256:
            missing.append("manifest_sha256")
        if not self.recorded_at:
            missing.append("recorded_at")
        if self.dataset == "unspecified":
            missing.append("dataset")
        if self.split == "unspecified":
            missing.append("split")
        return missing


def extract_provenance(benchmark: dict, source_name: str = "") -> ProvenanceRecord:
    return ProvenanceRecord(benchmark, source_name)
