"""Explicit model-confidence semantics: logits in, softmax confidence out.

The landmark FP16/INT8 classifiers emit float32 *logits* (verified: outputs
lie outside [0, 1] and do not sum to 1), while the S3D INT8 models emit
quantized logits dequantized via (q - zp) * scale. In `src/main.py` both paths
apply softmax over the dequantized logits and report `confidence` as the
top-1 softmax probability. Confidence is therefore a softmax probability
derived from logits — never a raw model output and never a calibrated
probability unless calibration has been measured (it has not).

These helpers make the semantics explicit and testable without a model file:
given raw logits they reproduce exactly the post-processing in main.py.
"""

from __future__ import annotations

import math


def softmax_confidence(logits):
    """Return (top1_index, top1_probability, top5_indices) via softmax.

    Mirrors src/main.py: exp(logit - max) / sum(exp(logit - max)), descending
    order. Probabilities sum to 1; the top-1 probability is the reported
    `confidence`. Works on any finite numeric sequence (no numpy dependency
    so the semantics can be unit-tested anywhere).
    """
    values = [float(v) for v in logits]
    if not values:
        raise ValueError("logits must be non-empty")
    if any(not math.isfinite(v) for v in values):
        raise ValueError("logits must be finite")
    peak = max(values)
    exps = [math.exp(v - peak) for v in values]
    total = sum(exps)
    probs = [e / total for e in exps]
    order = sorted(range(len(probs)), key=lambda i: -probs[i])
    return order[0], probs[order[0]], order[:5]


def describe_output_semantics(*, output_scale=0.0, output_zero_point=0,
                               output_dtype="float32", observed=None):
    """Describe whether a model's raw output is logits or probabilities.

    `observed` is an optional list of raw-output samples (e.g. from one
    real inference). If every sample lies in [0, 1] and sums to ~1 the output
    behaves like probabilities; otherwise it is logits and confidence must
    come from softmax. When no sample is supplied the quantization params
    alone decide: a nonzero output scale means quantized logits that
    dequantize to logits; scale 0 with float32 IO is treated as logits
    unless a sample proves otherwise (the landmark models are in this class).
    """
    semantics = "logits"
    basis = "default: float32 output treated as logits; softmax required for confidence"
    if observed:
        values = [float(v) for v in observed]
        in_range = all(0.0 <= v <= 1.0 for v in values)
        sums_to_one = abs(sum(values) - 1.0) < 1e-3
        if in_range and sums_to_one:
            semantics = "probabilities"
            basis = "observed sample lies in [0,1] and sums to 1"
        else:
            semantics = "logits"
            basis = "observed sample outside [0,1] or does not sum to 1"
    elif output_scale:
        semantics = "logits"
        basis = f"quantized output (scale={output_scale}, zp={output_zero_point}) dequantizes to logits"
    return {
        "output_semantics": semantics,
        "confidence_definition": (
            "top-1 softmax probability over dequantized logits"
            if semantics == "logits" else "raw top-1 output value"),
        "calibrated": False,
        "calibration_note": (
            "No calibration (temperature scaling / ECE) has been measured; "
            "confidence is an uncalibrated softmax score, not a probability of correctness."),
        "basis": basis,
        "output_dtype": output_dtype,
    }
