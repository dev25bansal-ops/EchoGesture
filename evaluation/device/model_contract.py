"""Validate a Dev model bundle before integration.

The manifest is a local integration contract, not a claim that the organizer
accepts the vocabulary. All paths are relative to the manifest's directory.

Strictness rules (A05, 8 Nov 2026):
  - No silent fallback: a missing / mismatched artifact raises
    CompatibilityError; this module never substitutes another model, vocab,
    or quantization variant.
  - Quantization is identified from the actual TFLite tensor census, never
    from a filename. The shipped FP16 artifact (float16 weights, float32 IO)
    is ``fp16_float32_io`` and must never be reported as INT8.
  - Output semantics (logits vs probabilities) are probed by invoking the
    interpreter on deterministic inputs, not trusted from the manifest alone.

Candidate under test (integration state / origin/hackathon/dev-model):
  models/landmark_classifier_fp16.tflite
    SHA-256 a012f2b5f7a1110961079f19eb0e28c9abacb95e4d0787352e4157ebca7044a4
    832888 bytes, input [1,24,255] float32, output [1,100] float32 logits,
    feature dim 255, frames 24, classes 100, FP16 weights / float32 IO.
  models/landmark_labels.json (ordered gloss -> index, 100 entries)
  models/landmark_model_meta.json  {frames:24, feature_dim:255, classes:100,
    quantization:"float16"}
  models/hand_landmarker.task (MediaPipe detector, ZIP/TFLite-task asset)

The INT8 alternative (models/landmark_classifier_int8.tflite,
SHA 9d24640a23244a3cd00c9fa62257733819f86ed1da5a10194c9b1cb64323700b,
509720 bytes) is identified as ``int8_mixed_float_io``: float32 IO with
82 int8-quantized internal tensors. It is a different artifact and is
rejected as the FP16 candidate (and vice versa) by exact SHA + census.
"""

import hashlib
import json
from collections import Counter
from pathlib import Path

REQUIRED = (
    "model_sha256", "model_size_bytes", "model_format", "input_tensor_shape",
    "input_dtype", "input_scale", "input_zero_point", "output_tensor_shape",
    "output_dtype", "output_scale", "output_zero_point",
    "output_is_logits_or_probabilities", "ordered_vocabulary",
    "feature_config_version", "feature_shape", "frame_count", "sampling_method",
    "normalization", "handedness", "presence_masks", "coordinate_system",
    "missing_landmark_policy",
)

CANDIDATE_VERSION = "landmark-fp16-v1"

CANDIDATE_SPEC = {
    "version": CANDIDATE_VERSION,
    "model_filename": "landmark_classifier_fp16.tflite",
    "model_sha256": "a012f2b5f7a1110961079f19eb0e28c9abacb95e4d0787352e4157ebca7044a4",
    "model_size_bytes": 832888,
    "model_format": "tflite",
    "input_tensor_shape": [1, 24, 255],
    "input_dtype": "float32",
    "output_tensor_shape": [1, 100],
    "output_dtype": "float32",
    "output_is_logits_or_probabilities": "logits",
    "feature_shape": [24, 255],
    "frame_count": 24,
    "feature_dim": 255,
    "classes": 100,
    "quantization": "float16",  # as declared in landmark_model_meta.json
    "quantization_classification": "fp16_float32_io",
    "labels_filename": "landmark_labels.json",
    "labels_sha256": "e78f5028e97e2eaf86aff65af403af65061eebef010f0fbd41012293e8019a4a",
    "meta_filename": "landmark_model_meta.json",
    "detector_filename": "hand_landmarker.task",
    "detector_sha256": "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1",
}

# Filename for the rejected-as-candidate alternative, kept for explicit
# identification tests. Its presence is never an error; selecting it as the
# candidate is.
INT8_ALT_SPEC = {
    "model_filename": "landmark_classifier_int8.tflite",
    "model_sha256": "9d24640a23244a3cd00c9fa62257733819f86ed1da5a10194c9b1cb64323700b",
    "model_size_bytes": 509720,
    "quantization_classification": "int8_mixed_float_io",
}


class CompatibilityError(Exception):
    pass


def tensor_contract(interpreter):
    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()
    if len(inputs) != 1 or len(outputs) != 1:
        raise CompatibilityError("Expected exactly one model input and one output")
    result = {}
    for name, detail in (("input", inputs[0]), ("output", outputs[0])):
        scale, zero = detail["quantization"]
        result[f"{name}_tensor_shape"] = [int(size) for size in detail["shape"]]
        result[f"{name}_dtype"] = detail["dtype"].__name__
        result[f"{name}_scale"] = float(scale)
        result[f"{name}_zero_point"] = int(zero)
    return result


def quantization_census(interpreter):
    """Count internal tensor dtypes via LiteRT's full tensor details.

    Returns a dict with per-dtype counts, how many int8 tensors carry a
    non-zero scale (i.e. are truly quantized, not just int8 constants),
    and how many float16 tensors exist. A census is required to tell
    FP16 (float16 weights) apart from INT8 (int8 quantized weights) when
    both expose float32 IO -- the IO tensors alone cannot distinguish them.
    """
    try:
        details = interpreter.get_tensor_details()
    except Exception as exc:  # fake interpreters in unit tests
        raise CompatibilityError(f"Cannot census tensor details: {exc}") from exc
    counts = Counter(d["dtype"].__name__ for d in details)
    int8_quantized = sum(
        1
        for d in details
        if d["dtype"].__name__ == "int8" and float(d["quantization"][0]) != 0.0
    )
    return {
        "total_tensors": len(details),
        "dtype_counts": dict(counts),
        "float16_tensors": int(counts.get("float16", 0)),
        "int8_tensors": int(counts.get("int8", 0)),
        "int8_quantized_tensors": int(int8_quantized),
        "float32_tensors": int(counts.get("float32", 0)),
    }


def identify_quantization(interpreter):
    """Classify an artifact's quantization from its actual tensors.

    Never claims FP16 is INT8. Classifications:
      - ``fp16_float32_io``: float16 internal tensors present, no int8
        quantized tensors, float32 IO.
      - ``int8_mixed_float_io``: int8 quantized internal tensors present
        but IO is float32 (Dev's INT8 alternative).
      - ``int8_io``: int8 IO (the S3D baseline).
      - ``float32``: no fp16/int8 tensors.
      - ``mixed_unknown``: anything else (caller should reject).
    """
    census = quantization_census(interpreter)
    contract = tensor_contract(interpreter)
    io_float = contract["input_dtype"] == "float32" and contract["output_dtype"] == "float32"
    io_int8 = contract["input_dtype"] == "int8" and contract["output_dtype"] == "int8"
    if census["float16_tensors"] > 0 and census["int8_quantized_tensors"] == 0 and io_float:
        classification = "fp16_float32_io"
    elif census["int8_quantized_tensors"] > 0 and io_float:
        classification = "int8_mixed_float_io"
    elif io_int8 and census["int8_quantized_tensors"] > 0:
        classification = "int8_io"
    elif census["float16_tensors"] == 0 and census["int8_quantized_tensors"] == 0:
        classification = "float32"
    else:
        classification = "mixed_unknown"
    return {"classification": classification, "census": census, **contract}


def probe_output_semantics(interpreter, num_probes=3):
    """Invoke the model on deterministic inputs and classify the output.

    Logits: finite, contain negatives (or at least are not confined to
    [0,1]), and do not sum to 1. Probabilities: all in [0,1] and summing
    to 1 within tolerance. Uses zeros plus seeded random inputs so the
    result is reproducible. Returns a dict; raises CompatibilityError if
    the model cannot be invoked or produces non-finite outputs.
    """
    import numpy as np

    contract = tensor_contract(interpreter)
    shape = contract["input_tensor_shape"]
    dtype_name = contract["input_dtype"]
    dtype = {"float32": np.float32, "float16": np.float16, "int8": np.int8}.get(dtype_name)
    if dtype is None:
        raise CompatibilityError(f"Unsupported input dtype for semantics probe: {dtype_name}")
    rng = np.random.RandomState(1234)
    probes = [np.zeros(shape, dtype=dtype)]
    for _ in range(max(0, num_probes - 1)):
        if dtype == np.int8:
            probes.append(rng.randint(-128, 128, size=shape).astype(dtype))
        else:
            probes.append(rng.randn(*shape).astype(dtype))
    outputs = []
    for x in probes:
        try:
            interpreter.set_tensor(interpreter.get_input_details()[0]["index"], x)
            interpreter.invoke()
            out = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])
        except Exception as exc:
            raise CompatibilityError(f"Cannot invoke model for semantics probe: {exc}") from exc
        outputs.append(np.asarray(out, dtype=np.float32))
    if not all(np.isfinite(o).all() for o in outputs):
        raise CompatibilityError("Model produced non-finite outputs during semantics probe")
    flat = np.concatenate([o.reshape(-1) for o in outputs])
    sums = [float(o.sum()) for o in outputs]
    in_unit = bool((flat >= 0).all() and (flat <= 1).all())
    sums_to_one = all(abs(s - o.size / o.shape[-1] * 1.0) < 1e-3 or abs(s - 1.0) < 1e-3 for o, s in zip(outputs, sums))
    # Per-output sums: each output row is a single distribution over classes.
    row_sums_to_one = all(abs(float(o.reshape(-1)[-o.shape[-1]:].sum()) - 1.0) < 1e-3 for o in outputs)
    has_negative = bool((flat < 0).any())
    if in_unit and row_sums_to_one and not has_negative:
        looks_like = "probabilities"
    elif has_negative or not in_unit:
        looks_like = "logits"
    else:
        looks_like = "unknown"
    return {
        "looks_like": looks_like,
        "finite": True,
        "has_negative": has_negative,
        "in_unit_interval": in_unit,
        "row_sums_to_one": bool(row_sums_to_one),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "probes": len(outputs),
    }


def _load_json_file(path, what):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CompatibilityError(f"Invalid {what}: {exc}") from exc


def _ordered_vocab_from_label_map(label_map):
    if not isinstance(label_map, dict) or not label_map:
        raise CompatibilityError("Vocabulary must be a non-empty label-to-index object")
    try:
        indices = sorted(int(v) for v in label_map.values())
    except (TypeError, ValueError) as exc:
        raise CompatibilityError(f"Vocabulary indices must be integers: {exc}") from exc
    if indices != list(range(len(label_map))):
        raise CompatibilityError("Vocabulary indices must be contiguous from zero")
    ordered = [None] * len(label_map)
    for label, index in label_map.items():
        ordered[int(index)] = str(label).upper()
    if any(not isinstance(label, str) or label != label.upper() for label in ordered):
        raise CompatibilityError("Ordered vocabulary must contain unique uppercase strings")
    if len(set(ordered)) != len(ordered):
        raise CompatibilityError("Ordered vocabulary must contain unique uppercase strings")
    return ordered


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_bundle(directory, interpreter=None):
    directory = Path(directory)
    manifest_path = directory / "integration_manifest.json"
    model_path = directory / "classifier.tflite"
    if not manifest_path.is_file() or not model_path.is_file():
        raise CompatibilityError("integration_manifest.json and classifier.tflite are required")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CompatibilityError(f"Invalid integration manifest: {exc}") from exc
    missing = [key for key in REQUIRED if key not in manifest]
    if missing:
        raise CompatibilityError(f"Missing handoff fields: {', '.join(missing)}")
    if manifest["model_format"] != "tflite":
        raise CompatibilityError("Only TFLite classifier artifacts are supported")
    if manifest["output_is_logits_or_probabilities"] not in ("logits", "probabilities"):
        raise CompatibilityError("Output semantics must be logits or probabilities")
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if digest != manifest["model_sha256"]:
        raise CompatibilityError(f"Model SHA mismatch: expected {manifest['model_sha256']}, got {digest}")
    if model_path.stat().st_size != manifest["model_size_bytes"]:
        raise CompatibilityError("Model byte size mismatch")
    labels = manifest["ordered_vocabulary"]
    if (not isinstance(labels, list) or not labels
            or any(not isinstance(label, str) or label != label.upper() for label in labels)
            or len(set(labels)) != len(labels)):
        raise CompatibilityError("Ordered vocabulary must contain unique uppercase strings")
    if manifest["frame_count"] <= 0:
        raise CompatibilityError("frame_count must be positive")
    if len(manifest["feature_shape"]) != 2 or manifest["feature_shape"][0] != manifest["frame_count"]:
        raise CompatibilityError("feature_shape must be [frame_count, feature_dim]")
    # Optional strict checks, only when the manifest declares them (keeps
    # older manifests valid while letting new handoffs pin them).
    for asset in manifest.get("required_assets", []):
        if not (directory / asset).is_file():
            raise CompatibilityError(f"Missing required asset: {asset}")
    if "labels_file" in manifest:
        labels_path = directory / manifest["labels_file"]
        if not labels_path.is_file():
            raise CompatibilityError(f"Missing labels file: {manifest['labels_file']}")
        declared = _ordered_vocab_from_label_map(_load_json_file(labels_path, "labels file"))
        if declared != labels:
            raise CompatibilityError("Labels file ordering does not match ordered_vocabulary")
    if interpreter is not None:
        actual = tensor_contract(interpreter)
        for field in ("input_tensor_shape", "input_dtype", "input_scale", "input_zero_point",
                      "output_tensor_shape", "output_dtype", "output_scale", "output_zero_point"):
            if actual[field] != manifest[field]:
                raise CompatibilityError(f"{field} mismatch: manifest {manifest[field]!r}, model {actual[field]!r}")
        if actual["output_tensor_shape"][-1] != len(labels):
            raise CompatibilityError("Output class count does not match ordered vocabulary")
        if manifest["input_tensor_shape"][-1] != manifest["feature_shape"][-1]:
            raise CompatibilityError("Model feature width does not match feature_shape")
        if "quantization" in manifest:
            identified = identify_quantization(interpreter)["classification"]
            declared_quant = str(manifest["quantization"]).lower()
            mapping = {"float16": "fp16_float32_io", "fp16": "fp16_float32_io",
                       "int8": "int8_mixed_float_io", "float32": "float32"}
            expected_class = mapping.get(declared_quant)
            if expected_class is not None and identified != expected_class:
                raise CompatibilityError(
                    f"quantization mismatch: manifest declares {manifest['quantization']!r} "
                    f"({expected_class}), actual artifact is {identified}")
        if "quantization_classification" in manifest:
            identified = identify_quantization(interpreter)["classification"]
            if identified != manifest["quantization_classification"]:
                raise CompatibilityError(
                    f"quantization_classification mismatch: manifest "
                    f"{manifest['quantization_classification']!r}, actual {identified!r}")
        semantics = probe_output_semantics(interpreter)
        if semantics["looks_like"] != manifest["output_is_logits_or_probabilities"]:
            raise CompatibilityError(
                f"output_is_logits_or_probabilities mismatch: manifest "
                f"{manifest['output_is_logits_or_probabilities']!r}, probed {semantics['looks_like']!r}")
    return {"model_sha256": digest, "feature_config_version": manifest["feature_config_version"],
            "classes": len(labels), "status": "compatible"}


def validate_candidate_bundle(models_dir, interpreter=None, spec=None):
    """Strictly validate the versioned landmark candidate bundle.

    Checks, with no fallback:
      - all four assets exist and are non-empty (model, labels, meta,
        detector); a missing asset raises instead of falling back to S3D.
      - model SHA-256 + size match the pinned candidate spec exactly
        (the INT8 alternative therefore fails here as the wrong artifact).
      - meta frames/feature_dim/classes/quantization match the spec.
      - labels file hashes to the pinned SHA and its ordering matches the
        model class count (100) and output width.
      - if an interpreter is supplied: actual input/output tensors match
        [1,24,255]/[1,100] float32, quantization census identifies
        ``fp16_float32_io`` (never INT8), and probed output semantics are
        logits.
    """
    spec = dict(CANDIDATE_SPEC if spec is None else spec)
    models_dir = Path(models_dir)
    model_path = models_dir / spec["model_filename"]
    labels_path = models_dir / spec["labels_filename"]
    meta_path = models_dir / spec["meta_filename"]
    detector_path = models_dir / spec["detector_filename"]
    for path, role in ((model_path, "model"), (labels_path, "labels"),
                       (meta_path, "model metadata"), (detector_path, "detector")):
        if not path.is_file() or path.stat().st_size == 0:
            raise CompatibilityError(f"Missing candidate asset ({role}): {path}")
    if b"PK\x03\x04" not in detector_path.read_bytes()[:64]:
        raise CompatibilityError("Detector asset is not a valid MediaPipe task (ZIP) archive")
    digest = _sha256(model_path)
    if digest != spec["model_sha256"]:
        raise CompatibilityError(
            f"Model SHA mismatch: expected {spec['model_sha256']}, got {digest} "
            f"(refusing to substitute another artifact)")
    if model_path.stat().st_size != spec["model_size_bytes"]:
        raise CompatibilityError("Model byte size mismatch")
    labels_digest = _sha256(labels_path)
    if labels_digest != spec["labels_sha256"]:
        raise CompatibilityError(
            f"Labels SHA mismatch: expected {spec['labels_sha256']}, got {labels_digest}")
    detector_digest = _sha256(detector_path)
    if detector_digest != spec["detector_sha256"]:
        raise CompatibilityError(
            f"Detector SHA mismatch: expected {spec['detector_sha256']}, got {detector_digest}")
    meta = _load_json_file(meta_path, "model metadata")
    for key, expected in (("frames", spec["frame_count"]), ("feature_dim", spec["feature_dim"]),
                          ("classes", spec["classes"]), ("quantization", spec["quantization"])):
        if meta.get(key) != expected:
            raise CompatibilityError(f"Model metadata {key} mismatch: {meta.get(key)!r} != {expected!r}")
    ordered = _ordered_vocab_from_label_map(_load_json_file(labels_path, "labels file"))
    if len(ordered) != spec["classes"]:
        raise CompatibilityError("Labels class count does not match candidate spec")
    result = {"status": "compatible", "version": spec["version"],
              "model_sha256": digest, "labels_sha256": labels_digest,
              "classes": len(ordered), "ordered_vocabulary": ordered}
    if interpreter is not None:
        actual = tensor_contract(interpreter)
        if actual["input_tensor_shape"] != spec["input_tensor_shape"]:
            raise CompatibilityError(f"input_tensor_shape mismatch: {actual['input_tensor_shape']!r}")
        if actual["output_tensor_shape"] != spec["output_tensor_shape"]:
            raise CompatibilityError(f"output_tensor_shape mismatch: {actual['output_tensor_shape']!r}")
        if actual["input_dtype"] != spec["input_dtype"] or actual["output_dtype"] != spec["output_dtype"]:
            raise CompatibilityError("Candidate requires float32 input/output tensors")
        if actual["input_tensor_shape"][-1] != spec["feature_dim"]:
            raise CompatibilityError("Model feature width does not match feature_dim")
        if actual["output_tensor_shape"][-1] != len(ordered):
            raise CompatibilityError("Output class count does not match ordered vocabulary")
        identified = identify_quantization(interpreter)
        if identified["classification"] != spec["quantization_classification"]:
            raise CompatibilityError(
                f"quantization mismatch: candidate requires "
                f"{spec['quantization_classification']}, actual artifact is "
                f"{identified['classification']} (FP16 candidate must not be INT8)")
        semantics = probe_output_semantics(interpreter)
        if semantics["looks_like"] != spec["output_is_logits_or_probabilities"]:
            raise CompatibilityError(
                f"Output semantics mismatch: candidate requires logits, probed "
                f"{semantics['looks_like']}")
        result["quantization"] = identified
        result["output_semantics"] = semantics
        result["tensor_contract"] = actual
    return result
