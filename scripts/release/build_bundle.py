"""Build and verify a provisional Track 2 bundle from explicit local artifacts.

Verification treats every bundle artifact as untrusted input:
paths are checked before any file is read or extracted, model/vocabulary
evidence is hash- and shape-checked, prediction IDs are validated against
the organizer manifest when present, and predictor output is checked
against the same vocabulary/range contract as a predictions file.

The ``--official`` path deliberately stays blocked without real
organizer-manifest predictions: an official build requires
``results/predictions.jsonl`` in staging (which the builder does not
fabricate) and verification requires both the predictions and the
organizer ``manifest.json`` with an exact ID match.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile

REQUIRED = ("src/main.py", "src/requirements.txt", "src/README.md", "results/results.csv", "release.json")
REQUIRED_RESULT_COLUMNS = ("metric", "target", "measured", "unit", "device", "runtime", "pass", "notes")
ALLOWED_SCHEMES = ("http", "https")  # retained for compatibility; schemes are not trusted paths
_ALLOWED_RESULT_STATUS = ("PASS", "FAIL", "NOT_MEASURED", "NOT_APPLICABLE")
_ALLOWED_EXTRA_ROOTS = ("src", "models", "scripts", "evaluation")
# Extra runtime files the landmark backend imports at predict() time but which
# the base copy list does not stage. The s3d backend needs none of these.
_LANDMARK_RUNTIME_CLOSURE = (
    "src/features.py",
    "src/landmark_frontend.py",
    # The opt-in parallel tracker. Without it ECHOGESTURE_TRACKER=parallel
    # fails inside an extracted bundle, which is what made the 99.8 ms figure
    # unreachable for anyone running the shipped package.
    "src/parallel_tracker.py",
    "models/hand_landmarker.task",
    "models/landmark_model_meta.json",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DRIVE_RE = re.compile(r"^[A-Za-z]:")
_ABS_LEAK_RE = re.compile(r"(/Users/|/home/|/private/|/tmp/|[A-Za-z]:\\)")


def _landmark_runtime_files(model, labels):
    """Return the runtime closure the landmark backend needs, or () for s3d.

    The backend is identified from the selected artifacts rather than from
    ambient state: the landmark model/vocabulary names carry ``landmark``
    while the s3d model names (``*_int8.tflite``) and ``label100.json`` do
    not. The s3d backend therefore keeps its existing minimal copy list.
    """
    if "landmark" not in Path(model).name.lower() and "landmark" not in Path(labels).name.lower():
        return ()
    return _LANDMARK_RUNTIME_CLOSURE


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_member(name):
    """True only for a plain relative POSIX member with no traversal/tricks."""
    if not isinstance(name, str) or not name or "\x00" in name:
        return False
    if "\\" in name or _DRIVE_RE.match(name):
        return False
    # Directory entries end with '/'. Trailing slash is stripped for checks.
    stripped = name.rstrip("/")
    if not stripped:
        return False
    if name.startswith("/"):
        return False
    # Split on '/' manually: Path.parts normalises '.' and '//' away,
    # hiding exactly the components this check must see.
    parts = tuple(stripped.split("/"))
    if not parts or ".." in parts or any(part in ("", ".") for part in parts):
        return False
    # Reject colon in ANY component: drive/UNC-style ("C:") and NTFS
    # alternate-data-stream forms ("file:stream", "dir/file.mp4:stream")
    # are unsafe on Windows. Colon-free ZIP members are valid.
    if any(part.endswith(":") or ":" in part for part in parts):
        return False
    return True


def safe_relative(root, relative):
    """Resolve ``relative`` under ``root`` and require the result to stay inside."""
    if not isinstance(relative, str):
        raise ValueError(f"Unsafe relative artifact path: {relative!r}")
    if not safe_member(relative):
        raise ValueError(f"Unsafe relative artifact path: {relative}")
    root_resolved = Path(root).resolve()
    target = (root_resolved / relative).resolve()
    try:
        target.relative_to(root_resolved)
    except ValueError:
        raise ValueError(f"Artifact path escapes bundle root: {relative}") from None
    if target == root_resolved:
        raise ValueError(f"Artifact path is not a file inside bundle root: {relative}")
    return target


def _load_json(path, label):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {exc}") from exc


def declared_members(metadata, label):
    """Validate and return an ensemble member list from model metadata.

    Members are bare filenames that live next to the selected model. They are
    checked with the same guards as ZIP members: no separators, no traversal,
    no drive/UNC/ADS colon forms. An absent or empty ``members`` key means the
    single-model layout, which keeps older bundles valid.
    """
    if not isinstance(metadata, dict):
        raise ValueError(f"{label} must be a JSON object")
    raw = metadata.get("members")
    if raw is None or raw == []:
        return []
    if not isinstance(raw, list):
        raise ValueError(f"{label} field 'members' must be a list")
    members = []
    seen = set()
    for name in raw:
        if not isinstance(name, str) or not name:
            raise ValueError(f"{label} member names must be non-empty strings")
        if "/" in name:
            raise ValueError(f"Unsafe ensemble member name: {name!r}")
        if not safe_member(name) or name in (".", ".."):
            raise ValueError(f"Unsafe ensemble member name: {name!r}")
        if not name.lower().endswith(".tflite"):
            raise ValueError(f"Ensemble member must be a .tflite file: {name!r}")
        if name in seen:
            raise ValueError(f"Duplicate ensemble member: {name}")
        seen.add(name)
        members.append(name)
    return members


def _require_str(mapping, key, label):
    value = mapping.get(key) if isinstance(mapping, dict) else None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} field {key!r} must be a non-empty string")
    return value


def _require_file(path, label):
    if not Path(path).is_file():
        raise ValueError(f"{label} is missing: {Path(path).name}")
    if Path(path).stat().st_size == 0:
        raise ValueError(f"{label} is empty: {Path(path).name}")


def _validate_labels(labels_path):
    labels_map = _load_json(labels_path, "Vocabulary")
    if not isinstance(labels_map, dict) or not labels_map:
        raise ValueError("Vocabulary must be a non-empty label-to-index object")
    indices = []
    for label, index in labels_map.items():
        if not isinstance(label, str) or not label.strip() or "\\" in label:
            raise ValueError(f"Invalid vocabulary label: {label!r}")
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError(f"Vocabulary index for {label!r} must be an integer")
        indices.append(index)
    if sorted(indices) != list(range(len(indices))):
        raise ValueError("Vocabulary indices must be contiguous from zero")
    return labels_map, {label.upper() for label in labels_map}


def _scan_no_absolute_leak(value, label):
    if isinstance(value, str) and _ABS_LEAK_RE.search(value):
        raise ValueError(f"Private or absolute path leaked in {label}: {value!r}")


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_prediction_dict(value, allowed_labels, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    for field in ("clip_id", "prediction", "confidence", "latency_ms"):
        if field not in value:
            raise ValueError(f"{label} is incomplete: missing {field}")
    clip_id = value["clip_id"]
    if not isinstance(clip_id, str) or not clip_id.strip() or "/" in clip_id or "\\" in clip_id:
        raise ValueError(f"{label} has an invalid clip_id: {clip_id!r}")
    if value["prediction"] not in allowed_labels:
        raise ValueError(f"{label} prediction outside selected vocabulary: {value['prediction']!r}")
    if not _is_number(value["confidence"]) or not _is_number(value["latency_ms"]):
        raise ValueError(f"{label} confidence and latency must be numeric")
    if not 0 <= value["confidence"] <= 1 or not 0 <= value["latency_ms"]:
        raise ValueError(f"{label} has invalid confidence or latency")


def _load_manifest_ids(root):
    """Return ordered clip IDs from manifest.json, or None if absent."""
    manifest_path = Path(root) / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        with manifest_path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None or "clip_id" not in reader.fieldnames:
                raise ValueError("manifest.json must be a CSV with a clip_id column")
            ids = [row["clip_id"] for row in reader if row.get("clip_id") not in (None, "")]
    except (OSError, csv.Error) as exc:
        raise ValueError(f"manifest.json is unreadable: {exc}") from exc
    if not ids:
        raise ValueError("manifest.json contains no clip_id rows")
    if len(ids) != len(set(ids)):
        raise ValueError("manifest.json contains duplicate clip_id values")
    for clip_id in ids:
        if not isinstance(clip_id, str) or "/" in clip_id or "\\" in clip_id:
            raise ValueError(f"manifest.json clip_id looks like a path: {clip_id!r}")
    return ids


def _validate_results(root, digest):
    results_path = Path(root) / "results/results.csv"
    with results_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("results/results.csv is missing a header")
        missing = [c for c in REQUIRED_RESULT_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"results/results.csv missing columns: {', '.join(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("results/results.csv contains no rows")
    for row in rows:
        for column, value in row.items():
            _scan_no_absolute_leak(value, "results/results.csv")
            if column == "pass":
                if not isinstance(value, str) or value.upper() not in _ALLOWED_RESULT_STATUS:
                    raise ValueError(f"Invalid result status: {value}")
        if "model_sha256" in row and row["model_sha256"] not in ("", digest):
            raise ValueError("Result evidence belongs to another model")


def _validate_predictions(root, allowed_labels, official, digest, manifest_model_sha=None):
    predictions = Path(root) / "results/predictions.jsonl"
    if official and not predictions.is_file():
        raise ValueError("Official release requires organizer-manifest predictions.jsonl")
    manifest_ids = _load_manifest_ids(root)
    if official and manifest_ids is None:
        raise ValueError("Official release requires organizer manifest.json for ID validation")
    if predictions.exists():
        _require_file(predictions, "predictions.jsonl")
        _scan_no_absolute_leak(predictions.read_text(encoding="utf-8"), "predictions.jsonl")
        actual = []
        seen = set()
        for line in predictions.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            value = _load_json_text(line, "Prediction line")
            _validate_prediction_dict(value, allowed_labels, "Prediction line")
            clip_id = value["clip_id"]
            if clip_id in seen:
                raise ValueError(f"Duplicate prediction clip_id: {clip_id}")
            seen.add(clip_id)
            actual.append(clip_id)
        if not actual:
            raise ValueError("predictions.jsonl contains no prediction rows")
        if manifest_ids is None:
            raise ValueError("predictions.jsonl requires manifest.json for ID validation")
        if actual != manifest_ids:
            raise ValueError("Prediction IDs do not match the manifest exactly")


def _load_json_text(text, label):
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {exc}") from exc


def verify_directory(root, clip=None, model_sha=None, official=False):
    root = Path(root)
    missing = [name for name in REQUIRED if not (root / name).is_file()]
    if missing:
        raise ValueError(f"Missing bundle artifacts: {', '.join(missing)}")
    for name in REQUIRED:
        _require_file(root / name, f"Bundle artifact {name}")

    manifest = _load_json(root / "release.json", "release.json")
    if not isinstance(manifest, dict):
        raise ValueError("release.json must be a JSON object")
    _scan_no_absolute_leak(json.dumps(manifest), "release.json")
    model_rel = _require_str(manifest, "model", "release.json")
    labels_rel = _require_str(manifest, "labels", "release.json")
    conversion_rel = _require_str(manifest, "conversion_script", "release.json")
    declared_sha = _require_str(manifest, "model_sha256", "release.json")
    if not _SHA256_RE.match(declared_sha):
        raise ValueError("release.json model_sha256 must be a 64-char lowercase hex SHA-256")
    if official and manifest.get("official_submission") is not True:
        raise ValueError("Official verification requires release.json official_submission=true")

    model = safe_relative(root, model_rel)
    labels = safe_relative(root, labels_rel)
    conversion = safe_relative(root, conversion_rel)
    if model.suffix != ".tflite":
        raise ValueError("Selected model must be a .tflite file")
    if labels.suffix != ".json":
        raise ValueError("Vocabulary must be a .json file")
    if conversion.suffix != ".py":
        raise ValueError("Model conversion script must be a .py file")
    _require_file(model, "Selected model")
    _require_file(labels, "Vocabulary")
    _require_file(conversion, "Model conversion script")
    _require_file(root / "Dockerfile", "Docker setup")
    if not (root / "Dockerfile").read_text(encoding="utf-8").strip():
        raise ValueError("Docker setup is empty")

    labels_map, allowed_labels = _validate_labels(labels)
    digest = sha256(model)
    if declared_sha != digest:
        raise ValueError("release.json model hash does not match the selected artifact")

    # Ensemble members are declared in release.json by the builder and must
    # agree with the shipped meta sidecar. Verification never trusts either
    # list alone: both must match and every member must be present and hash-correct.
    declared_member_hashes = manifest.get("members")
    metad_rel = manifest.get("model_metadata")
    meta_members = []
    if metad_rel is not None:
        metad_rel = _require_str(manifest, "model_metadata", "release.json")
        metadata = safe_relative(root, metad_rel)
        _require_file(metadata, "Model metadata")
        meta_members = declared_members(_load_json(metadata, "Model metadata"), "Model metadata")
    if declared_member_hashes is None:
        declared_member_hashes = []
    if not isinstance(declared_member_hashes, list):
        raise ValueError("release.json 'members' must be a list")
    if meta_members:
        if len(meta_members) != len(declared_member_hashes) or any(
                not isinstance(m, dict) or m.get("name") != name
                for m, name in zip(declared_member_hashes, meta_members)):
            raise ValueError("release.json members do not match the metadata members list")
    verified_members = []
    for entry in declared_member_hashes:
        if not isinstance(entry, dict):
            raise ValueError("release.json member entries must be objects")
        name = _require_str(entry, "name", "release.json member")
        member_sha = _require_str(entry, "sha256", "release.json member")
        if "/" in name or not safe_member(name):
            raise ValueError(f"Unsafe ensemble member path: {name!r}")
        if not _SHA256_RE.match(member_sha):
            raise ValueError(f"release.json member sha256 for {name!r} must be 64-char lowercase hex")
        member_path = safe_relative(root, f"{Path(model_rel).parent.as_posix()}/{name}")
        _require_file(member_path, f"Ensemble member {name}")
        if sha256(member_path) != member_sha:
            raise ValueError(f"Ensemble member hash mismatch: {name}")
        verified_members.append({"name": name, "sha256": member_sha})
    models_dir = root / "models"
    if not models_dir.is_dir():
        raise ValueError("Bundle is missing models/ directory for total-size accounting")
    total_model_bytes = sum(p.stat().st_size for p in models_dir.rglob("*") if p.is_file())
    if total_model_bytes >= 20e6:
        raise ValueError(f"Total shipped model assets exceed 20 MB: {total_model_bytes} bytes")
    if model_sha:
        if not _SHA256_RE.match(model_sha):
            raise ValueError("--model-sha must be a 64-char lowercase hex SHA-256")
        if digest != model_sha:
            raise ValueError("Bundle model does not match requested model SHA")

    _validate_results(root, digest)
    _validate_predictions(root, allowed_labels, official, digest)

    if clip:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(root / "src")
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        completed = subprocess.run(
            [sys.executable, "-I", "-c", "import sys, json; sys.path.insert(0, 'src'); import main; print(json.dumps(main.predict(sys.argv[1])))", str(Path(clip).resolve())],
            cwd=root, env=env, capture_output=True, text=True, timeout=180)
        if completed.returncode:
            raise ValueError(f"Extracted predictor failed: {completed.stderr.strip()}")
        prediction = _load_json_text(completed.stdout.strip().splitlines()[-1], "Predictor output")
        _validate_prediction_dict(prediction, allowed_labels, "Predictor output")
        if prediction["clip_id"] != Path(clip).stem:
            raise ValueError("Extracted predictor returned an incorrect clip ID")
        return prediction
    return {"model_sha256": digest, "members": verified_members, "status": "structure_verified"}


def verify_zip(path, clip=None, model_sha=None, official=False):
    path = Path(path)
    with tempfile.TemporaryDirectory(prefix="echogesture-verify-") as directory:
        dest_root = Path(directory).resolve()
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            names = [m.filename for m in members]
            if len(names) != len(set(names)):
                raise ValueError("ZIP contains duplicate member names")
            for member in members:
                if not safe_member(member.filename):
                    raise ValueError(f"Unsafe bundle path: {member.filename}")
                mode = member.external_attr >> 16
                if stat.S_ISLNK(mode) or (mode and not (stat.S_ISREG(mode) or stat.S_ISDIR(mode))):
                    raise ValueError(f"Unsafe ZIP member type: {member.filename}")
                target = (dest_root / member.filename).resolve()
                try:
                    target.relative_to(dest_root)
                except ValueError:
                    raise ValueError(f"Unsafe bundle path: {member.filename}") from None
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(member) as source_stream, target.open("wb") as out_stream:
                        shutil.copyfileobj(source_stream, out_stream)
        return verify_directory(dest_root, clip, model_sha, official)


def build(source, model, labels, results, output, official=False, extra=(), metadata=None):
    """Stage, verify and ZIP one bundle; returns the clean-extraction result.

    ``metadata`` optionally overrides the ``landmark_model_meta.json`` sidecar
    shipped next to the model. The builder never edits source: a release that
    declares ensemble ``members`` passes the intended sidecar here instead of
    mutating a tracked file.
    """
    source = Path(source).resolve()
    if not (source / "src/main.py").is_file():
        raise ValueError("Source must contain src/main.py")
    if not safe_member(model):
        raise ValueError(f"Unsafe model path: {model}")
    if not safe_member(labels):
        raise ValueError(f"Unsafe labels path: {labels}")
    selected_model = safe_relative(source, model)
    selected_labels = safe_relative(source, labels)
    if not selected_model.is_file():
        raise ValueError("Selected model is missing")
    if not selected_labels.is_file():
        raise ValueError("Selected labels are missing")
    # The meta sidecar next to the model declares the ensemble (if any). It is
    # the same file ``main.predict`` reads, so members are staged exactly where
    # the runtime looks for them.
    meta_rel = f"{Path(model).parent.as_posix()}/landmark_model_meta.json"
    if not safe_member(meta_rel):
        raise ValueError(f"Unsafe model metadata path: {meta_rel}")
    if metadata is not None:
        metadata = Path(metadata).resolve()
        if not metadata.is_file():
            raise ValueError("Model metadata override is missing")
        members = declared_members(_load_json(metadata, "Model metadata"), "Model metadata")
    else:
        meta_path = selected_model.parent / "landmark_model_meta.json"
        members = []
        if meta_path.is_file():
            members = declared_members(_load_json(meta_path, "Model metadata"), "Model metadata")
    member_sources = []
    member_hashes = {}
    for name in members:
        member_src = safe_relative(source, f"{Path(model).parent.as_posix()}/{name}")
        _require_file(member_src, f"Ensemble member {name}")
        member_hashes[name] = sha256(member_src)
        member_sources.append((name, member_src))
    model_sha = sha256(selected_model)
    # A caller-supplied metadata override may pin expected member hashes. Without
    # a pin there is nothing to compare against, but the computed hashes are
    # still recorded in release.json so an evaluator can detect substitution.
    if metadata is not None:
        expected_members = _load_json(metadata, "Model metadata").get("member_sha256")
        if isinstance(expected_members, dict):
            for name, expected in expected_members.items():
                actual_member_sha = member_hashes.get(name)
                if actual_member_sha is None:
                    raise ValueError(f"Declared member hash for unknown member {name}")
                if actual_member_sha != expected:
                    raise ValueError(
                        f"Ensemble member {name} hash mismatch: {actual_member_sha} "
                        f"!= declared {expected}")
        elif expected_members is not None:
            raise ValueError("Model metadata field 'member_sha256' must be an object")
    # A member whose bytes differ from the shipped artifact changes the averaged
    # logits exactly as much as a wrong selected model does.
    for name, digest in sorted({n: sha256(p) for n, p in member_sources}.items()):
        if digest != member_hashes[name]:
            raise ValueError(f"Ensemble member {name} changed during staging")
    results = Path(results)
    if not results.is_file() or results.stat().st_size == 0:
        raise ValueError("Results file is missing or empty")
    for name in ("main.py", "requirements.txt", "README.md"):
        _require_file(source / "src" / name, f"Source file src/{name}")
    if official:
        # An official build is possible, but only from real inputs: an organizer
        # manifest and a predictions file generated by running THIS model over
        # that manifest's clips. The builder never invents either. Requiring
        # both here means the failure is a clear "supply the inputs" message
        # rather than a rejected flag.
        for name, flag in (("manifest.json", "--manifest"),
                           ("results/predictions.jsonl", "--predictions")):
            candidate = (
                source / name if name == "manifest.json"
                else results.parent / "predictions.jsonl"
            )
            if not candidate.is_file():
                raise ValueError(
                    f"Official bundle requires {name} (pass {flag}); the builder "
                    "does not fabricate organizer clip IDs or predictions")

    with tempfile.TemporaryDirectory(prefix="echogesture-build-") as directory:
        staging = Path(directory) / "bundle"
        (staging / "src").mkdir(parents=True)
        (staging / "results").mkdir()
        for name in ("main.py", "requirements.txt", "README.md"):
            shutil.copy2(source / "src" / name, staging / "src" / name)
        out_model = staging / model
        out_labels = staging / labels
        out_model.parent.mkdir(parents=True, exist_ok=True)
        out_labels.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(selected_model, out_model)
        shutil.copy2(selected_labels, out_labels)
        staged_members = []
        for name, member_src in member_sources:
            out_member = out_model.parent / name
            out_member.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(member_src, out_member)
            staged_members.append({"name": name, "sha256": sha256(out_member)})
        for runtime_rel in _landmark_runtime_files(model, labels):
            runtime_src = safe_relative(source, runtime_rel)
            _require_file(runtime_src, f"Landmark runtime asset {runtime_rel}")
            if metadata is not None and runtime_rel == meta_rel:
                runtime_src = metadata
            out_runtime = staging / runtime_rel
            out_runtime.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(runtime_src, out_runtime)
        conversion_rel = None
        for extra_path in extra:
            extra_path = Path(extra_path).resolve()
            if not extra_path.is_file():
                raise ValueError(f"Extra artifact is missing: {extra_path.name}")
            relative = extra_path.relative_to(source)
            if relative.parts[0] not in _ALLOWED_EXTRA_ROOTS:
                raise ValueError(f"Unsupported extra artifact: {relative}")
            if not safe_member(relative.as_posix()):
                raise ValueError(f"Unsafe extra artifact: {relative}")
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(extra_path, destination)
            if extra_path.name.startswith("export") and extra_path.suffix == ".py":
                conversion_rel = relative.as_posix()
        if conversion_rel is None:
            raise ValueError("Add a model conversion/export script with --extra")
        shutil.copy2(results, staging / "results/results.csv")
        # p.12 puts predictions.json in the results/ folder, and p.11 requires a
        # single predictions.jsonl. Stage a predictions file only when it can be
        # validated against an organizer manifest: predictions without a
        # manifest cannot be checked, and a fabricated manifest would defeat the
        # ID check this builder exists to perform.
        sibling_predictions = results.parent / "predictions.jsonl"
        source_manifest = source / "manifest.json"
        if sibling_predictions.is_file() and source_manifest.is_file():
            shutil.copy2(sibling_predictions, staging / "results/predictions.jsonl")
            # _load_manifest_ids reads manifest.json from the bundle root, so it
            # must be staged or official verification fails on the copied tree
            # even though the source tree validated.
            shutil.copy2(source_manifest, staging / "manifest.json")
            # p.12 names predictions.json; p.11 mandates a JSON-Lines payload.
            # Emit both so either reading of the deck resolves, with identical
            # content: JSON Lines is valid as newline-delimited JSON.
            shutil.copy2(sibling_predictions, staging / "results/predictions.json")
        if official and not (staging / "results/predictions.jsonl").is_file():
            raise ValueError(
                "Official bundle requires organizer-generated predictions.jsonl "
                "beside results.csv, validated against manifest.json")
        _require_file(staging / conversion_rel, "Model conversion script")
        # Use the repository's Dockerfile when it has content. A generated stub
        # would silently ship a different image than the one the README
        # documents.
        root_dockerfile = source / "Dockerfile"
        if root_dockerfile.is_file() and root_dockerfile.read_text(encoding="utf-8").strip():
            shutil.copy2(root_dockerfile, staging / "Dockerfile")
        else:
            (staging / "Dockerfile").write_text(
                "FROM python:3.11-slim\nWORKDIR /app\nCOPY . /app\n"
                "RUN pip install --no-cache-dir -r src/requirements.txt\n"
                'CMD ["python", "src/main.py"]\n', encoding="utf-8")
        manifest = {
            "model": model, "labels": labels,
            "model_sha256": sha256(out_model),
            "conversion_script": conversion_rel,
            "official_submission": official,
            "layout": "provisional pending organizer starter kit",
        }
        # Only ensemble bundles gain these keys; single-model release.json keeps
        # its exact previous shape.
        if staged_members:
            manifest["members"] = staged_members
        if (staging / meta_rel).is_file():
            manifest["model_metadata"] = meta_rel
        (staging / "release.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        verify_directory(staging, model_sha=sha256(out_model), official=official)
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for file in sorted(staging.rglob("*")):
                if file.is_file():
                    archive.write(file, file.relative_to(staging))
    return verify_zip(output, model_sha=model_sha, official=official)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--source", type=Path, required=True)
    build_parser.add_argument("--model", required=True)
    build_parser.add_argument("--labels", required=True)
    build_parser.add_argument("--results", type=Path, required=True)
    build_parser.add_argument("--output", type=Path, required=True)
    build_parser.add_argument("--extra", type=Path, action="append", default=[])
    build_parser.add_argument("--metadata", type=Path,
                              help="override the shipped landmark_model_meta.json sidecar")
    build_parser.add_argument("--official", action="store_true")
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("bundle", type=Path)
    verify_parser.add_argument("--clip", type=Path)
    verify_parser.add_argument("--model-sha")
    verify_parser.add_argument("--official", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            result = build(args.source, args.model, args.labels, args.results, args.output, args.official, args.extra, args.metadata)
        else:
            result = verify_zip(args.bundle, args.clip, args.model_sha, args.official)
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile, subprocess.SubprocessError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
