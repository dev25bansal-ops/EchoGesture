"""Provisional release builder for the selected landmark backend.

Why this wrapper instead of ``build_bundle.build()``: ``build_bundle.build``
only auto-copies ``src/main.py`` + ``requirements.txt`` + ``README.md``, the
chosen model/labels, one conversion script, and ``results.csv``. It does NOT
auto-include ``src/features.py``, ``src/landmark_frontend.py``, or the
detector/meta assets that ``predict(video_path)`` actually imports for the
landmark backend. A naive build therefore fails clean-extraction the moment
``main.predict`` reaches ``from features import ...`` or needs
``models/hand_landmarker.task``.

This module stages the *complete* runtime closure below and zips it, reusing
``build_bundle``'s trusted ``safe_member`` / ``sha256`` helpers for path and
hash integrity but performing its own verification so that the honest
evidence in ``results/results.csv`` (``pass`` values ``n/a`` /
``not_instrumented``) is not rejected by ``build_bundle``'s stricter enum.

The selected artifact is pinned: any change to the FP16 classifier's bytes
or SHA aborts the build rather than silently shipping a different model.

Deliverables under this directory:
  - echogesture_landmark_release.zip  (the provisional bundle)
  - REPORT.md                         (build + clean-extraction evidence)

Standard library only. Run with the project interpreter:
  .venv/bin/python -I evaluation/device/release_final/release_build.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "release"))
from build_bundle import sha256, safe_member  # noqa: E402  (reused, not modified)

# --- The complete runtime closure for predict(video_path), landmark backend ---
# Ordered so the manifest reads like a dependency list. Every entry is a
# relative POSIX path rooted at the repository root.
#
# Ensemble members are declared by models/landmark_model_meta.json, not hardcoded
# here: the runtime refuses to start when a declared member is absent, so a
# closure that shipped only the single classifier staged a bundle that could not
# predict. Derive them so the closure cannot drift from the metadata again.
_META_REL = "models/landmark_model_meta.json"


def _declared_members(repo=REPO):
    """Member basenames declared by the model metadata, validated for safety."""
    meta_path = Path(repo) / _META_REL
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    raw = meta.get("members")
    if not isinstance(raw, list) or not raw:
        raise ValueError(
            f"{_META_REL} must declare a non-empty 'members' list for the ensemble runtime")
    members = []
    for name in raw:
        if not isinstance(name, str) or not name or "/" in name or "\\" in name:
            raise ValueError(f"Unsafe ensemble member name in {_META_REL}: {name!r}")
        members.append(f"models/{name}")
    return tuple(members)


CLOSURE = (
    "src/main.py",
    "src/requirements.txt",
    "src/README.md",
    "src/features.py",
    "src/landmark_frontend.py",
    "models/landmark_classifier_fp16.tflite",
    *_declared_members(),
    "models/landmark_labels.json",
    _META_REL,
    "models/hand_landmarker.task",
    "scripts/ml/export_tflite_tf.py",
    "results/results.csv",
)

SELECTED_MODEL_REL = "models/landmark_classifier_fp16.tflite"
SELECTED_LABELS_REL = "models/landmark_labels.json"
CONVERSION_REL = "scripts/ml/export_tflite_tf.py"

# Pinned identity of the selected artifact (verified independently at build).
EXPECTED_MODEL_SHA = "a012f2b5f7a1110961079f19eb0e28c9abacb95e4d0787352e4157ebca7044a4"
EXPECTED_MODEL_BYTES = 832888

# Permitted real clip for clean-extraction verification, relative to REPO.
CLIP_REL = "evaluation/device/v5_accuracy/data/val_30/0283886564670357-WHAT FOR.mp4"

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "echogesture_landmark_release.zip"

_MAX_MODEL_BYTES = 20 * 1000 * 1000  # 20 MB organizer ceiling, mirrored from build_bundle
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def check_complete(staging, closure=CLOSURE):
    """Raise unless every closure asset is present and non-empty."""
    staging = Path(staging)
    missing = [rel for rel in closure
               if not (staging / rel).is_file() or (staging / rel).stat().st_size == 0]
    if missing:
        raise ValueError("Incomplete closure (missing or empty): " + ", ".join(map(str, missing)))
    return staging


def model_dir_bytes(staging):
    """Total bytes under models/, enforcing the 20 MB organizer ceiling."""
    models = Path(staging) / "models"
    if not models.is_dir():
        raise ValueError("staging is missing models/ directory for size accounting")
    total = sum(p.stat().st_size for p in models.rglob("*") if p.is_file())
    if total >= _MAX_MODEL_BYTES:
        raise ValueError(f"Total shipped model assets exceed 20 MB: {total} bytes")
    return total


def write_release_json(staging):
    """Emit release.json in build_bundle's schema (provisional, not official)."""
    staging = Path(staging)
    model_sha = sha256(staging / SELECTED_MODEL_REL)
    manifest = {
        "model": SELECTED_MODEL_REL,
        "labels": SELECTED_LABELS_REL,
        "model_sha256": model_sha,
        "conversion_script": CONVERSION_REL,
        "official_submission": False,
        "layout": "provisional pending organizer starter kit",
    }
    (staging / "release.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def write_dockerfile(staging):
    (staging / "Dockerfile").write_text(
        "FROM python:3.11-slim\nWORKDIR /app\nCOPY . /app\n"
        "RUN pip install --no-cache-dir -r src/requirements.txt\n"
        'CMD ["python", "src/main.py"]\n', encoding="utf-8")


def check_release_hash(staging):
    """Reject a declared/actual hash mismatch and a non-pinned model."""
    staging = Path(staging)
    release = staging / "release.json"
    if not release.is_file():
        raise ValueError("release.json is missing")
    manifest = json.loads(release.read_text(encoding="utf-8"))
    declared = manifest.get("model_sha256")
    if not isinstance(declared, str) or not _SHA256_RE.match(declared):
        raise ValueError("release.json model_sha256 must be a 64-char lowercase hex SHA-256")
    model_rel = manifest.get("model")
    actual = sha256(staging / model_rel)
    if declared != actual:
        raise ValueError("release.json model hash does not match the selected artifact")
    if declared != EXPECTED_MODEL_SHA:
        raise ValueError("selected model does not match the pinned landmark FP16 artifact")
    return manifest


def assemble(staging, closure=CLOSURE):
    """Copy the complete closure plus release.json/Dockerfile; verify it."""
    staging = Path(staging)
    staging.mkdir(parents=True, exist_ok=True)
    for rel in closure:
        rel_path = Path(rel)
        src = REPO / rel_path
        if not src.is_file() or src.stat().st_size == 0:
            raise ValueError(f"Closure asset missing or empty: {rel}")
        if not safe_member(rel_path.as_posix()):
            raise ValueError(f"Unsafe closure path: {rel}")
        dest = staging / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    write_release_json(staging)
    write_dockerfile(staging)
    check_complete(staging, closure)
    check_release_hash(staging)
    model_dir_bytes(staging)
    return staging


def zip_staging(staging, output):
    """ZIP the staged bundle, emitting regular-file members with safe names only."""
    staging = Path(staging)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(staging.rglob("*")):
            if file.is_file():
                member = file.relative_to(staging).as_posix()
                if not safe_member(member):
                    raise ValueError(f"Unsafe staging path: {member}")
                archive.write(file, member)
    return output


def safe_extract(zip_path, dest_root):
    """Extract a ZIP into dest_root, rejecting traversal and non-regular members.

    Mirrors build_bundle.verify_zip's member loop so cleand-extraction never
    writes outside dest_root or materialises a symlink/device member.
    """
    zip_path = Path(zip_path)
    dest_root = Path(dest_root).resolve()
    dest_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
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
                with archive.open(member) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
    return dest_root


def validate_prediction(pred, expected_clip_id, allowed_uppercase_labels):
    """Enforce the evaluator output contract on a real predictor result."""
    if not isinstance(pred, dict):
        raise ValueError("predictor output is not a JSON object")
    for field in ("clip_id", "prediction", "confidence", "latency_ms"):
        if field not in pred:
            raise ValueError(f"predictor output is missing field {field!r}")
    clip_id = pred["clip_id"]
    if not isinstance(clip_id, str) or not clip_id.strip() or "/" in clip_id or "\\" in clip_id:
        raise ValueError(f"predictor returned an invalid clip_id: {clip_id!r}")
    if clip_id != expected_clip_id:
        raise ValueError(f"predictor clip_id {clip_id!r} != expected {expected_clip_id!r}")
    if pred["prediction"] not in allowed_uppercase_labels:
        raise ValueError(f"prediction outside vocabulary or not uppercase: {pred['prediction']!r}")
    confidence = pred["confidence"]
    latency = pred["latency_ms"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError("confidence must be numeric")
    if isinstance(latency, bool) or not isinstance(latency, (int, float)):
        raise ValueError("latency_ms must be numeric")
    if not 0 <= confidence <= 1:
        raise ValueError(f"confidence out of range: {confidence!r}")
    if latency < 0:
        raise ValueError(f"negative latency_ms: {latency!r}")
    return pred


def verify_extracted_predict(zip_path, clip_path, python=None):
    """Clean-extraction check: extract into a FRESH temp dir and run the real predictor.

    Mirrors build_bundle.verify_zip's subprocess shape: ``sys.executable -I``,
    ``cwd`` = extract root, ``PYTHONPATH`` = extract root's src/. The clip path
    is resolved to an absolute value because the extract root is an unrelated
    fresh temp dir.
    """
    zip_path = Path(zip_path)
    clip_path = Path(clip_path).resolve()
    python = Path(python) if python else Path(sys.executable)
    with tempfile.TemporaryDirectory(prefix="eg-release-final-") as directory:
        root = Path(directory).resolve()
        safe_extract(zip_path, root)
        labels_map = json.loads(
            (root / SELECTED_LABELS_REL).read_text(encoding="utf-8"))
        allowed = {str(label).upper() for label in labels_map}
        expected_clip_id = clip_path.stem
        env = os.environ.copy()
        env["PYTHONPATH"] = str(root / "src")
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        command = [
            str(python), "-I", "-c",
            "import sys, json; sys.path.insert(0, 'src'); import main; "
            "print(json.dumps(main.predict(sys.argv[1])))",
            str(clip_path),
        ]
        completed = subprocess.run(command, cwd=root, env=env,
                                   capture_output=True, text=True, timeout=180)
        if completed.returncode:
            raise ValueError(
                f"Extracted predictor failed (rc={completed.returncode}): "
                f"{completed.stderr.strip()}"
            )
        line = completed.stdout.strip().splitlines()[-1]
        prediction = json.loads(line)
        validate_prediction(prediction, expected_clip_id, allowed)
        return {
            "prediction": prediction,
            "verbatim": line,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }


def build_release(output=DEFAULT_OUTPUT, clip_path=REPO / CLIP_REL, python=None, verify=True):
    """Assemble, zip, and (by default) verify by clean extraction."""
    output = Path(output).resolve()
    with tempfile.TemporaryDirectory(prefix="eg-assemble-") as directory:
        staging = Path(directory) / "bundle"
        assemble(staging)
        zip_staging(staging, output)
    info = {
        "zip": str(output),
        "zip_sha256": sha256(output),
        "zip_bytes": output.stat().st_size,
    }
    if verify:
        info["extraction"] = verify_extracted_predict(output, clip_path, python)
    return info


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Destination ZIP (default: release_final/echogesture_landmark_release.zip)")
    parser.add_argument("--clip", type=Path, default=REPO / CLIP_REL,
                        help="Real clip for clean-extraction verification")
    parser.add_argument("--python", default=None,
                        help="Interpreter for the extracted predictor (default: sys.executable)")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip clean-extraction prediction check")
    args = parser.parse_args(argv)
    info = build_release(args.output, args.clip, args.python, verify=not args.no_verify)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()