"""Evidence & export sanitisation (D6 + D7 fixes).

Two related jobs:

1. **Hostname / private-path redaction** (D6).  ``run_manifest`` writes
   ``platform.node()`` as ``device_model`` and absolute resolved paths
   for every artifact/manifest into ``benchmark.json``.  That file is
   not safe to paste into submission notes, ``results/results.csv`` or
   a bundle (``build_bundle`` already *rejects* absolute-path leaks at
   verify time via ``_ABS_LEAK_RE`` — so unsanitised evidence simply
   fails verification, or worse, leaks when copied by hand).

   This module provides:

   * :func:`sanitize_benchmark_dict` — a pure dict transform used by
     the patched generator (see ``patches/run_manifest.patch``) and by
     any caller holding a benchmark dict.  ``device_model`` becomes a
     generic label (never a hostname), artifact/manifest paths become
     verifier-safe relative paths (repo-relative when the path is
     under the repo root, otherwise basename only), and string values
     that embed private absolute paths are redacted.
   * :func:`sanitize_benchmark_file` — load / sanitise / atomic-write
     for existing ``benchmark.json`` files.

   Sanitisation never mutates its input.  All other evidence fields
   (SHAs, counts, timings, accuracies, code commit, dataset/split)
   pass through byte-identical so provenance binding
   (``evidence_lab.provenance``) is preserved.

2. **Participant-ID exclusion from public exports** (D7).  ASL Citizen
   per-participant rows (``evaluation/asl_citizen_predictions.csv``,
   robustness ``signer_id`` / ``signer_ids``, per-signer accuracy rows
   in ``generate_results`` output) are personal-adjacent metadata.
   Redistribution terms are unresolved, so public exports must not
   carry raw participant IDs:

   * :func:`sanitize_predictions_csv` — drop the ``participant``
     column (and any ``signer_id`` / ``signer`` column).  Rows are
     otherwise verbatim.
   * :func:`hash_signer_id` / :func:`sanitize_signer_value` — when a
     signer linkage is needed for local aggregation, replace the raw
     ID with a salted HMAC pseudonym.  The salt never ships.
   * :func:`strip_signer_fields` — deep-copy a summary dict with
     ``signer_id`` / ``signer_ids`` / ``per_signer`` keys removed or
     pseudonymised (counts are kept; the key list is not).

Standard library only; Python 3.11 compatible.
"""

from __future__ import annotations

import csv
import hashlib
import hmac
import json
import os
import re
import tempfile
from pathlib import Path

GENERIC_DEVICE_MODEL = "generic-device"

# Private absolute-path prefixes that must never appear in shipped
# evidence.  Mirrors (and extends) build_bundle._ABS_LEAK_RE so a file
# passing this module also passes bundle verification.
_PRIVATE_PATH_RE = re.compile(
    r"(?:/Users/|/home/[^/\s]*|/private/|/tmp/|/var/folders/|/[^/\s]+/Documents/|[A-Za-z]:\\(?:Users|home|tmp|var|private))"
)
_WIN_ABS_RE = re.compile(r"^[A-Za-z]:\\")
_POSIX_ABS_RE = re.compile(r"^/")

_SIGNER_COLUMNS = ("participant", "signer_id", "signer", "Participant ID", "participant_id")


# --------------------------------------------------------------------------
# Path handling
# --------------------------------------------------------------------------

def _is_private_absolute(text: str) -> bool:
    if _PRIVATE_PATH_RE.search(text):
        return True
    if _WIN_ABS_RE.match(text) or _POSIX_ABS_RE.match(text):
        # Any absolute path is private on its own; only repo-relative
        # paths (no leading slash / drive) are safe to ship.
        return True
    return False


def redact_private_paths(value, *, repo_root=None):
    """Deep-copy ``value`` replacing private absolute paths with safe forms.

    * ``str`` values that are absolute paths are relativised: when
      ``repo_root`` is given and the path is under it, the repo-relative
      POSIX path is used; otherwise only the basename is kept.  Any
      private-path substring inside a longer string is replaced with
      ``[redacted-path]``.
    * Dicts/lists are rebuilt recursively; keys are left untouched
      (evidence keys are fixed schema names, never paths).
    * All other scalars pass through unchanged.
    """
    if isinstance(value, str):
        return _redact_string(value, repo_root=repo_root)
    if isinstance(value, dict):
        return {k: redact_private_paths(v, repo_root=repo_root) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_private_paths(v, repo_root=repo_root) for v in value]
    return value


def _redact_string(text: str, *, repo_root=None) -> str:
    stripped = text.strip()
    # Whole-string path (the common artifacts/manifest case).
    if stripped and (_WIN_ABS_RE.match(stripped) or _POSIX_ABS_RE.match(stripped)):
        # Windows-absolute paths are not parsed as such on POSIX, so
        # extract the basename with backslash awareness first.
        _win_base = stripped.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
        if _WIN_ABS_RE.match(stripped):
            return _win_base
        candidate = Path(stripped)
        if repo_root is not None:
            try:
                rel = candidate.resolve().relative_to(Path(repo_root).resolve())
                return rel.as_posix()
            except (ValueError, OSError):
                pass
        return candidate.name
    # Embedded private path inside prose / notes.
    if _PRIVATE_PATH_RE.search(text):
        return "[redacted-path]"
    return text


def sanitize_environment(env: dict, *, repo_root=None) -> dict:
    """Return an environment block with hostname and paths removed."""
    out = dict(env) if isinstance(env, dict) else {}
    # Hostname is never evidence; it identifies the operator's machine.
    if "device_model" in out:
        out["device_model"] = GENERIC_DEVICE_MODEL
    out.pop("hostname", None)
    out.pop("machine_hostname", None)
    for key in ("platform",):
        # platform.platform() embeds the kernel release but not a
        # private path; keep it for reproducibility, redacted if it
        # ever embeds one.
        value = out.get(key)
        if isinstance(value, str) and _PRIVATE_PATH_RE.search(value):
            out[key] = "[redacted-path]"
    return redact_private_paths(out, repo_root=repo_root)


def sanitize_artifacts(artifacts: dict, *, repo_root=None) -> dict:
    """Relativise artifact ``path`` fields, keep SHAs/sizes verbatim."""
    if not isinstance(artifacts, dict):
        return artifacts
    out = {}
    for name, entry in artifacts.items():
        if isinstance(entry, dict):
            copied = dict(entry)
            if "path" in copied and isinstance(copied["path"], str):
                copied["path"] = _redact_string(copied["path"], repo_root=repo_root)
            out[name] = redact_private_paths(copied, repo_root=repo_root)
        else:
            out[name] = entry
    return out


def sanitize_manifest_block(manifest: dict, *, repo_root=None) -> dict:
    """Relativise the manifest ``path``; keep its SHA/dataset/split."""
    if not isinstance(manifest, dict):
        return flatten_signer_block(manifest)
    out = dict(manifest)
    if "path" in out and isinstance(out["path"], str):
        out["path"] = _redact_string(out["path"], repo_root=repo_root)
    return out


# --------------------------------------------------------------------------
# Benchmark dict / file sanitisation
# --------------------------------------------------------------------------

def sanitize_benchmark_dict(benchmark: dict, *, repo_root=None) -> dict:
    """Return a submission-safe copy of a ``benchmark.json`` dict.

    Drops the operator hostname and every absolute artifact/manifest
    path.  Preserves SHAs, bytes, counts, timings, code commit,
    dataset/split and timestamps so stale-evidence checks in
    ``generate_results`` / ``evidence_lab`` still bind correctly.
    """
    if not isinstance(benchmark, dict):
        raise ValueError("benchmark must be a JSON object")
    out = dict(benchmark)
    if isinstance(out.get("environment"), dict):
        out["environment"] = sanitize_environment(out["environment"], repo_root=repo_root)
    if "artifacts" in out and isinstance(out["artifacts"], dict):
        out["artifacts"] = sanitize_artifacts(out["artifacts"], repo_root=repo_root)
    if "manifest" in out and isinstance(out["manifest"], dict):
        out["manifest"] = sanitize_manifest_block(out["manifest"], repo_root=repo_root)
    # Any other top-level string that embeds a private absolute path is
    # redacted (defence in depth; the schema above should cover it).
    for key in list(out):
        if key in ("environment", "artifacts", "manifest"):
            continue
        out[key] = redact_private_paths(out[key], repo_root=repo_root)
    return out


def sanitize_benchmark_file(src_path, dst_path=None, *, repo_root=None) -> dict:
    """Sanitise a benchmark JSON file; write atomically when ``dst_path`` set.

    When ``dst_path`` is None the source file is rewritten in place
    (atomic temp + ``os.replace``).  Returns the sanitised dict.
    """
    src = Path(src_path)
    try:
        data = json.loads(src.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{src.name} is not valid JSON: {exc}") from exc
    clean = sanitize_benchmark_dict(data, repo_root=repo_root)
    dest = Path(dst_path) if dst_path is not None else src
    text = json.dumps(clean, indent=2, ensure_ascii=False) + "\n"
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{dest.name}.", dir=dest.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_name, dest)
    finally:
        Path(tmp_name).unlink(missing_ok=True)
    return clean


def assert_sanitized(value) -> list[str]:
    """List any remaining private-path leaks (empty list when clean).

    Walks every string in a JSON-shaped value and reports hits with
    their key path.  Used by tests and by ``verify``-style gates.
    """
    leaks: list[str] = []
    if isinstance(value, str):
        if _PRIVATE_PATH_RE.search(value):
            leaks.append(value[:120])
    elif isinstance(value, dict):
        for key, item in value.items():
            for hit in assert_sanitized(item):
                leaks.append(f"{key}/{hit}")
    elif isinstance(value, list):
        for item in value:
            leaks.extend(assert_sanitized(item))
    return leaks


# --------------------------------------------------------------------------
# Participant-ID exclusion
# --------------------------------------------------------------------------

def hash_signer_id(signer_id: str, salt: str | bytes) -> str:
    """Salted HMAC pseudonym for a participant ID (local use only)."""
    if not isinstance(signer_id, str):
        signer_id = str(signer_id)
    key = salt if isinstance(salt, bytes) else str(salt).encode("utf-8")
    digest = hmac.new(key, signer_id.encode("utf-8"), hashlib.sha256).hexdigest()
    return "S" + digest[:16]


def sanitize_signer_value(value, *, salt=None):
    """Map one signer value to a safe public form.

    ``None`` / empty / ``"unknown"`` pass through (they carry no
    identity).  Otherwise the value is replaced with a salted pseudonym
    when ``salt`` is given, or with ``"redacted"`` when it is not.
    Never returns the raw ID.
    """
    if value is None:
        return value
    text = str(value).strip()
    if not text or text.lower() in ("unknown", "unspecified"):
        return "unknown" if text.lower() == "unknown" else value
    if salt is None:
        return "redacted"
    return hash_signer_id(text, salt)


def flatten_signer_block(value):
    return value


def strip_signer_fields(summary: dict, *, salt=None) -> dict:
    """Copy a summary dict with signer linkage removed or pseudonymised.

    * ``signer_id`` (scalar) → :func:`sanitize_signer_value`.
    * ``signer_ids`` (list) → count-preserving pseudonyms; when
      ``salt`` is None the list is replaced by ``["redacted"]`` *and*
      the ``signer_ids`` key is dropped when it would be a 1-element
      redaction — use ``with_signer_count=True`` in
      :func:`sanitize_robustness_summary` when the count itself must be
      preserved.  Here the list maps element-wise.
    * ``per_signer`` (dict keyed by participant) → keys pseudonymised
      the same way.

    Raw participant IDs never appear in the output.
    """
    if not isinstance(summary, dict):
        raise ValueError("summary must be a JSON object")
    out = {}
    for key, item in summary.items():
        if key == "signer_id":
            out[key] = sanitize_signer_value(item, salt=salt)
        elif key == "signer_ids" and isinstance(item, list):
            out[key] = [sanitize_signer_value(v, salt=salt) for v in item]
            # De-duplicate pseudonyms caused by salt-less redaction.
            if salt is None and len(out[key]) > 1:
                out[key] = ["redacted"]
        elif key == "per_signer" and isinstance(item, dict):
            out[key] = {
                sanitize_signer_value(k, salt=salt): redact_private_paths(v)
                for k, v in item.items()
            }
        else:
            out[key] = redact_private_paths(item)
    return out


def sanitize_robustness_summary(summary: dict, *, salt=None) -> dict:
    """Public-safe robustness summary: signer keys dropped, paths redacted."""
    return strip_signer_fields(summary, salt=salt)


def sanitize_predictions_csv(src_path, dst_path, *, salt=None) -> dict:
    """Write a public copy of a per-clip predictions CSV without participant IDs.

    * Drops ``participant`` / ``signer_id`` / ``signer`` /
      ``Participant ID`` columns entirely (aggregation downstream
      recomputes counts without linkage).
    * When ``salt`` is given *and* a dropped column was the only
      signer linkage, a ``signer_pseudonym`` column is added instead
      so local joins keep working without exposing raw IDs.
    * All other columns (video, glosses, predictions, correctness)
      are byte-identical, in original order.

    Returns ``{"rows": n, "dropped_columns": [...], "added_columns": [...]}``.
    """
    src = Path(src_path)
    dst = Path(dst_path)
    with src.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("predictions CSV is missing a header")
        original_fields = list(reader.fieldnames)
        dropped = [c for c in original_fields if c in _SIGNER_COLUMNS]
        signer_source = next(
            (c for c in original_fields if c in _SIGNER_COLUMNS), None
        )
        kept_fields = [c for c in original_fields if c not in _SIGNER_COLUMNS]
        added: list[str] = []
        if salt is not None and signer_source is not None:
            kept_fields.append("signer_pseudonym")
            added.append("signer_pseudonym")
        rows = []
        for row in reader:
            clean = {c: (row.get(c) or "") for c in kept_fields if c in original_fields}
            if "signer_pseudonym" in added:
                clean["signer_pseudonym"] = hash_signer_id(
                    row.get(signer_source) or "", salt
                )
            rows.append(clean)

    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{dst.name}.", dir=dst.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=kept_fields)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_name, dst)
    finally:
        Path(tmp_name).unlink(missing_ok=True)
    return {"rows": len(rows), "dropped_columns": dropped, "added_columns": added}


def sanitize_pairs_jsonl(src_path, dst_path, *, salt=None) -> dict:
    """Write a public copy of ``pairs.jsonl`` without raw signer linkage."""
    src = Path(src_path)
    dst = Path(dst_path)
    rows = []
    for line in src.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            record = strip_signer_fields(record, salt=salt)
        rows.append(record)
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{dst.name}.", dir=dst.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            for record in rows:
                stream.write(json.dumps(record, allow_nan=False, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_name, dst)
    finally:
        Path(tmp_name).unlink(missing_ok=True)
    return {"rows": len(rows)}
