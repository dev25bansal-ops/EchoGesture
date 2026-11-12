"""Publish-safe sanitisation of restricted evidence exports (A2 Fix 1/2).

The tracked research evidence under ``evaluation/`` carries personal-adjacent
metadata that must never reach a public bundle or a public repository mirror:

* ``evaluation/asl_citizen_predictions.csv`` has a ``participant`` column of
  raw ASL Citizen participant IDs (``P15``-style);
* ``evaluation/asl_citizen_{ptq,qat}_results.json`` embed ``per_signer`` dicts
  keyed by those same raw IDs;
* any benchmark / manifest dict may embed the operator hostname or absolute
  private paths.

This module writes **new** publishable copies.  It never edits, moves, or
deletes the originals — those remain local research inputs whose visibility
and eventual removal is a separate owner decision (see NOTICE §3).

Guarantees:

* stdlib only, Python 3.11 compatible, import-safe under ``python -I``;
* reads the originals, writes only to a caller-chosen output directory that
  can never be the source file (checked by resolved path);
* **fails closed**: if any emitted value still contains a raw participant
  token (``P<digits>``), an operator hostname, or an absolute private path,
  the file is not written and :class:`PublishLeakError` is raised;
* deterministic: the same input + key always yields byte-identical output.

Pseudonymisation is a keyed HMAC-SHA-256 truncated to 8 hex characters and
prefixed ``signer_`` — non-identifying and non-reversible without the key.
The key is never written to disk by this module.  Without a key the linkage
is removed rather than pseudonymised (CSV column dropped; JSON ``per_signer``
keys replaced by stable ``signer_redacted_<n>`` labels).

Public API used by ``scripts/release/compliance_final.py``:

    leaks_in_text(text) -> list[str]        # reasons, empty when safe
    leaks_in_document(path) -> list[str]
    scan_publish_dir(publish_dir) -> dict   # {"ok", "files", "leaks"}
    sanitize_document(obj, key=None) -> dict
    sanitize_file(src, dst, key=None) -> dict
    sanitize_all(root, outdir, key=None) -> dict

CLI (run from the repo root):

    python3 -I scripts/release/sanitize_export.py [--output DIR] [--key KEY]

Exit 0 on success, 1 when sanitisation refused (leak detected), 2 on usage
or I/O error.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import io
import json
import os
import re
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Tracked evidence files that must never be published verbatim.
DEFAULT_SOURCES = (
    "evaluation/asl_citizen_predictions.csv",
    "evaluation/asl_citizen_ptq_results.json",
    "evaluation/asl_citizen_qat_results.json",
)
DEFAULT_OUTDIR = "evaluation/device/compliance_final/publish"

# Raw participant-ID token: uppercase P + 1-3 digits.  An underscore before
# the P is allowed ("top1_signer_P15") because metric/JSON keys use that form;
# a leading letter is not, so glosses like PULLCONVINCE and lowercase latency
# percentiles (p50/p95/p99) never match.
PARTICIPANT_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9])P\d{1,3}(?![A-Za-z0-9_])")

# Operator hostname heuristics: dotted local-only suffixes.  A bare hostname
# with no suffix cannot be distinguished from ordinary text by inspection;
# field names (below) cover that case.
HOSTNAME_VALUE_RE = re.compile(
    r"\b[A-Za-z0-9][A-Za-z0-9._-]*\.(?:local|localdomain|lan|home|internal|corp)\b",
    re.IGNORECASE,
)
ABS_PATH_RE = re.compile(
    r"(?:/Users/|/home/[^/\s]+|/private/|/tmp/|/var/folders/|[A-Za-z]:\\|(?<![\w.])/(?:Volumes|opt|etc)/)"
)

SIGNER_COLUMNS = ("participant", "signer_id", "signer", "participant_id", "Participant ID")
SIGNER_VALUE_KEYS = ("signer_id", "signer", "participant", "participant_id")
SIGNER_LIST_KEYS = ("signer_ids", "participants")
SIGNER_CONTAINER_KEYS = ("per_signer", "per_signer_counts", "per_signer_accuracy", "signers")
HOSTNAME_KEYS = ("hostname", "machine_hostname", "device_model", "machine")
GENERIC_DEVICE = "generic-device"


class PublishLeakError(ValueError):
    """Raised when sanitised output would still carry private identifiers."""


# --------------------------------------------------------------------------
# Leak detection (shared with the compliance checker)
# --------------------------------------------------------------------------

def leaks_in_text(text: str) -> list[str]:
    """Return human-readable leak reasons for one string (empty when safe)."""
    if not isinstance(text, str):
        return []
    reasons: list[str] = []
    for match in PARTICIPANT_TOKEN_RE.finditer(text):
        reasons.append(f"raw participant token {match.group(0)!r}")
    for match in HOSTNAME_VALUE_RE.finditer(text):
        reasons.append(f"hostname-like value {match.group(0)!r}")
    for match in ABS_PATH_RE.finditer(text):
        reasons.append(f"absolute private path {match.group(0)!r}")
    return reasons


def _walk_strings(value, path=""):
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, dict):
        for key, item in value.items():
            # Keys are scanned too: a per_signer key is exactly the risk.
            yield from _walk_strings(str(key), f"{path}/{key}")
            yield from _walk_strings(item, f"{path}/{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk_strings(item, f"{path}[{index}]")


def leaks_in_document(value) -> list[str]:
    """Return leak reasons for a JSON-shaped value or a file path.

    Strings are checked for participant tokens, hostname suffixes and
    absolute private paths; keys are checked identically.
    """
    if isinstance(value, (str, Path)) and not isinstance(value, dict):
        path = Path(value)
        if path.is_file():
            if path.suffix == ".json":
                try:
                    value = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    return [f"{path}: unreadable JSON ({exc})"]
            else:
                value = path.read_text(encoding="utf-8", errors="ignore")
    out: list[str] = []
    for where, text in _walk_strings(value):
        for reason in leaks_in_text(text):
            out.append(f"{where or '<root>'}: {reason}")
    return out


# --------------------------------------------------------------------------
# Value transforms
# --------------------------------------------------------------------------

def _key_bytes(key):
    if key is None:
        return None
    return key if isinstance(key, bytes) else str(key).encode("utf-8")


def pseudonymise_signer(value, key, width: int = 8) -> str:
    """Deterministic keyed pseudonym ``signer_<hex>``; never reversible."""
    if not isinstance(key, (bytes, str)) or not key:
        raise ValueError("pseudonymise_signer requires a non-empty key")
    digest = hmac.new(_key_bytes(key), str(value).encode("utf-8"),
                      hashlib.sha256).hexdigest()
    return f"signer_{digest[:width]}"


def _sanitize_scalar(text: str, *, repo_root=None) -> str:
    """Redact one string value: paths become repo-relative/basename."""
    stripped = text.strip()
    if not stripped:
        return text
    # Whole-string absolute path.
    if stripped.startswith("/") or re.match(r"^[A-Za-z]:\\", stripped):
        candidate = Path(stripped.replace("\\", "/"))
        if repo_root is not None:
            try:
                return candidate.resolve().relative_to(
                    Path(repo_root).resolve()).as_posix()
            except (ValueError, OSError):
                pass
        return candidate.name or "[redacted-path]"
    # Embedded private path inside prose.
    if ABS_PATH_RE.search(text):
        return ABS_PATH_RE.sub("[redacted-path]", text)
    # Embedded hostname suffix.
    if HOSTNAME_VALUE_RE.search(text):
        return HOSTNAME_VALUE_RE.sub("[redacted-host]", text)
    return text


def _mapped_signer(value, key, index):
    text = "" if value is None else str(value).strip()
    if not text or text.lower() in ("unknown", "unspecified"):
        return text or "unknown"
    if key:
        return pseudonymise_signer(text, key)
    # Stable, non-identifying label; ordering is sorted by the caller.
    return f"signer_redacted_{index:02d}"


def sanitize_document(value, *, key=None, repo_root=None):
    """Deep-copy ``value`` with signer linkage and private paths removed.

    * CSV-shaped dicts (a header + rows) are not handled here; see
      :func:`sanitize_predictions_csv`.
    * ``per_signer``-style dicts: keys pseudonymised (with ``key``) or
      replaced by stable ``signer_redacted_<n>`` labels (without).
    * ``signer_id`` / ``signer`` / ``participant`` values likewise.
    * hostname-bearing fields become ``generic-device`` or are dropped.
    * every other string is path/hostname redacted.

    Raises :class:`PublishLeakError` if the result still carries a leak.
    """
    def _walk(item):
        if isinstance(item, dict):
            out = {}
            signer_container = None
            for name, sub in item.items():
                low = str(name).lower()
                if low in HOSTNAME_KEYS:
                    if low == "device_model":
                        out[name] = GENERIC_DEVICE
                    # hostname / machine / machine_hostname are dropped.
                    continue
                if low in SIGNER_CONTAINER_KEYS and isinstance(sub, dict):
                    signer_container = (name, sub)
                    continue
                if low in SIGNER_VALUE_KEYS and isinstance(sub, str):
                    out[name] = _mapped_signer(sub, key, 0)
                    continue
                if low in SIGNER_LIST_KEYS and isinstance(sub, list):
                    mapped = sorted(sub, key=str)
                    # Without a key, many raw IDs would collide into one
                    # label; keep the count but drop the identities.
                    if key:
                        out[name] = [_mapped_signer(v, key, i) for i, v in enumerate(mapped)]
                    else:
                        out[name] = ["redacted"] * len(mapped)
                    continue
                out[name] = _walk(sub)
            if signer_container is not None:
                name, sub = signer_container
                # Sorted original keys keep the label order deterministic.
                mapped = {}
                for index, signer in enumerate(sorted(sub, key=str)):
                    label = _mapped_signer(signer, key, index)
                    mapped[label] = _walk(sub[signer])
                out[name] = mapped
            return out
        if isinstance(item, list):
            return [_walk(sub) for sub in item]
        if isinstance(item, str):
            return _sanitize_scalar(item, repo_root=repo_root)
        return item

    clean = _walk(value)
    problems = leaks_in_document(clean)
    if problems:
        raise PublishLeakError("refusing to emit unsafe export: " + "; ".join(problems[:5]))
    return clean


# --------------------------------------------------------------------------
# File-level writers (atomic; original never touched)
# --------------------------------------------------------------------------

def _atomic_write_text(dst: Path, text: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{dst.name}.", dir=dst.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_name, dst)
    finally:
        Path(tmp_name).unlink(missing_ok=True)


def _assert_distinct(src: Path, dst: Path) -> None:
    if src.resolve() == dst.resolve():
        raise ValueError(f"refusing to overwrite the original: {src}")
    try:
        dst.resolve().relative_to(src.resolve())
    except ValueError:
        return
    raise ValueError(f"refusing to write inside the source path: {dst}")


def sanitize_predictions_csv(src_path, dst_path, *, key=None, repo_root=None) -> dict:
    """Write a publish copy of a per-clip predictions CSV without raw IDs.

    Signer columns are dropped entirely when ``key`` is None; with ``key`` a
    single ``signer_pseudonym`` column is appended instead, so local
    aggregation keeps working without exposing the raw participant.
    """
    src, dst = Path(src_path), Path(dst_path)
    _assert_distinct(src, dst)
    dropped: list[str] = []
    rows: list[dict] = []
    with src.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"{src}: missing CSV header")
        fields = list(reader.fieldnames)
        signer_source = next((c for c in fields if c.lower() in
                              {s.lower() for s in SIGNER_COLUMNS}), None)
        dropped = [c for c in fields if c.lower() in
                   {s.lower() for s in SIGNER_COLUMNS}]
        kept = [c for c in fields if c not in dropped]
        added = []
        if key and signer_source is not None:
            kept = kept + ["signer_pseudonym"]
            added = ["signer_pseudonym"]
        for row in reader:
            clean = {c: _sanitize_scalar(row.get(c) or "", repo_root=repo_root)
                     for c in kept if c in fields}
            if added:
                clean["signer_pseudonym"] = pseudonymise_signer(
                    row.get(signer_source) or "", key)
            rows.append(clean)

    # Fail closed before writing anything.
    problems = leaks_in_document(rows) + leaks_in_document(kept)
    if problems:
        raise PublishLeakError("refusing to emit unsafe CSV: " + "; ".join(problems[:5]))

    sink = io.StringIO()
    writer = csv.DictWriter(sink, fieldnames=kept, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_write_text(dst, sink.getvalue())
    return {"source": str(src), "output": str(dst), "rows": len(rows),
            "dropped_columns": dropped, "added_columns": added}


def sanitize_json(src_path, dst_path, *, key=None, repo_root=None) -> dict:
    """Write a publish copy of a JSON evidence document."""
    src, dst = Path(src_path), Path(dst_path)
    _assert_distinct(src, dst)
    try:
        data = json.loads(src.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{src}: invalid JSON ({exc})") from exc
    clean = sanitize_document(data, key=key, repo_root=repo_root)
    _atomic_write_text(dst, json.dumps(clean, indent=2, ensure_ascii=False) + "\n")
    redacted = [k for k in data
                if isinstance(k, str) and k.lower() in SIGNER_CONTAINER_KEYS]
    return {"source": str(src), "output": str(dst), "redacted_keys": redacted}


def sanitize_file(src_path, dst_path, *, key=None, repo_root=None) -> dict:
    src = Path(src_path)
    if src.suffix.lower() == ".csv":
        return sanitize_predictions_csv(src, dst_path, key=key, repo_root=repo_root)
    if src.suffix.lower() in (".json", ".jsonl"):
        if src.suffix.lower() == ".jsonl":
            raise ValueError("JSONL evidence is not part of the tracked export set")
        return sanitize_json(src, dst_path, key=key, repo_root=repo_root)
    raise ValueError(f"unsupported evidence type: {src}")


def sanitize_all(root=None, outdir=None, *, key=None, sources=DEFAULT_SOURCES) -> dict:
    """Sanitise every tracked restricted export into ``outdir``.

    Missing sources are reported, not fatal.  Returns a report with one
    entry per source plus ``ok`` (all found sources sanitised cleanly).
    """
    root = Path(root or _REPO_ROOT).resolve()
    out = Path(outdir or (root / DEFAULT_OUTDIR)).resolve()
    if out == root:
        raise ValueError("output directory must not be the repo root")
    report = {"root": str(root), "outdir": str(out), "files": []}
    for rel in sources:
        src = root / rel
        if not src.is_file():
            report["files"].append({"source": rel, "missing": True})
            continue
        entry = sanitize_file(src, out / src.name, key=key, repo_root=root)
        entry["source"] = rel
        report["files"].append(entry)
    report["ok"] = all(not f.get("missing") for f in report["files"])
    return report


# --------------------------------------------------------------------------
# Publisher-side verification (called by the compliance checker)
# --------------------------------------------------------------------------

def scan_publish_dir(publish_dir) -> dict:
    """Check a publish directory for any surviving private identifier."""
    base = Path(publish_dir)
    result = {"dir": str(base), "exists": base.is_dir(), "files": [], "leaks": [],
              "ok": False}
    if not base.is_dir():
        return result
    for path in sorted(base.rglob("*")):
        if not path.is_file():
            continue
        result["files"].append(path.relative_to(base).as_posix())
        result["leaks"].extend(f"{path.name}: {reason}"
                               for reason in leaks_in_document(path))
    result["ok"] = bool(result["files"]) and not result["leaks"]
    return result


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=_REPO_ROOT)
    ap.add_argument("--output", type=Path, default=None,
                    help=f"Publish output directory (default: {DEFAULT_OUTDIR})")
    ap.add_argument("--key", default=None,
                    help="Pseudonymisation key (HMAC). Omit to drop linkage. "
                         "May also come from ECHOGESTURE_EXPORT_KEY.")
    args = ap.parse_args(argv)
    key = args.key or os.environ.get("ECHOGESTURE_EXPORT_KEY") or None
    try:
        report = sanitize_all(args.root, args.output, key=key)
    except (PublishLeakError, ValueError, OSError) as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}, indent=2))
        return 1
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())