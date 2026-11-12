"""Reusable final compliance checker for EchoGesture Track 2 bundles.

A single, stdlib-only, read-only sweep the lead can call during bundle
verification to gate the publishable surface.  It composes the existing
hardened primitives instead of reimplementing them:

* ``build_bundle`` (sibling module) for ZIP member / extraction safety
  (traversal, absolute paths, symlink/hardlink/external attributes,
  duplicates) and its private-path leak regex;
* ``evaluation/device/v8_privacy/sanitize_evidence`` for evidence
  sanitisation semantics (participant ID / hostname / absolute-path
  redaction while preserving non-private provenance).

Coverage (one finding per check):

    C01  offline inference  -- no network/cloud imports in src/+app/;
                               the only URL literal is a download-hint
                               error string, never a call.
    C02  secrets            -- credential patterns, excluding known
                               avoidable false positives (ASL gloss
                               ``secretary``, ``queue.Queue`` attrs,
                               the scanner's own regex source).
    C03  evidence hygiene   -- participant IDs, usernames/hostnames and
                               absolute paths must not survive in
                               publishable evidence. Audits the sanitised
                               publish surface by default; ``--raw`` audits
                               the tracked originals.
    C04  manifest paths     -- ``run_manifest`` resolves clips inside an
                               allowed input root (no ``../``/absolute
                               escape).
    C05  ZIP safety         -- traversal, absolute, symlink/hardlink/
                               external-attr, duplicate members, per-file
                               and total size limits (any passed bundle
                               plus the committed provisional zips).
    C06  temp media         -- demo temp clips deleted on success AND on
                               predictor exception; private 0700 dir.
    C07  screenshot policy  -- demo-lab HUD PNG/JSON must default to an
                               ephemeral private directory (or require an
                               explicit --outdir), never a persistent
                               shared location.
    C08  raw footage        -- no tracked raw video/audio/array caches or
                               screenshots; excluded datasets stay out of
                               git and out of the bundle.
    C09  restricted data    -- ASL Citizen / WLASL derived metadata must
                               not carry raw participant IDs in the publish
                               surface; raw local originals are reported
                               as local-only.
    C10  LICENSE/NOTICE     -- repo root must carry a licence + NOTICE
                               with a provenance row for every models/
                               artifact and an explicit rights statement.
    C11  model licensing    -- restricted weights are NOT_CLEARED unless
                               their NOTICE row records written
                               authorization (machine-checkable marker).

Exit 0 on a completed sweep (FAILs and NOT_CLEARED findings included);
exit 2 only on an audit error.  ``--strict`` turns the diagnostic sweep into
a release gate: it exits 1 when any mandatory gate is FAIL / NOT_CLEARED /
NOT_MEASURED, and prints an explicit "NOT legally cleared for distribution"
line while C10/C11 remain unresolved.  Stdout is JSON; ``--out`` writes the
same JSON (atomic) for CI.  Run with the interpreter flags the data warrants:

    python3 -I scripts/release/compliance_final.py --root . [--zip bundle.zip] [--out findings.json] [--strict]

Stdlib only; Python 3.11 compatible.  Never writes into the audited tree
and performs no network access.

Release-gate checks are evaluated against a *publish surface*: by default the
sanitised copies under ``evaluation/device/compliance_final/publish/`` (when
present, produced by ``scripts/release/sanitize_export.py``) are what C03/C09
audit for publication.  ``--raw`` audits the tracked originals instead, which
is the stricter local-hygiene view and still reports the raw participant IDs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import sys
import tempfile
import zipfile
from pathlib import Path

# --------------------------------------------------------------------------
# Path setup: make sibling ``build_bundle`` and the v8_privacy sanitizer
# importable regardless of how the lead invokes this module.
# --------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PARENT = str(_REPO_ROOT)
_sys_path = __import__("sys")
_sys_path.path.insert(0, str(Path(__file__).resolve().parent))          # scripts/release
_sys_path.path.insert(0, str(_REPO_ROOT / "evaluation" / "device"))     # v8_privacy + run_manifest

import build_bundle  # noqa: E402  (sibling; safe_member, _ABS_LEAK_RE)
from v8_privacy import sanitize_evidence  # noqa: E402
import sanitize_export  # noqa: E402  (sibling; publish-surface leak scan)

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

SHIPPED_DIRS = ("src", "app")
# Directory subtrees known to hold research media/weights; never scanned for
# text evidence, only cheaply checked for git-tracking (via git, below).
SKIP_DIRS = {
    ".git", ".venv", "venv", "__pycache__", ".scratch", "tmp", ".temp",
    "node_modules", ".idea", ".vscode",
}
# Subtrees that contain downloaded/derived research media; excluded from
# recursive text scans to bound runtime and disk reads.
MEDIA_LAB_DIRS = {
    "v5_accuracy", "v7_android", "v4_hardware", "v4_performance",
    "release_lab", "v5_release", "v7_release", "v8_release",
}
SCAN_SUFFIXES = (".py", ".md", ".txt", ".json", ".csv", ".yml", ".yaml", ".toml", ".cfg")
# Files larger than this are never read as text (Keept out of evidence scans).
MAX_TEXT_SCAN_BYTES = 5 * 1024 * 1024

SIGNER_COLUMNS = {"participant", "signer_id", "signer", "Participant ID", "participant_id"}
# Metric-name / dict-key participant-ID forms: bare "P15" keys, and
# underscore-prefixed suffixes ("top1_accuracy_signer_P15", "_P15").
# Case-sensitive on the "P" prefix: ASL Citizen participant IDs are uppercase
# (P15, P17), while latency percentiles in evidence are lowercase
# (p50, p95, p99) and must not match.
PARTICIPANT_METRIC_RE = re.compile(r"\bP\d{1,3}\b|_P\d{1,3}\b")
HOSTNAME_INDICATOR_RE = re.compile(
    r"\bplatform\.node\(\)|\bsocket\.gethostname\(\)|\bsocket\.getfqdn\(\)|\bgetpass\.getuser\(\)"
)
# Absolute local-path leak detection (mirrors build_bundle._ABS_LEAK_RE plus
# the v8 sanitizer's extended private prefixes).
ABS_LEAK_RE = getattr(build_bundle, "_ABS_LEAK_RE", re.compile(r"(/Users/|/home/|/private/|/tmp/|[A-Za-z]:\\)"))

# Credential patterns with deliberate false-positive avoidance: the ASL gloss
# "secretary", Queue attributes, and the scanner's own regex source are
# excluded by the line-level filter in ``_secret_hit``.
SECRET_RE = re.compile(
    r"(api[_-]?key|client[_-]?secret|auth[_-]?token|access[_-]?token|passwd|password\s*=|"
    r"aws_secret|AWS_SECRET|secret[_-]?key|private[_-]?key|BEGIN [A-Z ]*PRIVATE KEY|"
    r"Bearer [A-Za-z0-9._-]{16,})",
    re.IGNORECASE,
)
NETWORK_RE = re.compile(
    r"(?:^|[^.\w])(?:import|from)\s+(urllib|socket|requests|httpx|http\.client|"
    r"firebase|google\.cloud)|urlopen\b|websocket\b",
)
# Local URL-scheme text that would indicate a runtime download vs. a
# documented/hint string. Separately treated as informational.
URL_SCHEME_LITERAL_RE = re.compile(r"https?://")

RESEARCH_FETCHERS = {
    "evaluation/eval_asl_citizen.py",
    "evaluation/device/fetch_smoke_clip.py",
    "evaluation/device/fetch_validation_batch.py",
}
RESTRICTED_METADATA = {
    "evaluation/asl_citizen_predictions.csv",
    "evaluation/asl_citizen_ptq_results.json",
    "evaluation/asl_citizen_qat_results.json",
}
PUBLISH_DIR = "evaluation/device/compliance_final/publish"
# Mandatory release gates.  C08 is mandatory only when git tracking is
# measurable; NOT_MEASURED is handled explicitly in the gate evaluation.
MANDATORY_GATES = (
    "C01-offline-inference", "C02-secrets", "C03-evidence-hygiene",
    "C04-manifest-paths", "C05-zip-safety", "C06-temp-media",
    "C07-screenshot-policy", "C08-raw-footage", "C09-restricted-data",
    "C10-license-notice", "C11-model-licensing",
)
# Gate states that block a strict release.
BLOCKING_STATES = ("FAIL", "NOT_CLEARED")
# Model artifacts whose redistribution rights are NOT established by the repo.
RESTRICTED_MODEL_NAMES = (
    "landmark_classifier_fp16.tflite", "landmark_classifier_int8.tflite",
    "landmark_labels.json", "s3d_conv1d_qat_int8.tflite", "ptq_100_int8.tflite",
    "label100.json",
)
# A restricted artifact is clear only when its NOTICE row records written
# authorization.  This is the machine-checkable marker; prose alone is not.
AUTHORIZATION_MARKER = "AUTHORIZATION RECORDED"
TRACKED_BUNDLE_ZIPS = (
    "scripts/release/release_lab/provisional.zip",
    "scripts/release/v5_release/provisional.zip",
    "scripts/release/v8_release/provisional.zip",
    "evaluation/device/v7_release/provisional.zip",
)
RAW_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".npy", ".npz", ".pt", ".pth")
SCREENSHOT_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")
MODEL_EXTENSIONS = (".task", ".tflite", ".onnx", ".pb")

DEFAULT_MAX_MEMBER_BYTES = 20 * 1024 * 1024   # per-uncompressed-member cap
DEFAULT_MAX_TOTAL_BYTES = 100 * 1024 * 1024   # total uncompressed payload cap
DEFAULT_MAX_MEMBERS = 4096


# --------------------------------------------------------------------------
# Small read helpers (bounded, deterministic, non-recursive where possible)
# --------------------------------------------------------------------------

def _read_text(path: Path) -> str:
    try:
        if path.stat().st_size > MAX_TEXT_SCAN_BYTES:
            return ""
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def _git_ls_files(root: Path) -> list[str]:
    """Return the git-tracked path list, or [] when not a git repo."""
    import subprocess  # local import: audit also usable via python3 -I

    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "ls-files"],
            capture_output=True, text=True, timeout=30, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if proc.returncode != 0:
        return []
    return [line for line in proc.stdout.splitlines() if line.strip()]


def _iter_text_files(root: Path, dirs):
    """Yield *.py/md/txt/json/csv files under ``dirs``, skipping banned dirs."""
    for name in dirs:
        base = root / name
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.suffix not in SCAN_SUFFIXES:
                continue
            # Only skip directories *inside* the scanned root. Absolute
            # ancestors must not count: on Linux the temp dir is /tmp/...,
            # so matching "tmp" against the whole path made every synthetic
            # repo scan as empty and every finding pass.
            if any(part in SKIP_DIRS for part in path.relative_to(root).parts):
                continue
            if path.suffix in RAW_EXTENSIONS or path.suffix in SCREENSHOT_EXTENSIONS:
                continue
            yield path


def _line_hits(text: str, pattern, rel: str) -> list[str]:
    out: list[str] = []
    for n, line in enumerate(text.splitlines(), 1):
        if pattern.search(line):
            out.append(f"{rel}:{n}: {line.strip()[:100]}")
    return out


# --------------------------------------------------------------------------
# Individual checks (each returns a finding dict)
# --------------------------------------------------------------------------

def _check_offline(root: Path) -> dict:
    hits: list[str] = []
    for p in _iter_text_files(root, SHIPPED_DIRS):
        rel = p.relative_to(root).as_posix()
        for n, line in enumerate(_read_text(p).splitlines(), 1):
            if NETWORK_RE.search(line):
                hits.append(f"{rel}:{n}: {line.strip()[:90]}")
    # Informational: URL literals that are download-hint strings, not calls.
    url_hints: list[str] = []
    if not hits:
        for p in _iter_text_files(root, SHIPPED_DIRS):
            for n, line in enumerate(_read_text(p).splitlines(), 1):
                if URL_SCHEME_LITERAL_RE.search(line):
                    url_hints.append(f"{p.relative_to(root).as_posix()}:{n}: {line.strip()[:90]}")
    status = "PASS" if not hits else "FAIL"
    detail = (
        f"Network/cloud sinks in src+app: {len(hits)}. "
        "src/main.py and app/offline_demo.py infer locally (TFLite/mediapipe)."
    )
    if url_hints and not hits:
        detail += f" URL literals (hint strings, not calls): {len(url_hints)}."
    return {"check": "C01-offline-inference", "status": status, "detail": detail,
            "where": (hits or url_hints)[:12]}


def _secret_hit(line: str) -> str | None:
    """Skip avoidable false positives and return the offending pattern or None."""
    low = line.lower()
    if "secretary" in low:
        return None
    if "queue." in low:
        return None
    # The scanner's own pattern source and regex-reference prose.
    if "BEGIN [A-Z ]*PRIVATE KEY" in line and ("re.compile" in line or "SECRET_RE" in line):
        return None
    m = SECRET_RE.search(line)
    return m.group(0) if m else None


def _check_secrets(root: Path) -> dict:
    # Audit-artifact subtrees and test files deliberately contain planted
    # samples and the scanner's own regex source; exclude them. The checker's
    # own source literally spells the credential patterns — never scan it.
    skip_parts = {"privacy_lab", "v7_compliance", "v8_privacy", "__pycache__"}
    hits: list[str] = []
    dirs = list(SHIPPED_DIRS) + ["scripts", "models"]
    for p in _iter_text_files(root, dirs):
        if p.name == "compliance_final.py" or p.name.startswith("test_"):
            continue
        if any(part in skip_parts for part in p.parts):
            continue
        rel = p.relative_to(root).as_posix()
        for n, line in enumerate(_read_text(p).splitlines(), 1):
            pat = _secret_hit(line)
            if pat:
                hits.append(f"{rel}:{n}: {pat}")
    return {"check": "C02-secrets", "status": "PASS" if not hits else "FAIL",
            "detail": "Hardcoded credential patterns in the shipped surface "
                      "(src/, app/, scripts/, model label JSON), excluding the ASL "
                      "gloss 'secretary' and the scanner's own regex source.",
            "where": hits[:12]}


def _check_evidence(root: Path, *, raw: bool = False) -> dict:
    """Participant ID / hostname / username / absolute-path leak scan.

    By default this audits the *publish surface*: ``results/results.csv`` plus,
    when present, the sanitised export tree
    ``evaluation/device/compliance_final/publish/``.  With ``raw=True`` it
    audits the tracked restricted originals directly, which reports the raw
    participant IDs that remain local-only by design (their removal is a
    separate owner decision; see NOTICE section 3).
    """
    hits: list[str] = []
    publish = root / PUBLISH_DIR
    publish_scan = sanitize_export.scan_publish_dir(publish) if publish.is_dir() else None
    audited = ["results/results.csv"]
    if raw or publish_scan is None:
        targets = ["results/results.csv", "evaluation/asl_citizen_predictions.csv",
                   "evaluation/asl_citizen_ptq_results.json",
                   "evaluation/asl_citizen_qat_results.json"]
    else:
        targets = ["results/results.csv"]
        audited += [f"{PUBLISH_DIR}/{name}" for name in publish_scan["files"]]
        hits.extend(f"{PUBLISH_DIR}: {leak}" for leak in publish_scan["leaks"])
        if not publish_scan["files"]:
            hits.append(f"{PUBLISH_DIR}: empty publish surface (run sanitize_export.py)")
    for rel in targets:
        p = root / rel
        if not p.is_file():
            continue
        text = _read_text(p)
        # Absolute path / username leaks.
        for n, line in enumerate(text.splitlines(), 1):
            if ABS_LEAK_RE.search(line):
                hits.append(f"{rel}:{n}: abs-path/username")
        # Participant IDs in metric-name strings and per_signer keys.
        for m in PARTICIPANT_METRIC_RE.finditer(text):
            hits.append(f"{rel}: participant-id {m.group(0)!r}")
            if len(hits) > 60:
                break
    # Hostname/username-embedding indicators belong in shipped code only
    # (research eval scripts legitimately call platform.node(); they are
    # sanitized by sanitize_evidence before any evidence leaves the machine).
    for p in _iter_text_files(root, SHIPPED_DIRS):
        for n, line in enumerate(_read_text(p).splitlines(), 1):
            if HOSTNAME_INDICATOR_RE.search(line):
                hits.append(f"{p.relative_to(root).as_posix()}:{n}: hostname indicator")
    # Dedupe while preserving first occurrence.
    seen, unique = set(), []
    for h in hits:
        if h not in seen:
            seen.add(h)
            unique.append(h)
    # Exclude genuinely-benign evidence: the sanitizer module itself holds
    # participant tokens in fixtures/docstrings and is not publishable evidence.
    leaks = [h for h in unique if "v8_privacy" not in h.split(":")[0]]
    mode = "raw tracked originals" if raw or publish_scan is None else "publish surface"
    return {"check": "C03-evidence-hygiene", "status": "PASS" if not leaks else "FAIL",
            "detail": f"Participant IDs, usernames/hostnames and absolute paths must not "
                      f"survive in publishable evidence (audited: {mode}; "
                      f"{len(audited)} file group(s)). Local research originals keep raw "
                      f"IDs unless --raw is passed.",
            "where": leaks[:15]}


def _check_manifest(root: Path) -> dict:
    text = _read_text(root / "evaluation/device/run_manifest.py")
    contained = ("relative_to" in text) and ("escapes" in text or "allowed input root" in text)
    generic = '"_GENERIC_DEVICE_MODEL"' in text or "_GENERIC_DEVICE_MODEL" in text
    return {"check": "C04-manifest-paths", "status": "PASS" if (contained and generic) else "FAIL",
            "detail": "run_manifest resolves clip paths inside an allowed input root and "
                      "writes a generic (non-hostname) device model.",
            "where": ["evaluation/device/run_manifest.py"]}


# --------------------------------------------------------------------------
# ZIP member / extraction safety (reusable, not bundle-structure specific)
# --------------------------------------------------------------------------

def zip_safety_violations(path, *, max_member_bytes=DEFAULT_MAX_MEMBER_BYTES,
                          max_total_bytes=DEFAULT_MAX_TOTAL_BYTES,
                          max_members=DEFAULT_MAX_MEMBERS) -> list[str]:
    """Return human-readable violations for one ZIP, or [] when safe.

    Checks: duplicate member names, unsafe member names (traversal, absolute,
    drive/UNC/NUL/backslash), symlink / hardlink / non-regular external
    attributes, per-file and total uncompressed-size limits, and member-count
    limit.  Uses ``build_bundle.safe_member`` so the result matches the
    builder's own verifier.  Pure inspection -- nothing is extracted.
    """
    path = Path(path)
    out: list[str] = []
    if not path.is_file():
        return [f"{path}: missing"]
    try:
        archive = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        return [f"{path}: not a valid zip: {exc}"]
    with archive:
        members = archive.infolist()
        if len(members) > max_members:
            out.append(f"{path}: {len(members)} members exceeds limit {max_members}")
        names = [m.filename for m in members]
        if len(names) != len(set(names)):
            out.append(f"{path}: duplicate member names")
        total = 0
        for member in members:
            total += member.file_size
            if member.file_size > max_member_bytes:
                out.append(f"{member.filename}: uncompressed {member.file_size} exceeds {max_member_bytes}")
            if not build_bundle.safe_member(member.filename):
                out.append(f"{member.filename}: unsafe name (traversal/absolute/drive/colon/NUL)")
                continue
            mode = member.external_attr >> 16
            ftype = mode & 0o170000
            if ftype == stat.S_IFLNK:
                out.append(f"{member.filename}: symlink member")
            elif ftype not in (0, stat.S_IFREG, stat.S_IFDIR):
                # A zero file-type field (bare permission bits, e.g. from
                # ZipFile.writestr) denotes a regular file; a non-regular,
                # non-directory type (FIFO, device, socket, ...) is unsafe.
                out.append(f"{member.filename}: unsafe external attributes 0o{mode & 0o7777:o}")
        if total > max_total_bytes:
            out.append(f"{path}: total uncompressed {total} exceeds {max_total_bytes}")
    return out


def _check_zips(root: Path, zip_path=None) -> dict:
    targets = list(TRACKED_BUNDLE_ZIPS)
    if zip_path:
        targets.append(str(Path(zip_path)))
    hits: list[str] = []
    seen = set()
    for rel in targets:
        p = (root / rel) if not Path(rel).is_absolute() else Path(rel)
        if not p.is_file():
            continue
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        hits += [f"{p}: {v}" for v in zip_safety_violations(p)]
    status = "PASS" if targets and not hits else ("FAIL" if hits else "NOT_MEASURED")
    return {"check": "C05-zip-safety", "status": status,
            "detail": "ZIP traversal/absolute/symlink/hardlink/external-attr/duplicate "
                      "members and size limits (committed provisional zips plus any "
                      "passed --zip).", "where": hits[:15]}


def _check_temp_media(root: Path) -> dict:
    demo = _read_text(root / "app/offline_demo.py")
    ok = ("TemporaryDirectory" in demo and "mkstemp" in demo and
          "unlink(missing_ok=True)" in demo and "finally:" in demo)
    return {"check": "C06-temp-media", "status": "PASS" if ok else "FAIL",
            "detail": "app/offline_demo.py writes one temp clip via mkstemp in a private "
                      "TemporaryDirectory (0700) and unlinks it in finally (success AND "
                      "predictor exception).", "where": ["app/offline_demo.py"]}


def _check_screenshot(root: Path) -> dict:
    dl = _read_text(root / "app/demo_lab/__init__.py")
    if not dl:
        return {"check": "C07-screenshot-policy", "status": "NOT_MEASURED",
                "detail": "app/demo_lab/__init__.py not present; screenshot policy "
                          "cannot be evaluated.", "where": ["app/demo_lab/__init__.py"]}
    # The defect: a persistent shared default output location
    # (tempfile.gettempdir()/echogesture-demo-lab) for a HUD PNG that embeds a
    # rendered video frame.
    persistent_default = ("gettempdir" in dl) and ("echogesture-demo-lab" in dl)
    # The fix: ephemeral per-run TemporaryDirectory, or an explicit --outdir
    # required before anything is written.
    ephemeral_default = ("TemporaryDirectory" in dl) and (
        "ephemeral_outdir" in dl or "args.outdir" in dl)
    writes_png = "cv2.imwrite" in dl
    ok = writes_png and ephemeral_default and not persistent_default
    return {"check": "C07-screenshot-policy", "status": "PASS" if ok else "FAIL",
            "detail": "app/demo_lab renders a HUD PNG (a rendered video frame) + JSON. "
                      "The default output location must be ephemeral (private per-run "
                      "TemporaryDirectory) or require an explicit --outdir; a persistent "
                      "shared tempfile.gettempdir()/echogesture-demo-lab default FAILS.",
            "where": ["app/demo_lab/__init__.py"]}


def _check_raw_footage(root: Path) -> dict:
    tracked = _git_ls_files(root)
    bad: list[str] = []
    for rel in tracked:
        low = rel.lower()
        if low.endswith(RAW_EXTENSIONS):
            bad.append(f"{rel}: raw footage/cache")
        elif low.endswith(SCREENSHOT_EXTENSIONS):
            bad.append(f"{rel}: screenshot")
    no_git = not tracked
    return {"check": "C08-raw-footage", "status": "PASS" if (tracked and not bad) else ("NOT_MEASURED" if no_git else "FAIL"),
            "detail": "No raw video/audio/array caches or screenshots tracked in git. "
                      "Research media (v5_accuracy/mp4, v7_android npz) is left on disk "
                      "but untracked (skipped from text scans).",
            "where": bad[:12]}


def _raw_restricted_flagged(root: Path) -> list[str]:
    """Return the tracked restricted originals that still carry raw IDs."""
    flagged: list[str] = []
    csvp = root / "evaluation/asl_citizen_predictions.csv"
    if csvp.is_file():
        header = _read_text(csvp).splitlines()[0] if _read_text(csvp) else ""
        if any(c in SIGNER_COLUMNS for c in (header.split(",") if header else [])):
            flagged.append("evaluation/asl_citizen_predictions.csv (participant column)")
    for rel in ("evaluation/asl_citizen_ptq_results.json",
                "evaluation/asl_citizen_qat_results.json"):
        p = root / rel
        if p.is_file() and '"per_signer"' in _read_text(p):
            flagged.append(f"{rel} (per_signer keys)")
    return flagged


def _check_restricted(root: Path) -> dict:
    """The *publishable* surface must carry no raw participant IDs.

    Local research originals are expected to still hold raw IDs; that is
    reported explicitly and is not what ships.  Publication is gated on the
    sanitised export tree produced by ``scripts/release/sanitize_export.py``.
    """
    raw_flagged = _raw_restricted_flagged(root)
    publish = sanitize_export.scan_publish_dir(root / PUBLISH_DIR)
    if not publish["exists"]:
        status = "FAIL"
        where = [f"{PUBLISH_DIR}: missing (run scripts/release/sanitize_export.py)"]
    elif not publish["files"]:
        status = "FAIL"
        where = [f"{PUBLISH_DIR}: empty publish surface"]
    elif publish["leaks"]:
        status = "FAIL"
        where = [f"{PUBLISH_DIR}: {leak}" for leak in publish["leaks"]]
    else:
        status = "PASS"
        where = []
    detail = ("ASL Citizen- / WLASL-derived metadata must not carry raw participant "
              f"IDs in the publish surface ({PUBLISH_DIR}, {len(publish['files'])} file(s) "
              "scanned). Raw IDs remain in local research originals: "
              f"{len(raw_flagged)} file(s), local/private only, removal is a separate "
              "owner decision (NOTICE section 3).")
    return {"check": "C09-restricted-data", "status": status, "detail": detail,
            "where": where[:12], "raw_local_only": raw_flagged}


def _model_artifacts(root: Path) -> list[str]:
    models_dir = root / "models"
    if not models_dir.is_dir():
        return []
    return sorted(p.name for p in models_dir.iterdir()
                  if p.is_file() and p.suffix.lower() in MODEL_EXTENSIONS + (".json",))


def _authorized_artifacts(notice_text: str) -> list[str]:
    """Return restricted artifact names whose NOTICE row records authorization.

    A restricted artifact clears only on the machine-checkable marker, so a
    passing gate always corresponds to an actual recorded permission rather
    than prose that merely sounds permissive.
    """
    cleared = []
    for line in notice_text.splitlines():
        if AUTHORIZATION_MARKER in line:
            cleared.extend(n for n in RESTRICTED_MODEL_NAMES if n in line)
    return sorted(set(cleared))


def _check_license(root: Path) -> dict:
    license_files = [p for p in root.iterdir()
                     if p.name.upper() in ("LICENSE", "LICENSE.MD", "LICENSE.TXT", "COPYING")
                     and p.is_file()]
    notice = root / "NOTICE"
    license_text = _read_text(license_files[0]) if license_files else ""
    notice_text = _read_text(notice)
    artifacts = _model_artifacts(root)
    missing_assets = [name for name in artifacts if name not in notice_text]
    normalized_license = re.sub(r"\s+", " ", license_text.lower())
    has_scope = "mit" in normalized_license and "does not" in normalized_license
    restricted_present = [n for n in RESTRICTED_MODEL_NAMES if n in artifacts]
    authorized = set(_authorized_artifacts(notice_text))
    unauthorized = [n for n in restricted_present if n not in authorized]
    # The rights statement is satisfied when every restricted artifact is
    # explicitly not-cleared in prose, or when every one is authorized.
    not_cleared_ok = ("no confirmed redistribution permission" in notice_text.lower()
                      or "no confirmed redistribution permission" in license_text.lower())
    rights_ok = not_cleared_ok or not unauthorized
    ok = (bool(license_files) and notice.is_file() and has_scope
          and not missing_assets and rights_ok)
    where = []
    if not license_files:
        where.append("(repo root): LICENSE missing")
    elif not has_scope:
        where.append("(repo root)/LICENSE: missing MIT scope limitation")
    if not notice.is_file():
        where.append("(repo root): NOTICE missing")
    elif missing_assets:
        where.append(f"NOTICE: model artifacts without provenance rows: {', '.join(missing_assets)}")
    elif not rights_ok:
        where.append("NOTICE: restricted weights neither not-cleared nor authorized")
    return {"check": "C10-license-notice", "status": "PASS" if ok else "FAIL",
            "detail": "Repo root carries LICENSE (MIT, repo-authored code only) and a NOTICE "
                      "with a per-asset provenance row for every models/ artifact and an "
                      "explicit rights statement (not-cleared, or recorded authorization).",
            "where": where[:12]}


def _check_model_licensing(root: Path) -> dict:
    artifacts = _model_artifacts(root)
    notice_text = _read_text(root / "NOTICE")
    missing = [name for name in artifacts if name not in notice_text]
    restricted = [n for n in RESTRICTED_MODEL_NAMES if n in artifacts]
    authorized = set(_authorized_artifacts(notice_text))
    unauthorized = [n for n in restricted if n not in authorized]
    if missing:
        status, detail = "FAIL", (
            "NOTICE lacks provenance rows for model artifacts: "
            + ", ".join(missing) + ".")
    elif restricted and unauthorized:
        status, detail = "NOT_CLEARED", (
            "NOT CLEARED for release. models/hand_landmarker.task = MediaPipe "
            "Apache-2.0 (re-verify and carry attribution). The ASL Citizen-derived "
            "landmark classifier weights and the WLASL-derived S3D weights have NO "
            "confirmed redistribution permission; they are local/private only pending "
            "written authorization recorded in NOTICE section 2a.")
    else:
        status, detail = "PASS", (
            "Every restricted model artifact has a recorded written authorization "
            "in NOTICE section 2a.")
    return {"check": "C11-model-licensing", "status": status, "detail": detail,
            "where": [f"models/{name}" for name in restricted]}


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------

def gate_report(result: dict) -> dict:
    """Evaluate the mandatory release gates from a :func:`run_checks` result.

    A gate blocks when it is FAIL or NOT_CLEARED, or when it is NOT_MEASURED
    and listed in MANDATORY_GATES (C08 is mandatory only if measurable; the
    current tree always measures it).  ``legal_clearance`` is False while
    C10/C11 are unresolved, which is the case that must never be described as
    cleared for distribution.
    """
    by_name = {f["check"]: f for f in result["findings"]}
    blocking = []
    for name in MANDATORY_GATES:
        finding = by_name.get(name)
        if finding is None:
            blocking.append({"check": name, "status": "MISSING"})
        elif finding["status"] in BLOCKING_STATES:
            blocking.append({"check": name, "status": finding["status"]})
        elif finding["status"] == "NOT_MEASURED" and name != "C08-raw-footage":
            blocking.append({"check": name, "status": "NOT_MEASURED"})
    unresolved_rights = [name for name in ("C10-license-notice", "C11-model-licensing")
                         if by_name.get(name, {}).get("status") != "PASS"]
    gated = {**result, "gate": {
        "strict_ok": not blocking,
        "blocking": blocking,
        "legal_clearance": not unresolved_rights,
        "unresolved_rights": unresolved_rights,
    }}
    return gated


def run_checks(root=None, zip_path=None, *, max_member_bytes=DEFAULT_MAX_MEMBER_BYTES,
               max_total_bytes=DEFAULT_MAX_TOTAL_BYTES, raw=False) -> dict:
    root = Path(root or _REPO_ROOT).resolve()
    findings = [
        _check_offline(root),
        _check_secrets(root),
        _check_evidence(root, raw=raw),
        _check_manifest(root),
        _check_zips(root, zip_path),
        _check_temp_media(root),
        _check_screenshot(root),
        _check_raw_footage(root),
        _check_restricted(root),
        _check_license(root),
        _check_model_licensing(root),
    ]
    summary = {
        "root": str(root),
        "zip": str(Path(zip_path).resolve()) if zip_path else None,
        "checks": len(findings),
        "pass": sum(1 for f in findings if f["status"] == "PASS"),
        "fail": sum(1 for f in findings if f["status"] == "FAIL"),
        "not_cleared": sum(1 for f in findings if f["status"] == "NOT_CLEARED"),
        "not_measured": sum(1 for f in findings if f["status"] == "NOT_MEASURED"),
        "findings": findings,
    }
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=_REPO_ROOT,
                    help="Repository / bundle root to audit (default: repo root).")
    ap.add_argument("--zip", type=Path, default=None,
                    help="Additional bundle ZIP to check for extraction safety.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Write the JSON findings here (atomic) for CI.")
    ap.add_argument("--strict", action="store_true",
                    help="Release gate: exit nonzero when any mandatory gate is FAIL / "
                         "NOT_CLEARED / NOT_MEASURED. Default is report-only exit 0.")
    ap.add_argument("--raw", action="store_true",
                    help="Audit the tracked restricted originals instead of the sanitised "
                         "publish surface (reports remaining raw participant IDs).")
    ap.add_argument("--max-member-bytes", type=int, default=DEFAULT_MAX_MEMBER_BYTES)
    ap.add_argument("--max-total-bytes", type=int, default=DEFAULT_MAX_TOTAL_BYTES)
    args = ap.parse_args(argv)

    try:
        result = gate_report(run_checks(
            args.root, args.zip,
            max_member_bytes=args.max_member_bytes,
            max_total_bytes=args.max_total_bytes,
            raw=args.raw))
    except Exception as exc:  # audit error only; findings FAILs exit 0
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}, indent=2))
        return 2

    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=f".{out.name}.", dir=out.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(text + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp, out)
        finally:
            Path(tmp).unlink(missing_ok=True)

    gate = result["gate"]
    if not gate["legal_clearance"]:
        # Always printed: this is the load-bearing rights statement, whether
        # the sweep is report-only or strict.
        print("NOT legally cleared for distribution: "
              + ", ".join(gate["unresolved_rights"]) + " unresolved "
              "(see NOTICE section 2a permission requests).", file=sys.stderr)
    if args.strict:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if not gate["strict_ok"]:
            blocking = ", ".join(f"{b['check']}={b['status']}" for b in gate["blocking"])
            print(f"STRICT GATE FAILED: {blocking}", file=sys.stderr)
            return 1
        print("STRICT GATE PASSED: all mandatory gates cleared.", file=sys.stderr)
        return 0

    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())