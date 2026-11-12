"""Secure manifest reader with configurable input roots (D2 fix).

Drop-in replacement for ``evaluation/device/run_manifest.py::read_manifest``.

The original accepts ``../`` escapes and absolute paths anywhere on disk.
This module constrains every clip path to an explicit allow-list of roots:

* By default the only allowed root is the manifest file's own directory
  (after ``resolve()``).  This preserves the legitimate
  ``fetch_validation_batch.py`` flow, which writes ``manifest.csv`` next
  to the clips it downloaded, and the existing test fixtures that place
  clips beside (or under) the manifest.
* A judge or operator whose clips live elsewhere (for example an
  organizer clip directory with absolute paths in the manifest) passes
  ``allowed_roots=[...]`` or sets the ``ECHOGESTURE_INPUT_ROOT``
  environment variable (``os.pathsep``-separated, e.g. ``:`` on POSIX,
  ``;`` on Windows).  Any root listed there is an additional containment
  boundary; paths inside it are legitimate and accepted.

Containment is checked on *resolved* paths, so ``..``, absolute paths
outside the roots, and symlink escapes are all rejected with a
line-numbered :class:`SecureManifestError`.  ``~`` expansion happens
before resolution, so ``~/...`` outside the roots is also rejected.

Python 3.11 compatible.  Standard library only.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

REQUIRED_FIELDS = ("clip_id", "path")

ENV_INPUT_ROOT = "ECHOGESTURE_INPUT_ROOT"


class SecureManifestError(ValueError):
    """Manifest is malformed or a clip path escapes the allowed roots.

    Subclasses ValueError so existing ``except EvaluationError`` /
    ``except ValueError`` handlers in callers keep working during the
    migration; the patched ``run_manifest.py`` maps it to its own
    ``EvaluationError`` (see ``patches/run_manifest.patch``).
    """


def _resolve(path: Path) -> Path:
    # resolve() follows symlinks and normalises ".." without requiring
    # the target to exist.  strict=False is the default in 3.11.
    return path.resolve()


def _is_within(child: Path, parent: Path) -> bool:
    """True when resolved ``child`` is ``parent`` itself or under it.

    Uses ``Path.relative_to`` instead of ``is_relative_to`` so behaviour
    is identical across 3.11 patch releases.
    """
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def parse_allowed_roots(allowed_roots=None, *, env=None) -> list[Path]:
    """Normalise the ``allowed_roots`` argument plus the environment.

    ``allowed_roots`` may be ``None`` (no explicit roots), a single path,
    or an iterable of paths.  Entries from ``ECHOGESTURE_INPUT_ROOT``
    (when ``env`` is None, ``os.environ`` is used) are appended.  Each
    entry is resolved; duplicate roots collapse.
    """
    env = os.environ if env is None else env
    raw: list = []
    if allowed_roots is None:
        pass
    elif isinstance(allowed_roots, (str, os.PathLike)):
        raw.append(allowed_roots)
    else:
        raw.extend(allowed_roots)
    env_value = env.get(ENV_INPUT_ROOT, "") or ""
    if env_value.strip():
        for part in env_value.split(os.pathsep):
            part = part.strip()
            if part:
                raw.append(part)
    resolved: list[Path] = []
    seen: set[Path] = set()
    for entry in raw:
        candidate = _resolve(Path(entry).expanduser())
        if candidate not in seen:
            seen.add(candidate)
            resolved.append(candidate)
    return resolved


def effective_roots(manifest_path, allowed_roots=None, *, env=None) -> list[Path]:
    """Manifest directory plus any explicitly configured input roots."""
    manifest_dir = _resolve(Path(manifest_path).expanduser()).parent
    roots = parse_allowed_roots(allowed_roots, env=env)
    if manifest_dir not in roots:
        roots = [manifest_dir, *roots]
    return roots


def _contain(candidate: Path, roots: list[Path]) -> Path | None:
    """Return the matching root when ``candidate`` is inside it, else None."""
    for root in roots:
        if _is_within(candidate, root) and candidate != root:
            return root
    return None


def resolve_clip_path(relative_path: str, manifest_path, allowed_roots=None,
                      *, env=None) -> Path:
    """Resolve one ``path`` column entry and enforce containment.

    Returns the resolved absolute path.  Raises SecureManifestError when
    the entry is empty, contains a NUL byte, or resolves outside every
    allowed root.
    """
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise SecureManifestError("clip path is empty")
    if "\x00" in relative_path:
        raise SecureManifestError(f"clip path contains NUL byte: {relative_path!r}")
    manifest_dir = _resolve(Path(manifest_path).expanduser()).parent
    roots = effective_roots(manifest_path, allowed_roots, env=env)
    expanded = Path(relative_path.strip()).expanduser()
    if expanded.is_absolute():
        candidate = expanded
    else:
        candidate = manifest_dir / expanded
    resolved = _resolve(candidate)
    if _contain(resolved, roots) is None:
        joined = ", ".join(str(r) for r in roots)
        raise SecureManifestError(
            f"clip path escapes allowed input root: {relative_path!r} "
            f"(resolved={resolved}; allowed roots: {joined})"
        )
    return resolved


def read_manifest_secure(path, allowed_roots=None, *, env=None) -> list[tuple]:
    """Read a manifest CSV, rejecting any clip path outside allowed roots.

    Same return shape as ``run_manifest.read_manifest``:
    ``[(clip_id, clip_path, ground_truth, signer_id), ...]`` where
    ``clip_path`` is the resolved absolute path.  ``clip_id`` is
    preserved verbatim (including outer whitespace), matching the
    original.  Missing files inside the roots raise
    ``SecureManifestError`` with ``clip not found`` in the message, so
    existing ``assertRaisesRegex(..., "clip not found")`` checks keep
    working.
    """
    manifest_path = Path(path)
    try:
        with manifest_path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            missing = [f for f in REQUIRED_FIELDS if f not in (reader.fieldnames or [])]
            if missing:
                raise SecureManifestError(
                    f"Manifest missing columns: {', '.join(missing)}"
                )
            rows = list(reader)
    except (OSError, csv.Error) as exc:
        raise SecureManifestError(f"Cannot read manifest {path}: {exc}") from exc
    if not rows:
        raise SecureManifestError("Manifest contains no clips")

    seen: set[str] = set()
    items: list[tuple] = []
    for line, row in enumerate(rows, start=2):
        clip_id = row.get("clip_id") or ""
        relative_path = (row.get("path") or "").strip()
        if not clip_id.strip() or not relative_path:
            raise SecureManifestError(
                f"Manifest line {line}: clip_id and path are required"
            )
        if clip_id in seen:
            raise SecureManifestError(
                f"Manifest line {line}: duplicate clip_id {clip_id!r}"
            )
        seen.add(clip_id)
        try:
            clip_path = resolve_clip_path(
                relative_path, manifest_path, allowed_roots, env=env
            )
        except SecureManifestError as exc:
            raise SecureManifestError(f"Manifest line {line}: {exc}") from exc
        if not clip_path.is_file():
            raise SecureManifestError(
                f"Manifest line {line}: clip not found: {clip_path}"
            )
        signer = (row.get("signer_id") or "").strip()
        truth = (row.get("ground_truth") or "").strip().upper()
        items.append((clip_id, clip_path, truth, signer))
    return items


# Drop-in alias so patched call sites can use the familiar name.
read_manifest = read_manifest_secure
