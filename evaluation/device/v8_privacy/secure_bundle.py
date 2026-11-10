"""Hardened bundle-member check (D3 residual fix).

The current ``build_bundle.safe_member`` rejects a colon in the *first*
path component (``C:``-style drives and ``data:file``), but accepts a
colon in any *later* component:

    safe_member("foo/bar:baz")  == True
    safe_member("results/predictions.jsonl:extra") == True

On Windows, ``file:stream`` is an NTFS alternate-data-stream name; a
ZIP member spelling like ``dir/file.mp4:stream`` is therefore a
traversal-adjacent hazard the A15 lab flagged (repro D3).  This module
rejects a colon in *any* component, matching the proposed fix in
``evaluation/device/privacy_lab/fixes.py`` D3:

    any(part.endswith(":") or ":" in part for part in path.parts)

plus the additional Windows-sensitive guards required for a shipped
bundle verifier: backslashes, drive letters, UNC prefixes, absolute
names, ``..`` / ``.`` / empty components, NUL bytes, and NUL-ish
traversal via ``\x00``.  Reuses the original's
``_DRIVE_RE`` / ``safe_relative`` contract shape so the patched
``build_bundle.py`` diff is small (see ``patches/build_bundle.patch``).

Standard library only; Python 3.11 compatible.
"""

from __future__ import annotations

import re
from pathlib import Path

_DRIVE_RE = re.compile(r"^[A-Za-z]:")


def safe_member(name) -> bool:
    """True only for a plain relative POSIX member with no traversal/tricks.

    Reject-on-any-colon: a colon in *any* part (``data:file``,
    ``foo/bar:baz``, ``dir/file.mp4:stream``) is unsafe.  Colon-free
    ZIP members remain valid per the ZIP spec.
    """
    if not isinstance(name, str) or not name or "\x00" in name:
        return False
    if "\\" in name or _DRIVE_RE.match(name):
        return False
    # UNC-style "//server/share" never reaches Path safely.
    if name.startswith("//"):
        return False
    stripped = name.rstrip("/")
    if not stripped:
        return False
    if name.startswith("/"):
        return False
    # Split on '/' manually: Path.parts normalises '.' and '//' away,
    # hiding exactly the components this check must see (the original
    # safe_member had the same hole: safe_member('a/./b') == True).
    parts = tuple(stripped.split("/"))
    if not parts or ".." in parts or any(part in ("", ".") for part in parts):
        return False
    # Drive/ADS check on every component (D3 fix).
    if any(part.endswith(":") or ":" in part for part in parts):
        return False
    return True


def safe_relative(root, relative):
    """Resolve ``relative`` under ``root`` and require containment (D3-fixed).

    Drops recognition: ``relative`` itself must satisfy the hardened
    :func:`safe_member` (so a colon-bearing relative is rejected before
    any filesystem access), then the resolved target must stay inside
    the resolved root and not equal the root.
    """
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
