"""Total artifact size accounting (classifier + detector + auxiliary assets).

`generate_results.py` historically reported `model_size` from classifier
bytes only. For the landmark candidate the shipped payload also includes the
MediaPipe hand landmarker (.task, ~7.8 MB), label/metadata JSON and the
feature/frontend code. This module inventories every artifact the benchmark
recorded (plus any extra files supplied explicitly) and reports both the
classifier-only and the total byte counts, so the 20 MB cap can be checked
against the real shipped payload. Missing files are reported, never zeroed.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class SizeBreakdown:
    def __init__(self, entries):
        # entries: list of (name, path_str, sha256|None, bytes|None)
        self.entries = entries

    @property
    def classifier_bytes(self):
        for name, _path, _sha, size in self.entries:
            if name == "model" or name.endswith("classifier"):
                return size
        return None

    @property
    def total_bytes(self):
        sizes = [size for _n, _p, _s, size in self.entries]
        if any(size is None for size in sizes):
            return None
        return sum(sizes)

    def as_dict(self):
        return {
            "classifier_bytes": self.classifier_bytes,
            "total_bytes": self.total_bytes,
            "total_mb": (self.total_bytes / 1e6) if self.total_bytes is not None else None,
            "items": [
                {"name": name, "path": path, "sha256": sha, "bytes": size}
                for name, path, sha, size in self.entries
            ],
            "complete": self.total_bytes is not None,
        }


def artifact_inventory(benchmark_artifacts=None, extra_paths=None, root=None):
    """Build a SizeBreakdown from benchmark artifacts + explicit extra files.

    benchmark_artifacts: mapping name -> {"path", "sha256", "bytes"} as
    recorded by run_manifest.py. Extra paths (e.g. hand_landmarker.task,
    label JSON, feature code) are hashed/sized from disk; a missing extra
    file yields bytes=None so the total is honestly incomplete.
    """
    entries = []
    seen_paths = set()
    if benchmark_artifacts:
        for name, entry in benchmark_artifacts.items():
            if not isinstance(entry, dict):
                continue
            path = entry.get("path", "")
            entries.append((name, path, entry.get("sha256"), entry.get("bytes")))
            if path:
                seen_paths.add(str(Path(path).resolve()))
    for item in extra_paths or []:
        name, path = (item if isinstance(item, tuple) else (Path(item).name, item))
        path = Path(path)
        if not path.is_absolute() and root is not None:
            path = Path(root) / path
        resolved = str(path.resolve()) if path.exists() else str(path)
        if resolved in seen_paths:
            continue
        if path.is_file():
            entries.append((name, resolved, sha256_of(path), path.stat().st_size))
        else:
            entries.append((name, resolved, None, None))
        seen_paths.add(resolved)
    return SizeBreakdown(entries)


def total_model_bytes(benchmark_artifacts=None, extra_paths=None, root=None):
    return artifact_inventory(benchmark_artifacts, extra_paths, root).total_bytes
