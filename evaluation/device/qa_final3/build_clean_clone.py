"""Build a clean-clone simulation: git archive HEAD + working-tree overlay.

Read-only with respect to the repository; writes only into the target dir
passed as argv[1].
"""
import fnmatch
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

REPO = Path("$REPO")
DEST = Path(sys.argv[1])

EXCLUDE_GLOBS = [
    "models/*.tflite", "models/*.task", "*.zip",
    "build/*", "*/build/*", "assets/*", "*/assets/*",
    "__pycache__/*", "*/__pycache__/*", ".venv/*",
    "evaluation/device/qa_final3/*",
]

def excluded(rel):
    parts = rel.split("/")
    if "__pycache__" in parts:
        return True
    if rel.startswith("evaluation/device/qa_final3/"):
        return True
    for g in EXCLUDE_GLOBS:
        if fnmatch.fnmatch(rel, g):
            return True
        # match any path segment build/ or assets/
    if "build" in parts or "assets" in parts:
        return True
    return False

DEST.mkdir(parents=True, exist_ok=True)

# 1) git archive HEAD
with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tmp:
    tar_path = Path(tmp.name)
subprocess.run(
    ["git", "-C", str(REPO), "archive", "--format=tar", "-o", str(tar_path), "HEAD"],
    check=True)
with tarfile.open(tar_path) as tf:
    tf.extractall(DEST)
tar_path.unlink()

# 2) overlay working-tree tracked-eligible files
out = subprocess.run(
    ["git", "-C", str(REPO), "ls-files", "--cached", "--others", "--exclude-standard"],
    check=True, capture_output=True, text=True)
files = [line for line in out.stdout.splitlines() if line]
copied = 0
for rel in files:
    if excluded(rel):
        continue
    src = REPO / rel
    if not src.is_file():
        continue
    dst = DEST / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    copied += 1
print(f"overlay copied: {copied}")
print(f"total files in dest: {sum(1 for _ in DEST.rglob('*') if _.is_file())}")