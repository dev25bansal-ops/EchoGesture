"""A15 suggested fixes (proposed, NOT applied to Abhijeet-owned files).

Apply with the owning agent and re-run privacy_lab to confirm closure.
Kept separate so this lab never silently rewrites eval/release code.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

# ---------------------------------------------------------------- D1
# evaluation/device/robustness.py — paired_evaluate():
# Wrap the per-clip loop so the workdir is always removed.
#
#   workdir = output_dir / "work"
#   workdir.mkdir(exist_ok=True)
#   try:
#       for index, ... in enumerate(items):
#           ...existing loop body (predict, decode, write corrupted, predict)...
#   finally:
#       shutil.rmtree(workdir, ignore_errors=True)
#
# decode_video() raising ValueError on an unreadable clip currently aborts
# the run and leaves workdir/ behind. The finally also covers
# KeyboardInterrupt. pairs.jsonl / robustness.json still get written by
# the caller only on success.

# --------------------------------------------------------------- D2
# evaluation/device/run_manifest.py — read_manifest():
# Contain relative clip paths inside the manifest's directory.
#
#   clip_path = Path(relative_path).expanduser()
#   if not clip_path.is_absolute():
#       clip_path = (path.parent / clip_path).resolve()
#       if clip_path != path.parent.resolve() and path.parent.resolve() not in clip_path.parents:
#           raise EvaluationError(f"Manifest line {line}: clip path escapes manifest directory: {relative_path}")
#
# Absolute paths: keep allowed only if they exist locally AND are
# documented as operator-supplied infrastructure paths; otherwise also
# constrain them to the manifest subtree. The organizer manifest may
# legitimately place clips beside the CSV, so subtree containment is the
# correct default. fetch_validation_batch.py writes manifest.csv next to
# the clips it downloaded, so this change does not break that flow.

# --------------------------------------------------------------- D3
# scripts/release/build_bundle.py — safe_member():
# Reject colon-bearing members on any platform.
#
#   def safe_member(name):
#       path = Path(name)
#       return bool(name) and not path.is_absolute() and ".." not in path.parts \
#           and not any(part.endswith(":") or ":" in part for part in path.parts)
#
# The current parts[0].endswith(":") only catches single-letter drives
# ("C:"); "data:file" passes (reproduced in D3). A colon-free ZIP is valid
# per the ZIP spec and avoids Windows NTFS alternate-data-stream
# interpretation of "file:stream" names.

# --------------------------------------------------------------- D4
# app/offline_demo.py — PredictionWorker._run():
# Create temp clips under a private subdir instead of the shared temp dir.
#
#   private_dir = Path(tempfile.mkdtemp(prefix="echogesture-"))  # 0700
#   ... NamedTemporaryFile(dir=private_dir) ...
#   finally:
#       path.unlink(missing_ok=True)
#       # rmdir private_dir when empty (ignore_errors)
#
# tempfile already makes each file 0600 + O_EXCL, so this is defence in
# depth (name non-listability) plus a single cleanup point for
# crash-orphaned clips if a future janitor sweep is added.

# --------------------------------------------------------------- D6
# evaluation/device/run_manifest.py — summary["environment"]:
#   "device_model": "generic-device" instead of platform.node(), and emit
#   artifact paths relative to the repo root (or basename only) so
#   benchmark.json is safe to paste into submission notes.

# --------------------------------------------------------------- D7
# evaluation/asl_citizen_predictions.csv:
# Strip the participant column (or hash it) before shipping, pending
# confirmation of ASL Citizen research-licence redistribution terms for
# per-participant result rows. Raw clips already stay outside Git via
# .gitignore in the original checkout — keep that exclusion and add the
# same to any new dataset dir. Do not add *.csv as a blanket ignore;
# results/results.csv must be tracked.
