"""Import parent modules read-only for v8_privacy patches/tests.

The v8_privacy tree lives inside the isolated audit copy
(``.../evaluation/device/v8_privacy/``).  Parent modules
(``run_manifest``, ``robustness``, ``app.offline_demo``,
``scripts.release.build_bundle``, ``evidence_lab``) are imported
read-only; this package never writes into them.

Callers should insert the repo root on ``sys.path`` *before* importing
this package (tests do ``sys.path.insert(0, str(REPO_ROOT))``), or use
the ``ensure_parent_on_path()`` helper below.
"""

from __future__ import annotations

import sys
from pathlib import Path

# parents: v8_privacy -> device -> evaluation -> repo root
REPO_ROOT = Path(__file__).resolve().parents[3]
_EVALUATION_DEVICE = Path(__file__).resolve().parent.parent
_APP_ROOT = REPO_ROOT / "app"
_RELEASE_ROOT = REPO_ROOT / "scripts" / "release"


def ensure_parent_on_path() -> Path:
    """Prepend the directories parent modules live in; return repo root."""
    for entry in (str(REPO_ROOT), str(_EVALUATION_DEVICE), str(_APP_ROOT), str(_RELEASE_ROOT)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    return REPO_ROOT
