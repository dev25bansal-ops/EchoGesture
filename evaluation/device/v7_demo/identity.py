"""Model-identity checks for the V7 demo harness."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.offline_demo import model_identity  # noqa: E402

EXPECTED_FP16_SHA256 = "a012f2b5f7a1110961079f19eb0e28c9abacb95e4d0787352e4157ebca7044a4"
EXPECTED_LABELS_SHA256 = "e78f5028e97e2eaf86aff65af403af65061eebef010f0fbd41012293e8019a4a"


def _sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def model_identity_report(root=ROOT) -> dict:
    identity = model_identity(root)
    model_path = Path(root) / "models/landmark_classifier_fp16.tflite"
    labels_path = Path(root) / "models/landmark_labels.json"
    report = dict(identity)
    report["expected_fp16_sha256"] = EXPECTED_FP16_SHA256
    report["model_full_sha256"] = _sha256(model_path)
    report["labels_full_sha256"] = _sha256(labels_path)
    report["fp16_matches_expected"] = report["model_full_sha256"] == EXPECTED_FP16_SHA256
    report["labels_matches_expected"] = report["labels_full_sha256"] == EXPECTED_LABELS_SHA256
    return report


def check_model_identity(root=ROOT) -> dict:
    report = model_identity_report(root)
    ok = (
        report["backend"] == "landmark"
        and report["fp16_matches_expected"]
        and report["labels_matches_expected"]
        and "provisional" in report["threshold_status"]
    )
    return {"ok": ok, "report": report}
