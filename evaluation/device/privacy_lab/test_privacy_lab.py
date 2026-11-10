"""A15 privacy_lab unit tests (stdlib unittest, no network, no real clips)."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

LAB = Path(__file__).resolve().parent
ROOT = LAB.parents[2]
sys.path.insert(0, str(LAB))
sys.path.insert(0, str(ROOT / "app"))

import audit  # noqa: E402
import reproduce_defects  # noqa: E402


class AuditTest(unittest.TestCase):
    def test_audit_finds_abs_paths_but_no_secrets(self):
        result = audit.audit(ROOT)
        by_id = {f["check"]: f for f in result["findings"]}
        self.assertIn("P1-absolute-paths", by_id)
        self.assertEqual(by_id["P2-secrets"]["status"], "PASS",
                         "secretary label must not count as a secret")
        self.assertIn("P3-network-sinks", by_id)
        self.assertIn("P8-models", by_id)
        self.assertEqual(by_id["P8-models"]["status"], "NOT_CLEARED")
        self.assertGreater(result["files_scanned"], 5)

    def test_audit_is_readonly_and_json_serialisable(self):
        result = audit.audit(ROOT)
        json.dumps(result, ensure_ascii=False)  # must not raise
        self.assertEqual(list((ROOT / "evaluation/device/privacy_lab").rglob("*.mp4")), [])


class ReproTest(unittest.TestCase):
    """Regression guards that the previously-reproduced defects are now CLOSED.

    The lab originally *reproduced* each defect. The hardening has since
    shipped, so these assert the fixed behaviour: the defect must NOT
    reproduce. If one starts reproducing again, that is a real regression.
    """

    def test_d2_manifest_traversal_is_blocked(self):
        r = reproduce_defects.repro_manifest_traversal()
        self.assertFalse(r["reproduced"], r["evidence"])

    def test_d3_colon_bypass_is_blocked(self):
        r = reproduce_defects.repro_bundle_colon()
        self.assertFalse(r["reproduced"], r["evidence"])

    def test_d5_temp_is_deleted_on_exception(self):
        # This one is a control: the good behaviour ("reproduced") must hold.
        r = reproduce_defects.repro_temp_deleted_on_exception()
        self.assertTrue(r["reproduced"], r["evidence"])

    def test_d6_hostname_is_not_written(self):
        r = reproduce_defects.repro_hostname_in_benchmark()
        self.assertFalse(r["reproduced"], r["evidence"])


if __name__ == "__main__":
    unittest.main()
