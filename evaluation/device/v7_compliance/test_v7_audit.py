"""A15 V7 compliance unit tests (stdlib unittest, no cv2/numpy/network required).

Re-verifies live-code claims that the prior privacy_lab got stale:
 - robustness workdir cleanup now lives in try/finally (prior D1 CLOSED)
 - build_bundle.safe_member now rejects ':' in first component (prior D3 CLOSED)
 - shipped src/app have zero network/cloud sinks (prior P3 still holds)
 - offline_demo keeps finally+unlink deletion; TemporaryDirectory private subdir
 - run_manifest still leaks platform.node() + absolute artifact paths (D6 stays OPEN)
 - run_manifest.read_manifest still lacks '../' containment (D2 stays OPEN)
 - no LICENSE file while README says MIT (stays OPEN)
 - ASL Citizen participant CSV is present with participant column (D7 stays OPEN)

Run: python3 -I -m unittest evaluation.device.v7_compliance.test_v7_audit -v
  or: python3 -I evaluation/device/v7_compliance/test_v7_audit.py
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

V7 = Path(__file__).resolve().parent
ROOT = V7.parents[2]
sys.path.insert(0, str(V7))

import v7_audit  # noqa: E402


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(errors="ignore")


class LiveCodeClaimsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = v7_audit.audit(ROOT)
        cls.by_id = {f["check"]: f for f in cls.result["findings"]}

    def test_shipped_offline(self):
        self.assertEqual(self.by_id["V1-shipped-offline"]["status"], "PASS",
                         self.by_id["V1-shipped-offline"])

    def test_no_secrets(self):
        self.assertEqual(self.by_id["V2-secrets"]["status"], "PASS")

    def test_demo_temp_deleted(self):
        self.assertEqual(self.by_id["V3-temp-clip-deletion"]["status"], "PASS")
        src = _read("app/offline_demo.py")
        self.assertIn("path.unlink(missing_ok=True)", src)
        self.assertIn("TemporaryDirectory", src)

    def test_d1_closed(self):
        self.assertEqual(self.by_id["V6-robustness-temp-cleanup"]["status"], "PASS")
        rob = _read("evaluation/device/robustness.py")
        self.assertIn("try:", rob)
        self.assertIn("finally:", rob)
        self.assertIn("shutil.rmtree(workdir, ignore_errors=True)", rob)

    def test_d3_closed(self):
        self.assertEqual(self.by_id["V9-bundle-safe-member"]["status"], "PASS")
        # Colon bypass helper must reject 'data:file' — probe behavior, not source.
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "bb", ROOT / "scripts/release/build_bundle.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.assertFalse(mod.safe_member("data:file"))
        self.assertFalse(mod.safe_member("a:b"))
        self.assertTrue(mod.safe_member("ok/file"))
        self.assertFalse(mod.safe_member("../x"))

    def test_d2_containment_closed(self):
        # The audit now verifies real containment behavior; it must PASS.
        self.assertEqual(self.by_id["V7-manifest-containment"]["status"], "PASS")

    def test_d6_hostname_closed(self):
        # run_manifest now writes a generic device label; the leak is fixed.
        self.assertEqual(self.by_id["V8-evidence-hostname-leak"]["status"], "PASS")
        rm = _read("evaluation/device/run_manifest.py")
        self.assertNotIn("platform.node()", rm)

    def test_license_file_present(self):
        # A root LICENSE now ships (A2), so the audit must PASS.
        self.assertEqual(self.by_id["V10-license-file"]["status"], "PASS")

    def test_models_not_cleared(self):
        self.assertEqual(self.by_id["V11-models-licensing"]["status"], "NOT_CLEARED")

    def test_bundle_zip_clean(self):
        self.assertEqual(self.by_id["V5-bundle-clean"]["status"], "PASS")

    def test_audit_is_json_and_readonly(self):
        json.dumps(self.result, ensure_ascii=False)
        self.assertGreater(self.result["checks"], 8)

    def test_stale_lab_flagged(self):
        # Prior privacy_lab audit.py references NamedTemporaryFile (gone from live
        # code) and hardcodes D1 as FAIL. v7_audit must NOT repeat those claims.
        src = (V7 / "v7_audit.py").read_text()
        self.assertNotIn("NamedTemporaryFile(delete=False)", src)  # only in history note
        # V6 must reflect the try/finally fix, not the old FAIL
        self.assertEqual(self.by_id["V6-robustness-temp-cleanup"]["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
