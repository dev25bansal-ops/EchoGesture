"""Focused deterministic tests for scripts/release/compliance_final.py.

Every failure path is exercised against a synthetic directory so the
suite depends on no repo artifacts, no models, and no network.  The two
rewrite-only files this checker inspects (offline demo temp handling and
demo-lab screenshot policy) are stubbed by a ``--root`` override, so the
tests remain true unit tests that never read or mutate the real repo.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent          # evaluation/device/compliance_final/
_REPO = HERE.parent.parent.parent               # repo root
_RUNNER = _REPO / "scripts" / "release" / "compliance_final.py"

sys.path.insert(0, str(_RUNNER.parent))         # scripts/release (for build_bundle + compliance_final)

import compliance_final as cf  # noqa: E402  (imported from scripts/release)
import sanitize_export as se  # noqa: E402  (sibling sanitizer)


def _write(path: Path, text) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(text, bytes):
        path.write_bytes(text)
    else:
        path.write_text(text, encoding="utf-8")
    return path


def _mk_repo(*, fixed_demo_lab=True) -> Path:
    """Create a synthetic repo with shipped files good enough for C01/C02."""
    repo = Path(tempfile.mkdtemp(prefix="cf-repo-"))
    _write(repo / "src" / "main.py", "import json\n\ndef predict(video_path, *, clip_id=None):\n    return {'clip_id': clip_id}\n")
    _write(repo / "app" / "offline_demo.py",
           "import tempfile\n\ndef make():\n    tempfile.TemporaryDirectory()\n    t = tempfile.mkstemp()\n    try:\n        pass\n    finally:\n        p.unlink(missing_ok=True)\n")
    if fixed_demo_lab:
        # Post-fix shape: ephemeral default, no persistent shared dir.
        demo_lab = (
            "import tempfile\nimport cv2\n"
            "def ephemeral_outdir():\n"
            "    with tempfile.TemporaryDirectory(prefix='echogesture-demo-lab-') as tmp:\n"
            "        yield tmp\n"
            "def main(argv=None):\n"
            "    if args.outdir is not None:\n"
            "        outdir = args.outdir\n"
            "    cv2.imwrite(str(outdir / 'hud.png'), image)\n")
    else:
        # Pre-fix defect: persistent shared default + HUD write.
        demo_lab = ("import tempfile\nimport cv2\n"
                    "outdir = tempfile.gettempdir() / 'echogesture-demo-lab'\n"
                    "cv2.imwrite(str(outdir / 'hud.png'), image)\n")
    _write(repo / "app" / "demo_lab" / "__init__.py", demo_lab)
    _write(repo / "evaluation" / "device" / "run_manifest.py",
           "def read_manifest(manifest_path, allowed_roots=None):\n"
           "    x = Path(p).resolve().relative_to(Path(r).resolve())\n"
           "    if 'escapes' in str(x): raise ValueError('escapes allowed input root')\n"
           "_GENERIC_DEVICE_MODEL = 'generic-device'\n")
    _write(repo / "models" / "hand_landmarker.task", b"x" * 8)
    _write(repo / "models" / "landmark_classifier_fp16.tflite", b"y" * 8)
    _write(repo / "models" / "s3d_conv1d_qat_int8.tflite", b"z" * 8)
    return repo


def _write_publish(repo: Path, *, poison: str = "") -> Path:
    """Create a clean sanitised publish surface under the synthetic repo."""
    publish = repo / "evaluation" / "device" / "compliance_final" / "publish"
    _write(publish / "asl_citizen_predictions.csv",
           "video,true_gloss,pred_gloss,correct\n" + poison + "a.mp4,A,B,0\n")
    _write(publish / "asl_citizen_ptq_results.json",
           json.dumps({"top1": 0.1, "per_signer": {"signer_redacted_00": [1, 2]}}))
    return publish


def _write_license_notice(repo: Path, *, authorize: bool = False) -> None:
    _write(repo / "LICENSE",
           "MIT License\n\nPermission is hereby granted...\n"
           "This grant does not cover model artifacts or datasets.\n")
    rows = [
        "hand_landmarker.task | MediaPipe Apache-2.0",
        "landmark_classifier_fp16.tflite | ASL Citizen-derived",
        "s3d_conv1d_qat_int8.tflite | WLASL-derived",
    ]
    if authorize:
        rows = [r + " | " + cf.AUTHORIZATION_MARKER + " 2026-01-01"
                for r in rows]
        body = "Rights: written authorization recorded for every restricted artifact.\n"
    else:
        body = "Rights: NO confirmed redistribution permission; local/private only.\n"
    _write(repo / "NOTICE", "NOTICE\n" + body + "\n".join(rows) + "\n")


class OfflineAndSecretTests(unittest.TestCase):
    def setUp(self):
        self.repo = _mk_repo()

    def test_offline_pass_no_network(self):
        findings = {f["check"]: f for f in cf.run_checks(self.repo)["findings"]}
        self.assertEqual(findings["C01-offline-inference"]["status"], "PASS")

    def test_offline_detects_network_import(self):
        _write(self.repo / "src" / "main.py", "import requests\nprint('ok')\n")
        findings = {f["check"]: f for f in cf.run_checks(self.repo)["findings"]}
        self.assertEqual(findings["C01-offline-inference"]["status"], "FAIL")

    def test_secret_pass_excluding_secretary_gloss(self):
        _write(self.repo / "models" / "x.json", "gloss=secretary\nsecret key here?\n")
        findings = {f["check"]: f for f in cf.run_checks(self.repo)["findings"]}
        self.assertEqual(findings["C02-secrets"]["status"], "PASS")

    def test_secret_detects_api_key(self):
        _write(self.repo / "src" / "leak.py", "api_key = 'abc1234'\n")
        findings = {f["check"]: f for f in cf.run_checks(self.repo)["findings"]}
        self.assertEqual(findings["C02-secrets"]["status"], "FAIL")


class EvidenceAndManifestTests(unittest.TestCase):
    def setUp(self):
        self.repo = _mk_repo()

    def test_evidence_passes_clean_results(self):
        _write(self.repo / "results" / "results.csv",
               "metric,target,measured,unit,device,runtime,pass,notes\n"
               "top1_accuracy,>=70,50.0,%,generic-device,x,PASS,ok\n")
        findings = {f["check"]: f for f in cf.run_checks(self.repo)["findings"]}
        self.assertEqual(findings["C03-evidence-hygiene"]["status"], "PASS")

    def test_evidence_ignores_lowercase_percentiles(self):
        # p50/p95/p99 are latency percentiles, not participant IDs.
        _write(self.repo / "results" / "results.csv",
               "metric,notes\nlatency,p50=54.1 ms p95=117.8 p99=146.3\n")
        findings = {f["check"]: f for f in cf.run_checks(self.repo)["findings"]}
        self.assertEqual(findings["C03-evidence-hygiene"]["status"], "PASS")

    def test_evidence_flags_uppercase_participant(self):
        _write(self.repo / "results" / "results.csv",
               "metric,notes\ntop1,signer_P15 accuracy\n")
        findings = {f["check"]: f for f in cf.run_checks(self.repo)["findings"]}
        self.assertEqual(findings["C03-evidence-hygiene"]["status"], "FAIL")

    def test_evidence_flags_participant_and_abs_path(self):
        _write(self.repo / "results" / "results.csv",
               "metric,notes\nx,/Users/example/clip.mp4\n")
        _write(self.repo / "evaluation" / "asl_citizen_predictions.csv",
               "video,participant,correct\na,P15,1\n")
        findings = {f["check"]: f for f in cf.run_checks(self.repo)["findings"]}
        self.assertEqual(findings["C03-evidence-hygiene"]["status"], "FAIL")
        self.assertTrue(findings["C09-restricted-data"]["where"])

    def test_restricted_participant_csv_fails_without_publish(self):
        _write(self.repo / "evaluation" / "asl_citizen_predictions.csv",
               "video,true_gloss,pred_gloss,participant,correct\n"
               "a.mp4,A,B,P15,0\n")
        findings = {f["check"]: f for f in cf.run_checks(self.repo)["findings"]}
        self.assertEqual(findings["C09-restricted-data"]["status"], "FAIL")

    def test_publish_surface_clears_c03_and_c09_despite_raw_originals(self):
        # Raw IDs stay in the local research originals...
        _write(self.repo / "evaluation" / "asl_citizen_predictions.csv",
               "video,participant,correct\na,P15,1\n")
        _write(self.repo / "evaluation" / "asl_citizen_ptq_results.json",
               json.dumps({"per_signer": {"P15": [1, 2]}}))
        # ...but the sanitised publish surface is what ships.
        _write_publish(self.repo)
        findings = {f["check"]: f for f in cf.run_checks(self.repo)["findings"]}
        self.assertEqual(findings["C03-evidence-hygiene"]["status"], "PASS")
        self.assertEqual(findings["C09-restricted-data"]["status"], "PASS")
        self.assertTrue(findings["C09-restricted-data"]["raw_local_only"])

    def test_publish_surface_with_hostname_or_path_fails(self):
        _write_publish(self.repo, poison="evil,/Users/example/x.mp4\n")
        findings = {f["check"]: f for f in cf.run_checks(self.repo)["findings"]}
        self.assertEqual(findings["C03-evidence-hygiene"]["status"], "FAIL")
        self.assertEqual(findings["C09-restricted-data"]["status"], "FAIL")

    def test_publish_surface_with_participant_token_fails(self):
        _write_publish(self.repo, poison="evil,P15\n")
        findings = {f["check"]: f for f in cf.run_checks(self.repo)["findings"]}
        self.assertEqual(findings["C09-restricted-data"]["status"], "FAIL")

    def test_raw_mode_flags_tracked_originals(self):
        _write(self.repo / "evaluation" / "asl_citizen_predictions.csv",
               "video,participant,correct\na,P15,1\n")
        findings = {f["check"]: f for f in cf.run_checks(self.repo, raw=True)["findings"]}
        self.assertEqual(findings["C03-evidence-hygiene"]["status"], "FAIL")

    def test_manifest_containment_and_generic_device(self):
        findings = {f["check"]: f for f in cf.run_checks(self.repo)["findings"]}
        self.assertEqual(findings["C04-manifest-paths"]["status"], "PASS")


class ZipSafetyTests(unittest.TestCase):
    def test_clean_zip_passes(self):
        with tempfile.TemporaryDirectory() as td:
            zip_path = Path(td) / "clean.zip"
            with zipfile.ZipFile(zip_path, "w") as z:
                z.writestr("a.txt", "hello")
            self.assertEqual(cf.zip_safety_violations(zip_path), [])

    def test_traversal_member_flagged(self):
        with tempfile.TemporaryDirectory() as td:
            zip_path = Path(td) / "traversal.zip"
            with zipfile.ZipFile(zip_path, "w") as z:
                z.writestr("../escape.txt", "x")
            violations = cf.zip_safety_violations(zip_path)
            self.assertTrue(any("unsafe name" in v for v in violations))

    def test_absolute_member_flagged(self):
        with tempfile.TemporaryDirectory() as td:
            zip_path = Path(td) / "abs.zip"
            with zipfile.ZipFile(zip_path, "w") as z:
                z.writestr("/etc/passwd", "x")
            violations = cf.zip_safety_violations(zip_path)
            self.assertTrue(any("unsafe name" in v for v in violations))

    def test_duplicate_members_flagged(self):
        with tempfile.TemporaryDirectory() as td:
            zip_path = Path(td) / "dup.zip"
            with zipfile.ZipFile(zip_path, "w") as z:
                z.writestr("a.txt", "1")
                z.writestr("a.txt", "2")
            violations = cf.zip_safety_violations(zip_path)
            self.assertTrue(any("duplicate" in v for v in violations))

    def test_symlink_member_flagged(self):
        with tempfile.TemporaryDirectory() as td:
            zip_path = Path(td) / "link.zip"
            info = zipfile.ZipInfo("link.txt")
            info.create_system = 3  # unix
            info.external_attr = (0o120777 << 16)  # symlink file type
            with zipfile.ZipFile(zip_path, "w") as z:
                z.writestr(info, b"target")
            violations = cf.zip_safety_violations(zip_path)
            self.assertTrue(any("symlink" in v for v in violations))

    def test_oversized_member_flagged(self):
        with tempfile.TemporaryDirectory() as td:
            zip_path = Path(td) / "big.zip"
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as z:
                z.writestr("big.bin", b"x" * 1024)
            violations = cf.zip_safety_violations(zip_path, max_member_bytes=512)
            self.assertTrue(any("exceeds" in v for v in violations))


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.repo = _mk_repo()

    def test_temp_media_pass_and_fixed_screenshot_policy(self):
        findings = {f["check"]: f for f in cf.run_checks(self.repo)["findings"]}
        self.assertEqual(findings["C06-temp-media"]["status"], "PASS")
        self.assertEqual(findings["C07-screenshot-policy"]["status"], "PASS")

    def test_screenshot_policy_flags_persistent_shared_default(self):
        legacy = _mk_repo(fixed_demo_lab=False)
        findings = {f["check"]: f for f in cf.run_checks(legacy)["findings"]}
        self.assertEqual(findings["C07-screenshot-policy"]["status"], "FAIL")

    def test_screenshot_policy_not_measured_without_demo_lab(self):
        repo = _mk_repo()
        (repo / "app" / "demo_lab" / "__init__.py").unlink()
        findings = {f["check"]: f for f in cf.run_checks(repo)["findings"]}
        self.assertEqual(findings["C07-screenshot-policy"]["status"], "NOT_MEASURED")

    def test_license_missing_fails(self):
        findings = {f["check"]: f for f in cf.run_checks(self.repo)["findings"]}
        # synthetic repo has no LICENSE/NOTICE at root
        self.assertEqual(findings["C10-license-notice"]["status"], "FAIL")

    def test_license_and_notice_pass_but_models_not_cleared(self):
        _write_license_notice(self.repo)
        findings = {f["check"]: f for f in cf.run_checks(self.repo)["findings"]}
        self.assertEqual(findings["C10-license-notice"]["status"], "PASS")
        self.assertEqual(findings["C11-model-licensing"]["status"], "NOT_CLEARED")

    def test_model_artifact_missing_from_notice_fails(self):
        _write_license_notice(self.repo)
        _write(self.repo / "models" / "new_weights.tflite", b"q" * 8)
        findings = {f["check"]: f for f in cf.run_checks(self.repo)["findings"]}
        self.assertEqual(findings["C11-model-licensing"]["status"], "FAIL")

    def test_recorded_authorization_clears_models(self):
        _write_license_notice(self.repo, authorize=True)
        findings = {f["check"]: f for f in cf.run_checks(self.repo)["findings"]}
        self.assertEqual(findings["C11-model-licensing"]["status"], "PASS")

    def test_raw_footage_pass_when_no_git(self):
        # No git repo -> _git_ls_files returns [] -> NOT_MEASURED, not FAIL.
        findings = {f["check"]: f for f in cf.run_checks(self.repo)["findings"]}
        self.assertEqual(findings["C08-raw-footage"]["status"], "NOT_MEASURED")

    def test_summary_counts_consistent(self):
        result = cf.run_checks(self.repo)
        counts = {f["status"] for f in result["findings"]}
        self.assertEqual(result["pass"] + result["fail"] + result["not_cleared"]
                         + result["not_measured"], result["checks"])
        self.assertEqual(result["checks"], len(result["findings"]))


class SafeMemberBehaviourTests(unittest.TestCase):
    """The checker reuses build_bundle.safe_member; lock its contract via
    cf.zip_safety_violations (no direct import of build_bundle needed)."""

    def test_colon_rejected_in_later_component(self):
        self.assertFalse(cf.build_bundle.safe_member("a/b:c"))

    def test_dot_and_dotdot_rejected(self):
        self.assertFalse(cf.build_bundle.safe_member("a/./b"))
        self.assertFalse(cf.build_bundle.safe_member("a/../b"))

    def test_plain_relative_ok(self):
        self.assertTrue(cf.build_bundle.safe_member("models/hand_landmarker.task"))


class StrictGateTests(unittest.TestCase):
    """``--strict`` exits nonzero while mandatory gates are unresolved."""

    def _clean_repo(self) -> Path:
        repo = _mk_repo()
        _write(repo / "results" / "results.csv",
               "metric,target,measured,unit,device,runtime,pass,notes\n"
               "top1_accuracy,>=70,50.0,%,generic-device,x,PASS,ok\n")
        _write_publish(repo)
        _write_license_notice(repo)
        return repo

    def _run_main(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cf.main(list(args))
        return code, out.getvalue(), err.getvalue()

    def test_strict_returns_nonzero_on_unresolved_model_rights(self):
        repo = self._clean_repo()  # LICENSE+NOTICE present but models NOT_CLEARED
        code, _, err = self._run_main("--root", str(repo), "--strict")
        self.assertEqual(code, 1)
        self.assertIn("NOT legally cleared for distribution", err)
        self.assertIn("STRICT GATE FAILED", err)
        result = cf.gate_report(cf.run_checks(repo))
        self.assertFalse(result["gate"]["strict_ok"])
        self.assertFalse(result["gate"]["legal_clearance"])
        self.assertIn("C11-model-licensing", result["gate"]["unresolved_rights"])

    def test_strict_returns_zero_when_every_gate_cleared(self):
        repo = self._clean_repo()
        _write_license_notice(repo, authorize=True)
        code, _, err = self._run_main("--root", str(repo), "--strict")
        self.assertEqual(code, 0)
        self.assertNotIn("NOT legally cleared", err)
        self.assertIn("STRICT GATE PASSED", err)
        result = cf.gate_report(cf.run_checks(repo))
        self.assertTrue(result["gate"]["strict_ok"])
        self.assertTrue(result["gate"]["legal_clearance"])

    def test_strict_returns_nonzero_on_failed_gate(self):
        repo = self._clean_repo()
        _write_license_notice(repo, authorize=True)
        _write(repo / "src" / "leak.py", "api_key = 'abc1234'\n")
        code, _, err = self._run_main("--root", str(repo), "--strict")
        self.assertEqual(code, 1)
        self.assertIn("C02-secrets=FAIL", err)

    def test_default_mode_reports_only_and_exits_zero(self):
        repo = self._clean_repo()
        code, out, err = self._run_main("--root", str(repo))
        self.assertEqual(code, 0)
        self.assertIn("NOT legally cleared for distribution", err)
        self.assertEqual(json.loads(out)["gate"]["strict_ok"], False)

    def test_gate_blocks_not_measured_mandatory_check(self):
        repo = self._clean_repo()
        _write_license_notice(repo, authorize=True)
        (repo / "app" / "demo_lab" / "__init__.py").unlink()
        result = cf.gate_report(cf.run_checks(repo))
        self.assertIn({"check": "C07-screenshot-policy", "status": "NOT_MEASURED"},
                      result["gate"]["blocking"])


class SanitizeExportTests(unittest.TestCase):
    """Fix 1 + Fix 2: participant-ID and env hygiene in exported evidence."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="cf-sanitize-"))
        self.src_csv = self.tmp / "preds.csv"
        self.src_csv.write_text(
            "video,true_gloss,pred_gloss,participant,correct\n"
            "a.mp4,APPLE,TIME,P15,0\n"
            "b.mp4,JACKET,GO,P17,1\n", encoding="utf-8")
        self.src_json = self.tmp / "ptq.json"
        self.src_json.write_text(json.dumps(
            {"top1": 0.01, "n": 2, "ok": 0,
             "per_signer": {"signer_e0f4874f": [3, 161], "signer_5f5b9d6a": [2, 145]}}), encoding="utf-8")

    def test_csv_default_drops_participant_column(self):
        dst = self.tmp / "out" / "preds.csv"
        report = se.sanitize_predictions_csv(self.src_csv, dst)
        self.assertEqual(report["dropped_columns"], ["participant"])
        text = dst.read_text(encoding="utf-8")
        self.assertNotIn("P15", text)
        self.assertIn("video,true_gloss,pred_gloss,correct", text)

    def test_csv_keyed_pseudonym_is_deterministic(self):
        dst1 = self.tmp / "out1.csv"
        dst2 = self.tmp / "out2.csv"
        se.sanitize_predictions_csv(self.src_csv, dst1, key="k")
        se.sanitize_predictions_csv(self.src_csv, dst2, key="k")
        self.assertEqual(dst1.read_bytes(), dst2.read_bytes())
        self.assertIn("signer_", dst1.read_text(encoding="utf-8"))
        self.assertNotIn("P15", dst1.read_text(encoding="utf-8"))

    def test_json_per_signer_keys_redacted(self):
        dst = self.tmp / "out" / "ptq.json"
        se.sanitize_json(self.src_json, dst)
        text = dst.read_text(encoding="utf-8")
        self.assertNotIn("P15", text)
        self.assertIn("signer_redacted_", text)

    def test_originals_are_never_modified(self):
        before = self.src_csv.read_bytes(), self.src_json.read_bytes()
        se.sanitize_all(self.tmp, self.tmp / "publish",
                        sources=["preds.csv", "ptq.json"])
        after = self.src_csv.read_bytes(), self.src_json.read_bytes()
        self.assertEqual(before, after)

    def test_refuses_to_overwrite_source(self):
        with self.assertRaises(ValueError):
            se.sanitize_predictions_csv(self.src_csv, self.src_csv)

    def test_benchmark_hostname_and_path_cleaned(self):
        doc = {"environment": {"device_model": "example-host.local"},
               "artifacts": {"m": {"path": "/Users/example/x/model.tflite"}}}
        clean = se.sanitize_document(doc)
        self.assertEqual(clean["environment"]["device_model"], "generic-device")
        self.assertEqual(clean["artifacts"]["m"]["path"], "model.tflite")
        self.assertEqual(se.leaks_in_document(clean), [])

    def test_hostname_leak_is_detected(self):
        leaks = se.leaks_in_document({"environment": {"hostname": "example-box.local"}})
        self.assertTrue(leaks)

    def test_absolute_path_leak_is_detected(self):
        leaks = se.leaks_in_document({"notes": "/Users/example/clip.mp4"})
        self.assertTrue(any("absolute" in reason for reason in leaks))

    def test_participant_token_leak_is_detected(self):
        leaks = se.leaks_in_document({"metric": "top1_signer_P15"})
        self.assertTrue(any("participant" in reason for reason in leaks))

    def test_lowercase_percentile_is_not_a_leak(self):
        self.assertEqual(se.leaks_in_document({"latency": "p50,p95,p99"}), [])

    def test_publish_scan_clean_and_poisoned(self):
        publish = self.tmp / "publish"
        se.sanitize_all(self.tmp, publish, sources=["preds.csv", "ptq.json"])
        self.assertTrue(se.scan_publish_dir(publish)["ok"])
        poison = publish / "leak.json"
        poison.write_text('{"per_signer": {"P15": [1, 2]}}', encoding="utf-8")
        scan = se.scan_publish_dir(publish)
        self.assertFalse(scan["ok"])
        self.assertTrue(scan["leaks"])


class RealRepoEvidenceTests(unittest.TestCase):
    """The committed publish surface must be leak-free (read-only check)."""

    def test_committed_publish_surface_is_clean(self):
        publish = _REPO / cf.PUBLISH_DIR
        if not publish.is_dir():
            self.skipTest("publish surface not generated")
        scan = se.scan_publish_dir(publish)
        self.assertTrue(scan["ok"], scan["leaks"])
        self.assertEqual(scan["leaks"], [])


if __name__ == "__main__":
    unittest.main()