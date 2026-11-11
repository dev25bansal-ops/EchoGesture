"""Regression tests for evidence sanitisation + participant-ID exclusion.

Hostname/absolute-path leaks (D6) and tracked participant IDs (D7)
must not survive into public exports; legitimate evidence fields
(SHAs, counts, timings, code commit, dataset/split) must pass through
byte-identical so stale-evidence binding stays valid.
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # evaluation/device (contains v8_privacy + run_manifest)

from v8_privacy.sanitize_evidence import (  # noqa: E402
    GENERIC_DEVICE_MODEL,
    assert_sanitized,
    hash_signer_id,
    sanitize_benchmark_dict,
    sanitize_benchmark_file,
    sanitize_predictions_csv,
    sanitize_robustness_summary,
    sanitize_signer_value,
    strip_signer_fields,
)

A15_BENCH = {
    "status": "passed_schema",
    "clip_count": 30,
    "successful_predictions": 30,
    "artifacts": {
        "model": {
            "path": "/Users/dev25/projects/EchoGesture/models/s3d_conv1d_qat_int8.tflite",
            "sha256": "a" * 64,
            "bytes": 901200,
        },
        "predictor": {
            "path": "/private/tmp/echogesture-agents/A15/src/main.py",
            "sha256": "b" * 64,
            "bytes": 12000,
        },
    },
    "manifest": {
        "path": "/var/folders/d5/xxx/manifest.csv",
        "sha256": "c" * 64,
        "dataset": "asl-citizen",
        "split": "validation",
    },
    "code_commit": "f81dd30",
    "environment": {
        "python": "3.11.18",
        "platform": "macOS-14.0-arm64",
        "machine": "arm64",
        "device_model": "my-laptop-hostname",
        "inference_runtime": "ai-edge-litert 2.3.0",
    },
    "accuracy": {"top1": 0.78, "representative_accuracy_measured": True},
    "latency": {"all_calls": {"p95_ms": 42.0}},
    "recorded_at": "2026-11-08T00:00:00+00:00",
}


class BenchmarkSanitizerTest(unittest.TestCase):
    def test_hostname_replaced_with_generic_label(self):
        clean = sanitize_benchmark_dict(A15_BENCH)
        self.assertEqual(clean["environment"]["device_model"], GENERIC_DEVICE_MODEL)
        self.assertNotEqual(
            clean["environment"]["device_model"], A15_BENCH["environment"]["device_model"]
        )

    def test_absolute_paths_relativised(self):
        clean = sanitize_benchmark_dict(A15_BENCH)
        for entry in clean["artifacts"].values():
            self.assertNotIn("/Users/", entry["path"])
            self.assertNotIn("/private/", entry["path"])
            self.assertFalse(Path(entry["path"]).is_absolute())
        self.assertNotIn("/var/folders/", clean["manifest"]["path"])

    def test_repo_relative_when_under_repo(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            model = repo / "models" / "m.tflite"
            model.parent.mkdir()
            model.write_bytes(b"x")
            bench = {
                "artifacts": {"model": {"path": str(model), "sha256": "d" * 64, "bytes": 1}},
                "environment": {"device_model": "host"},
                "manifest": {"path": str(repo / "manifest.csv"), "sha256": "e" * 64},
            }
            clean = sanitize_benchmark_dict(bench, repo_root=repo)
            self.assertEqual(clean["artifacts"]["model"]["path"], "models/m.tflite")

    def test_evidence_fields_preserved_verbatim(self):
        clean = sanitize_benchmark_dict(A15_BENCH)
        self.assertEqual(clean["artifacts"]["model"]["sha256"], "a" * 64)
        self.assertEqual(clean["artifacts"]["model"]["bytes"], 901200)
        self.assertEqual(clean["manifest"]["sha256"], "c" * 64)
        self.assertEqual(clean["code_commit"], "f81dd30")
        self.assertEqual(clean["accuracy"], A15_BENCH["accuracy"])
        self.assertEqual(clean["latency"], A15_BENCH["latency"])
        self.assertEqual(clean["recorded_at"], "2026-11-08T00:00:00+00:00")
        self.assertEqual(clean["environment"]["python"], "3.11.18")

    def test_input_not_mutated(self):
        import copy

        before = copy.deepcopy(A15_BENCH)
        sanitize_benchmark_dict(A15_BENCH)
        self.assertEqual(A15_BENCH, before)

    def test_no_leaks_remain(self):
        clean = sanitize_benchmark_dict(A15_BENCH)
        self.assertEqual(assert_sanitized(clean), [])

    def test_assert_sanitized_catches_leaks(self):
        leaks = assert_sanitized(A15_BENCH)
        self.assertTrue(leaks)

    def test_windows_path_relativised(self):
        bench = {
            "artifacts": {
                "model": {
                    "path": "C:\\Users\\dev25\\models\\m.tflite",
                    "sha256": "f" * 64,
                    "bytes": 1,
                }
            },
            "environment": {"device_model": "DESKTOP-7F3K2"},
            "manifest": {"path": "C:\\Users\\dev25\\m.csv", "sha256": "1" * 64},
        }
        clean = sanitize_benchmark_dict(bench)
        self.assertEqual(clean["environment"]["device_model"], GENERIC_DEVICE_MODEL)
        self.assertNotIn("C:\\", clean["artifacts"]["model"]["path"])

    def test_sanitize_benchmark_file_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "benchmark.json"
            src.write_text(json.dumps(A15_BENCH), encoding="utf-8")
            clean = sanitize_benchmark_file(src)
            self.assertEqual(clean["environment"]["device_model"], GENERIC_DEVICE_MODEL)
            reloaded = json.loads(src.read_text(encoding="utf-8"))
            self.assertEqual(reloaded["environment"]["device_model"], GENERIC_DEVICE_MODEL)
            self.assertEqual(assert_sanitized(reloaded), [])

    def test_sanitize_benchmark_file_to_new_path(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "in.json"
            dst = Path(td) / "out" / "benchmark.json"
            src.write_text(json.dumps(A15_BENCH), encoding="utf-8")
            sanitize_benchmark_file(src, dst)
            self.assertTrue(dst.is_file())
            self.assertEqual(
                json.loads(dst.read_text(encoding="utf-8"))["environment"]["device_model"],
                GENERIC_DEVICE_MODEL,
            )


class ParticipantSanitizerTest(unittest.TestCase):
    def test_predictions_csv_drops_participant_column(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "pred.csv"
            dst = Path(td) / "public.csv"
            with src.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["video", "true_gloss", "pred_gloss", "participant", "correct"])
                writer.writerow(["a.mp4", "APPLE", "APPLE", "P15", "1"])
                writer.writerow(["b.mp4", "COOL", "TABLE", "P17", "0"])
            info = sanitize_predictions_csv(src, dst)
            self.assertEqual(info["dropped_columns"], ["participant"])
            with dst.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                self.assertNotIn("participant", reader.fieldnames)
                rows = list(reader)
                self.assertEqual(len(rows), 2)
                self.assertEqual(rows[0]["true_gloss"], "APPLE")
                text = dst.read_text(encoding="utf-8")
                self.assertNotIn("P15", text)
                self.assertNotIn("P17", text)

    def test_predictions_csv_with_salt_adds_pseudonym_not_raw(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "pred.csv"
            dst = Path(td) / "public.csv"
            with src.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["video", "true_gloss", "pred_gloss", "Participant ID", "correct"])
                writer.writerow(["a.mp4", "APPLE", "APPLE", "P15", "1"])
            info = sanitize_predictions_csv(src, dst, salt="local-salt-never-ship")
            self.assertIn("Participant ID", info["dropped_columns"])
            self.assertIn("signer_pseudonym", info["added_columns"])
            with dst.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
                self.assertIn("signer_pseudonym", rows[0])
                self.assertNotEqual(rows[0]["signer_pseudonym"], "P15")
                self.assertNotIn("P15", dst.read_text(encoding="utf-8"))

    def test_signer_list_sanitised_without_salt(self):
        summary = {"signer_ids": ["P15", "P17", "P19"], "signer_count": 3}
        clean = strip_signer_fields(summary)
        self.assertNotIn("P15", json.dumps(clean))
        self.assertEqual(clean["signer_count"], 3)

    def test_signer_list_with_salt_is_stable_and_distinct(self):
        summary = {"signer_ids": ["P15", "P17"]}
        clean = strip_signer_fields(summary, salt="s")
        self.assertEqual(len(clean["signer_ids"]), 2)
        self.assertNotEqual(clean["signer_ids"][0], clean["signer_ids"][1])
        again = strip_signer_fields(summary, salt="s")
        self.assertEqual(clean["signer_ids"], again["signer_ids"])

    def test_hash_signer_id_stable(self):
        self.assertEqual(hash_signer_id("P15", "s"), hash_signer_id("P15", "s"))
        self.assertNotEqual(hash_signer_id("P15", "s"), hash_signer_id("P17", "s"))
        self.assertNotIn("P15", hash_signer_id("P15", "s"))

    def test_sanitize_signer_value_never_returns_raw(self):
        self.assertEqual(sanitize_signer_value("P15", salt=None), "redacted")
        self.assertNotEqual(sanitize_signer_value("P15", salt="s"), "P15")
        self.assertEqual(sanitize_signer_value("unknown"), "unknown")

    def test_robustness_summary_drops_raw_ids(self):
        summary = {
            "signer_ids": ["P15"],
            "signer_count": 1,
            "per_signer": {"signer_e0f4874f": {"sample_count": 10, "clean_top1": 0.8}},
            "model_sha256": "a" * 64,
        }
        clean = sanitize_robustness_summary(summary)
        text = json.dumps(clean)
        self.assertNotIn("P15", text)
        self.assertEqual(clean["model_sha256"], "a" * 64)
        self.assertEqual(clean["signer_count"], 1)


if __name__ == "__main__":
    unittest.main()
