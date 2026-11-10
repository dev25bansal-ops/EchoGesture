"""Tests for hardened generate_results.py (A12 evidence work).

These cover: stale-evidence rejection, no-PASS-from-smoke, total size
accounting, confidence semantics, per-signer derivation, and provenance
tracing. They fabricate no model results — all fixtures are synthetic JSON
shaped like the real run_manifest output.
"""

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/release"))
sys.path.insert(0, str(ROOT / "evaluation/device"))
from generate_results import generate

SHA = "a" * 64
OTHER_SHA = "b" * 64


def benchmark_dict(*, sha=SHA, top1=None, p95=10.0, clips=1, representative=False,
                   accuracy_representative=None, with_artifacts=True, per_signer=None):
    bench = {
        "artifacts": {"model": {"sha256": sha, "bytes": 1000, "path": "/tmp/model.tflite"}}
        if with_artifacts else {},
        "environment": {"device_model": "test-device", "inference_runtime": "test-runtime"},
        "code_commit": "commit123",
        "manifest": {"sha256": "manifest456", "path": "/tmp/manifest.csv"},
        "dataset": "test-dataset",
        "split": "validation",
        "fixture_type": "real",
        "recorded_at": "2026-11-08T00:00:00+00:00",
        "clip_count": clips,
        "successful_predictions": clips,
        "accuracy": {"top1": top1,
                     "representative_accuracy_measured": (
                         representative if accuracy_representative is None
                         else accuracy_representative)},
        "latency": {"all_calls": {"p95_ms": p95}},
        "representative_performance_measured": representative,
    }
    if per_signer is not None:
        bench["per_signer"] = per_signer
    return bench


def write_json(path, data):
    Path(path).write_text(json.dumps(data), encoding="utf-8")
    return Path(path)


class GenerateResultsHardeningTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.out = self.root / "results.csv"

    def tearDown(self):
        self.temp.cleanup()

    def generate_rows(self, bench, **kwargs):
        bench_path = write_json(self.root / "benchmark.json", bench)
        return generate(bench_path, [], self.out, SHA, **kwargs)

    def by_metric(self, rows):
        return {r["metric"]: r for r in rows}

    def test_smoke_run_cannot_pass_accuracy_or_latency(self):
        rows = self.generate_rows(benchmark_dict(top1=1.0, p95=1.0, clips=3))
        m = self.by_metric(rows)
        self.assertEqual(m["top1_accuracy"]["pass"], "NOT_MEASURED")
        self.assertEqual(m["latency_p95"]["pass"], "NOT_MEASURED")
        self.assertEqual(m["top1_accuracy"]["measured"], 100.0)

    def test_representative_run_can_pass_when_thresholds_met(self):
        rows = self.generate_rows(
            benchmark_dict(top1=0.80, p95=40.0, clips=30, representative=True))
        m = self.by_metric(rows)
        self.assertEqual(m["top1_accuracy"]["pass"], "PASS")
        self.assertEqual(m["latency_p95"]["pass"], "PASS")

    def test_representative_run_fails_when_below_threshold(self):
        rows = self.generate_rows(
            benchmark_dict(top1=0.50, p95=90.0, clips=30, representative=True))
        m = self.by_metric(rows)
        self.assertEqual(m["top1_accuracy"]["pass"], "FAIL")
        self.assertEqual(m["latency_p95"]["pass"], "FAIL")

    def test_stale_model_sha_rejected(self):
        bench = benchmark_dict(sha=OTHER_SHA)
        with self.assertRaisesRegex(ValueError, "Stale benchmark"):
            self.generate_rows(bench)

    def test_benchmark_without_model_sha_rejected(self):
        bench = benchmark_dict(with_artifacts=False)
        with self.assertRaisesRegex(ValueError, "no model SHA|Stale"):
            self.generate_rows(bench)

    def test_stale_robustness_rejected(self):
        bench_path = write_json(self.root / "benchmark.json", benchmark_dict())
        rob = write_json(self.root / "robustness.json", {
            "model_sha256": OTHER_SHA,
            "accuracy_drop_percentage_points": 1.0,
            "representative_robustness_measured": True,
            "sample_count": 30,
            "corruption_type": "low_light",
        })
        with self.assertRaisesRegex(ValueError, "Stale robustness"):
            generate(bench_path, [rob], self.out, SHA)

    def test_total_size_includes_detector_asset(self):
        detector = self.root / "hand_landmarker.task"
        detector.write_bytes(b"x" * 5000)
        rows = self.generate_rows(
            benchmark_dict(),
            extra_artifacts=[("detector", str(detector))])
        m = self.by_metric(rows)
        self.assertEqual(m["model_size"]["measured"], 0.006)  # 1000 + 5000
        self.assertEqual(m["model_size"]["pass"], "PASS")
        self.assertIn("detector=5000B", m["model_size"]["notes"])
        self.assertIn("classifier-only", m["model_size"]["notes"])

    def test_missing_artifact_makes_size_not_measured(self):
        rows = self.generate_rows(
            benchmark_dict(),
            extra_artifacts=[("detector", str(self.root / "missing.task"))])
        m = self.by_metric(rows)
        self.assertEqual(m["model_size"]["pass"], "NOT_MEASURED")
        self.assertIn("incomplete", m["model_size"]["notes"])

    def test_oversized_total_fails(self):
        big = self.root / "detector.task"
        big.write_bytes(b"x" * 25_000_000)
        rows = self.generate_rows(
            benchmark_dict(), extra_artifacts=[("detector", str(big))])
        m = self.by_metric(rows)
        self.assertEqual(m["model_size"]["pass"], "FAIL")

    def test_confidence_semantics_row_is_explicit(self):
        rows = self.generate_rows(benchmark_dict())
        m = self.by_metric(rows)
        self.assertEqual(m["confidence_semantics"]["pass"], "NOT_MEASURED")
        self.assertIn("logits", m["confidence_semantics"]["notes"])
        self.assertIn("softmax", m["confidence_semantics"]["notes"])
        self.assertIn("calibrated=False", m["confidence_semantics"]["notes"])

    def test_per_signer_rows_from_benchmark_per_signer(self):
        rows = self.generate_rows(benchmark_dict(
            top1=0.5, representative=True, clips=30,
            per_signer={"signer_e0f4874f": [1, 4], "signer_5f5b9d6a": [3, 4]}))
        m = self.by_metric(rows)
        self.assertIn("top1_accuracy_signer_signer_e0f4874f", m)
        self.assertEqual(m["top1_accuracy_signer_signer_e0f4874f"]["measured"], 25.0)
        # Per-signer rows never claim PASS.
        self.assertEqual(m["top1_accuracy_signer_signer_e0f4874f"]["pass"], "NOT_MEASURED")

    def test_per_signer_not_supplied_is_honest(self):
        rows = self.generate_rows(benchmark_dict())
        m = self.by_metric(rows)
        self.assertEqual(m["top1_accuracy_per_signer"]["pass"], "NOT_MEASURED")
        self.assertIn("not derivable", m["top1_accuracy_per_signer"]["notes"])

    def test_accuracy_row_traces_provenance(self):
        rows = self.generate_rows(benchmark_dict(top1=None))
        m = self.by_metric(rows)
        for token in ("model_sha256=" + SHA, "code_commit=commit123",
                      "manifest_sha256=manifest456", "device=generic-device",
                      "test-runtime", "2026-11-08T00:00:00", "validation"):
            self.assertIn(token, m["top1_accuracy"]["notes"], token)

    def test_csv_columns_unchanged(self):
        self.generate_rows(benchmark_dict())
        with self.out.open(newline="", encoding="utf-8") as stream:
            header = next(csv.reader(stream))
        self.assertEqual(header, ["metric", "target", "measured", "unit",
                                  "device", "runtime", "pass", "notes"])


class EvidenceLabUnitsTest(unittest.TestCase):
    def test_softmax_matches_main_py_postprocessing(self):
        sys.path.insert(0, str(ROOT / "evaluation/device"))
        from evidence_lab import softmax_confidence
        top, prob, top5 = softmax_confidence([1.0, 3.0, 2.0])
        self.assertEqual(top, 1)
        self.assertAlmostEqual(prob, 0.6652409558, places=6)
        self.assertEqual(top5, [1, 2, 0])

    def test_output_semantics_logits_from_observed_sample(self):
        sys.path.insert(0, str(ROOT / "evaluation/device"))
        from evidence_lab import describe_output_semantics
        # Landmark outputs: outside [0,1], do not sum to 1.
        desc = describe_output_semantics(observed=[-3.0, 4.8, 10.2])
        self.assertEqual(desc["output_semantics"], "logits")
        self.assertFalse(desc["calibrated"])
        probs = describe_output_semantics(observed=[0.2, 0.8])
        self.assertEqual(probs["output_semantics"], "probabilities")

    def test_provenance_missing_fields_listed(self):
        sys.path.insert(0, str(ROOT / "evaluation/device"))
        from evidence_lab import extract_provenance
        prov = extract_provenance({"artifacts": {"model": {"sha256": SHA, "bytes": 1}}},
                                  "sparse.json")
        missing = prov.missing_provenance()
        self.assertIn("code_commit", missing)
        self.assertIn("dataset", missing)


if __name__ == "__main__":
    unittest.main()
