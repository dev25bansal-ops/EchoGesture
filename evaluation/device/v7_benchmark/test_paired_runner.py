"""Tests for paired_runner.compute_paired / build_paired_summary (no heavy inference).

These tests use synthetic cell dicts in the exact shape run_variant_cell emits,
so they verify pairing logic, p50/p95 extraction, and the target-device gating
without executing any model.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmark_lab"))

import paired_runner  # noqa: E402


def _cell(variant, per_clip):
    wall = [r["wall_ms"] for r in per_clip]
    lat = paired_runner.matrix._timing_summary(wall)
    return {
        "cell": f"fresh_manifest_{variant}",
        "variant": variant,
        "config": {"repeats": 1, "warmup": 1, "profile": True, "repeat_mode": "fresh_manifest"},
        "clip_count": len(per_clip),
        "successful_predictions": len(per_clip),
        "failures": [],
        "failed_predictions": [],
        "accuracy": {"clips_with_ground_truth": len(per_clip), "correct": len(per_clip),
                     "top1": 1.0, "representative_accuracy_measured": True},
        "latency_all_calls": lat,
        "cold_call": paired_runner.matrix._timing_summary(wall[:1]),
        "warm_calls": paired_runner.matrix._timing_summary(wall[1:]),
        "stage_aggregates": {"landmark_tracking": paired_runner.matrix._timing_summary([w * 0.9 for w in wall])},
        "stage_extras": {},
        "profiles_count": len(per_clip),
        "all_wall_ms": wall,
        "per_clip": per_clip,
    }


def _clip(cid, wall, pred, top5):
    return {"clip_id": cid, "wall_ms": wall, "prediction": pred,
            "confidence": 0.9, "top5": top5, "ground_truth": None}


class ComputePairedTest(unittest.TestCase):
    def setUp(self):
        self.fp16 = _cell("fp16", [_clip("a", 100.0, "CANDY", ["CANDY", "RIGHT"]),
                                  _clip("b", 200.0, "RIGHT", ["RIGHT", "CANDY"])])
        self.int8 = _cell("int8", [_clip("a", 90.0, "CANDY", ["CANDY", "RIGHT"]),
                                  _clip("b", 180.0, "CANDY", ["CANDY", "RIGHT"])])

    def test_identical_inputs_required(self):
        self.int8["per_clip"][1]["clip_id"] = "zzz"
        with self.assertRaises(ValueError):
            paired_runner.compute_paired(self.fp16, self.int8)

    def test_paired_fields(self):
        paired = paired_runner.compute_paired(self.fp16, self.int8)
        self.assertEqual(paired["n"], 2)
        self.assertEqual(paired["prediction_agreement"], 0.5)
        self.assertAlmostEqual(paired["top5_overlap_mean"], 1.0)
        # delta = int8 - fp16 per clip: [-10, -20] sorted [-20, -10];
        # nearest-rank p50 = ceil(0.5*2)-1 = 0 -> -20, mean = -15.
        self.assertAlmostEqual(paired["wall_delta_ms"]["p50"], -20.0)
        self.assertAlmostEqual(paired["wall_delta_ms"]["mean"], -15.0)
        self.assertEqual(paired["per_variant_latency_all_calls"]["first"]["p50_ms"], 100.0)
        self.assertEqual(paired["per_variant_latency_all_calls"]["first"]["p95_ms"], 200.0)
        self.assertIn("landmark_tracking", paired["stage_aggregate_delta_ms"])

    def test_empty_cells_report_nulls(self):
        paired = paired_runner.compute_paired(_cell("fp16", []), _cell("int8", []))
        self.assertEqual(paired["n"], 0)
        self.assertIsNone(paired["prediction_agreement"])
        self.assertIsNone(paired["wall_delta_ms"])

    def test_target_device_gating(self):
        manifest = Path("/tmp/does/not/matter.csv")
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text("clip_id,path\n", encoding="utf-8")
        artifacts = {}
        # Non-target run: must be flagged as blocked, never as measured target evidence.
        summary = paired_runner.build_paired_summary(
            {"fp16": self.fp16, "int8": self.int8},
            manifest_path=manifest, artifacts=artifacts,
            hardware_note="Mac laptop — NOT target device",
            hardware_is_target=False, device_model=None,
        )
        self.assertFalse(summary["hardware_is_target_device"])
        self.assertEqual(summary["target_device_status"], "blocked_not_target_hardware")
        self.assertIn("NOT_MEASURED", summary["power"])
        self.assertIn("paired", summary)

        # Target run without device model must not be constructible by the CLI gate
        # (checked in _run_paired, not here); here we only verify the flag propagates.
        summary2 = paired_runner.build_paired_summary(
            {"fp16": self.fp16, "int8": self.int8},
            manifest_path=manifest, artifacts=artifacts,
            hardware_note="Target device: Example XR headset",
            hardware_is_target=True, device_model="Example XR headset",
        )
        self.assertTrue(summary2["hardware_is_target_device"])
        self.assertEqual(summary2["target_device_status"], "measured")


if __name__ == "__main__":
    unittest.main()
