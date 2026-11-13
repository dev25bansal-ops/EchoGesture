"""The official (organizer-manifest) release path: what it accepts and refuses.

Every test here uses synthetic fixtures built in a temp directory. A synthetic
manifest is a manifest-shaped CSV used to exercise validation logic; it is never
labelled an organizer manifest and nothing here contacts the organizers.

The property under test is that an official bundle can only be produced from
inputs that actually line up: a manifest whose clip IDs match the prediction
file exactly, predictions drawn from the shipped vocabulary, and a prediction
object in the p.11 schema. Any mismatch must fail loudly rather than ship a
package whose IDs the evaluator will reject.
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "scripts" / "release"))

import build_bundle as bb  # noqa: E402

MODEL = "models/landmark_ens_rob71_fp16.tflite"
LABELS = "models/landmark_labels.json"
CONVERSION = "scripts/ml/export_tflite_tf.py"


def _weights_present() -> bool:
    meta = REPO / "models" / "landmark_model_meta.json"
    if not meta.is_file():
        return False
    try:
        members = json.loads(meta.read_text(encoding="utf-8")).get("members")
    except (OSError, ValueError):
        return False
    needed = ["hand_landmarker.task"]
    if isinstance(members, list) and members:
        needed += list(members)
    return all((REPO / "models" / n).is_file() for n in needed)


class OfficialBuildTest(unittest.TestCase):
    """A complete, self-consistent synthetic input set."""

    def setUp(self):
        if not _weights_present():
            self.skipTest("model weights are not present (not redistributable)")
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.source = self.root / "source"
        # A minimal but real source tree: the builder stages src/ and needs the
        # conversion script as an --extra under an allowed root.
        (self.source / "src").mkdir(parents=True)
        (self.source / "results").mkdir()
        (self.source / "models").mkdir()
        (self.source / "scripts" / "ml").mkdir(parents=True)
        for name in ("main.py", "requirements.txt", "README.md"):
            (self.source / "src" / name).write_text("x", encoding="utf-8")
        # The builder stages the landmark runtime closure, which imports these.
        for name in ("features.py", "landmark_frontend.py", "parallel_tracker.py"):
            shutil.copy2(REPO / "src" / name, self.source / "src" / name)
        for name in ("landmark_ens_tcn21_fp16.tflite",
                     "landmark_ens_tcnm61_fp16.tflite",
                     "landmark_ens_tcnm62_fp16.tflite",
                     "landmark_ens_rob71_fp16.tflite",
                     "hand_landmarker.task",
                     "landmark_labels.json",
                     "landmark_model_meta.json"):
            shutil.copy2(REPO / "models" / name, self.source / "models" / name)
        shutil.copy2(REPO / CONVERSION, self.source / CONVERSION)
        shutil.copy2(REPO / "Dockerfile", self.source / "Dockerfile")
        (self.source / "results" / "results.csv").write_text(
            "metric,target,measured,unit,device,runtime,pass,notes\n"
            "top1_accuracy,>=70,72.78,%,dev,rt,PASS,note\n", encoding="utf-8")

        self.labels = sorted(json.loads(
            (REPO / LABELS).read_text(encoding="utf-8")), key=lambda k: json.loads(
                (REPO / LABELS).read_text(encoding="utf-8"))[k])
        self.clip_ids = ["clip_a", "clip_b", "clip_c"]

    def tearDown(self):
        self._tmp.cleanup()

    # -- fixture writers ---------------------------------------------------

    def write_manifest(self, ids=None):
        ids = self.clip_ids if ids is None else ids
        with (self.source / "manifest.json").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["clip_id"])
            for i in ids:
                w.writerow([i])

    def write_predictions(self, ids=None, label="WHATFOR1"):
        ids = self.clip_ids if ids is None else ids
        with (self.source / "results" / "predictions.jsonl").open("w", encoding="utf-8") as f:
            for i in ids:
                f.write(json.dumps({
                    "clip_id": i,
                    "prediction": label,
                    "top5": [label, self.labels[0], self.labels[1],
                             self.labels[2], self.labels[3]],
                    "confidence": 0.5,
                    "latency_ms": 1.0,
                }) + "\n")

    def build(self, official=True, predictions=None):
        return bb.build(self.source, MODEL, LABELS, self.source / "results" / "results.csv",
                        self.root / "out.zip", official=official,
                        extra=[self.source / CONVERSION],
                        predictions=predictions)

    # -- the happy path ----------------------------------------------------

    def test_complete_synthetic_inputs_build_an_official_bundle(self):
        self.write_manifest()
        self.write_predictions()
        info = self.build(official=True)
        self.assertEqual(info["status"], "structure_verified")

        import zipfile
        with zipfile.ZipFile(self.root / "out.zip") as z:
            names = set(z.namelist())
        # Both spellings the deck uses, plus the manifest the IDs are checked against.
        self.assertIn("results/predictions.jsonl", names)
        self.assertIn("results/predictions.json", names)
        self.assertIn("manifest.json", names)

    # -- the refusals ------------------------------------------------------

    def test_missing_manifest_fails(self):
        self.write_predictions()
        with self.assertRaises(ValueError) as ctx:
            self.build(official=True)
        self.assertIn("manifest.json", str(ctx.exception))

    def test_missing_predictions_fails(self):
        self.write_manifest()
        with self.assertRaises(ValueError) as ctx:
            self.build(official=True)
        self.assertIn("predictions.jsonl", str(ctx.exception))

    def test_missing_clip_fails(self):
        self.write_manifest(["clip_a", "clip_b", "clip_c"])
        self.write_predictions(["clip_a", "clip_b"])   # clip_c absent
        with self.assertRaises(ValueError) as ctx:
            self.build(official=True)
        self.assertIn("do not match the manifest", str(ctx.exception))

    def test_extra_prediction_id_fails(self):
        self.write_manifest(["clip_a", "clip_b"])
        self.write_predictions(["clip_a", "clip_b", "clip_c"])
        with self.assertRaises(ValueError) as ctx:
            self.build(official=True)
        self.assertIn("do not match the manifest", str(ctx.exception))

    def test_duplicate_clip_ids_fail(self):
        self.write_manifest(["clip_a", "clip_a"])
        self.write_predictions(["clip_a", "clip_a"])
        with self.assertRaises(ValueError) as ctx:
            self.build(official=True)
        self.assertIn("duplicate", str(ctx.exception).lower())

    def test_unknown_label_fails(self):
        self.write_manifest()
        self.write_predictions(label="NOT_A_REAL_GLOSS")
        with self.assertRaises(ValueError):
            self.build(official=True)

    def test_wrong_output_schema_fails(self):
        self.write_manifest()
        with (self.source / "results" / "predictions.jsonl").open("w", encoding="utf-8") as f:
            f.write(json.dumps({"clip_id": "clip_a", "prediction": "WHATFOR1"}) + "\n")
        with self.assertRaises(ValueError):
            self.build(official=True)

    def test_wrong_model_identity_fails(self):
        """A member that does not match release.json is refused at verification.

        The builder records each member's SHA-256 in release.json. Substituting
        the bytes afterwards and re-verifying is the real attack: it is what an
        evaluator would hit if a different checkpoint shipped under an expected
        filename. Verification must catch it, not trust the name.
        """
        self.write_manifest()
        self.write_predictions()
        self.build(official=True)

        import zipfile
        bundle = self.root / "out.zip"
        with zipfile.ZipFile(bundle) as z:
            entries = {n: z.read(n) for n in z.namelist()}
        victim = "models/landmark_ens_rob71_fp16.tflite"
        self.assertIn(victim, entries)
        entries[victim] = b"substituted model bytes"
        tampered = self.root / "tampered.zip"
        with zipfile.ZipFile(bundle) as src, \
                zipfile.ZipFile(tampered, "w", zipfile.ZIP_DEFLATED) as z:
            for info in src.infolist():
                data = entries[info.filename]
                # Preserve the original mode: verify_zip rejects members whose
                # external_attr is not a regular file, which a bare writestr
                # would trip before the hash check could run.
                z.writestr(info, data)

        with self.assertRaises(ValueError) as ctx:
            bb.verify_zip(tampered, official=True)
        self.assertIn("hash", str(ctx.exception).lower())

    # -- provisional path must keep working --------------------------------

    def test_provisional_build_still_works_without_a_manifest(self):
        self.write_predictions()          # native predictions, no manifest
        info = self.build(official=False)
        self.assertEqual(info["status"], "structure_verified")
        import zipfile
        with zipfile.ZipFile(self.root / "out.zip") as z:
            names = set(z.namelist())
        # Without --predictions nothing claims to be organizer predictions.
        self.assertNotIn("results/predictions.jsonl", names)
        self.assertNotIn("manifest.json", names)

    def test_provisional_build_can_stage_explicit_predictions(self):
        # Deck p.11 requires predictions in results/ even though their clip IDs
        # cannot be checked without an organizer manifest. The flag stages them
        # for a PROVISIONAL bundle only; the bundle stays unofficial.
        self.write_predictions()
        info = self.build(official=False,
                          predictions=self.source / "results" / "predictions.jsonl")
        self.assertEqual(info["status"], "structure_verified")
        self.assertFalse(info.get("official_submission", False))
        import zipfile
        with zipfile.ZipFile(self.root / "out.zip") as z:
            names = set(z.namelist())
        self.assertIn("results/predictions.jsonl", names)
        self.assertIn("results/predictions.json", names)
        # No manifest was supplied, so none may be invented.
        self.assertNotIn("manifest.json", names)

    def test_staged_predictions_json_is_a_single_valid_json_document(self):
        # Regression: an earlier version copied the .jsonl bytes to .json, which
        # produces "Extra data" on parse. The .json file must be ONE JSON value.
        self.write_predictions(ids=["a", "b"])
        self.build(official=False,
                   predictions=self.source / "results" / "predictions.jsonl")
        import zipfile
        with zipfile.ZipFile(self.root / "out.zip") as z:
            raw = z.read("results/predictions.json").decode("utf-8")
        parsed = json.loads(raw)          # raises if it is still JSON Lines
        self.assertIsInstance(parsed, list)
        self.assertEqual([r["clip_id"] for r in parsed], ["a", "b"])

    def test_staged_predictions_jsonl_stays_json_lines(self):
        # The .jsonl must remain line-delimited; one object per line.
        self.write_predictions(ids=["a", "b"])
        self.build(official=False,
                   predictions=self.source / "results" / "predictions.jsonl")
        import zipfile
        with zipfile.ZipFile(self.root / "out.zip") as z:
            raw = z.read("results/predictions.jsonl").decode("utf-8")
        lines = [l for l in raw.splitlines() if l.strip()]
        self.assertEqual(2, len(lines))
        for line in lines:
            self.assertIsInstance(json.loads(line), dict)

    def test_provisional_predictions_with_unknown_label_are_refused(self):
        # The structural check must catch a label outside the shipped
        # vocabulary even when there is no manifest to check IDs against.
        self.write_predictions(label="NOTAREALGLOSS")
        with self.assertRaises(ValueError) as ctx:
            self.build(official=False,
                       predictions=self.source / "results" / "predictions.jsonl")
        self.assertIn("uppercase label", str(ctx.exception))

    def test_official_build_still_refuses_predictions_without_a_manifest(self):
        # The flag must not open a back door: official builds still require the
        # manifest that makes ID validation possible.
        self.write_predictions()
        with self.assertRaises(ValueError) as ctx:
            self.build(official=True,
                       predictions=self.source / "results" / "predictions.jsonl")
        self.assertIn("manifest", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)