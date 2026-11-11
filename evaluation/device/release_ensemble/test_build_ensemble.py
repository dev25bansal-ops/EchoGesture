"""Release-builder tests for a multi-member (ensemble) model bundle.

Covers the ensemble extension to ``scripts/release/build_bundle.py``:

* every member declared in ``landmark_model_meta.json`` is staged;
* ``release.json`` records each member with its own SHA-256;
* verification rejects a member hash mismatch, a traversal member name and a
  missing member;
* the 20 MB ``models/`` ceiling still applies to the whole ensemble;
* a real build -> clean extract -> ``predict`` round trip on the permitted clip.

The round-trip test skips when the local weights are absent: they carry no
redistribution permission and are never committed or redistributed here.
"""

import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "scripts" / "release"))

import build_bundle as bb  # noqa: E402

MODEL = "models/landmark_classifier_fp16.tflite"
LABELS = "models/landmark_labels.json"
META = "models/landmark_model_meta.json"
CONVERSION = "scripts/ml/export_tflite_tf.py"
MEMBERS = (
    "landmark_ens_tcn21_fp16.tflite",
    "landmark_ens_tcnm61_fp16.tflite",
    "landmark_ens_tcnm62_fp16.tflite",
    "landmark_ens_rob71_fp16.tflite",
)
CLIP = REPO / "evaluation/device/v5_accuracy/data/val_30/0283886564670357-WHAT FOR.mp4"
PYTHON = REPO / ".venv" / "bin" / "python"


def _meta(members=None):
    value = {"frames": 24, "feature_dim": 255, "classes": 100,
             "quantization": "float16"}
    if members is not None:
        value["members"] = list(members)
    return value


def _write_bundle(root, members=(), member_bytes=None, model_bytes=b"model",
                  member_hash_override=None, write_members=True):
    """Fabricate a minimal bundle directory accepted by verify_directory()."""
    root = Path(root)
    (root / "src").mkdir(parents=True)
    (root / "models").mkdir()
    (root / "results").mkdir()
    (root / "scripts/ml").mkdir(parents=True)
    for name in ("main.py", "requirements.txt", "README.md"):
        (root / "src" / name).write_text("x", encoding="utf-8")
    (root / "scripts/ml/export_tflite_tf.py").write_text("# export", encoding="utf-8")
    (root / "Dockerfile").write_text("FROM python:3.11-slim\n", encoding="utf-8")
    (root / LABELS).write_text(json.dumps({"A": 0, "B": 1}), encoding="utf-8")
    (root / MODEL).write_bytes(model_bytes)
    (root / "results/results.csv").write_text(
        "metric,target,measured,unit,device,runtime,pass,notes\n"
        "m,1,1,u,d,r,PASS,note\n", encoding="utf-8")
    (root / META).write_text(json.dumps(_meta(members)), encoding="utf-8")
    entries = []
    for name in members:
        if write_members:
            (root / "models" / name).write_bytes(
                (member_bytes or {}).get(name, b"member"))
        entries.append({
            "name": name,
            "sha256": (member_hash_override or {}).get(
                name, bb.sha256(root / "models" / name) if write_members else "a" * 64),
        })
    manifest = {
        "model": MODEL, "labels": LABELS,
        "model_sha256": bb.sha256(root / MODEL),
        "conversion_script": "scripts/ml/export_tflite_tf.py",
        "official_submission": False,
        "layout": "provisional pending organizer starter kit",
        "model_metadata": META,
    }
    if members:
        manifest["members"] = entries
    (root / "release.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


class EnsembleStagingTest(unittest.TestCase):
    def test_build_stages_every_member(self):
        metadata = _meta(MEMBERS)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            meta_override = directory / "meta.json"
            meta_override.write_text(json.dumps(metadata), encoding="utf-8")
            output = directory / "bundle.zip"
            info = bb.build(REPO, MODEL, LABELS, REPO / "results/results.csv",
                            output, extra=[REPO / CONVERSION],
                            metadata=meta_override)
            with zipfile.ZipFile(output) as archive:
                names = set(archive.namelist())
                manifest = json.loads(archive.read("release.json"))
            for name in MEMBERS:
                self.assertIn(f"models/{name}", names)
            self.assertIn(META, names)
            declared = [entry["name"] for entry in manifest["members"]]
            self.assertEqual(declared, list(MEMBERS))
            for entry in manifest["members"]:
                self.assertEqual(
                    entry["sha256"],
                    bb.sha256(REPO / "models" / entry["name"]))
            self.assertEqual(manifest["model_metadata"], META)
            self.assertEqual([m["name"] for m in info["members"]], list(MEMBERS))

    def test_single_model_manifest_has_no_members_key(self):
        """A meta sidecar without 'members' keeps the old single-model shape."""
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "repo"
            (source / "src").mkdir(parents=True)
            (source / "models").mkdir()
            (source / "results").mkdir()
            (source / "scripts/ml").mkdir(parents=True)
            for name in ("main.py", "requirements.txt", "README.md",
                         "features.py", "landmark_frontend.py"):
                (source / "src" / name).write_text("x", encoding="utf-8")
            (source / MODEL).write_bytes(b"single")
            (source / LABELS).write_text(json.dumps({"A": 0, "B": 1}), encoding="utf-8")
            (source / META).write_text(json.dumps(_meta()), encoding="utf-8")
            (source / "models/hand_landmarker.task").write_bytes(b"task")
            (source / "results/results.csv").write_text(
                "metric,target,measured,unit,device,runtime,pass,notes\n"
                "m,1,1,u,d,r,PASS,note\n", encoding="utf-8")
            (source / CONVERSION).write_text("# export", encoding="utf-8")
            output = directory / "bundle.zip"
            bb.build(source, MODEL, LABELS, source / "results/results.csv",
                     output, extra=[source / CONVERSION])
            with zipfile.ZipFile(output) as archive:
                manifest = json.loads(archive.read("release.json"))
            self.assertNotIn("members", manifest)
            self.assertIn("model_metadata", manifest)


class EnsembleVerificationTest(unittest.TestCase):
    def test_valid_ensemble_verifies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = _write_bundle(Path(directory) / "bundle", members=MEMBERS[:2])
            info = bb.verify_directory(root)
            self.assertEqual(info["status"], "structure_verified")
            self.assertEqual([m["name"] for m in info["members"]], list(MEMBERS[:2]))

    def test_member_hash_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = _write_bundle(
                Path(directory) / "bundle", members=MEMBERS[:1],
                member_hash_override={MEMBERS[0]: "b" * 64})
            with self.assertRaisesRegex(ValueError, "member hash mismatch"):
                bb.verify_directory(root)

    def test_traversal_member_name_rejected(self):
        for bad in ("../evil.tflite", "sub/evil.tflite", "..", "C:evil.tflite",
                    "evil.tflite:stream", ""):
            with self.subTest(member=bad):
                with self.assertRaises(ValueError):
                    bb.declared_members(_meta([bad]), "Model metadata")

    def test_missing_member_in_metadata_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            meta_override = Path(directory) / "meta.json"
            meta_override.write_text(json.dumps(_meta(["landmark_ens_missing_fp16.tflite"])),
                                     encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Ensemble member .* is missing"):
                bb.build(REPO, MODEL, LABELS, REPO / "results/results.csv",
                         Path(directory) / "bundle.zip",
                         extra=[REPO / CONVERSION], metadata=meta_override)

    def test_missing_member_in_bundle_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = _write_bundle(Path(directory) / "bundle", members=MEMBERS[:1],
                                 write_members=False)
            with self.assertRaisesRegex(ValueError, "Ensemble member .* is missing"):
                bb.verify_directory(root)

    def test_metadata_member_order_must_match_release_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = _write_bundle(Path(directory) / "bundle", members=MEMBERS[:2])
            manifest_path = root / "release.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["members"].reverse()
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "do not match the metadata"):
                bb.verify_directory(root)

    def test_size_ceiling_covers_whole_ensemble(self):
        with tempfile.TemporaryDirectory() as directory:
            root = _write_bundle(Path(directory) / "bundle")
            (root / "models" / "bulk.tflite").write_bytes(b"x" * (20 * 1000 * 1000))
            with self.assertRaisesRegex(ValueError, "exceed 20 MB"):
                bb.verify_directory(root)


class EnsembleRoundTripTest(unittest.TestCase):
    """Real build -> fresh extract -> predict. Skips without local weights."""

    def _weights_available(self):
        return (all((REPO / "models" / name).is_file() for name in MEMBERS)
                and (REPO / MODEL).is_file() and (REPO / LABELS).is_file()
                and (REPO / "models/hand_landmarker.task").is_file())

    def test_build_extract_predict_round_trip(self):
        if not PYTHON.is_file():
            self.skipTest("project interpreter not available")
        if not CLIP.is_file():
            self.skipTest("real clip not available")
        if not self._weights_available():
            self.skipTest("ensemble weights not present locally (not redistributable)")

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            meta_override = directory / "meta.json"
            meta_override.write_text(json.dumps(_meta(MEMBERS)), encoding="utf-8")
            output = directory / "bundle.zip"
            bb.build(REPO, MODEL, LABELS, REPO / "results/results.csv",
                     output, extra=[REPO / CONVERSION], metadata=meta_override)
            prediction = bb.verify_zip(output, clip=CLIP)

            self.assertEqual(prediction["clip_id"], CLIP.stem)
            labels = json.loads((REPO / LABELS).read_text(encoding="utf-8"))
            self.assertIn(prediction["prediction"], {label.upper() for label in labels})
            self.assertTrue(0 <= prediction["confidence"] <= 1)
            self.assertGreaterEqual(prediction["latency_ms"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)