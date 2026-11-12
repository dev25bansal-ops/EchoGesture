"""V7 QA: Unicode / CSV edge cases in clip IDs and labels.

Verified behavior (probed against current production):
  * Exact clip IDs round-trip through manifest -> evaluate -> prediction,
    including emoji, CJK with fullwidth comma, comma-in-quotes, leading
    and trailing whitespace variants, and combining characters.
  * Whitespace variants ("a" vs " a") are distinct clips, not duplicates.
  * Unicode IDs are distinct keys: NFC vs NFD spellings are NOT merged
    (read_manifest keys on the exact string, no Unicode normalization).
  * A BOM-prefixed manifest header is accepted (utf-8-sig) and a
    quoted-comma clip_id parses as one field.
  * Unicode clip IDs are written verbatim (ensure_ascii=False) by the
    atomic evidence writer and read back identically.
  * Label ordering matters: validate_bundle rejects a labels file whose
    index order disagrees with the manifest's ordered_vocabulary (label
    permutation is an error, not a silent remap).
"""

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

DEVICE_DIR = Path(__file__).resolve().parents[1]
ROOT = DEVICE_DIR.parents[1]
for _p in (str(DEVICE_DIR), str(ROOT / "scripts" / "release")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import hashlib  # noqa: E402

import run_manifest  # noqa: E402
from model_contract import CompatibilityError, validate_bundle  # noqa: E402


class UnicodeClipIdTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.labels = {"CANDY", "RIGHT"}

    def _clip(self, name):
        path = self.root / name
        path.write_bytes(b"fixture")
        return str(path)

    def _write_manifest(self, rows):
        manifest = self.root / "manifest.csv"
        with manifest.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["clip_id", "path", "ground_truth"])
            writer.writerows(rows)
        return manifest

    def _predict(self, clip_id, **updates):
        value = {"clip_id": clip_id, "prediction": "CANDY", "top5": ["CANDY", "RIGHT"],
                 "confidence": 0.9, "latency_ms": 1}
        value.update(updates)
        return value

    def test_unicode_clip_ids_roundtrip_through_evaluate(self):
        rows = [
            ("sign-✋-01", self._clip("a.mp4"), ""),
            ("标题，1", self._clip("b.mp4"), ""),
            ("éclair", self._clip("c.mp4"), ""),
        ]
        manifest = self._write_manifest(rows)
        items = run_manifest.read_manifest(manifest)
        self.assertEqual([i[0] for i in items], ["sign-✋-01", "标题，1", "éclair"])
        predictions, _, _ = run_manifest.evaluate(
            items, lambda _, *, clip_id=None: self._predict(clip_id=clip_id), self.labels)
        self.assertEqual([p["clip_id"] for p in predictions],
                         ["sign-✋-01", "标题，1", "éclair"])

    def test_quoted_comma_clip_id_is_single_field(self):
        manifest = self.root / "manifest.csv"
        manifest.write_text('clip_id,path\na,b\n', encoding="utf-8")
        # Unquoted comma makes the header row three fields; the data row's
        # "a" has path "b" only if quoted properly -- write quoted form:
        manifest.write_text('clip_id,path\n"a,b",c.mp4\n', encoding="utf-8")
        self._clip("c.mp4")
        items = run_manifest.read_manifest(manifest)
        self.assertEqual(items[0][0], "a,b")

    def test_whitespace_variant_ids_are_distinct(self):
        rows = [("a", self._clip("a.mp4"), ""), (" a", self._clip("b.mp4"), ""),
                ("a ", self._clip("c.mp4"), "")]
        items = run_manifest.read_manifest(self._write_manifest(rows))
        self.assertEqual([i[0] for i in items], ["a", " a", "a "])
        predictions, _, _ = run_manifest.evaluate(
            items, lambda _, *, clip_id=None: self._predict(clip_id=clip_id), self.labels)
        self.assertEqual([p["clip_id"] for p in predictions], ["a", " a", "a "])

    def test_nfc_and_nfd_forms_are_distinct_keys(self):
        nfc = "café"          # é as single codepoint U+00E9
        nfd = "café"      # e + combining acute U+0301
        self.assertNotEqual(nfc, nfd)
        rows = [(nfc, self._clip("a.mp4"), ""), (nfd, self._clip("b.mp4"), "")]
        items = run_manifest.read_manifest(self._write_manifest(rows))
        self.assertEqual([i[0] for i in items], [nfc, nfd])

    def test_bom_manifest_accepted(self):
        manifest = self.root / "manifest.csv"
        manifest.write_text("﻿clip_id,path\nbomid,c.mp4\n", encoding="utf-8")
        self._clip("c.mp4")
        items = run_manifest.read_manifest(manifest)
        self.assertEqual(items[0][0], "bomid")

    def test_unicode_predictions_written_verbatim_and_read_back(self):
        output = self.root / "out"
        run_manifest.write_pair_atomically(
            output, [self._predict(clip_id="sign-✋-01")], {"status": "passed_schema"})
        lines = (output / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertIn("sign-✋-01", lines[0])
        self.assertEqual(json.loads(lines[0])["clip_id"], "sign-✋-01")


class LabelPermutationTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.bundle = self.root / "bundle"
        self.bundle.mkdir()
        (self.bundle / "classifier.tflite").write_bytes(b"test model")
        self.manifest = {
            "model_sha256": hashlib.sha256(b"test model").hexdigest(),
            "model_size_bytes": 10, "model_format": "tflite",
            "input_tensor_shape": [1, 24, 255], "input_dtype": "float32",
            "input_scale": 0.0, "input_zero_point": 0,
            "output_tensor_shape": [1, 2], "output_dtype": "float32",
            "output_scale": 0.0, "output_zero_point": 0,
            "output_is_logits_or_probabilities": "logits",
            "ordered_vocabulary": ["A", "B"],
            "feature_config_version": "v1", "feature_shape": [24, 255],
            "frame_count": 24, "sampling_method": "uniform full clip",
            "normalization": "documented", "handedness": "left/right",
            "presence_masks": True, "coordinate_system": "normalized image xy",
            "missing_landmark_policy": "explicit presence mask",
        }

    def test_permated_labels_file_rejected(self):
        # labels file says B=0, A=1 (permuted vs manifest ["A", "B"])
        (self.bundle / "labels.json").write_text(json.dumps({"b": 0, "a": 1}))
        manifest = dict(self.manifest, labels_file="labels.json")
        (self.bundle / "integration_manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(CompatibilityError, "Labels file ordering"):
            validate_bundle(self.bundle)

    def test_consistent_labels_file_accepted(self):
        (self.bundle / "labels.json").write_text(json.dumps({"a": 0, "b": 1}))
        manifest = dict(self.manifest, labels_file="labels.json")
        (self.bundle / "integration_manifest.json").write_text(json.dumps(manifest))
        self.assertEqual(validate_bundle(self.bundle)["status"], "compatible")

    def test_case_only_difference_in_labels_accepted_as_upper(self):
        # Production upper-cases labels-file keys before comparing order.
        (self.bundle / "labels.json").write_text(json.dumps({"a": 0, "b": 1}))
        manifest = dict(self.manifest, labels_file="labels.json")
        (self.bundle / "integration_manifest.json").write_text(json.dumps(manifest))
        self.assertEqual(validate_bundle(self.bundle)["classes"], 2)


if __name__ == "__main__":
    unittest.main()
