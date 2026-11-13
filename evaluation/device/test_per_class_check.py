import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import per_class_check


class CanonTests(unittest.TestCase):
    def test_normalisation_bridges_gloss_spellings(self):
        # The split writes "WHAT FOR"; the vocabulary says "WHATFOR1". Both
        # normalise to WHATFOR, and the vocabulary lookup is prefix-tolerant
        # only through the canon dictionary, so this must match exactly after
        # stripping the digit suffix the vocabulary uses.
        self.assertEqual("WHATFOR", per_class_check.canon("WHAT FOR"))
        self.assertEqual("WHATFOR1", per_class_check.canon("WHATFOR1"))

    def test_case_and_punctuation_are_ignored(self):
        self.assertEqual(per_class_check.canon("pull-convince"),
                         per_class_check.canon("PULL CONVINCE"))
        self.assertEqual(per_class_check.canon("HURDLE/TRIP1"),
                         per_class_check.canon("hurdle trip 1"))


class MappingTests(unittest.TestCase):
    def _labels(self):
        import json
        return json.loads((Path(per_class_check.ROOT) / "models/landmark_labels.json").read_text())

    def test_every_label_is_unique_after_normalisation(self):
        # If two labels collided, a gloss could map to the wrong class and any
        # per-class table would be silently wrong.
        labels = self._labels()
        seen = {}
        for l in labels:
            c = per_class_check.canon(l)
            self.assertNotIn(c, seen, f"collision between {seen.get(c)} and {l}")
            seen[c] = l

    def test_vocabulary_has_one_hundred_unique_labels(self):
        labels = self._labels()
        self.assertEqual(100, len(labels))
        self.assertEqual(100, len(set(labels)))


class PctTests(unittest.TestCase):
    def test_zero_denominator_is_not_a_crash(self):
        self.assertEqual("n/a", per_class_check.pct(0, 0))

    def test_rounds_to_one_decimal(self):
        self.assertEqual("33.3%", per_class_check.pct(1, 3))


if __name__ == "__main__":
    unittest.main()