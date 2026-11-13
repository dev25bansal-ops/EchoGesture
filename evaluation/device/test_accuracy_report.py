import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import accuracy_report


def _write(tmp: Path, name: str, text: str) -> Path:
    p = tmp / name
    p.write_text(text, encoding="utf-8")
    return p


def _diag(frames, hand_frames, pred, score, top5, dur=3700):
    pres = "1" * max(hand_frames, 0) + "0" * max(frames - hand_frames, 0)
    return (
        f"frames={frames} dur_ms={dur} fps=30.0 hand_frames={hand_frames} "
        f"presence={pres} pred={pred} score={score} top5={top5} "
        f"infer_ms={43 if pred else -1} total_ms={51 if pred else 0}"
    )


class ParseTests(unittest.TestCase):
    def test_header_and_blank_lines_are_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            p = _write(tmp, "d.txt", "# header\n\n" + _diag(32, 20, "EAT1", "0.529", "EAT1|X") + "\n")
            rows = accuracy_report.parse_diagnostics(p)
        self.assertEqual(1, len(rows))
        self.assertEqual("EAT1", rows[0]["pred"])
        self.assertEqual(["EAT1", "X"], rows[0]["top5"])

    def test_empty_score_parses_as_zero_not_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            p = _write(tmp, "d.txt", "frames=32 dur_ms=1 fps=1.0 hand_frames=0 presence=00 pred= score= top5= infer_ms=-1 total_ms=0\n")
            rows = accuracy_report.parse_diagnostics(p)
        self.assertEqual(0.0, rows[0]["score"])
        self.assertEqual([], rows[0]["top5"])


class CategorizeTests(unittest.TestCase):
    def _row(self, frames, hand, pred, score):
        return {"frames": frames, "hand_frames": hand, "pred": pred, "score": score}

    def test_low_coverage_is_a_failed_capture_not_a_model_error(self):
        # 2/32 is well under the 15% floor: the app never classifies this.
        self.assertEqual(
            "failed capture", accuracy_report.categorize(self._row(32, 2, "", 0.0), "EAT1")
        )

    def test_no_prediction_is_a_failed_capture(self):
        self.assertEqual(
            "failed capture", accuracy_report.categorize(self._row(32, 20, "", 0.0), "EAT1")
        )

    def test_wrong_label_above_cut_is_an_accepted_error(self):
        # The real failure mode: a confident wrong answer is worse than a
        # rejection, and must be counted separately from one.
        self.assertEqual(
            "incorrect (accepted)",
            accuracy_report.categorize(self._row(32, 20, "TAIL1", 0.455), "EAT1"),
        )

    def test_wrong_label_below_cut_is_a_rejection(self):
        self.assertEqual(
            "incorrect (rejected)",
            accuracy_report.categorize(self._row(32, 20, "TAIL1", 0.20), "EAT1"),
        )

    def test_correct_low_score_still_counts_as_correct(self):
        # Accuracy is about correctness, not confidence.
        self.assertEqual(
            "correct", accuracy_report.categorize(self._row(32, 20, "EAT1", 0.18), "EAT1")
        )


class ReportTests(unittest.TestCase):
    def test_mismatched_pairing_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            diag = _write(tmp, "d.txt", _diag(32, 20, "EAT1", "0.5", "EAT1") + "\n")
            ann = _write(tmp, "a.txt", "1 EAT1 plain\n2 EAT1 plain\n")
            self.assertEqual(2, accuracy_report.main([str(diag), str(ann)]))

    def test_margin_is_used_when_present(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            diag = _write(
                tmp,
                "d.txt",
                "frames=32 dur_ms=1 fps=30.0 hand_frames=20 presence=" + "1"*20 + "0"*12
                + " pred=EAT1 score=0.2200 top5=EAT1|X margin=0.3000 infer_ms=40 total_ms=50\n",
            )
            ann = _write(tmp, "a.txt", "1 EAT1 plain\n")
            rows = accuracy_report.parse_diagnostics(diag)
        self.assertAlmostEqual(0.30, rows[0]["margin"])
        # 0.22 score would fail the old 0.35 cut; a 0.30 margin must accept it.
        self.assertTrue(rows[0]["margin"] >= accuracy_report.MARGIN_CUT)

    def test_report_prints_top1_and_acceptance(self):
        import contextlib
        import io

        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            diag = _write(
                tmp,
                "d.txt",
                _diag(32, 20, "EAT1", "0.529", "EAT1") + "\n"
                + _diag(32, 20, "TAIL1", "0.455", "TAIL1") + "\n",
            )
            ann = _write(tmp, "a.txt", "1 EAT1 plain\n2 EAT1 plain\n")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.assertEqual(0, accuracy_report.main([str(diag), str(ann)]))
            out = buf.getvalue()
        self.assertIn("top-1                    1/2 = 50.0%", out)
        self.assertIn("accuracy at accepted     1/2 = 50.0%", out)
        self.assertIn("EAT1             -> TAIL1", out)


if __name__ == "__main__":
    unittest.main()