"""Tests for the A13 v7_release bundle: structure, real-clip inference, honesty gates."""
import csv, hashlib, json, subprocess, sys, tempfile, zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
ZIP = HERE / "provisional.zip"
MODEL_SHA = "a012f2b5f7a1110961079f19eb0e28c9abacb95e4d0787352e4157ebca7044a4"
DETECTOR_SHA = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"
EXPECTED = {"Dockerfile", "release.json", "results/results.csv",
            "models/landmark_classifier_fp16.tflite", "models/landmark_labels.json",
            "models/landmark_model_meta.json", "models/hand_landmarker.task",
            "scripts/ml/export_tflite_tf.py", "src/main.py", "src/requirements.txt",
            "src/README.md", "src/features.py", "src/landmark_frontend.py"}

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def test_zip_hashes_and_members():
    with zipfile.ZipFile(ZIP) as z:
        names = set(z.namelist())
        assert names == EXPECTED, names ^ EXPECTED
        assert not any(n.startswith("/") or ".." in n for n in names)
        with tempfile.TemporaryDirectory() as d:
            z.extractall(d)
            assert sha(Path(d) / "models/landmark_classifier_fp16.tflite") == MODEL_SHA
            assert sha(Path(d) / "models/hand_landmarker.task") == DETECTOR_SHA
            rel = json.loads((Path(d) / "release.json").read_text())
            assert rel["model_sha256"] == MODEL_SHA and rel["official_submission"] is False
            assert rel["conversion_script"] == "scripts/ml/export_tflite_tf.py"
            rows = list(csv.DictReader(open(Path(d) / "results/results.csv")))
            assert rows and {r["pass"] for r in rows} <= {"PASS", "FAIL", "NOT_MEASURED", "NOT_APPLICABLE"}

def test_official_blocked_and_verify_real_clip():
    root = Path("/Users/abhijeetmiskin/AppData/EchoGesture")
    py = str(root / ".venv/bin/python")
    builder = str(root / "scripts/release/build_bundle.py")
    r = subprocess.run([py, "-I", builder, "build", "--source", str(root),
        "--model", "models/landmark_classifier_fp16.tflite",
        "--labels", "models/landmark_labels.json",
        "--results", str(HERE / "results.csv"), "--output", str(HERE / "official_blocked.zip"),
        "--official"], capture_output=True, text=True)
    assert r.returncode == 2 and "Official bundle requires organizer-generated predictions.jsonl" in r.stderr
    assert not (HERE / "official_blocked.zip").exists()
    clip = root / "evaluation/device/v5_accuracy/data/val_30/17092802473126545-SQUEEZE.mp4"
    if clip.exists():
        v = subprocess.run([py, "-I", builder, "verify", str(ZIP), "--clip", str(clip),
                            "--model-sha", MODEL_SHA], capture_output=True, text=True, timeout=300)
        assert v.returncode == 0, v.stderr
        out = json.loads(v.stdout)
        assert out["clip_id"] == "17092802473126545-SQUEEZE" and out["prediction"] == "SQUEEZE"
        assert 0 <= out["confidence"] <= 1 and out["latency_ms"] > 0
    else:
        import pytest; pytest.skip("no real clip available")

if __name__ == "__main__":
    test_zip_hashes_and_members(); print("structure ok")
    test_official_blocked_and_verify_real_clip(); print("inference+official-block ok")
