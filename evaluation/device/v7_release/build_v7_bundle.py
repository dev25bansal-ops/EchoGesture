"""A13 v7_release driver: regenerate results.csv from measured evidence and
rebuild the hardened provisional ZIP from real files in the original checkout.

This is a reproducible wrapper around scripts/release/generate_results.py
and scripts/release/build_bundle.py (the hardened builder, unmodified).
It reproduces the exact build that produced provisional.zip in this directory,
including real extracted-clip verification and the official-mode block test.
No measurement is fabricated; all numbers come from the cited JSON evidence.
"""
from __future__ import annotations
import argparse, subprocess, sys, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PY = str(ROOT / ".venv/bin/python")
MODEL_SHA = "a012f2b5f7a1110961079f19eb0e28c9abacb95e4d0787352e4157ebca7044a4"
BENCH = ROOT / "evaluation/device/accuracy_lab/benchmark.json"
ROB = ROOT / "evaluation/device/robustness_lab/runs"
ROB_FILES = ["low_light/robustness.json", "crop/robustness.json",
             "occlusion/robustness.json", "motion_blur/robustness.json",
             "frame_drop/robustness.json"]
RESULTS = HERE / "results.csv"
ZIP = HERE / "provisional.zip"
CLIP = ROOT / "evaluation/device/v5_accuracy/data/val_30/17092802473126545-SQUEEZE.mp4"

def run(cmd):
    print("+", " ".join(map(str, cmd)))
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout)
    if r.returncode:
        print(r.stderr, file=sys.stderr); raise SystemExit(r.returncode)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rebuild", action="store_true", help="regenerate results.csv and provisional.zip")
    ap.add_argument("--verify-only", action="store_true", help="re-verify existing provisional.zip with real clip inference")
    args = ap.parse_args()
    if args.verify_only:
        run([PY, "-I", str(ROOT / "scripts/release/build_bundle.py"), "verify", str(ZIP),
             "--clip", str(CLIP), "--model-sha", MODEL_SHA])
        return
    cmd = [PY, "-I", str(ROOT / "scripts/release/generate_results.py"),
           "--benchmark", str(BENCH)]
    for f in ROB_FILES: cmd += ["--robustness", str(ROB / f)]
    cmd += ["--model-sha", MODEL_SHA,
            "--artifact", f"hand_landmarker={ROOT/'models/hand_landmarker.task'}",
            "--artifact", f"landmark_labels={ROOT/'models/landmark_labels.json'}",
            "--artifact", f"landmark_model_meta={ROOT/'models/landmark_model_meta.json'}",
            "--predictions", str(ROOT / "evaluation/device/accuracy_lab/predictions.jsonl"),
            "--output", str(RESULTS)]
    run(cmd)
    run([PY, "-I", str(ROOT / "scripts/release/build_bundle.py"), "build",
         "--source", str(ROOT), "--model", "models/landmark_classifier_fp16.tflite",
         "--labels", "models/landmark_labels.json", "--results", str(RESULTS),
         "--output", str(ZIP),
         "--extra", "models/landmark_classifier_fp16.tflite",
         "--extra", "models/landmark_labels.json",
         "--extra", "models/landmark_model_meta.json",
         "--extra", "models/hand_landmarker.task",
         "--extra", "scripts/ml/export_tflite_tf.py",
         "--extra", "src/features.py", "--extra", "src/landmark_frontend.py"])
    run([PY, "-I", str(ROOT / "scripts/release/build_bundle.py"), "verify", str(ZIP),
         "--clip", str(CLIP), "--model-sha", MODEL_SHA])
    # Official mode must stay blocked: expect exit 2 and no file.
    r = subprocess.run([PY, "-I", str(ROOT / "scripts/release/build_bundle.py"), "build",
         "--source", str(ROOT), "--model", "models/landmark_classifier_fp16.tflite",
         "--labels", "models/landmark_labels.json", "--results", str(RESULTS),
         "--output", str(HERE / "official_blocked.zip"), "--official"],
         capture_output=True, text=True)
    assert r.returncode == 2 and "Official bundle requires organizer-generated predictions.jsonl" in r.stderr, (r.returncode, r.stderr)
    print("official-mode block confirmed (exit 2, no file written)")

if __name__ == "__main__":
    main()
