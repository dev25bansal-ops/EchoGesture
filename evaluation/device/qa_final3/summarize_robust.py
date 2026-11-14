"""Read-only summarizer for robustness_ensemble result JSONs."""
import glob
import json
import os

SKIP = {"clip_ids", "model_meta", "model_sha256", "manifest", "predictions",
        "per_clip", "clips", "clip_results"}

for path in sorted(glob.glob("evaluation/device/robustness_ensemble/results/*.json")):
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    print("==", os.path.basename(path))
    for key, value in data.items():
        if key in SKIP:
            continue
        text = repr(value)
        if len(text) > 240:
            text = text[:240] + "..."
        print("   ", key, "=", text)
    print()