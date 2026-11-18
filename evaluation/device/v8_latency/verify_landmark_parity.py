"""Bit-exact landmark parity: parallel_4 vs IMAGE singleton baseline.

IMAGE mode is stateless per frame. Splitting frames across independent
tracker instances in threads must produce bit-identical landmarks to
serial singleton. This script proves that on all 30 clips.
"""
import json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def _load_sibling(name: str):
    """Import `name` from THIS file's directory, by path, not by module name.

    Two sibling directories (v7_latency, v8_latency) ship same-named helper
    modules with different contents, so a plain `import name` resolves to
    whichever was imported first in the process. Loading by path makes the file
    next to this one authoritative.
    """
    import importlib.util
    path = HERE / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_local_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
cand = _load_sibling("v7_candidates")
sys.path.insert(0, str(HERE))
cand = _load_sibling("v7_candidates")
MANIFEST = Path("/tmp/echogesture-validation-30/manifest.csv")

import csv
rows = list(csv.DictReader(MANIFEST.open()))

results = []
try:
    for row in rows:
        clip = str(MANIFEST.parent / row["path"])
        frames = cand._INTERNALS.get("frontend")
        cand._ensure_classifier()
        frontend = cand._INTERNALS["frontend"]
        frames = frontend.read_frames(clip, max_frames=32, size=256)
        lm_singleton = cand._singleton("image")
        baseline_landmarks, baseline_present = cand._detect_image(frames, lm_singleton)
        meta = {}
        parallel_landmarks, parallel_present = cand._detect_parallel(frames, 4, meta)
        identical_l = np.array_equal(baseline_landmarks, parallel_landmarks)
        identical_p = np.array_equal(baseline_present, parallel_present)
        results.append({
            "clip_id": row["clip_id"],
            "identical_landmarks": bool(identical_l),
            "identical_present": bool(identical_p),
            "max_abs_diff": float(np.max(np.abs(baseline_landmarks - parallel_landmarks))),
        })
        if not (identical_l and identical_p):
            print(f"MISMATCH: {row['clip_id']}")
finally:
    cand.shutdown()

n_ok = sum(1 for r in results if r["identical_landmarks"] and r["identical_present"])
print(f"{n_ok}/{len(results)} clips bit-identical landmarks + present flags")
out = Path(HERE) / "results" / "landmark_parity_parallel4.json"
out.write_text(json.dumps({"clips": results, "n_identical": n_ok, "n_total": len(results)}, indent=2))
