"""Evaluate whether short internal missing-hand gaps are safe to repair.

The only candidate retained here is an internal one-frame gap whose two
surrounding frames have exactly one observed hand in the same canonical slot.
No slot relabelling, no extrapolation, and no new coordinates are introduced.
The candidate is intentionally not the default predictor.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/tmp/echogesture-v7-agents/A03")
sys.path.insert(0, str(ROOT / "evaluation/device/v7_tracking"))
sys.path.insert(0, str(ROOT / "evaluation/device"))
sys.path.insert(0, str(ROOT / "src"))
from v6_robustness.diagnose import get_interpreters, setup_interp, run_classifier, probs, read_labels  # noqa: E402
from features import build_features  # noqa: E402
from candidates import short_gap_interpolation  # noqa: E402


def main():
    run_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "evaluation/device/v7_tracking/results/clean_346/run.json"
    run = json.loads(run_path.read_text())
    lm = np.load(run_path.parent / "raw_landmarks.npy")
    pr = np.load(run_path.parent / "raw_present.npy")
    state = []
    for name, (_p, interp) in zip(("fp16", "int8"), get_interpreters()):
        state.append((name, interp, *setup_interp(interp)))
    labels = read_labels()
    changes = []
    for i, r in enumerate([x for x in run["records"] if "candidates" in x]):
        p = pr[i] > 0.5
        for h in range(2):
            t = 0
            while t < len(p):
                if p[t, h]:
                    t += 1
                    continue
                start = t
                while t < len(p) and not p[t, h]:
                    t += 1
                end = t - 1
                if start == 0 or t == len(p) or end - start + 1 != 1:
                    continue
                if not (p[start - 1, h] and p[t, h]):
                    continue
                # Require the other slot to be absent before and after. This
                # avoids filling a missing hand during a one-hand phase.
                if p[start - 1, 1 - h] or p[t, 1 - h]:
                    continue
                if p[start].sum() != 0 or p[end].sum() != 0:
                    continue
                trial = short_gap_interpolation(lm[i], pr[i], max_gap=1)
                tensor, _ = build_features(trial[0], trial[1], expected_frames=24)
                base_tensor, _ = build_features(lm[i], pr[i], expected_frames=24)
                pred = {}
                for name, interp, ind, outd in state:
                    b = run_classifier(interp, ind, outd, base_tensor[None])
                    c = run_classifier(interp, ind, outd, tensor[None])
                    pred[name] = {"base": labels[int(np.argmax(b))], "cand": labels[int(np.argmax(c))]}
                changes.append({"clip_id": r["clip_id"], "ground_truth": r["ground_truth"],
                                "hand": h, "frame": start, "pred": pred})
    print(json.dumps({"n_events": len(changes), "events": changes}, indent=2))


if __name__ == "__main__":
    main()
