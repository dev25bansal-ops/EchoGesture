"""Measure what each capture mode actually delivers to the four-model ensemble.

Both modes are simulated against genuine ASL Citizen clips, using the SAME
production feature contract (src/features.py) and the SAME frozen ensemble.
The only difference between the rows is WHICH FRAMES each mode selects.

  EXPLICIT  Start → perform → Finish. Frames from [start, finish], sampled 32
            evenly across the interval, resampled to 24. This mirrors
            SignRecorder + RecordingFeatures exactly.

  QUICK SCAN
            Perform → tap. The 3.7 s window ENDING at the tap, sampled at 32
            points, resampled to 24. This mirrors HandTracker.selectCapture.

Because a training clip IS one sign from start to finish, "explicit" on a clip
means the whole clip. Quick scan is then given the same clip with the tap at
its end — the best case for quick scan — and also with the tap delayed, which
is what a real user does.

Usage:
  python simulate_modes.py --clips DIR --manifest manifest.json --out results.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/tmp/eg-rc")
sys.path.insert(0, str(ROOT / "src"))

# The Python models/ dir has no FP16 ensemble; the Android assets hold the
# four pinned members the app actually ships. Point the evaluator at those
# exact bytes so this measurement uses the production artifacts, not a copy.
import main as _eg_bootstrap  # noqa: E402
_eg_bootstrap._landmark_config = (
    lambda model_dir, frames_hint, feature_dim_hint: (
        json.loads((ROOT / "app/android_final/app/src/main/assets/landmark_model_meta.json").read_text())["members"],
        24, 255,
    )
)

# The production feature contract. Imported, never re-implemented: a copy that
# drifts from features.py would make this whole measurement meaningless.
from features import build_features  # noqa: E402
import landmark_frontend as lf  # noqa: E402
import main as eg  # noqa: E402

SAMPLED = 32
MODEL_STEPS = 24
QUICK_SCAN_MS = 3700
FPS_ASSUMED = 30.0


def clip_frames(path):
    """Every frame of the clip at its native rate, with real timestamps."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return [], 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS_ASSUMED
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(cv2.resize(f, (256, 256)), cv2.COLOR_BGR2RGB))
    cap.release()
    return frames, fps


def detect_all(frames, landmarker):
    """Hand landmarks for EVERY frame, once, so both modes see identical data."""
    return lf.detect_hands(frames, landmarker, mode="image")


def sample_indices(n, target):
    """np.linspace(0, n-1, target) — the training sampling rule."""
    if n <= target:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, target).astype(int))


def features_from(landmarks, present, idx):
    """Sample the given frame indices, then build the 24x255 tensor."""
    lm = landmarks[idx]
    pr = present[idx]
    feats, mask = build_features(lm, pr, expected_frames=MODEL_STEPS)
    return feats, mask


def motion_stats(feats):
    """What the model is actually told about movement."""
    block = 84
    vel = feats[:, block:2 * block]
    return {
        "mean_abs_vel": float(np.abs(vel).mean()),
        "max_abs_vel": float(np.abs(vel).max()),
        "nonzero_vel_frac": float((np.abs(vel) > 1e-6).mean()),
    }


def predict(feats):
    """Exactly the production aggregation: mean logits, one softmax, top-5.

    Uses src/main.py's own _run_ensemble and its own softmax rather than a
    re-implementation, so a drift between this measurement and the evaluator
    is impossible.
    """
    tensor = feats[None, ...].astype(np.float32)
    logits, _ = eg._run_ensemble(tensor)
    e = np.exp(logits - logits.max())
    probs = e / e.sum()
    top = np.argsort(-probs)[:5]
    top5 = [eg._IDX_TO_LABEL.get(int(i), f"UNKNOWN_{i}") for i in top]
    return {
        "prediction": top5[0],
        "top5": top5,
        "confidence": float(probs[top[0]]),
        # The app's top-two margin gate; kept here so accepted accuracy can be
        # reported on the same rule the product uses.
        "margin": float(probs[top[0]] - probs[top[1]]),
    }


def run_one(entry, clips_dir, landmarker, interpreter_bundle, labels):
    path = clips_dir / entry["file"]
    frames, fps = clip_frames(path)
    if not frames:
        return {"file": entry["file"], "error": "unreadable"}
    landmarks, present = detect_all(frames, landmarker)
    n = len(frames)
    duration_ms = n / fps * 1000.0

    rows = {}

    # --- EXPLICIT: the whole clip is the recording ---
    idx = sample_indices(n, SAMPLED)
    feats, mask = features_from(landmarks, present, idx)
    rows["explicit"] = {
        "frames_in_interval": n,
        "sampled": int(len(idx)),
        "duration_ms": duration_ms,
        "hand_coverage": float(present[idx].max(axis=1).mean()),
        "span_ms": float((idx[-1] - idx[0]) / fps * 1000.0),
        **motion_stats(feats),
    }

    # --- QUICK SCAN, tap at the end (best case for quick scan) ---
    tail_n = int(round(QUICK_SCAN_MS / 1000.0 * fps))
    tail_n = min(tail_n, n)
    tail_idx = sample_indices(tail_n, SAMPLED) + (n - tail_n)
    feats_q, _ = features_from(landmarks, present, tail_idx)
    rows["quick_tap_at_end"] = {
        "frames_in_interval": tail_n,
        "sampled": int(len(tail_idx)),
        "duration_ms": tail_n / fps * 1000.0,
        "hand_coverage": float(present[tail_idx].max(axis=1).mean()),
        "span_ms": float((tail_idx[-1] - tail_idx[0]) / fps * 1000.0),
        **motion_stats(feats_q),
    }

    # --- QUICK SCAN, tap 1.5 s late (a realistic error the brief names) ---
    delay_frames = int(round(1.5 * fps))
    late_end = max(1, n - delay_frames)
    late_tail = min(int(round(QUICK_SCAN_MS / 1000.0 * fps)), late_end)
    late_idx = sample_indices(late_tail, SAMPLED) + (late_end - late_tail)
    feats_late, _ = features_from(landmarks, present, late_idx)
    rows["quick_tap_late"] = {
        "frames_in_interval": late_tail,
        "sampled": int(len(late_idx)),
        "duration_ms": late_tail / fps * 1000.0,
        "hand_coverage": float(present[late_idx].max(axis=1).mean()),
        "span_ms": float((late_idx[-1] - late_idx[0]) / fps * 1000.0),
        **motion_stats(feats_late),
    }

    # One inference per tensor, through the production ensemble.
    preds = {}
    for mode, f in (("explicit", feats), ("quick_tap_at_end", feats_q), ("quick_tap_late", feats_late)):
        r = predict(f)
        preds[mode] = {
            "pred": r["prediction"],
            "top5": r["top5"],
            "score": r["confidence"],
            "correct": r["prediction"] == entry["gloss"],
        }
        rows[mode]["pred"] = r["prediction"]
        rows[mode]["correct"] = r["prediction"] == entry["gloss"]
        rows[mode]["score"] = r["confidence"]
        rows[mode]["margin"] = r["margin"]
        rows[mode]["top5"] = r["top5"]

    return {
        "file": entry["file"],
        "gloss": entry["gloss"],
        "signer": entry["signer"],
        "clip_frames": n,
        "clip_fps": fps,
        "clip_duration_ms": duration_ms,
        "modes": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    clips_dir = Path(args.clips)
    entries = json.loads(Path(args.manifest).read_text())
    labels = json.loads((ROOT / "models/landmark_labels.json").read_text())

    eg._init_model()
    bundle = eg._INTERPRETERS
    landmarker = lf.get_landmarker(mode="image")

    results = []
    for i, e in enumerate(entries, 1):
        try:
            results.append(run_one(e, clips_dir, landmarker, bundle, labels))
        except Exception as exc:  # noqa: BLE001
            results.append({"file": e["file"], "error": f"{type(exc).__name__}: {exc}"})
        print(f"[{i}/{len(entries)}] {e.get('gloss')} {e.get('signer')}", flush=True)

    Path(args.out).write_text(json.dumps(results, indent=1))
    ok = [r for r in results if "error" not in r]
    print(f"\n{len(ok)}/{len(results)} clips analysed")
    for mode in ("explicit", "quick_tap_at_end", "quick_tap_late"):
        hits = sum(1 for r in ok if r["modes"][mode]["correct"])
        top5 = 0
        print(f"  {mode:18s} top1 {hits}/{len(ok)} = {100.0*hits/max(1,len(ok)):.1f}%")


if __name__ == "__main__":
    main()