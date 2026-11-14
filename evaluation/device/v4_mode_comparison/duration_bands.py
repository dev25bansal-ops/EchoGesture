"""Does idle time inside a recording dilute the temporal features?

The V4 recorder lets the signer choose the interval. If they hold the record
button through seconds of idle before or after the sign, the 32 samples spread
across that whole interval and only some of them land on the sign. This
measures exactly that, on real clips, with the production feature contract and
the production ensemble.

Each clip is a real sign. Idle is simulated by prepending/appending frames with
NO detected hands — which is what an idle camera actually produces — rather than
by fabricating landmarks.

Usage:
  python duration_bands.py --clips DIR --manifest manifest.json --out results.json
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/tmp/eg-rc")
sys.path.insert(0, str(ROOT / "src"))

import main as eg  # noqa: E402
from features import build_features  # noqa: E402
import landmark_frontend as lf  # noqa: E402

SAMPLED = 32
STEPS = 24


def load_clip(path, max_frames=400):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return [], 30.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while len(frames) < max_frames:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(cv2.resize(f, (256, 256)), cv2.COLOR_BGR2RGB))
    cap.release()
    return frames, fps


def detect(frames, landmarker):
    return lf.detect_hands(frames, landmarker, mode="image")


def predict(feats):
    tensor = feats[None, ...].astype(np.float32)
    logits, _ = eg._run_ensemble(tensor)
    e = np.exp(logits - logits.max())
    p = e / e.sum()
    top = np.argsort(-p)[:5]
    return {
        "prediction": eg._IDX_TO_LABEL[int(top[0])],
        "top5": [eg._IDX_TO_LABEL[int(i)] for i in top],
        "score": float(p[top[0]]),
        "margin": float(p[top[0]] - p[top[1]]),
    }


def build(lm, pr):
    """Production two-stage contract: 32 even samples, then 32 -> 24."""
    n = len(lm)
    idx = np.unique(np.linspace(0, n - 1, SAMPLED).astype(int)) if n > SAMPLED else np.arange(n)
    f, _ = build_features(lm[idx], pr[idx], expected_frames=STEPS)
    return f


def idle_frames(count):
    """Frames with no hand: what an idle camera produces."""
    return (np.zeros((count, 2, 21, 3), np.float32), np.zeros((count, 2), np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    entries = json.loads(Path(a.manifest).read_text())
    clips = Path(a.clips)
    eg._init_model()
    landmarker = lf.get_landmarker(mode="image")

    rows = []
    for i, e in enumerate(entries, 1):
        frames, fps = load_clip(clips / e["file"])
        if not frames:
            continue
        lm, pr = detect(frames, landmarker)
        n = len(frames)
        sign_ms = n / fps * 1000.0

        # The sign alone.
        base = predict(build(lm, pr))
        row = {
            "gloss": e["gloss"], "signer": e["signer"],
            "sign_ms": sign_ms, "sign_frames": n,
            "sign_coverage": float(pr.max(axis=1).mean()),
            "base": {**base, "correct": base["prediction"] == e["gloss"]},
        }

        # Idle padding on each side, as if the recording ran long.
        pads = {}
        for label, pre, post in (
            ("lead_1s", 1.0, 0.0), ("lead_2s", 2.0, 0.0), ("lead_3s", 3.0, 0.0),
            ("trail_1s", 0.0, 1.0), ("trail_2s", 0.0, 2.0), ("trail_3s", 0.0, 3.0),
            ("both_2s", 2.0, 2.0), ("both_4s", 4.0, 4.0),
        ):
            pre_n = int(round(pre * fps))
            post_n = int(round(post * fps))
            il, ip = idle_frames(pre_n + post_n)
            L = np.concatenate([il[:pre_n], lm, il[pre_n:]])
            P = np.concatenate([ip[:pre_n], pr, ip[pre_n:]])
            r = predict(build(L, P))
            total_frames = pre_n + n + post_n
            pads[label] = {
                **r, "correct": r["prediction"] == e["gloss"],
                "recording_ms": total_frames / fps * 1000.0,
                "coverage": float(P.max(axis=1).mean()),
                "sign_fraction": n / total_frames,
            }
        row["pads"] = pads
        rows.append(row)
        print(f"[{i}/{len(entries)}] {e['gloss']:14s} {e['signer']:4s} {sign_ms:5.0f}ms", flush=True)

    Path(a.out).write_text(json.dumps(rows, indent=1))

    ok = [r for r in rows]
    print(f"\n{len(ok)} clips")
    print(f"{'condition':12s} {'top1':>10s} {'top5':>10s} {'mean dur':>9s} {'sign frac':>10s}")
    b = sum(1 for r in ok if r["base"]["correct"])
    print(f"{'sign only':12s} {b:3d}/{len(ok):<6d} "
          f"{sum(1 for r in ok if r['gloss'] in r['base']['top5']):3d}/{len(ok):<6d} "
          f"{np.mean([r['sign_ms'] for r in ok]):8.0f}ms {1.0:10.2f}")
    for k in ("lead_1s","lead_2s","lead_3s","trail_1s","trail_2s","trail_3s","both_2s","both_4s"):
        h = sum(1 for r in ok if r["pads"][k]["correct"])
        t5 = sum(1 for r in ok if r["gloss"] in r["pads"][k]["top5"])
        d = np.mean([r["pads"][k]["recording_ms"] for r in ok])
        sf = np.mean([r["pads"][k]["sign_fraction"] for r in ok])
        print(f"{k:12s} {h:3d}/{len(ok):<6d} {t5:3d}/{len(ok):<6d} {d:8.0f}ms {sf:10.2f}")


if __name__ == "__main__":
    main()