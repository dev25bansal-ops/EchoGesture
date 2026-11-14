"""Evaluate sign-bearing-segment selection on labelled clips.

The problem: a recording bracketed by the signer's presses can contain idle
seconds on either side. Spreading 32 samples across the whole interval then
dilutes the motion features and can crop a long sign's onset.

The rule tested here chooses a sub-interval using ONLY information a phone
has — hand presence and landmark movement — never the model's confidence
(using confidence would be selecting the answer by how much the model likes
itself, which is not evidence).

Rules compared:
  full     no segmentation: the whole recording (V4 today)
  presence trim to the first..last frame with a hand, plus padding
  motion   trim to where landmark displacement exceeds a threshold,
           expanded to cover the sign, then the same padding
  energy   fixed-length window centred on peak motion energy

Measured against ground truth on labelled clips, including how often each rule
accidentally removes part of the real sign.
"""

import json
import sys
from pathlib import Path

import numpy as np
import cv2

ROOT = Path("/tmp/eg-rc")
sys.path.insert(0, str(ROOT / "src"))
import main as eg  # noqa: E402
from features import build_features  # noqa: E402
import landmark_frontend as lf  # noqa: E402

S, T = 32, 24
STILL_HANDS = False


def build(lm, pr):
    n = len(lm)
    if n == 0:
        return None
    idx = np.unique(np.linspace(0, n - 1, S).astype(int)) if n > S else np.arange(n)
    f, _ = build_features(lm[idx], pr[idx], expected_frames=T)
    return f


def predict(f):
    logits, _ = eg._run_ensemble(f[None, ...].astype(np.float32))
    e = np.exp(logits - logits.max())
    p = e / e.sum()
    t = np.argsort(-p)[:5]
    return {
        "prediction": eg._IDX_TO_LABEL[int(t[0])],
        "top5": [eg._IDX_TO_LABEL[int(i)] for i in t],
        "score": float(p[t[0]]),
        "margin": float(p[t[0]] - p[t[1]]),
    }


def per_frame_displacement(lm, pr, fps):
    """Mean landmark movement between consecutive frames where both have a hand."""
    n = len(lm)
    d = np.zeros(n, np.float32)
    for i in range(1, n):
        if pr[i - 1].max() > 0.5 and pr[i].max() > 0.5:
            d[i] = float(np.abs(lm[i] - lm[i - 1]).mean())
    return d


def trim_presence(lm, pr, fps, pad_s=0.4):
    """First..last frame with a hand, plus padding so onset/offset survive."""
    has = pr.max(axis=1) > 0.5
    if not has.any():
        return 0, len(lm)
    pad = int(round(pad_s * fps))
    a = max(0, int(np.argmax(has)) - pad)
    b = min(len(lm), len(lm) - int(np.argmax(has[::-1])) + pad)
    return a, b


def trim_motion(lm, pr, fps, thresh_frac=0.25, pad_s=0.5):
    """Trim to where landmarks actually move, then pad to cover the sign.

    The threshold is relative to this clip's own motion so a slow signer is
    not trimmed to nothing and a fast one is not over-trimmed.
    """
    d = per_frame_displacement(lm, pr, fps)
    if d.max() <= 0:
        return 0, len(lm)
    thr = d.max() * thresh_frac
    active = np.where(d >= thr)[0]
    if active.size == 0:
        return 0, len(lm)
    pad = int(round(pad_s * fps))
    a = max(0, int(active[0]) - pad)
    b = min(len(lm), int(active[-1]) + pad + 1)
    return a, b


def window_energy(lm, pr, fps, win_s=3.0):
    """Fixed-length window centred on the peak of smoothed motion energy."""
    d = per_frame_displacement(lm, pr, fps)
    k = max(1, int(round(0.2 * fps)))
    sm = np.convolve(d, np.ones(k) / k, mode="same")
    w = int(round(win_s * fps))
    if len(lm) <= w:
        return 0, len(lm)
    peak = int(np.argmax(sm))
    a = max(0, min(peak - w // 2, len(lm) - w))
    return a, a + w


def main():
    clips = Path(sys.argv[1])
    manifest = json.loads(Path(sys.argv[2]).read_text())
    out = sys.argv[3]
    # Optional idle padding, to test the condition the rule exists for.
    pre_s = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
    post_s = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
    global STILL_HANDS
    STILL_HANDS = (sys.argv[6] == "still") if len(sys.argv) > 6 else False

    eg._init_model()
    lmrk = lf.get_landmarker(mode="image")

    rules = {
        "full": None,
        "presence": trim_presence,
        "motion": trim_motion,
        "energy": window_energy,
    }
    rows = []
    for e in manifest:
        cap = cv2.VideoCapture(str(clips / e["file"]))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        fr = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            fr.append(cv2.cvtColor(cv2.resize(f, (256, 256)), cv2.COLOR_BGR2RGB))
        cap.release()
        if not fr:
            continue
        lm, pr = lf.detect_hands(fr, lmrk, mode="image")
        n_orig = len(lm)
        k1 = 0; k2 = 0
        if pre_s or post_s:
            k1 = int(round(pre_s * fps)); k2 = int(round(post_s * fps))
            if STILL_HANDS:
                # Hands in frame but not moving: the signer waiting with hands up.
                head = np.repeat(lm[:1], k1, axis=0)
                tail = np.repeat(lm[-1:], k2, axis=0)
                hp = np.repeat(pr[:1], k1, axis=0)
                tp = np.repeat(pr[-1:], k2, axis=0)
                lm = np.concatenate([head, lm, tail])
                pr = np.concatenate([hp, pr, tp])
            else:
                lm = np.concatenate([np.zeros((k1, 2, 21, 3), np.float32), lm,
                                     np.zeros((k2, 2, 21, 3), np.float32)])
                pr = np.concatenate([np.zeros((k1, 2), np.float32), pr,
                                     np.zeros((k2, 2), np.float32)])
        n = len(lm)

        row = {"gloss": e["gloss"], "signer": e["signer"], "frames": n,
               "fps": fps, "dur_ms": n / fps * 1000.0, "rules": {}}
        for name, fn in rules.items():
            a, b = (0, n) if fn is None else fn(lm, pr, fps)
            if b - a < 2:
                a, b = 0, n
            f = build(lm[a:b], pr[a:b])
            if f is None:
                continue
            r = predict(f)
            row["rules"][name] = {
                **r, "correct": r["prediction"] == e["gloss"],
                "start": a, "end": b,
                "kept_ms": (b - a) / fps * 1000.0,
                "kept_frac": (b - a) / n,
                # Did the rule cut into the real sign? The sign occupies
                # [k1, k1+n_orig) after padding.
                "cut_onset": a > k1,
                "cut_offset": b < k1 + n_orig,
            }
        rows.append(row)
        print(f"{e['gloss']:14s} {e['signer']:4s} {n:4d}f", flush=True)

    Path(out).write_text(json.dumps(rows, indent=1))
    print(f"\n{len(rows)} clips")
    print(f"{'rule':10s} {'top1':>9s} {'top5':>9s} {'mean kept':>10s}")
    for name in rules:
        h = sum(1 for x in rows if x["rules"].get(name, {}).get("correct"))
        t5 = sum(1 for x in rows if x["gloss"] in x["rules"].get(name, {}).get("top5", []))
        k = np.mean([x["rules"][name]["kept_frac"] for x in rows if name in x["rules"]])
        print(f"{name:10s} {h:3d}/{len(rows):<5d} {t5:3d}/{len(rows):<5d} {k:9.2f}")


if __name__ == "__main__":
    main()