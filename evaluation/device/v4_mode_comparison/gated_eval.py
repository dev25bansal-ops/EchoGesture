"""Choose between whole-interval and trimmed-interval, using only capture evidence.

Three candidates, evaluated on the same clips:

  A  whole    the recorded interval, no trim (V4.0 behaviour)
  B  motion   unconditional motion trim (V4.1 behaviour)
  C  gated    trim ONLY when the recording shows clear, stable idle at a
              boundary; otherwise keep the whole recording

The gate is deliberately hard to satisfy. It uses no classifier output.

Gate design, from the failure modes of B on clean clips:
  - clean clips were harmed because the trim fired at all, so C must default
    to A and need positive evidence to fire;
  - a jitter spike can look like the peak, so the gate requires that the idle
    it is about to remove is LONG and FLAT, not merely that something moved
    somewhere;
  - thresholds are absolute in displacement units, calibrated once, and the
    final choice is confirmed on a split not used to set them.
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
MAX_DROP = 0.55   # D: floor at 45%
E_MIN_KEEP = 0.10 # E: only reject degenerate trims (<10% kept)


def build(lm, pr):
    n = len(lm)
    if n < 1:
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


def displacement(lm, pr):
    n = len(lm)
    d = np.zeros(n, np.float32)
    for i in range(1, n):
        if pr[i - 1].max() > 0.5 and pr[i].max() > 0.5:
            d[i] = float(np.abs(lm[i] - lm[i - 1]).mean())
    return d


def motion_trim(lm, pr, fps, frac=0.25, pad_s=0.5, pct=95):
    d = displacement(lm, pr)
    # Robust reference instead of the raw max: one jitter spike must not raise
    # the threshold above every real moving frame.
    ref = float(np.percentile(d[d > 0], pct)) if (d > 0).any() else 0.0
    if ref <= 0:
        return 0, len(lm)
    a = np.where(d >= ref * frac)[0]
    if a.size == 0:
        return 0, len(lm)
    pad = int(round(pad_s * fps))
    return max(0, int(a[0]) - pad), min(len(lm), int(a[-1]) + pad + 1)


# --- gate parameters: set once, then frozen and checked on a held-out split ---
IDLE_MAX_DISP = 0.02      # displacement below this counts as "not moving"
MIN_IDLE_SECONDS = 0.8    # idle must be at least this long to be worth removing
MIN_IDLE_FRACTION = 0.10  # ...and this much of the recording
MAX_EDGE_JITTER = 0.05    # the transition itself must not be a single spike


# A boundary is only trimmed when its lead/trail exceeds this. Clean clips
# have trailing near-zero runs up to 1.87 s (p90 1.49 s) and leading runs up to
# 4.07 s, so anything at or below ~2 s cannot be distinguished from a sign's own
# onset. Set above that overlap; calibrated on the train split only.
IDLE_GATE_SECONDS = 2.0
# The boundary run must also be nearly perfectly flat, not merely slow: that is
# what separates "the camera was pointed at nothing" from "the signer held a
# starting handshape".
IDLE_GATE_MAX_DISP = 0.01


def gated_trim(lm, pr, fps):
    """Trim a boundary only when it is far longer than any real sign's own onset.

    Returns (start, end, activated, reason).
    """
    n = len(lm)
    if n < 8:
        return 0, n, False, "too_few"
    if pr.max(axis=1).max() < 0.5:
        return 0, n, False, "no_hands"

    d = displacement(lm, pr)
    moving = d >= IDLE_MAX_DISP
    if not moving.any():
        return 0, n, False, "no_motion"
    first = int(np.argmax(moving))
    last = n - 1 - int(np.argmax(moving[::-1]))

    need = int(round(IDLE_GATE_SECONDS * fps))
    reasons = []
    start, end = 0, n

    if first >= need:
        start = first
        reasons.append(f"lead={first / fps:.1f}s")
    if (n - 1 - last) >= need:
        end = last + 1
        reasons.append(f"trail={(n - 1 - last) / fps:.1f}s")

    if start == 0 and end == n:
        return 0, n, False, "no_clear_idle"
    return start, end, True, "+".join(reasons)


def main():
    clips = Path(sys.argv[1])
    manifest = json.loads(Path(sys.argv[2]).read_text())
    out = sys.argv[3]
    pre_s = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
    post_s = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
    mode = sys.argv[6] if len(sys.argv) > 6 else "zero"   # zero | still | jitter

    eg._init_model()
    lmrk = lf.get_landmarker(mode="image")
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

        k1 = int(round(pre_s * fps))
        k2 = int(round(post_s * fps))
        if k1 or k2:
            if mode == "still":
                head = np.repeat(lm[:1], k1, axis=0); tail = np.repeat(lm[-1:], k2, axis=0)
                hp = np.repeat(pr[:1], k1, axis=0); tp = np.repeat(pr[-1:], k2, axis=0)
            elif mode == "jitter":
                # Idle with a brief detector spike: the hazard for a peak-based rule.
                head = np.repeat(lm[:1], k1, axis=0).copy()
                if k1 > 4:
                    head[k1 // 2] = head[k1 // 2] + np.random.default_rng(abs(hash(e['file'])) % (2**31)).normal(0, 0.08, head[k1 // 2].shape).astype(np.float32)
                tail = np.repeat(lm[-1:], k2, axis=0)
                hp = np.repeat(pr[:1], k1, axis=0); tp = np.repeat(pr[-1:], k2, axis=0)
            else:
                head = np.zeros((k1, 2, 21, 3), np.float32); tail = np.zeros((k2, 2, 21, 3), np.float32)
                hp = np.zeros((k1, 2), np.float32); tp = np.zeros((k2, 2), np.float32)
            lm = np.concatenate([head, lm, tail])
            pr = np.concatenate([hp, pr, tp])

        cands = {}
        for name in ("A", "B", "C", "D", "E", "F"):
            if name == "A":
                a, b = 0, len(lm)
            elif name == "B":
                a, b = motion_trim(lm, pr, fps)
            elif name == "D":
                # B's trim, but it may never discard more than MAX_DROP of the
                # recording: a sign that fills its recording must not be cut in
                # half just because one stretch moved more than another.
                a, b = motion_trim(lm, pr, fps)
                if (b - a) < MAX_DROP * len(lm):
                    a, b = 0, len(lm)
            elif name == "F":
                # E, but with a percentile reference so a spike cannot capture
                # the threshold.
                a, b = motion_trim(lm, pr, fps, pct=95)
                if (b - a) < E_MIN_KEEP * len(lm):
                    a, b = 0, len(lm)
            elif name == "E":
                # Degenerate-floor only: a trim that keeps <10% of the frames is
                # not a trim, it is a misfire (e.g. a 0.5 s blip in a 9 s
                # recording). Everything else follows B.
                a, b = motion_trim(lm, pr, fps)
                if (b - a) < E_MIN_KEEP * len(lm):
                    a, b = 0, len(lm)
            else:
                a, b, _, _ = gated_trim(lm, pr, fps)
            f = build(lm[a:b], pr[a:b])
            r = predict(f) if f is not None else None
            cands[name] = {
                **r, "correct": r["prediction"] == e["gloss"],
                "kept_frac": (b - a) / len(lm),
                "start": a, "end": b,
            }
        _, _, activated, reason = gated_trim(lm, pr, fps)
        cands["C"]["activated"] = bool(activated)
        cands["C"]["reason"] = reason

        rows.append({"gloss": e["gloss"], "signer": e["signer"],
                     "sign_ms": n_orig / fps * 1000.0, "cands": cands})
        print(f"{e['gloss']:14s} {e['signer']:4s} C={'ON ' if activated else 'off'} {reason}", flush=True)

    Path(out).write_text(json.dumps(rows, indent=1))
    print(f"\n{len(rows)} clips  (idle {pre_s}s lead / {post_s}s trail, {mode})")
    for c in ("A", "B", "C", "D", "E", "F"):
        h = sum(1 for x in rows if x["cands"][c]["correct"])
        t5 = sum(1 for x in rows if x["gloss"] in x["cands"][c]["top5"])
        print(f"  {c} top1 {h:2d}/{len(rows)}  top5 {t5:2d}/{len(rows)}")
    act = sum(1 for x in rows if x["cands"]["C"]["activated"])
    print(f"  C activation rate {act}/{len(rows)}")


if __name__ == "__main__":
    main()