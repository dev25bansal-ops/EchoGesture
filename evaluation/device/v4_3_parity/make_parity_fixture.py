"""Dump the Python reference's features, masks and per-member logits.

Kotlin cannot call Python, so the parity test does the comparison the other way
round: this script writes a small binary fixture (landmark arrays in, feature
tensors, masks and logits out) and the Android test repeats the same computation
with the shipped Kotlin code and compares numbers.

Fixture layout, little-endian:
  int32   clipCount
  per clip:
    int32 srcFrames
    float32 present[srcFrames*2]
    float32 landmarks[srcFrames*2*21*3]
Then per clip, per width (255 then 381):
    float32 features[24*dim]
    float32 mask[24]
Then per clip, per member:
    float32 logits[100]
"""

import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/tmp/eg-rc/.pynref")
import main as eg  # noqa: E402
from features import build_features, feature_dim  # noqa: E402

OUT = Path("/tmp/eg-v4/parity_fixture.bin")


def main():
    manifest = json.loads(Path(sys.argv[1]).read_text())
    clips = Path(sys.argv[2])
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 6

    import cv2
    import landmark_frontend as lf

    eg._init_model()
    lmrk = lf.get_landmarker(mode="image")

    chosen = []
    for e in manifest:
        if len(chosen) >= limit:
            break
        cap = cv2.VideoCapture(str(clips / e["file"]))
        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(cv2.resize(f, (256, 256)), cv2.COLOR_BGR2RGB))
        cap.release()
        if len(frames) < 8:
            continue
        lm, pr = lf.detect_hands(frames, lmrk, mode="image")
        if pr.max() < 0.5:
            continue
        # Keep the fixture small: at most 200 source frames.
        if len(lm) > 200:
            idx = np.unique(np.linspace(0, len(lm) - 1, 200).astype(int))
            lm, pr = lm[idx], pr[idx]
        chosen.append({"file": e["file"], "gloss": e["gloss"], "lm": lm, "pr": pr})
        print(f"  {e['gloss']:14s} {e['signer']:4s} frames={len(lm)}", flush=True)

    if not chosen:
        raise SystemExit("no usable clips")

    rows = []
    for c in chosen:
        lm, pr = c["lm"], c["pr"]
        xy, mask = build_features(lm, pr, expected_frames=24, use_z=False)
        xyz, mask_z = build_features(lm, pr, expected_frames=24, use_z=True)
        assert np.allclose(mask, mask_z), "mask must not depend on the z flag"
        assert xy.shape == (24, 255), xy.shape
        assert xyz.shape == (24, 381), xyz.shape

        # Per-member logits through the reference interpreter path.
        logits = []
        for interp, (in_idx, out_idx, mask_idx), use_z in zip(
            eg._INTERPRETERS, eg._MEMBER_IO, eg._MEMBER_USE_Z
        ):
            feats = xyz if use_z else xy
            interp.set_tensor(in_idx, feats[None, ...].astype(np.float32))
            interp.set_tensor(mask_idx, mask[None, ...].astype(np.float32))
            interp.invoke()
            logits.append(interp.get_tensor(out_idx)[0].astype(np.float32))

        rows.append({
            "gloss": c["gloss"],
            "lm": lm, "pr": pr, "xy": xy, "xyz": xyz, "mask": mask,
            "logits": logits,
        })

    with OUT.open("wb") as f:
        f.write(struct.pack("<i", len(rows)))
        for r in rows:
            n = len(r["lm"])
            f.write(struct.pack("<i", n))
            f.write(r["pr"].astype("<f4").tobytes())
            f.write(r["lm"].astype("<f4").tobytes())
        for r in rows:
            f.write(r["xy"].astype("<f4").tobytes())
            f.write(r["xyz"].astype("<f4").tobytes())
            f.write(r["mask"].astype("<f4").tobytes())
        for r in rows:
            for lg in r["logits"]:
                f.write(lg.astype("<f4").tobytes())

    # A human-readable summary too, for the report.
    summary = []
    for r in rows:
        avg = np.mean(r["logits"], axis=0)
        e = np.exp(avg - avg.max())
        p = e / e.sum()
        top = np.argsort(-p)[:5]
        top5 = [eg._IDX_TO_LABEL[int(i)] for i in top]
        summary.append({
            "gloss": r["gloss"],
            "top1": top5[0],
            "top5": top5,
            "confidence": float(p[top[0]]),
            "correct": top5[0] == r["gloss"],
        })
    Path("/tmp/eg-v4/parity_python_summary.json").write_text(json.dumps(summary, indent=1))

    print(f"\nwrote {OUT} ({OUT.stat().st_size} bytes) for {len(rows)} clips")
    for s in summary:
        print(f"  {s['gloss']:14s} -> {s['top1']:14s} {s['confidence']:.3f} "
              f"{'OK' if s['correct'] else 'WRONG'}")


if __name__ == "__main__":
    main()