"""A/B/C model comparison and segmentation re-check with the new ensemble.

A  V4.2 frozen four-member ensemble (the OLD binaries, preserved).
B  New four masked x,y members.
C  New nine-member masked x,y,z ensemble (the release candidate).

All three are evaluated on identical clips with identical features, masks and
segmentation, so the only variable is the model set.

Also re-runs the trimmed-versus-untrimmed segmentation comparison under C, since
the attention mask and z representation can change how idle frames affect the
result.

Usage:
  python abc_compare.py --clips DIR --manifest M --out O [--pre S --post S]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, "/tmp/eg-rc/.pynref")
import main as eg  # noqa: E402
from features import build_features  # noqa: E402
import landmark_frontend as lf  # noqa: E402

S, T = 32, 24
OLD_DIR = Path("/tmp/eg-old-models")


def build(lm, pr, use_z):
    n = len(lm)
    if n < 1:
        return None, None
    idx = np.unique(np.linspace(0, n - 1, S).astype(int)) if n > S else np.arange(n)
    f, mask = build_features(lm[idx], pr[idx], expected_frames=T, use_z=use_z)
    return f, mask


def run(interps, feats_by_width, mask):
    """Mean logits across the given members, each fed its own width."""
    logs = []
    for interp, (in_idx, out_idx, mask_idx), use_z in interps:
        f = feats_by_width[use_z]
        interp.set_tensor(in_idx, f[None, ...].astype(np.float32))
        # mask_idx < 0 means this member has no mask input (the old bundle).
        if mask_idx >= 0:
            interp.set_tensor(mask_idx, mask[None, ...].astype(np.float32))
        interp.invoke()
        logs.append(interp.get_tensor(out_idx)[0].astype(np.float32))
    avg = np.mean(logs, axis=0)
    e = np.exp(avg - avg.max())
    p = e / e.sum()
    top = np.argsort(-p)[:5]
    return {
        "prediction": eg._IDX_TO_LABEL[int(top[0])],
        "top5": [eg._IDX_TO_LABEL[int(i)] for i in top],
        "score": float(p[top[0]]),
        "margin": float(p[top[0]] - p[top[1]]),
    }


def motion_trim(lm, pr, fps, frac=0.25, pad_s=0.5, pct=95):
    n = len(lm)
    d = np.zeros(n, np.float32)
    for i in range(1, n):
        if pr[i - 1].max() > 0.5 and pr[i].max() > 0.5:
            d[i] = float(np.abs(lm[i] - lm[i - 1]).mean())
    moved = d[d > 0]
    if moved.size == 0:
        return 0, n
    ref = float(np.percentile(moved, pct))
    a = np.where(d >= ref * frac)[0]
    if a.size == 0:
        return 0, n
    pad = int(round(pad_s * fps))
    s, e = max(0, int(a[0]) - pad), min(n, int(a[-1]) + pad + 1)
    if e - s < 0.10 * n:
        return 0, n
    return s, e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pre", type=float, default=0.0)
    ap.add_argument("--post", type=float, default=0.0)
    a = ap.parse_args()

    eg._init_model()
    all_members = list(zip(eg._INTERPRETERS, eg._MEMBER_IO, eg._MEMBER_USE_Z))
    by_width = [m for m in all_members if not m[2]]
    z_width = [m for m in all_members if m[2]]
    B = by_width                      # four masked x,y
    C = all_members                   # nine
    A = load_old_members()

    lmrk = lf.get_landmarker(mode="image")
    entries = json.loads(Path(a.manifest).read_text())
    rows = []

    for e in entries:
        cap = cv2.VideoCapture(str(Path(a.clips) / e["file"]))
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
        if a.pre or a.post:
            k1 = int(round(a.pre * fps)); k2 = int(round(a.post * fps))
            lm = np.concatenate([np.zeros((k1, 2, 21, 3), np.float32), lm,
                                 np.zeros((k2, 2, 21, 3), np.float32)])
            pr = np.concatenate([np.zeros((k1, 2), np.float32), pr,
                                 np.zeros((k2, 2), np.float32)])

        row = {"gloss": e["gloss"], "signer": e["signer"], "modes": {}}
        for seg_name, pick in (("full", lambda: (0, len(lm))),
                               ("trim", lambda: motion_trim(lm, pr, fps))):
            s, en = pick()
            fxy, mask = build(lm[s:en], pr[s:en], False)
            fz, _ = build(lm[s:en], pr[s:en], True)
            widths = {False: fxy, True: fz}
            for name, members in (("A", A), ("B", B), ("C", C)):
                try:
                    r = run(members, widths, mask)
                except Exception as exc:  # noqa: BLE001
                    r = {"prediction": f"ERR:{type(exc).__name__}", "top5": [], "score": 0.0, "margin": 0.0}
                r["correct"] = r["prediction"] == e["gloss"]
                row["modes"][f"{seg_name}_{name}"] = r
        rows.append(row)
        print(f"{e['gloss']:14s} {e['signer']:4s} n={len(lm)}", flush=True)

    Path(a.out).write_text(json.dumps(rows, indent=1))
    print(f"\n{len(rows)} clips  (idle {a.pre}s/{a.post}s)")
    for k in ("full_A", "full_B", "full_C", "trim_A", "trim_B", "trim_C"):
        h = sum(1 for r in rows if r["modes"][k]["correct"])
        t5 = sum(1 for r in rows if r["gloss"] in r["modes"][k]["top5"])
        print(f"  {k:9s} top1 {h:2d}/{len(rows)}  top5 {t5:2d}/{len(rows)}")


def load_old_members():
    """The V4.2 four-member bundle, read from the preserved old binaries."""
    from ai_edge_litert.interpreter import Interpreter
    out = []
    for name in ("landmark_ens_tcn21_fp16.tflite", "landmark_ens_tcnm61_fp16.tflite",
                 "landmark_ens_tcnm62_fp16.tflite", "landmark_ens_rob71_fp16.tflite"):
        path = OLD_DIR / name
        if not path.exists():
            raise SystemExit(f"old bundle missing {path}; extract it from V4.2 first")
        it = Interpreter(model_path=str(path))
        it.allocate_tensors()
        dets = it.get_input_details(); outs = it.get_output_details()
        f_idx = next(i["index"] for i in dets if "features" in i["name"])
        m_idx = next((i["index"] for i in dets if "mask" in i["name"]), -1)
        out.append((it, (f_idx, outs[0]["index"], m_idx), False))
    return out


if __name__ == "__main__":
    main()