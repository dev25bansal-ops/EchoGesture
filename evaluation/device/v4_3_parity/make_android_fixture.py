"""Regenerate the Android parity fixture for the nine-member ensemble.

The fixture format is fixed by `TfliteEnsembleInferenceTest.kt`:

    line 0            "<members> <classes>"
    line 1            space-separated member basenames, in meta order
    lines 2..2+M-1    one row per member: its raw logits
    line 2+M          averaged logits
    line 3+M          softmax probabilities
    line 4+M          top-5 indices
    line 5+M          top-5 labels
    line 6+M          confidence
    lines 7+M ...     "features <24*255 floats>" under a 255-wide header line

The previous fixture declared four members, so the two tests that index by
member count failed against the nine-member app.

Usage:
    python evaluation/device/v4_3_parity/make_android_fixture.py --clip PATH --out FILE
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import main as M  # noqa: E402


def main_() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    M._init_model()
    names = list(M._MEMBER_NAMES)
    if len(names) != 9:
        raise SystemExit(f"expected 9 members, loaded {len(names)}")

    from landmark_frontend import read_frames, detect_hands
    from features import build_features

    frames = read_frames(args.clip, max_frames=max(32, M._FRAMES))
    if not frames:
        raise SystemExit("no frames decoded")
    lm, present = detect_hands(frames, mode="image")
    feats, mask = build_features(lm, present, expected_frames=M._FRAMES, use_z=False)

    # Per-member raw logits through the same interpreter handles predict() uses.
    per_member = []
    for interp, (in_idx, out_idx, mask_idx, *_rest), use_z in zip(
            M._INTERPRETERS, M._MEMBER_IO, M._MEMBER_USE_Z):
        f, m = build_features(lm, present, expected_frames=M._FRAMES, use_z=use_z)
        interp.set_tensor(in_idx, f[None, ...].astype(np.float32))
        if mask_idx >= 0:
            interp.set_tensor(mask_idx, m[None, ...].astype(np.float32))
        interp.invoke()
        per_member.append(interp.get_tensor(out_idx)[0].astype(np.float32))

    avg = np.mean(per_member, axis=0)
    from features import feature_dim  # noqa: F401  (documented contract)
    exp = np.exp(avg - avg.max())
    probs = exp / exp.sum()
    order = np.argsort(-probs)[:5]
    labels = M._IDX_TO_LABEL
    top_labels = [labels[int(i)] for i in order]

    def fmt(a):
        return " ".join(f"{v:.9g}" for v in np.asarray(a, dtype=np.float32))

    # Line layout is fixed by TfliteEnsembleInferenceTest.kt:
    #   1  "<members> <classes>"
    #   2  member basenames in meta order
    #   3.. one raw-logit row per member
    #   +1 averaged logits, +1 probabilities, +1 top-5 indices, +1 top-5 labels,
    #   +1 confidence, +1 the full 100-label vocabulary (header), +1 features
    lines = [f"{len(names)} {len(avg)}", " ".join(names)]
    lines += [fmt(row) for row in per_member]
    lines.append(fmt(avg))
    lines.append(fmt(probs))
    lines.append(" ".join(str(int(i)) for i in order))
    lines.append(" ".join(top_labels))
    lines.append(f"{float(probs[order[0]]):.9g}")
    lines.append(" ".join(labels[i] for i in range(len(avg))))
    lines.append("features " + fmt(feats.reshape(-1)))
    feats_xyz, _ = build_features(lm, present, expected_frames=M._FRAMES, use_z=True)
    # The reference logits above were produced with the CLIP'S REAL presence
    # mask. A test that feeds an all-ones mask gets a different answer, so the
    # mask is emitted here rather than left for the test to invent.
    lines.append("mask " + fmt(mask.reshape(-1)))
    lines.append("features_xyz " + fmt(feats_xyz.reshape(-1)))
    lines.append(f"# members {len(names)} classes {len(avg)}")
    lines.append(f"# producer: evaluation/device/v4_3_parity/make_android_fixture.py")
    lines.append(f"# clip {args.clip}")

    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out}")
    print(f"  members={len(names)} classes={len(avg)} top1={top_labels[0]} "
          f"conf={float(probs[order[0]]):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_())