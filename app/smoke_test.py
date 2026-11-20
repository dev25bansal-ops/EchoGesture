"""Container entrypoint: prove the shipped pipeline actually runs.

A synthetic rectangle is enough to prove the decode->detect path executed,
because predict() REFUSES a clip with no hand (src/main.py's
_require_hand_coverage, MIN_HAND_FRACTION=0.15). That refusal is a positive
signal here: it can only fire after video decoding and MediaPipe detection have
both really run. So the smoke test asserts that exact ValueError, then exercises
the remaining stage — all nine interpreters — directly with a landmark array.

With a path argument it instead runs predict() on that video and prints the
returned JSON.

Exit status is 0 only if every stage it claims to check actually ran.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

# The script lives in app/ in the repo but at the container root beside src/,
# so resolve src/ relative to whichever directory actually contains it.
_here = Path(__file__).resolve().parent
for _cand in (_here / "src", _here.parent / "src"):
    if _cand.is_dir():
        sys.path.insert(0, str(_cand))
        break
else:
    raise SystemExit(f"cannot find src/ next to {_here} or its parent")

REQUIRED_KEYS = {"clip_id", "prediction", "top5", "confidence", "latency_ms"}


def synthesise_clip(path: Path, seconds: float = 3.0, fps: int = 24) -> None:
    """Write a short mp4 of a moving high-contrast rectangle (no hand in it)."""
    import cv2

    w, h = 320, 240
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not vw.isOpened():
        raise SystemExit(
            "could not open VideoWriter for the synthetic clip; this container's "
            "OpenCV cannot encode mp4v, so the smoke test cannot run here")
    n = int(seconds * fps)
    for i in range(n):
        frame = np.full((h, w, 3), 40, np.uint8)
        cx = 60 + (i * 160) // max(n, 1)
        cv2.rectangle(frame, (cx, 80), (cx + 60, 160), (230, 230, 230), -1)
        vw.write(frame)
    vw.release()


def _run_ensemble(M, lm, present):
    """Push one landmark array through every member; return (label, conf, logits).

    This is the shipped path's second half — the same feature build and the same
    mean-logit-then-one-softmax aggregation main.predict uses.
    """
    widths = {}
    for uz in set(M._MEMBER_USE_Z) or {False}:
        widths[uz] = M.build_features(lm, present, expected_frames=M._FRAMES, use_z=uz)

    logits = []
    for interp, (in_idx, out_idx, mask_idx), uz in zip(
            M._INTERPRETERS, M._MEMBER_IO, M._MEMBER_USE_Z):
        feats, mask = widths[uz]
        interp.set_tensor(in_idx, feats[None, ...].astype(np.float32))
        if mask_idx >= 0:
            interp.set_tensor(mask_idx, mask[None, ...].astype(np.float32))
        interp.invoke()
        logits.append(interp.get_tensor(out_idx)[0].astype(np.float32))

    mean_logits = np.mean(logits, axis=0)
    e = np.exp(mean_logits - mean_logits.max())
    probs = e / e.sum()
    idx = int(probs.argmax())
    return M._IDX_TO_LABEL[idx], float(probs[idx]), len(logits)


def _synthetic_landmarks():
    """A moving two-hand landmark tensor: (24, 2, 21, 3).

    Not a real sign. It exists only to make every interpreter allocate, be fed,
    and return a finite logit vector — the stage the hand-less clip cannot reach.
    """
    T = 24
    lm = np.zeros((T, 2, 21, 3), np.float32)
    present = np.zeros((T, 2), np.float32)
    for t in range(T):
        for hand in range(2):
            phase = 0.25 * np.sin(2 * np.pi * t / T + hand)
            base = np.array([0.5 + 0.2 * hand, 0.5 + phase, 0.0], np.float32)
            for j in range(21):
                ang = 2 * np.pi * j / 21
                lm[t, hand, j] = base + np.array(
                    [0.05 * np.cos(ang), 0.05 * np.sin(ang), 0.01 * np.sin(ang)],
                    np.float32)
            present[t, hand] = 1.0
    return lm, present


def stage_detector(path: Path) -> None:
    """Decode + MediaPipe must really run; a hand-less clip must be refused."""
    import main as M

    try:
        M.predict(str(path))
    except ValueError as e:
        if "No hand detected" in str(e):
            print(f"  [1/2] decode + MediaPipe ran; refused the hand-less clip "
                  f"as designed")
            return
        raise SystemExit(f"unexpected ValueError from predict(): {e}")
    except FileNotFoundError as e:
        raise SystemExit(f"model/label artifact missing: {e}")
    except Exception as e:
        raise SystemExit(f"predict() raised {type(e).__name__}: {e}")
    # A hand WAS detected in a rectangle clip, which would be surprising but is
    # not a failure of this container — report it rather than pretend.
    print("  [1/2] decode + MediaPipe ran (a hand was detected in the synthetic "
          "clip; unusual, but the detector is live)")


def stage_ensemble() -> None:
    """All nine interpreters must invoke and return finite logits."""
    import main as M

    lm, present = _synthetic_landmarks()
    try:
        label, conf, n = _run_ensemble(M, lm, present)
    except Exception as e:
        raise SystemExit(f"ensemble stage raised {type(e).__name__}: {e}")
    if n != len(M._INTERPRETERS):
        raise SystemExit(f"only {n} of {len(M._INTERPRETERS)} members ran")
    if not np.isfinite(conf):
        raise SystemExit(f"ensemble produced a non-finite confidence: {conf}")
    print(f"  [2/2] {n} interpreters invoked; argmax={label} conf={conf:.4f} "
          f"(argmax is meaningless on synthetic landmarks)")


def run_file(clip: Path) -> int:
    import main as M

    if not clip.exists():
        print(f"error: no such file: {clip}", file=sys.stderr)
        return 2
    try:
        result = M.predict(str(clip))
    except Exception as e:
        print(f"predict() raised {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    missing = REQUIRED_KEYS - set(result)
    if missing:
        print(f"predict() result missing keys: {sorted(missing)}\n{result}",
              file=sys.stderr)
        return 1
    if not np.isfinite(float(result["latency_ms"])):
        print(f"latency_ms is not finite: {result['latency_ms']!r}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nOK — predict() returned a well-formed result in "
          f"{float(result['latency_ms']):.1f} ms")
    return 0


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return 0

    if len(sys.argv) > 1:
        return run_file(Path(sys.argv[1]))

    print("EchoGesture container smoke test")
    tmpdir = Path(tempfile.mkdtemp())
    clip = tmpdir / "synthetic_smoke.mp4"
    synthesise_clip(clip)
    stage_detector(clip)
    stage_ensemble()
    print("\nOK — decode, MediaPipe detection, feature build and all ensemble "
          "members executed inside this container.")
    print("Accuracy is NOT measured here: that needs the ASL Citizen corpus, "
          "which is non-redistributable research data and is not in this image.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
