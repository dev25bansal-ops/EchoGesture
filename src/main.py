"""
EchoGesture: Sign Language Prediction on Edge Device
Track 2 — Samsung XR Hackathon 2026 (Samsung R&D Institute India, Bangalore)

Fully on-device, exocentric sign language translation pipeline.
Evaluator contract:
    def predict(video_path: str) -> dict
"""

import os
import sys
import time
import json
from pathlib import Path
import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from features import build_features, FEATURE_DIM  # noqa: E402
from landmark_frontend import clip_to_landmarks  # noqa: E402

# Universal TFLite interpreter loader
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
        Interpreter = tflite.Interpreter
    except ImportError:
        try:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter
        except ImportError:
            Interpreter = None

# Global model state (singleton pattern to avoid reloading per clip).
# The shipped artifact is an ENSEMBLE: several independently-trained models
# whose logits are averaged. Each member makes different errors on unseen
# signers, and the mean cancels part of that variance - measured 70.22% top-1
# versus 67.65% for the best single member, on 11 held-out signers.
_INTERPRETERS = []
_MEMBER_IO = []
# Per member: does it take the 381-wide x,y,z features? Legacy members do not.
_MEMBER_USE_Z = []
_INPUT_DETAILS = None
_OUTPUT_DETAILS = None
_INPUT_SCALE = 1.0
_INPUT_ZP = 0
_OUTPUT_SCALE = 1.0
_OUTPUT_ZP = 0
_IDX_TO_LABEL = {}
_FRAMES = 24
# Single flag set only after a FULL successful load. `if _INTERPRETERS: return`
# was wrong: the load loop appends as it goes, so a member that fails after the
# first left a non-empty but half-built state, and the retry short-circuited into
# a predict() with no per-member tensor indices. See _init_model below.
_INITIALIZED = False

# Mirrors CameraScreen.kt's MIN_HAND_FRACTION. The app refuses to classify below
# this share of frames carrying a hand; the Python entry point must not be the
# looser of the two on the same clip.
MIN_HAND_FRACTION = 0.15


def _init_model():
    """Load the shipped ensemble. All-or-nothing.

    Every global is assigned only after the whole load succeeds. The previous
    version appended to `_INTERPRETERS` inside the load loop and guarded re-entry
    with `if _INTERPRETERS: return`, so a failure at member N>1 left three
    interpreters in the global and made the SECOND call return normally with
    `_MEMBER_IO` empty - after which predict() died with `IndexError: invalid
    index to scalar variable` instead of re-raising the original error. A retry
    after a partial download or an unpack still in flight therefore entered a
    permanently broken state that reported success.
    """
    global _FRAMES
    global _INPUT_DETAILS, _OUTPUT_DETAILS, _IDX_TO_LABEL
    global _INTERPRETERS, _MEMBER_IO, _MEMBER_USE_Z, _INITIALIZED

    if _INITIALIZED:
        return

    if Interpreter is None:
        raise RuntimeError(
            "No TFLite interpreter found. Please install ai-edge-litert: "
            "pip install ai-edge-litert"
        )

    base_dir = Path(__file__).resolve().parent.parent
    model_dir = base_dir / "models"
    label_path = model_dir / "landmark_labels.json"
    meta_path = model_dir / "landmark_model_meta.json"

    if not label_path.exists():
        raise FileNotFoundError(f"Label file not found at {label_path}")

    with open(label_path, "r", encoding="utf-8") as f:
        labels_map = json.load(f)
    idx_to_label = {int(v): k.upper() for k, v in labels_map.items()}

    # Members come from the meta sidecar so adding or dropping one never
    # requires touching this file. `members_detail` carries a per-member
    # feature width, because a z-channel member is 381-wide and a legacy one
    # is 255-wide; absent means 255.
    #
    # A malformed sidecar raises with its own message and location. The old
    # code caught every exception here and fell through to a single-model
    # fallback whose file this repo does not ship, so one missing comma in a
    # sidecar next to nine healthy .tflite members reported "No model found"
    # and pointed the reader at the wrong artifact. That fallback is now only
    # used when the sidecar is ABSENT, which is the only case it was written for.
    members = []
    detail = {}
    if meta_path.exists():
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta_obj = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"{meta_path} is not valid JSON: {e.msg} at line {e.lineno} "
                f"column {e.colno}. The nine .tflite members are unaffected; "
                f"fix the sidecar.") from e
        members = list(meta_obj.get("members", []))
        for d in meta_obj.get("members_detail", []) or []:
            detail[d.get("file")] = bool(d.get("use_z", False))

    if not members:
        # Fall back to the single-model artifact only when the sidecar is
        # absent - never when it exists but declared no members.
        single = model_dir / "landmark_classifier_int8.tflite"
        if not single.exists():
            raise FileNotFoundError(
                f"No ensemble members declared in {meta_path} and the "
                f"single-model fallback {single.name} is absent from {model_dir}")
        members = [single.name]

    # Build into locals, publish to globals only on full success.
    interpreters, member_io, member_use_z = [], [], []
    for name in members:
        p = model_dir / name
        if not p.exists():
            raise FileNotFoundError(f"Ensemble member missing: {p}")
        try:
            interp = Interpreter(model_path=str(p))
            interp.allocate_tensors()
        except Exception as e:
            raise RuntimeError(f"Failed to load ensemble member {p}: {e}") from e
        interpreters.append(interp)

    # Tensor indices are per-interpreter and differ between architectures
    # (the conv member and the tcn members do not agree). Resolving them once
    # per model here avoids indexing one interpreter with another's handles.
    #
    # Inputs are matched by NAME, not position: the TFLite converter does not
    # preserve the order the Keras model declared them in, so index 0 is the
    # mask on some members and the features on others.
    for i, interp in enumerate(interpreters):
        dets = interp.get_input_details()
        f_det = next((d for d in dets if "features" in d["name"]), dets[0])
        m_det = next((d for d in dets if "mask" in d["name"]), None)
        member_io.append((f_det["index"],
                          interp.get_output_details()[0]["index"],
                          m_det["index"] if m_det else -1))
        member_use_z.append(detail.get(members[i], False))

    frames = 24
    if meta_path.exists():
        try:
            with open(meta_path, encoding="utf-8") as f:
                frames = int(json.load(f)["frames"])
        except Exception:
            # Sidecar already parsed above; a missing or bad "frames" key falls
            # back to the trained frame count rather than failing the load.
            pass

    # MediaPipe builds its graph lazily on the FIRST detect() call, not at
    # import. Measured on this machine that single call costs 134.8 s, versus
    # 325 ms to import everything and 93 ms to load nine TFLite interpreters.
    # Left alone, that one-time cost lands inside whichever predict() a caller
    # happens to time first, and the reported latency_ms becomes meaningless.
    # Pay it here instead, so predict() measures inference.
    try:
        import numpy as _np
        from landmark_frontend import detect_hands as _warm
        _warm([_np.zeros((256, 256, 3), _np.uint8)], mode="image")
    except Exception:
        # A warm-up failure must not break model loading; it only affects the
        # first call's timing, never correctness.
        pass

    # ---- commit: nothing above this line mutated global state
    _IDX_TO_LABEL = idx_to_label
    _FRAMES = frames
    _INTERPRETERS = interpreters
    _MEMBER_IO = member_io
    _MEMBER_USE_Z = member_use_z
    _INPUT_DETAILS = _INTERPRETERS[0].get_input_details()[0]
    _OUTPUT_DETAILS = _INTERPRETERS[0].get_output_details()[0]
    _INITIALIZED = True


def _require_hand_coverage(lm, present, video_path):
    """Refuse a clip the model cannot actually see.

    A clip with a hand in one sampled frame satisfies `present.max()`, but the
    resample to the model's frame count can then step over that frame entirely:
    measured, 8 of the 32 sampled positions are skipped by the 32->24 grid. When
    that happens every feature is zero and the mask is all-zero, yet the old
    gate passed it and the ensemble still returned a gloss (BREAKFAST1 at 0.0661
    on test). The app already refuses this - CameraScreen.kt requires
    MIN_HAND_FRACTION = 0.15 of frames to carry a hand - so the Python entry
    point was the looser of the two on the same input.
    """
    if lm.shape[0] == 0 or float(present.max()) < 0.5:
        raise ValueError(f"No hand detected in any frame of: {video_path}")
    fraction = float((present.max(axis=1) >= 0.5).mean())
    if fraction < MIN_HAND_FRACTION:
        raise ValueError(
            f"Hand detected in only {fraction:.1%} of sampled frames of "
            f"{video_path}; refusing below {MIN_HAND_FRACTION:.0%}. The model "
            "resamples frames, so a hand in one frame can be resampled away "
            "entirely and produce an all-zero, fully-masked input.")


def _preprocess_video(video_path: str, target_frames: int = 24, use_z: bool = False,
                      size=256):
    """Video -> the model's (T, FEATURE_DIM) landmark feature matrix and mask.

    MediaPipe extracts two hands per frame into canonical left/right slots,
    then features.py normalises them. The same functions run in training, so
    the two cannot drift.

    Returns (features, mask). The mask is not optional decoration: the models
    are trained with it applied to the attention, and scoring without it
    changes the operating point by ~1.8 points on test.

    Raises ValueError when no hand is visible in any sampled frame: predicting
    a gloss from an empty clip would be a guess dressed up as a measurement.
    """
    # `size` is a square edge (256 = the squash the checkpoints were extracted
    # at), or the string "native" for no resize - which is what the Android app
    # does and is worth ~8.5 points. The default is the extraction geometry, so
    # existing callers are unaffected.
    #
    # Note "native" is a distinct sentinel rather than None: clip_to_landmarks
    # uses None for its own native path, and overloading one value for both
    # "default" and "native" made size=None silently mean 256.
    native = (size == "native")
    lm, present = clip_to_landmarks(
        video_path, max_frames=max(32, target_frames),
        size=None if native else size)
    _require_hand_coverage(lm, present, video_path)

    feats, mask = build_features(lm, present, expected_frames=target_frames,
                                 use_z=use_z)
    return feats, mask


def predict(video_path: str) -> dict:
    """
    Return the predicted gloss/word for one video clip. Must run fully on-device.

    Return format:
    {
        "clip_id": "test_0412",
        "prediction": "HAPPY",
        "top5": ["HAPPY", "HAND", "HOUSE", "HELP", "HOME"],
        "confidence": 0.81,
        "latency_ms": 23.4
    }
    """
    _init_model()

    # Timed from here, not from before _init_model(). Cold-start loads nine
    # TFLite interpreters and a MediaPipe graph (~124 s measured on this
    # machine); including that in latency_ms reported a 200x inflated figure
    # for work that happens once per process. Cold start is reported separately.
    start_time = time.perf_counter()

    # Extract clip_id from video filename without extension
    clip_id = Path(video_path).stem

    # Landmarks are extracted and normalised ONCE, then turned into features
    # at each width the ensemble actually uses. Building them twice would run
    # the expensive MediaPipe pass twice for no benefit.
    lm, present = clip_to_landmarks(video_path, max_frames=max(32, _FRAMES))
    _require_hand_coverage(lm, present, video_path)
    widths = {}
    for uz in set(_MEMBER_USE_Z) or {False}:
        feats, mask = build_features(lm, present, expected_frames=_FRAMES, use_z=uz)
        # A single non-finite landmark coordinate propagates through the
        # normalise/velocity/acceleration stack into every ensemble member,
        # makes the softmax all-NaN, and used to return confidence=NaN - which
        # json.dumps then wrote as a bare `NaN` token that strict JSON parsers
        # reject. Refuse it here with the clip named, rather than emitting an
        # unparseable document. (Note the NaN-at-wrist case is silently zeroed
        # by _hand_scale, so this only fires on non-anchor landmarks or on a
        # second present hand - exactly the paths that reach the tensors.)
        if not np.isfinite(feats).all() or not np.isfinite(mask).all():
            raise ValueError(
                f"Non-finite landmark features for {video_path}: "
                f"{int((~np.isfinite(feats)).sum())} of {feats.size} feature "
                "values are NaN/Inf (one bad MediaPipe frame poisons the whole "
                "clip). Refusing to emit a NaN confidence.")
        widths[uz] = (feats, mask)

    # Average logits across ensemble members, then softmax once. This matches
    # src/main.py's contract and LandmarkClassifier.kt exactly, and it is what
    # was measured: the alternative (average per-member probabilities) scored
    # +0.9 test / -0.9 val differently and was rejected - see EXPERIMENTS.md.
    # Logit averaging preserves each member's confidence margin; softmaxing
    # first and averaging compresses every member toward uniform.
    member_logits = []
    for interp, (in_idx, out_idx, mask_idx), use_z in zip(
            _INTERPRETERS, _MEMBER_IO, _MEMBER_USE_Z):
        feats, mask = widths[use_z]
        interp.set_tensor(in_idx, feats[None, ...].astype(np.float32))
        if mask_idx >= 0:
            interp.set_tensor(mask_idx, mask[None, ...].astype(np.float32))
        interp.invoke()
        member_logits.append(interp.get_tensor(out_idx)[0].astype(np.float32))
    if not member_logits:
        # Cannot happen after a successful _init_model, but np.mean([], axis=0)
        # is NaN and np.argsort(-nan_probs)[:5] is a 0-d array whose indexing
        # raised the unhelpful "invalid index to scalar variable". Fail loudly
        # at the real cause instead.
        raise RuntimeError(
            f"Ensemble has {len(_INTERPRETERS)} interpreters but produced no "
            "logits; model state is inconsistent - restart the process")
    logits = np.mean(member_logits, axis=0)
    if not np.isfinite(logits).all():
        raise ValueError(
            f"Ensemble logits for {clip_id} are not finite; refusing to emit a "
            "NaN confidence")

    # Softmax for normalized confidence scores
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)

    # Sort descending
    top_indices = np.argsort(-probs)[:5]
    top5 = [_IDX_TO_LABEL.get(int(idx), f"UNKNOWN_{idx}") for idx in top_indices]
    prediction = top5[0]
    confidence = float(probs[top_indices[0]])
    if not np.isfinite(confidence):
        raise ValueError(f"confidence for {clip_id} is not finite; refusing to "
                         "emit a NaN confidence")

    latency_ms = (time.perf_counter() - start_time) * 1000.0

    return {
        "clip_id": clip_id,
        "prediction": prediction,
        "top5": top5,
        "confidence": round(confidence, 4),
        "latency_ms": round(latency_ms, 2)
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_video.mp4>")
        sys.exit(1)

    video_arg = sys.argv[1]
    res = predict(video_arg)
    # allow_nan=False: RFC 8259 JSON has no NaN/Infinity token, and a strict
    # parser on the evaluator's side rejects the bare `NaN` that the default
    # (permissive) json.dumps emits. predict() already refuses non-finite
    # values; this is the second line of defence at the serialization boundary.
    print(json.dumps(res, indent=2, allow_nan=False))
