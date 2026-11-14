"""Dump the Python reference's features, masks and per-member logits.

Kotlin cannot call Python, so the Android parity test compares numbers the other
way round: this script writes a binary fixture (landmark arrays in; both feature
widths, the mask and every member's raw logits out) and
`FeatureParityTest` repeats the computation with the shipped Kotlin code.

Portable by construction: every path is an argument or repository-relative, the
model set is verified against its pinned hashes before use, and the output path
is whatever the caller says. There is no developer-specific import and no
assumed temporary-directory layout.

Usage:
  python evaluation/device/v4_3_parity/make_parity_fixture.py \
      --clips DIR --manifest manifest.json --out /path/parity_fixture.bin [--limit 6]

The manifest is a JSON list of {"file": "...", "gloss": "...", "signer": "..."}.
Clips and the fixture are authorised research data: they stay outside the
repository and are never packaged for release.

Fixture layout, little-endian:
  int32 clipCount
  per clip:  int32 srcFrames; float32 present[n*2]; float32 landmarks[n*2*21*3]
  per clip:  float32 featuresXY[24*255]; float32 featuresXYZ[24*381];
             float32 mask[24]
  per clip:  9 x float32 logits[100]   (meta order)
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "src"
ASSETS = REPO / "app/android_final/app/src/main/assets"
MODELS = REPO / "models"
META_NAME = "landmark_model_meta.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_models_dir(explicit: str | None) -> Path:
    """Where the nine .tflite members live.

    Defaults to models/ (which the README tells you to populate from your
    authorised copy). Falls back to the Android asset directory so a checkout
    that has already staged them for a build works without duplication.
    """
    if explicit:
        d = Path(explicit).expanduser().resolve()
        if not d.is_dir():
            raise SystemExit(f"--models-dir not found: {d}")
        return d
    for candidate in (MODELS, ASSETS):
        if (candidate / META_NAME).is_file():
            return candidate
    raise SystemExit(
        f"No {META_NAME} found in {MODELS} or {ASSETS}. Supply your authorised "
        "model assets first (see models/README.md) or pass --models-dir.")


def verify_models(models_dir: Path):
    """Fail loudly unless the nine pinned members are the ones present."""
    meta = json.loads((models_dir / META_NAME).read_text(encoding="utf-8"))
    members = meta["members"]
    detail = {d["file"]: bool(d.get("use_z", False))
              for d in meta.get("members_detail", [])}

    missing = [m for m in members if not (models_dir / m).is_file()]
    if missing:
        raise SystemExit(f"model members missing from {models_dir}: {missing}")
    undetailed = [m for m in members if m not in detail]
    if undetailed:
        raise SystemExit(
            f"{META_NAME} has no members_detail entry for {undetailed}; "
            "per-member feature width would have to be guessed")

    # The fixture must describe the release bundle, not whatever happens to be
    # on disk. Import the pins from the production evaluator so there is one
    # source of truth.
    sys.path.insert(0, str(SRC))
    import main as eg  # noqa: E402

    hashes = {m: sha256_file(models_dir / m) for m in members}
    unpinned = [m for m in members if m not in eg.PINNED_MEMBERS]
    if unpinned:
        raise SystemExit(f"members not pinned in src/main.py: {unpinned}")
    drift = [m for m in members if hashes[m] != eg.PINNED_MEMBERS[m]]
    if drift:
        raise SystemExit(
            "member hash mismatch against src/main.py pins: " + ", ".join(drift))
    ordered = hashlib.sha256()
    for m in members:
        ordered.update((models_dir / m).read_bytes())
    if ordered.hexdigest() != eg.PINNED_ENSEMBLE_SHA256:
        raise SystemExit(
            f"ordered ensemble hash {ordered.hexdigest()[:16]}… != pinned "
            f"{eg.PINNED_ENSEMBLE_SHA256[:16]}…")
    detector = models_dir / "hand_landmarker.task"
    if detector.is_file() and eg.PINNED_DETECTOR_SHA256:
        got = sha256_file(detector)
        if got != eg.PINNED_DETECTOR_SHA256:
            raise SystemExit(f"detector hash {got[:16]}… != pinned detector")
    return meta, detail, hashes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True, help="directory holding the .mp4 files")
    ap.add_argument("--manifest", required=True,
                    help='JSON list of {"file","gloss","signer"}')
    ap.add_argument("--out", required=True, help="where to write the fixture")
    ap.add_argument("--limit", type=int, default=6, help="maximum clips to include")
    ap.add_argument("--models-dir", default=None)
    args = ap.parse_args()

    clips = Path(args.clips).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    if not clips.is_dir():
        raise SystemExit(f"--clips not found: {clips}")
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if not isinstance(manifest, list) or not manifest:
        raise SystemExit("--manifest must be a non-empty JSON list")

    models_dir = resolve_models_dir(args.models_dir)
    meta, detail, hashes = verify_models(models_dir)

    # The evaluator reads models/ by convention; point it at the verified dir so
    # the fixture is built from exactly the bytes that were checked.
    sys.path.insert(0, str(SRC))
    import main as eg  # noqa: E402
    from features import build_features  # noqa: E402

    orig_paths = eg._model_paths
    eg._model_paths = lambda: (
        "landmark", models_dir / "landmark_classifier_fp16.tflite",
        models_dir / "landmark_labels.json")
    try:
        eg._init_model()
    finally:
        eg._model_paths = orig_paths

    if len(eg._MEMBER_NAMES) != len(meta["members"]):
        raise SystemExit(
            f"evaluator loaded {len(eg._MEMBER_NAMES)} members, metadata lists "
            f"{len(meta['members'])}")

    import cv2
    import landmark_frontend as lf

    lmrk = lf.get_landmarker(mode="image")
    chosen = []
    for entry in manifest:
        if len(chosen) >= args.limit:
            break
        path = clips / entry["file"]
        if not path.is_file():
            print(f"  skip (absent): {entry['file']}", file=sys.stderr)
            continue
        cap = cv2.VideoCapture(str(path))
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
            print(f"  skip (no hands): {entry['file']}", file=sys.stderr)
            continue
        if len(lm) > 200:      # keep the fixture small
            idx = np.unique(np.linspace(0, len(lm) - 1, 200).astype(int))
            lm, pr = lm[idx], pr[idx]
        chosen.append({"file": entry["file"], "gloss": entry["gloss"],
                       "signer": entry.get("signer", ""), "lm": lm, "pr": pr})
        print(f"  {entry['gloss']:14s} {entry.get('signer',''):4s} "
              f"frames={len(lm)}", flush=True)

    if not chosen:
        raise SystemExit("no usable clips; nothing written")

    rows = []
    for c in chosen:
        lm, pr = c["lm"], c["pr"]
        xy, mask = build_features(lm, pr, expected_frames=24, use_z=False)
        xyz, mask_z = build_features(lm, pr, expected_frames=24, use_z=True)
        if not np.allclose(mask, mask_z):
            raise SystemExit("mask differs between widths; one landmark pass is broken")
        if xy.shape != (24, 255) or xyz.shape != (24, 381):
            raise SystemExit(f"unexpected widths {xy.shape} {xyz.shape}")

        logits = []
        for interp, io, use_z in zip(eg._INTERPRETERS, eg._MEMBER_IO, eg._MEMBER_USE_Z):
            in_idx, out_idx, mask_idx = io[0], io[1], io[2]
            feats = xyz if use_z else xy
            interp.set_tensor(in_idx, feats[None, ...].astype(np.float32))
            interp.set_tensor(mask_idx, mask[None, ...].astype(np.float32))
            interp.invoke()
            logits.append(interp.get_tensor(out_idx)[0].astype(np.float32))
        rows.append({"gloss": c["gloss"], "lm": lm, "pr": pr,
                     "xy": xy, "xyz": xyz, "mask": mask, "logits": logits})

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        f.write(np.int32(len(rows)).tobytes())
        for r in rows:
            n = len(r["lm"])
            f.write(np.int32(n).tobytes())
            f.write(r["pr"].astype("<f4").tobytes())
            f.write(r["lm"].astype("<f4").tobytes())
        for r in rows:
            f.write(r["xy"].astype("<f4").tobytes())
            f.write(r["xyz"].astype("<f4").tobytes())
            f.write(r["mask"].astype("<f4").tobytes())
        for r in rows:
            for lg in r["logits"]:
                f.write(lg.astype("<f4").tobytes())

    # Provenance beside the fixture, so a fixture can never be attributed to
    # the wrong model.
    prov = {
        "clips": len(rows),
        "signers": sorted({c["signer"] for c in chosen}),
        "glosses": [c["gloss"] for c in chosen],
        "frames_per_clip": [len(c["lm"]) for c in chosen],
        "models_dir": str(models_dir),
        "member_hashes": hashes,
        "ensemble_sha256": eg.PINNED_ENSEMBLE_SHA256,
        "feature_dims": {"xy": 255, "xyz": 381},
        "mask": "present.max(axis=1) > 0.5 over RESAMPLED presence",
        "source_files_sha256": {
            name: sha256_file(SRC / name)
            for name in ("main.py", "features.py", "landmark_frontend.py")
        },
        "fixture_sha256": sha256_file(out),
    }
    prov_path = out.with_suffix(out.suffix + ".provenance.json")
    prov_path.write_text(json.dumps(prov, indent=1), encoding="utf-8")

    summary = []
    for r in rows:
        avg = np.mean(r["logits"], axis=0)
        e = np.exp(avg - avg.max())
        p = e / e.sum()
        top = np.argsort(-p)[:5]
        top5 = [eg._IDX_TO_LABEL[int(i)] for i in top]
        summary.append({"gloss": r["gloss"], "top1": top5[0], "top5": top5,
                        "confidence": float(p[top[0]]),
                        "correct": top5[0] == r["gloss"]})
    out.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=1), encoding="utf-8")

    print(f"\nwrote {out} ({out.stat().st_size} bytes) for {len(rows)} clips")
    print(f"provenance: {prov_path}")
    for s_ in summary:
        print(f"  {s_['gloss']:14s} -> {s_['top1']:14s} {s_['confidence']:.3f} "
              f"{'OK' if s_['correct'] else 'WRONG'}")


if __name__ == "__main__":
    main()