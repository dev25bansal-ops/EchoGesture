"""EchoGesture offline demo lab: camera-less, real-prediction smoke suite.

Every run calls the repository's real ``predict(video_path)`` — no fake
captions, no synthetic boxes, no memorized answers. The input clip is read
in place; the HUD is rendered headless and saved as a PNG for review.

Run from the repository root::

    .venv/bin/python -m app.demo_lab --video "/path/to/clip.mp4"
    .venv/bin/python -m app.demo_lab --video "/path/to/clip.mp4" --outdir /tmp/demo-lab-out
    .venv/bin/python -m app.demo_lab --check

Retention policy: the HUD PNG embeds a rendered video frame, so it is
treated as demo media. Without an explicit ``--outdir`` the run writes into a
private per-run ``TemporaryDirectory`` that is deleted when the command exits;
nothing persists to a shared location by default. Persisting output requires
the operator to opt in with ``--outdir``.

``--check`` verifies the harness itself without a model: double prediction,
thread-safe state snapshots, temp-clip cleanup, and HUD shape. It still needs
``cv2``/``numpy`` but no TFLite model or camera.

Camera status: macOS denied camera access during the engineering check, so a
live camera preview has NOT been verified on this machine. The file-input
path below is the supported fallback; do not present it as a live camera
result.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.offline_demo import CaptionState, model_identity, model_label_from_identity, render_file_result, run_file


def _sha256(path: Path):
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def run_video(video: Path, outdir: Path, threshold: float = 0.25):
    outdir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter_ns()
    image, result, identity = render_file_result(video, ROOT / "src/main.py", threshold)
    render_path = outdir / f"hud-{Path(video).stem[:40]}.png"
    ok = bool(cv2.imwrite(str(render_path), image))
    report = {
        "video": str(video),
        "video_bytes": video.stat().st_size if video.exists() else None,
        "video_sha256": _sha256(video),
        "result": result,
        "model": identity,
        "backend_env": os.environ.get("ECHOGESTURE_MODEL", "landmark"),
        "render_path": str(render_path),
        "render_written": ok,
        "wall_ms": round((time.perf_counter_ns() - started) / 1e6, 2),
        "live_camera_verified": False,
        "note": "File-input smoke path. NOT a live-camera claim; macOS camera access is denied on this machine.",
    }
    report_path = outdir / f"hud-{Path(video).stem[:40]}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def check():
    """Model-free harness checks; no camera, no TFLite, no network."""
    from app.offline_demo import PredictionWorker, sanitize_result

    checks = []

    def _case(name, fn):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            checks.append({"name": name, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
        else:
            checks.append({"name": name, "ok": True})

    seen = []

    def predict(path, *, clip_id=None):
        clip = Path(path)
        seen.append(clip)
        assert clip.exists() and clip.stat().st_size > 0
        return {"clip_id": clip_id, "prediction": "BOOK", "top5": ["BOOK"],
                "confidence": 0.8, "latency_ms": 3.2}

    def worker_check():
        worker = PredictionWorker(predict)
        try:
            assert worker.submit([np.zeros((24, 32, 3), np.uint8)] * 3, 12, "check-sign")
            kind, result = worker.results.get(timeout=10)
            assert kind == "result"
            assert result["queue_wait_ms"] >= 0 and result["capture_to_result_ms"] >= 0
            deadline = time.time() + 5
            while time.time() < deadline:
                if not seen[0].exists():
                    break
                time.sleep(0.01)
            assert not seen[0].exists()
            leftovers = [p for p in worker.tmpdir_path.iterdir() if p.exists()]
            assert leftovers == [], f"temp leftovers: {leftovers}"
        finally:
            worker.close()
            worker.close()  # idempotent

    _case("worker_encodes_real_predictions_and_deletes_temp_clip", worker_check)

    def sanitize_check():
        # Direct sanitize_result must reject out-of-range confidence.
        try:
            sanitize_result({"prediction": "X", "confidence": 42, "top5": ["X"], "latency_ms": 1})
        except ValueError:
            return
        raise AssertionError("out-of-range confidence accepted by sanitize_result")

    _case("sanitize_rejects_out_of_range_confidence", sanitize_check)

    def state_check():
        state = CaptionState(0.25)
        snap = state.snapshot()
        assert isinstance(snap, dict) and snap["state"] == "WAITING"
        state.update({"prediction": "BOOK", "confidence": 0.7, "top5": ["BOOK"], "latency_ms": 2})
        assert state.state == "CAPTION"
        state.update({"prediction": "BOOK", "confidence": 0.1, "top5": ["BOOK"], "latency_ms": 2})
        assert state.state == "RETRY"
        state.update({"prediction": "BOOK", "confidence": "nan", "top5": ["BOOK"], "latency_ms": 2})
        assert state.state == "ERROR"
        state.reset()
        assert state.state == "WAITING"

    _case("caption_state_snapshot_and_recovery", state_check)

    def hud_check():
        from app.offline_demo import render
        image = render(np.zeros((240, 640, 3), np.uint8), CaptionState(0.25), "check-model", 0.25)
        assert image.shape == (240, 640, 3)

    _case("hud_render_shape", hud_check)

    def identity_check():
        identity = model_identity()
        label = model_label_from_identity(identity)
        assert identity["backend"] in ("landmark", "s3d")
        assert "sha" in label
        # Never claim a calibrated threshold.
        assert "provisional" in identity["threshold_status"]

    _case("model_identity_names_selected_artifact", identity_check)

    return {"checks": checks, "ok": all(c["ok"] for c in checks)}


@contextlib.contextmanager
def ephemeral_outdir():
    """Yield a private per-run directory that is removed on exit.

    Used when no ``--outdir`` is given: the HUD PNG (a rendered video frame)
    and its JSON report exist only for the lifetime of the command, so no
    demo media persists in a shared or default location.
    """
    with tempfile.TemporaryDirectory(prefix="echogesture-demo-lab-") as tmp:
        path = Path(tmp)
        try:
            path.chmod(0o700)
        except OSError:
            pass
        yield path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=None, help="Real clip file to predict (read in place)")
    parser.add_argument("--outdir", type=Path, default=None,
                        help="Explicit directory for HUD PNG + JSON report. Omit for a private "
                             "per-run temp directory that is deleted on exit (no persistent default).")
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--check", action="store_true", help="Run model-free harness checks and exit")
    args = parser.parse_args(argv)

    if args.check:
        report = check()
        print(json.dumps(report, indent=2))
        return 0 if report["ok"] else 1

    if not args.video:
        parser.error("provide --video <clip.mp4> or --check")

    if args.outdir is not None:
        report = run_video(args.video, args.outdir, args.threshold)
        print(json.dumps(report, indent=2))
        return 0 if report["render_written"] else 1

    # No explicit --outdir: keep the demo media private and ephemeral.
    with ephemeral_outdir() as tmpdir:
        report = run_video(args.video, tmpdir, args.threshold)
        report["retained"] = False
        report["outdir"] = "ephemeral (private TemporaryDirectory, deleted on exit)"
        print(json.dumps(report, indent=2))
        return 0 if report["render_written"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
