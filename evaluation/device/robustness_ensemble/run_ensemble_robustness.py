"""Real-video robustness of the shipped 4-model FP16 landmark ensemble.

Owner: A03.  This is a self-contained driver: it imports the existing
corruption functions from ``evaluation/device/corruptions.py`` but writes its
own results under ``evaluation/device/robustness_ensemble/`` and does not
modify ``robustness.py``, the lab, ``src/``, ``models/`` or ``scripts/``.

For every clip in the manifest and every condition (clean + one corruption
kind at a fixed severity) the driver:

1. decodes the clip, applies the seeded corruption to *every* frame,
2. writes a temporary mp4 and runs the REAL ``src/main.py`` ``predict()``,
3. records the predicted label and whether it matches the manifest truth,
4. classifies every incorrect corrupted result into the failure stage:
   ``no_hand`` (ValueError "No hand detected") vs ``wrong_prediction``.

Accuracy is reported over *scored* clips (prediction produced) and over
*eligible* clips (all clips, failures counted wrong) because a deployment
that cannot see a hand has failed too.

Usage (from repo root):

    .venv/bin/python evaluation/device/robustness_ensemble/run_ensemble_robustness.py \
        --limit 30 --workers 4

``--limit`` picks a deterministic prefix of the manifest (sorted by clip_id)
so a subset run is reproducible.  The heavy-benchmark lock is honoured:
if ``/tmp/echogesture-heavy-benchmark.lock`` is held the driver waits
briefly and then refuses rather than competing for CPU.
"""

from __future__ import annotations

import argparse
import csv
import errno
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEVICE_EVAL = ROOT / "evaluation" / "device"
sys.path.insert(0, str(DEVICE_EVAL))

from corruptions import CORRUPTION_VERSION, CorruptionSpec, corrupt_frames, decode_video, write_video
from robustness_lab.config import DEFAULT_SEVERITY, derive_clip_seed

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LOCK = Path("/tmp/echogesture-heavy-benchmark.lock")
NO_HAND_TOKEN = "No hand detected"

# Fixed, documented run configuration.  Change the version when the driver's
# semantics change so old result files can never be mistaken for new ones.
RUNNER_VERSION = 1
GLOBAL_SEED = 42
# "reencode" is the control: decode + re-encode with no pixel change, so the
# pure mp4 round-trip cost can be separated from the corruption effect.
CONDITIONS = ("clean", "reencode", "low_light", "motion_blur", "occlusion", "crop", "frame_drop")
CORRUPTIONS = ("low_light", "motion_blur", "occlusion", "crop", "frame_drop")
DEFAULT_MANIFEST = DEVICE_EVAL / "v5_accuracy" / "data" / "val_30" / "manifest.csv"


# --------------------------------------------------------------------------
# Pure aggregation math (unit-tested separately; never imports the model).
# --------------------------------------------------------------------------
def accuracy(correct, denominator):
    """Top-1 as a proportion; ``None`` for an empty denominator."""
    if denominator <= 0:
        return None
    return correct / denominator


def percentage_point_drop(clean_correct, corrupted_correct, denominator):
    """Clean minus corrupted accuracy in percentage points.

    Positive = corruption hurt.  Uses one shared denominator so a failed
    prediction on either side is encoded by ``correct`` being 0 there, not by
    silently shrinking N.
    """
    if denominator <= 0:
        return None
    return 100.0 * (clean_correct - corrupted_correct) / denominator


def failure_stage(prediction, error, truth):
    """Classify one prediction attempt.

    Returns one of ``correct``, ``no_hand``, ``wrong_prediction``, ``error``.
    ``no_hand`` means the pipeline produced no prediction because hand
    detection failed; that is a different deployment risk from a confident
    wrong label.
    """
    if error is not None:
        return "no_hand" if NO_HAND_TOKEN in error else "error"
    if prediction == truth:
        return "correct"
    if prediction is None:
        return "error"
    return "wrong_prediction"


def summarise(clip_rows, condition):
    """Aggregate per-clip rows for one condition.

    ``clip_rows`` are the flat records produced by :func:`run_condition` for
    this condition (each carries ``stage``/``correct``/``truth``).
    Both denominators are explicit:
      scored   = clips where the pipeline returned a label
      eligible = every clip in the manifest (failures counted wrong)
    """
    n = len(clip_rows)
    stages = Counter(row["stage"] for row in clip_rows)
    correct = stages.get("correct", 0)
    scored = n - stages.get("no_hand", 0) - stages.get("error", 0)
    return {
        "n_eligible": n,
        "n_scored": scored,
        "n_correct": correct,
        "n_no_hand": stages.get("no_hand", 0),
        "n_wrong": stages.get("wrong_prediction", 0),
        "n_error": stages.get("error", 0),
        "accuracy_scored": accuracy(correct, scored),
        "accuracy_eligible": accuracy(correct, n),
    }


# --------------------------------------------------------------------------
# Prediction + corruption plumbing.
# --------------------------------------------------------------------------
def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path):
    rows = []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            clip_id = (row.get("clip_id") or "").strip()
            clip_path = (row.get("path") or "").strip()
            truth = (row.get("ground_truth") or "").strip()
            if not clip_id or not clip_path or not truth:
                raise SystemExit(f"Manifest row missing clip_id/path/ground_truth: {row}")
            if not Path(clip_path).is_absolute():
                clip_path = str(path.parent / clip_path)
            rows.append({
                "clip_id": clip_id,
                "path": clip_path,
                "truth": truth,
                "signer_id": (row.get("signer_id") or "").strip(),
            })
    if not rows:
        raise SystemExit(f"Empty manifest: {path}")
    return rows


def load_predict():
    import importlib.util
    predictor = ROOT / "src" / "main.py"
    spec = importlib.util.spec_from_file_location("echogesture_predictor", predictor)
    module = importlib.util.module_from_spec(spec)
    sys.modules["echogesture_predictor"] = module
    spec.loader.exec_module(module)
    return module.predict


def predict_path(predict, video_path, clip_id):
    """Run the real predict(); return (prediction, confidence, latency_ms, error)."""
    try:
        result = predict(str(video_path), clip_id=clip_id)
        if not isinstance(result, dict) or "prediction" not in result:
            return None, None, None, f"contract: returned {type(result).__name__}, expected dict"
        return result["prediction"], result.get("confidence"), result.get("latency_ms"), None
    except Exception as exc:  # noqa: BLE001 - every failure becomes evidence
        return None, None, None, f"{type(exc).__name__}: {exc}"


def acquire_lock(timeout_s):
    """Take the heavy-benchmark lock or refuse after a short wait.

    This benchmark's own lock is a *file* containing the holder PID.  The
    shared path on this machine can also exist as an empty directory left by
    another tool; that shape can never be a live lock we wrote, so it is
    noted and stepped over rather than deleting someone else's path or
    blocking a run that no process is competing with.
    """
    if LOCK.is_dir() and not any(LOCK.iterdir()):
        print(f"NOTE: {LOCK} is an empty directory (stale, no holder); "
              "proceeding without touching it", file=sys.stderr)
        return
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise SystemExit(
                    f"Another benchmark holds {LOCK}; waited {timeout_s}s. "
                    "Re-run later or pass a smaller --limit without competing."
                )
            time.sleep(2.0)


def release_lock():
    if LOCK.is_dir():
        return  # never remove a path we did not create
    try:
        LOCK.unlink()
    except FileNotFoundError:
        pass


def run_condition(predict, clips, kind, severity, seed, workdir, condition):
    """One condition over every clip; returns per-clip rows."""
    rows = []
    for index, clip in enumerate(clips):
        clip_id = clip["clip_id"]
        if condition == "clean":
            video_path, corruption_error = Path(clip["path"]), None
        else:
            clip_seed = derive_clip_seed(seed, index, clip_id)
            try:
                frames, fps = decode_video(clip["path"])
                if condition == "reencode":
                    out_frames = frames
                else:
                    out_frames = corrupt_frames(frames, CorruptionSpec(kind, severity, clip_seed))
                video_path = workdir / f"{condition}-{index:04d}.mp4"
                write_video(video_path, out_frames, fps)
                corruption_error = None
            except (ValueError, OSError) as exc:
                video_path = None
                corruption_error = f"{type(exc).__name__}: {exc}"
        if corruption_error is not None:
            rows.append({**clip, "condition": condition, "prediction": None,
                         "correct": False, "stage": "error", "error": corruption_error,
                         "latency_ms": None, "confidence": None})
            continue
        started = time.perf_counter()
        prediction, confidence, latency_ms, error = predict_path(predict, video_path, clip_id)
        wall_ms = (time.perf_counter() - started) * 1000.0
        rows.append({
            **clip,
            "condition": condition,
            "prediction": prediction,
            "correct": prediction == clip["truth"],
            "stage": failure_stage(prediction, error, clip["truth"]),
            "error": error,
            "latency_ms": latency_ms if latency_ms is not None else round(wall_ms, 2),
            "confidence": confidence,
        })
        if video_path is not None and condition != "clean":
            video_path.unlink(missing_ok=True)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--limit", type=int, default=30,
                        help="Deterministic prefix of manifest sorted by clip_id (default 30 = all)")
    parser.add_argument("--severity", type=float, default=None,
                        help="Override every corruption's severity (default: lab defaults)")
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED)
    parser.add_argument("--workers", type=int, default=4,
                        help="ECHOGESTURE_WORKERS for the parallel tracker; 0 = singleton")
    parser.add_argument("--output-dir", type=Path, default=RESULTS)
    parser.add_argument("--lock-timeout", type=float, default=30.0)
    parser.add_argument("--skip-clean", action="store_true",
                        help="Reuse the clean result already in results/clean.json")
    args = parser.parse_args(argv)

    if args.workers > 0:
        os.environ["ECHOGESTURE_TRACKER"] = "parallel"
        os.environ["ECHOGESTURE_WORKERS"] = str(args.workers)
    else:
        os.environ.pop("ECHOGESTURE_TRACKER", None)

    clips = read_manifest(args.manifest)
    clips.sort(key=lambda row: row["clip_id"])
    clips = clips[: args.limit]
    if len(clips) < 20:
        print(f"WARNING: only {len(clips)} clips; this is not representative", file=sys.stderr)

    sev = {kind: (args.severity if args.severity is not None else DEFAULT_SEVERITY[kind])
           for kind in CORRUPTIONS}

    results_dir = args.output_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    acquire_lock(args.lock_timeout)
    started = time.perf_counter()
    try:
        workdir = Path(tempfile.mkdtemp(prefix="ens-rob-", dir=results_dir))
        try:
            predict = load_predict()
            model_shas = {name: sha256(ROOT / "models" / name) for name in (
                "landmark_ens_tcn21_fp16.tflite", "landmark_ens_tcnm61_fp16.tflite",
                "landmark_ens_tcnm62_fp16.tflite", "landmark_ens_rob71_fp16.tflite")}
            meta = json.loads((ROOT / "models" / "landmark_model_meta.json").read_text())

            per_condition = {}
            clean_rows = None
            if args.skip_clean and (results_dir / "clean.json").is_file():
                clean_rows = json.loads((results_dir / "clean.json").read_text())["clips"]
                per_condition["clean"] = clean_rows

            for condition in CONDITIONS:
                if condition in per_condition:
                    continue
                kind = None if condition in ("clean", "reencode") else condition
                print(f"[{condition}] {len(clips)} clips ...", file=sys.stderr, flush=True)
                rows = run_condition(predict, clips, kind,
                                     None if kind is None else sev[kind],
                                     args.seed, workdir, condition)
                per_condition[condition] = rows

            condition_summaries = {c: summarise(rows, c) for c, rows in per_condition.items()}
            clean_summary = condition_summaries["clean"]
            reencode_summary = condition_summaries["reencode"]
            clean_rows = per_condition["clean"]

            runtime_s = time.perf_counter() - started
            recorded_at = datetime.now(timezone.utc).isoformat()
            common = {
                "runner_version": RUNNER_VERSION,
                "seed": args.seed,
                "tracker": "parallel" if args.workers > 0 else "singleton",
                "workers": args.workers,
                "corruption_version": CORRUPTION_VERSION,
                "model_meta": meta,
                "model_sha256": model_shas,
                "manifest_sha256": sha256(args.manifest),
                "manifest": str(args.manifest.relative_to(ROOT)),
                "clip_count": len(clips),
                "clip_ids": [c["clip_id"] for c in clips],
                "signers": sorted({c["signer_id"] for c in clips if c["signer_id"]}),
                "recorded_at": recorded_at,
                "runner": str(Path(__file__).resolve().relative_to(ROOT)),
            }

            for condition, rows in per_condition.items():
                summary = condition_summaries[condition]
                if condition in ("clean", "reencode"):
                    payload = {**common, "condition": condition, "severity": None,
                               "summary": summary, "clips": rows}
                else:
                    clean_correct = clean_summary["n_correct"]
                    payload = {
                        **common,
                        "condition": condition,
                        "severity": sev[condition],
                        "summary": {
                            **summary,
                            "clean_n_correct": clean_correct,
                            "clean_accuracy_eligible": clean_summary["accuracy_eligible"],
                            "pp_drop_eligible": percentage_point_drop(
                                clean_correct, summary["n_correct"], summary["n_eligible"]),
                            "pp_drop_scored": percentage_point_drop(
                                clean_correct, summary["n_correct"], summary["n_scored"]),
                            # Drop against the re-encode control: isolates the
                            # corruption effect from the mp4 round-trip.
                            "reencode_n_correct": reencode_summary["n_correct"],
                            "pp_drop_vs_reencode": percentage_point_drop(
                                reencode_summary["n_correct"], summary["n_correct"],
                                summary["n_eligible"]),
                            "flipped_correct_to_wrong": sum(
                                1 for clean, corr in zip(clean_rows, rows)
                                if clean["correct"] and not corr["correct"]),
                            "flipped_wrong_to_correct": sum(
                                1 for clean, corr in zip(clean_rows, rows)
                                if not clean["correct"] and corr["correct"]),
                        },
                        "clips": rows,
                    }
                (results_dir / f"{condition}.json").write_text(json.dumps(payload, indent=2) + "\n")
                print(f"[{condition}] correct {summary['n_correct']}/{summary['n_eligible']} "
                      f"scored {summary['accuracy_scored']} eligible {summary['accuracy_eligible']} "
                      f"no_hand {summary['n_no_hand']}", file=sys.stderr, flush=True)

            runtime_payload = {
                "runtime_seconds": round(runtime_s, 2),
                "clip_count": len(clips),
                "conditions": list(per_condition),
                "predict_calls": len(clips) * len(per_condition),
                "recorded_at": recorded_at,
            }
            (results_dir / "runtime.json").write_text(json.dumps(runtime_payload, indent=2) + "\n")
            print(f"done in {runtime_s:.1f}s", file=sys.stderr)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
    finally:
        release_lock()


if __name__ == "__main__":
    main()