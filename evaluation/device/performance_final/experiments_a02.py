"""A02 candidate experiments on the 12 permitted clips.

Every variant is measured as the same work production `predict()` does for a
landmark clip: sample frames from the clip, hand-track every sampled frame,
build the feature tensor, invoke the ensemble.  Two kinds of variant:

* ``predict``  — call the real ``src/main.py:predict()`` in a fresh process with
                 its env knobs set.  Used for tracker configurations.
* ``mirror``   — replicate ``predict()``'s landmark branch inline so a change
                 that would need a ``src/`` edit (input size, frame count,
                 pipelined decode) can be measured without editing ``src/``.
                 ``mirror_4`` doubles as the control that the mirror itself is
                 faithful: it runs BOTH the mirror and real ``predict()`` and
                 reports whether prediction, top5 and the feature tensor match.

Each variant runs in its own child process (cold start is real, peak RSS is not
polluted by another variant).  The parent holds
``/tmp/echogesture-heavy-benchmark.lock`` for the whole run and removes it in
``finally``.  Nothing outside this file and ``results_candidates/`` is written.

Usage:
    .venv/bin/python evaluation/device/performance_final/experiments_a02.py \
        [--clips DIR] [--max-clips 12] [--variants baseline_4 workers_8 ...] \
        [--output results_candidates/experiments.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SRC = ROOT / "src"
DEFAULT_CLIPS = ROOT / "evaluation/device/v5_accuracy/data/val_30"
DEFAULT_OUTPUT = HERE / "results_candidates/experiments.json"

LOCK = Path("/tmp/echogesture-heavy-benchmark.lock")

# frames sampled by production predict() (max(32, _FRAMES))
PROD_SAMPLE_FRAMES = 32
PROD_FRAME_SIZE = 256

VARIANTS = {
    # --- tracker configuration (real predict()) ----------------------------
    "baseline_4": {"kind": "predict", "workers": 4},
    "workers_6": {"kind": "predict", "workers": 6},
    "workers_8": {"kind": "predict", "workers": 8},
    "workers_12": {"kind": "predict", "workers": 12},
    "workers_16": {"kind": "predict", "workers": 16},
    # --- mirror + control ---------------------------------------------------
    "mirror_4": {"kind": "mirror", "workers": 4, "sample_frames": 32, "size": 256},
    # --- candidate A: overlap decode with detection ------------------------
    "pipeline_4": {"kind": "mirror", "workers": 4, "sample_frames": 32, "size": 256,
                   "pipeline": True},
    "pipeline_8": {"kind": "mirror", "workers": 8, "sample_frames": 32, "size": 256,
                   "pipeline": True},
    "pipeline_12": {"kind": "mirror", "workers": 12, "sample_frames": 32, "size": 256,
                    "pipeline": True},
    # --- candidate B: smaller detection input ------------------------------
    "input_224_4": {"kind": "mirror", "workers": 4, "sample_frames": 32, "size": 224},
    "input_192_4": {"kind": "mirror", "workers": 4, "sample_frames": 32, "size": 192},
    # --- candidate C: fewer sampled frames --------------------------------
    "frames_24_4": {"kind": "mirror", "workers": 4, "sample_frames": 24, "size": 256},
}

# Variants compared against this control.  ``mirror_4`` is the same scheduling
# as production `predict()` at 4 workers, so every candidate is judged against
# the control that isolates the change being tested (not against a baseline
# that may have been measured under different machine load).
CONTROL = "mirror_4"


# ---------------------------------------------------------------------------
# shared
# ---------------------------------------------------------------------------

def percentile(values, q):
    if not values:
        return float("nan")
    ordered = sorted(values)
    k = (len(ordered) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    frac = k - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def _summ(values):
    return {
        "count": len(values),
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "min_ms": min(values) if values else None,
        "max_ms": max(values) if values else None,
        "mean_ms": statistics.mean(values) if values else None,
    }


def _peak_rss_mib():
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / (1024 * 1024) if sys.platform == "darwin" else raw / 1024


def _sha(arr) -> str:
    a = np.ascontiguousarray(arr)
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


def _resolve_clips(clip_dir: Path, limit: int):
    clips = sorted(clip_dir.glob("*.mp4"))
    if not clips:
        raise SystemExit(f"no .mp4 clips in {clip_dir}")
    return clips[:limit]


# ---------------------------------------------------------------------------
# child work
# ---------------------------------------------------------------------------

import numpy as np  # noqa: E402


def _load_main():
    import importlib.util
    spec = importlib.util.spec_from_file_location("eg_main_a02", SRC / "main.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _SizePool:
    """Production tracker scheduling, but accepting a non-256 frame size.

    Reuses ``ParallelImageTracker``'s workers/executor and only relaxes the
    frame-shape assertion, so the measured scheduling and per-frame MediaPipe
    call are exactly the production ones.
    """

    def __init__(self, workers: int, size: int):
        import tracker as tk
        self._size = size
        self._inner = tk.ParallelImageTracker(workers=workers)
        size_local = size

        def _validate(frames):
            frames = list(frames)
            for frame in frames:
                if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
                        or frame.shape != (size_local, size_local, 3)):
                    raise ValueError(f"bad frame {getattr(frame, 'shape', None)}")
            return frames

        self._inner._validate_frames = _validate

    def detect(self, frames):
        return self._inner.detect(frames)

    def close(self):
        self._inner.close()


class _PipelinedPool:
    """Decode each sampled frame and hand it to a worker as soon as it exists.

    The production path decodes all 32 frames (13.7 ms) and only then starts
    detection.  Here a producer thread runs the same grab/read/resize/cvtColor
    loop and submits frame j to worker ``j % workers`` immediately, so decode of
    later frames overlaps detection of earlier ones.  Per-frame work and worker
    assignment are identical to the production tracker; only the overlap is new.
    """

    def __init__(self, workers: int, size: int = PROD_FRAME_SIZE):
        import tracker as tk
        from concurrent.futures import ThreadPoolExecutor
        self._tk = tk
        self._size = size
        self._workers = workers
        self._pool = tk.ParallelImageTracker(workers=workers)
        self._executor = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="a02-pipe")

    def run(self, clip: Path, sample_frames: int):
        """Decode frame j and submit it to worker j%workers immediately.

        Worker threads call MediaPipe the moment their first frame exists, so
        decode of later frames overlaps detection of earlier ones.  The
        per-frame resize/cvtColor, the per-worker frame list and the final
        reassembly are byte-for-byte the production ones.
        """
        import cv2
        from concurrent.futures import Future

        local_workers = self._workers
        size = self._size
        inner = self._pool
        pending: list[Future] = []
        produced = 0

        def _emit(index, frame):
            pending.append(self._executor.submit(
                inner._workers[index % local_workers].detect, [frame]))

        decode_started = time.perf_counter_ns()
        cap = cv2.VideoCapture(str(clip))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open {clip}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        try:
            if total > sample_frames:
                targets = np.unique(np.linspace(0, total - 1, sample_frames).astype(int))
                cur = 0
                for t in targets:
                    while cur < t:
                        if not cap.grab():
                            break
                        cur += 1
                    ok, f = cap.read()
                    if not ok:
                        break
                    cur += 1
                    _emit(produced, cv2.cvtColor(
                        cv2.resize(f, (size, size)), cv2.COLOR_BGR2RGB))
                    produced += 1
            else:
                while produced < sample_frames:
                    ok, f = cap.read()
                    if not ok:
                        break
                    _emit(produced, cv2.cvtColor(
                        cv2.resize(f, (size, size)), cv2.COLOR_BGR2RGB))
                    produced += 1
        finally:
            cap.release()
        decode_ms = (time.perf_counter_ns() - decode_started) / 1e6

        landmarks = np.zeros((produced, 2, 21, 3), np.float32)
        present = np.zeros((produced, 2), np.float32)
        for index, future in enumerate(pending):
            lm, pr = future.result()
            landmarks[index] = lm[0]
            present[index] = pr[0]
        return landmarks, present, decode_ms

    def close(self):
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._pool.close()


def _mirror_predict(mainmod, clip: Path, *, workers: int, sample_frames: int,
                    size: int, pipeline: bool):
    """One full landmark-path prediction, stage-timed, without editing src/."""
    import landmark_frontend as lf
    from features import build_features

    stages = {}
    if pipeline:
        if not hasattr(_mirror_predict, "_pipe"):
            _mirror_predict._pipe = _PipelinedPool(workers=workers, size=size)
        t = time.perf_counter_ns()
        landmarks, present, decode_ms = _mirror_predict._pipe.run(clip, sample_frames)
        stages["pipelined_total_ms"] = (time.perf_counter_ns() - t) / 1e6
        stages["decode_in_producer_ms"] = decode_ms
        stages["read_ms"] = decode_ms
        stages["detect_ms"] = stages["pipelined_total_ms"] - decode_ms
    else:
        t = time.perf_counter_ns()
        frames = lf.read_frames(str(clip), max_frames=sample_frames, size=size)
        stages["read_ms"] = (time.perf_counter_ns() - t) / 1e6
        if not frames:
            raise ValueError(f"No valid frames decoded from: {clip}")
        t = time.perf_counter_ns()
        if size == PROD_FRAME_SIZE:
            if not hasattr(_mirror_predict, "_pool"):
                import tracker as tk
                _mirror_predict._pool = tk.ParallelImageTracker(workers=workers)
            landmarks, present = _mirror_predict._pool.detect(frames)
        else:
            if not hasattr(_mirror_predict, "_size_pools"):
                _mirror_predict._size_pools = {}
            pool = _mirror_predict._size_pools.get(size)
            if pool is None:
                pool = _SizePool(workers=workers, size=size)
                _mirror_predict._size_pools[size] = pool
            landmarks, present = pool.detect(frames)
        stages["detect_ms"] = (time.perf_counter_ns() - t) / 1e6

    t = time.perf_counter_ns()
    tensor, _ = build_features(landmarks, present, expected_frames=mainmod._FRAMES)
    stages["features_ms"] = (time.perf_counter_ns() - t) / 1e6
    tensor = tensor[None, ...].astype(np.float32)

    t = time.perf_counter_ns()
    logits, _ = mainmod._run_ensemble(tensor)
    stages["ensemble_ms"] = (time.perf_counter_ns() - t) / 1e6

    exp = np.exp(logits - np.max(logits))
    probs = exp / np.sum(exp)
    top = np.argsort(-probs)[:5]
    top5 = [mainmod._IDX_TO_LABEL.get(int(i), f"UNKNOWN_{int(i)}") for i in top]
    return {
        "prediction": top5[0],
        "top5": top5,
        "confidence": float(probs[top[0]]),
        "tensor_sha": _sha(tensor),
        "tensor_sum": float(tensor.sum()),
        "landmarks_sha": _sha(landmarks),
        "stages": stages,
    }


def _run_child(variant: str, clips):
    cfg = VARIANTS[variant]
    mainmod = _load_main()
    mainmod._init_model()

    records = []
    timings = []

    def record(path):
        if cfg["kind"] == "predict":
            t = time.perf_counter()
            result = mainmod.predict(path, profile=True)
            elapsed = (time.perf_counter() - t) * 1000.0
            prof = mainmod.get_last_profile() or {}
            stages = {k: v["milliseconds"] for k, v in
                      (prof.get("stages") or {}).items()}
            return elapsed, {
                "prediction": result["prediction"],
                "top5": result["top5"],
                "confidence": result["confidence"],
                "stages": stages,
                "tensor_sha": None,
            }
        t = time.perf_counter()
        mirror = _mirror_predict(
            mainmod, path, workers=cfg["workers"],
            sample_frames=cfg["sample_frames"], size=cfg["size"],
            pipeline=cfg.get("pipeline", False))
        elapsed = (time.perf_counter() - t) * 1000.0
        if variant == "mirror_4":
            real = mainmod.predict(path, profile=True)
            mirror["real_prediction"] = real["prediction"]
            mirror["real_top5"] = real["top5"]
            mirror["mirror_matches_predict"] = (
                real["prediction"] == mirror["prediction"]
                and real["top5"] == mirror["top5"])
        return elapsed, mirror

    # Warm up (unmeasured): model already loaded, now warm landmarkers/decoders.
    record(clips[0])
    for path in clips:
        elapsed, payload = record(path)
        timings.append(elapsed)
        records.append({"clip": Path(path).name, "ms": elapsed, **payload})

    return {
        "variant": variant,
        "config": {k: v for k, v in cfg.items() if k != "kind"},
        "kind": cfg["kind"],
        "env": {k: os.environ.get(k) for k in
                ("ECHOGESTURE_TRACKER", "ECHOGESTURE_WORKERS", "OPENBLAS_NUM_THREADS")},
        "warm": _summ(timings),
        "peak_rss_mib": _peak_rss_mib(),
        "records": records,
        "clip_count": len(clips),
    }


# ---------------------------------------------------------------------------
# parent
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", type=Path, default=DEFAULT_CLIPS)
    parser.add_argument("--max-clips", type=int, default=12)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--_child", help=argparse.SUPPRESS)
    parser.add_argument("--_child-clips", nargs="+", help=argparse.SUPPRESS)
    parser.add_argument("--_child-out", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args._child:
        payload = _run_child(args._child, args._child_clips)
        args._child_out.write_text(json.dumps(payload, indent=2) + "\n",
                                   encoding="utf-8")
        return 0

    clips = _resolve_clips(args.clips, args.max_clips)
    variants = [v for v in args.variants if v in VARIANTS]
    if not variants:
        parser.error(f"no valid variants in {args.variants}")
    if "baseline_4" not in variants:
        variants.insert(0, "baseline_4")
    if CONTROL not in variants:
        variants.insert(1, CONTROL)

    try:
        os.mkdir(str(LOCK))
    except FileExistsError:
        print(f"refusing to start: heavy benchmark lock held at {LOCK}",
              file=sys.stderr)
        return 2

    try:
        runs, errors = {}, {}
        rounds = [[] for _ in variants]
        with tempfile.TemporaryDirectory(prefix="a02_candidates_") as tmp:
            tmpdir = Path(tmp)
            # Rotate variant order each round: machine load drifts over minutes,
            # so a fixed order would charge the drift to whichever variant runs
            # last.  Round-robin order spreads it across all variants.
            for round_index in range(args.rounds):
                order = variants[round_index % len(variants):] + \
                    variants[:round_index % len(variants)]
                for variant in order:
                    env = os.environ.copy()
                    for key in ("ECHOGESTURE_TRACKER", "ECHOGESTURE_WORKERS"):
                        env.pop(key, None)
                    cfg = VARIANTS[variant]
                    if cfg["kind"] == "predict":
                        env["ECHOGESTURE_TRACKER"] = "parallel"
                        env["ECHOGESTURE_WORKERS"] = str(cfg["workers"])
                    out_path = tmpdir / f"{variant}-r{round_index}.json"
                    cmd = [sys.executable, str(Path(__file__).resolve()),
                           "--_child", variant,
                           "--_child-clips", *[str(p) for p in clips],
                           "--_child-out", str(out_path)]
                    proc = subprocess.run(cmd, env=env, capture_output=True,
                                          text=True, timeout=1800)
                    if proc.returncode != 0 or not out_path.is_file():
                        errors[f"{variant}-r{round_index}"] = \
                            (proc.stderr or proc.stdout).strip()[-1200:]
                        print(f"[{variant} r{round_index}] FAILED: "
                              f"{errors[f'{variant}-r{round_index}'][-300:]}")
                        continue
                    run = json.loads(out_path.read_text(encoding="utf-8"))
                    run["round"] = round_index
                    rounds[variants.index(variant)].append(run)

        for variant, variant_runs in zip(variants, rounds):
            if not variant_runs:
                continue
            merged = dict(variant_runs[0])
            merged["warm"] = _summ(
                [ms for r in variant_runs
                 for rec in r["records"] for ms in [rec["ms"]]])
            merged["round_p50"] = [r["warm"]["p50_ms"] for r in variant_runs]
            merged["round_p95"] = [r["warm"]["p95_ms"] for r in variant_runs]
            merged["peak_rss_mib"] = max(r["peak_rss_mib"] for r in variant_runs)
            merged["records"] = variant_runs[0]["records"]
            runs[variant] = merged
            print(f"[{variant}] p50={merged['warm']['p50_ms']:7.2f} "
                  f"p95={merged['warm']['p95_ms']:7.2f} "
                  f"rounds={['%.0f' % v for v in merged['round_p50']]} "
                  f"rss={merged['peak_rss_mib']:6.1f} MiB")

        baseline = runs.get(CONTROL)
        comparisons = {}
        if baseline:
            base = {r["clip"]: r for r in baseline["records"]}
            for variant, run in runs.items():
                if variant == CONTROL:
                    continue
                pred_diff, top5_diff, tensor_diff, lm_diff = [], [], [], []
                for r in run["records"]:
                    b = base[r["clip"]]
                    if r["prediction"] != b["prediction"]:
                        pred_diff.append({"clip": r["clip"], "baseline": b["prediction"],
                                          "variant": r["prediction"]})
                    if r["top5"] != b["top5"]:
                        top5_diff.append(r["clip"])
                    if r.get("tensor_sha") and b.get("tensor_sha"):
                        if r["tensor_sha"] != b["tensor_sha"]:
                            tensor_diff.append(r["clip"])
                    if r.get("landmarks_sha") and b.get("landmarks_sha"):
                        if r["landmarks_sha"] != b["landmarks_sha"]:
                            lm_diff.append(r["clip"])
                comparisons[variant] = {
                    "prediction_mismatches": pred_diff,
                    "top5_mismatches": top5_diff,
                    "tensor_mismatches": tensor_diff,
                    "landmark_mismatches": lm_diff,
                    "p50_delta_ms": run["warm"]["p50_ms"] - baseline["warm"]["p50_ms"],
                    "p95_delta_ms": run["warm"]["p95_ms"] - baseline["warm"]["p95_ms"],
                    # Worst single round: the honest claim must hold even when
                    # the machine was busiest.
                    "worst_round_p95_ms": max(run["round_p95"]),
                    "control_worst_round_p95_ms": max(baseline["round_p95"]),
                }

        report = {
            "git_head": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                       capture_output=True, text=True).stdout.strip(),
            "clip_dir": str(args.clips),
            "clip_count": len(clips),
            "clips": [p.name for p in clips],
            "variants": variants,
            "control": CONTROL,
            "rounds": args.rounds,
            "runs": runs,
            "comparisons_vs_baseline": comparisons,
            "errors": errors,
            "machine": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "cpu_count": os.cpu_count(),
            },
            "python": sys.version.split()[0],
            "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
        for variant, cmp in comparisons.items():
            print(f"{variant:14s} dp50={cmp['p50_delta_ms']:+7.2f} "
                  f"dp95={cmp['p95_delta_ms']:+7.2f} "
                  f"pred_mismatch={len(cmp['prediction_mismatches'])} "
                  f"top5_mismatch={len(cmp['top5_mismatches'])} "
                  f"tensor_mismatch={len(cmp['tensor_mismatches'])}")
        return 0 if not errors else 1
    finally:
        try:
            os.rmdir(str(LOCK))
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    sys.exit(main())