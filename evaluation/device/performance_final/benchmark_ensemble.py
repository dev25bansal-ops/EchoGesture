"""Final end-to-end benchmark of `predict()` across tracker configurations.

Measures the real clip-to-JSON cost of the shipped system (whatever model
path `src/main.py` currently provides, singleton or ensemble) for:

    * singleton IMAGE tracker (ECHOGESTURE_TRACKER=image)
    * parallel tracker, 2 workers
    * parallel tracker, 4 workers

Each configuration runs in a fresh child process so its "cold" first call is a
true cold start (interpreter allocation) and its peak RSS is not polluted by a
previous configuration.  The parent holds the atomic lock
``/tmp/echogesture-heavy-benchmark.lock`` for the whole run and removes it in
``finally``; a stale lock fails fast rather than starting a second heavy run.

Measured per configuration: cold first call, warm p50/p95/min/max of full
clip-to-JSON, mean landmark-tracking and mean model-inference stage from
``get_last_profile()``, peak RSS, failure count and per-clip predictions.
Parity is the set of clips whose prediction label differs across configs.

Usage:
    .venv/bin/python evaluation/device/performance_final/benchmark_ensemble.py \
        [--clips DIR] [--max-clips 12] [--output results_ensemble/benchmark.json]

Nothing outside this file and its results directory is written.
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
MODELS = ROOT / "models"
DEFAULT_CLIPS = ROOT / "evaluation/device/v5_accuracy/data/val_30"
DEFAULT_OUTPUT = HERE / "results_ensemble/benchmark.json"

LOCK = Path("/tmp/echogesture-heavy-benchmark.lock")

CONFIGS = {
    "singleton": {"ECHOGESTURE_TRACKER": "image"},
    "parallel_2": {"ECHOGESTURE_TRACKER": "parallel", "ECHOGESTURE_WORKERS": "2"},
    "parallel_4": {"ECHOGESTURE_TRACKER": "parallel", "ECHOGESTURE_WORKERS": "4"},
}

# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    k = (len(ordered) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    frac = k - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def _summ(values: list[float]) -> dict:
    return {
        "count": len(values),
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "min_ms": min(values) if values else None,
        "max_ms": max(values) if values else None,
        "mean_ms": statistics.mean(values) if values else None,
    }


def _peak_rss_mib() -> float:
    """Peak RSS in MiB; ru_maxrss is bytes on macOS, KiB on Linux."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return raw / (1024 * 1024)
    return raw / 1024


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _stage_ms(profile, names):
    if not profile:
        return None
    stages = profile.get("stages") or {}
    for name in names:
        entry = stages.get(name)
        if entry is not None:
            return float(entry["milliseconds"])
    return None


def _inference_stage(profile):
    """Exact `model_inference` if present, else the sum of ensemble stages."""
    exact = _stage_ms(profile, ["model_inference"])
    if exact is not None:
        return exact, ["model_inference"]
    if not profile:
        return None, []
    stages = profile.get("stages") or {}
    keys = [k for k in stages if "inference" in k or "member" in k]
    if not keys:
        return None, []
    return sum(float(stages[k]["milliseconds"]) for k in keys), sorted(keys)


def _main_module():
    """Load src/main.py by path (it is not a package on the evaluator path)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("echogesture_main", SRC / "main.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SRC / 'main.py'}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _resolve_clips(clip_dir: Path, limit: int) -> list[Path]:
    clips = sorted(clip_dir.glob("*.mp4"))
    if not clips:
        raise SystemExit(f"no .mp4 clips in {clip_dir}")
    return clips[:limit]


# ---------------------------------------------------------------------------
# child: one configuration in a fresh process
# ---------------------------------------------------------------------------

def _run_child(config: str, clips: list[str]) -> dict:
    mainmod = _main_module()
    timings: list[float] = []
    tracking_ms: list[float] = []
    inference_ms: list[float] = []
    inference_keys: set[str] = set()
    predictions: dict[str, str] = {}
    failures: list[dict] = []

    def call(path: str) -> tuple[float, dict | None]:
        started = time.perf_counter()
        try:
            result = mainmod.predict(path, profile=True)
        except Exception as exc:  # reported, never hidden
            failures.append({"clip": Path(path).name, "error": f"{type(exc).__name__}: {exc}"})
            return (time.perf_counter() - started) * 1000.0, None
        elapsed = (time.perf_counter() - started) * 1000.0
        profile = mainmod.get_last_profile()
        track = _stage_ms(profile, ["landmark_tracking"])
        inf, keys = _inference_stage(profile)
        if track is not None:
            tracking_ms.append(track)
        if inf is not None:
            inference_ms.append(inf)
            inference_keys.update(keys)
        return elapsed, result

    # Cold: the very first predict() in this process (model init included).
    cold_ms, cold_result = call(clips[0])
    if cold_result is not None:
        predictions[clips[0]] = cold_result["prediction"]
    # One unmeasured warmup, then all measured warm calls.
    call(clips[0])
    for path in clips:
        elapsed, result = call(path)
        timings.append(elapsed)
        if result is not None:
            predictions[path] = result["prediction"]

    return {
        "config": config,
        "env": {k: os.environ.get(k) for k in ("ECHOGESTURE_TRACKER", "ECHOGESTURE_WORKERS")},
        "cold_ms": cold_ms,
        "warm": _summ(timings),
        "tracking_stage_mean_ms": statistics.mean(tracking_ms) if tracking_ms else None,
        "model_inference_stage_mean_ms": statistics.mean(inference_ms) if inference_ms else None,
        "model_inference_stage_keys": sorted(inference_keys),
        "peak_rss_mib": _peak_rss_mib(),
        "failures": failures,
        "failure_count": len(failures),
        "predictions": {Path(k).name: v for k, v in predictions.items()},
        "stage_names": sorted((mainmod.get_last_profile() or {}).get("stages", {}).keys()),
    }


# ---------------------------------------------------------------------------
# parent: lock, orchestrate children, aggregate
# ---------------------------------------------------------------------------

def _detect_model_mode() -> tuple[str, list[str]]:
    """Ensemble is ACTIVE only if meta lists members and predict() averages them.

    A half-landed edit (loader present, still invoking one interpreter) must not
    be reported as an ensemble measurement.
    """
    text = (SRC / "main.py").read_text(encoding="utf-8")
    meta_path = MODELS / "landmark_model_meta.json"
    meta_members = []
    if meta_path.is_file():
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(metadata.get("members"), list):
                meta_members = metadata["members"]
        except (ValueError, OSError):
            pass
    reasoning = [
        f"src/main.py references _MEMBER_IO={'_MEMBER_IO' in text}",
        f"src/main.py loops over _INTERPRETERS={'for ' in text and '_INTERPRETERS' in text}",
        f"meta members={len(meta_members)}",
    ]
    active = bool(meta_members) and "for " in text and "_INTERPRETERS" in text
    return ("ensemble" if active else "single_model"), reasoning


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT), *args],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except Exception:
        return ""


def _model_shas() -> dict[str, str]:
    shas = {}
    for path in sorted(MODELS.glob("landmark_ens_*.tflite")):
        shas[path.name] = _sha256(path)
    classifier = MODELS / "landmark_classifier_fp16.tflite"
    if classifier.is_file():
        shas[classifier.name] = _sha256(classifier)
    return shas


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", type=Path, default=DEFAULT_CLIPS)
    parser.add_argument("--max-clips", type=int, default=12)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS))
    # internal: child mode
    parser.add_argument("--_child", choices=sorted(CONFIGS), help=argparse.SUPPRESS)
    parser.add_argument("--_child-clips", nargs="+", help=argparse.SUPPRESS)
    parser.add_argument("--_child-out", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args._child:
        payload = _run_child(args._child, args._child_clips)
        args._child_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return 0

    clips = _resolve_clips(args.clips, args.max_clips)
    configs = [c for c in args.configs if c in CONFIGS]
    if not configs:
        parser.error(f"no valid configs in {args.configs}")

    try:
        os.mkdir(str(LOCK))
    except FileExistsError:
        print(f"refusing to start: heavy benchmark lock held at {LOCK}", file=sys.stderr)
        return 2

    try:
        main_py = SRC / "main.py"
        main_sha_before = _sha256(main_py)
        model_mode, markers = _detect_model_mode()
        head = _git("rev-parse", "HEAD")

        runs: dict[str, dict] = {}
        errors: dict[str, str] = {}
        with tempfile.TemporaryDirectory(prefix="echogesture_ensemble_") as tmp:
            tmpdir = Path(tmp)
            for config in configs:
                env = os.environ.copy()
                for key in ("ECHOGESTURE_TRACKER", "ECHOGESTURE_WORKERS"):
                    env.pop(key, None)
                env.update(CONFIGS[config])
                out_path = tmpdir / f"{config}.json"
                cmd = [
                    sys.executable, str(Path(__file__).resolve()),
                    "--_child", config,
                    "--_child-clips", *[str(p) for p in clips],
                    "--_child-out", str(out_path),
                ]
                started = time.perf_counter()
                proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
                wall = time.perf_counter() - started
                if proc.returncode != 0 or not out_path.is_file():
                    errors[config] = (
                        f"exit {proc.returncode}: {(proc.stderr or proc.stdout).strip()[-800:]}"
                    )
                    continue
                run = json.loads(out_path.read_text(encoding="utf-8"))
                run["wall_seconds"] = wall
                runs[config] = run
                print(f"[{config}] warm p50={run['warm']['p50_ms']:.1f} ms "
                      f"p95={run['warm']['p95_ms']:.1f} ms "
                      f"rss={run['peak_rss_mib']:.1f} MiB "
                      f"failures={run['failure_count']}")

        main_sha_after = _sha256(main_py)
        if main_sha_after != main_sha_before:
            errors["src/main.py"] = "changed during the run; results are not from one source revision"

        # Parity: same label for every clip across every measured config.
        labels: dict[str, dict[str, str]] = {}
        for config, run in runs.items():
            for clip, label in run["predictions"].items():
                labels.setdefault(clip, {})[config] = label
        disagreements = {
            clip: preds for clip, preds in labels.items()
            if len(set(preds.values())) > 1
        }
        comparable = [c for c in labels if len(labels[c]) == len(runs)]
        parity = bool(runs) and not disagreements and len(comparable) == len(labels)

        report = {
            "mode": "ensemble" if model_mode == "ensemble" else
                    "single-model baseline, ensemble pending",
            "model_mode_evidence": markers,
            "git_head": head,
            "src_main_sha256": main_sha_before,
            "model_shas": _model_shas(),
            "clip_dir": str(args.clips),
            "clip_count": len(clips),
            "clips": [p.name for p in clips],
            "configs": configs,
            "runs": runs,
            "errors": errors,
            "parity": parity,
            "parity_clips_compared": len(comparable),
            "parity_disagreements": disagreements,
            "machine": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "cpu_count": os.cpu_count(),
            },
            "python": sys.version.split()[0],
            "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "note": "timings are full predict() clip-to-JSON with profile=True; "
                    "cold is the first call in a fresh process; warm is 12 clips "
                    "after one unmeasured warmup",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
        print(f"mode={report['mode']} parity={parity} "
              f"disagreements={len(disagreements)} errors={len(errors)}")
        return 0 if not errors else 1
    finally:
        try:
            os.rmdir(str(LOCK))
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    sys.exit(main())