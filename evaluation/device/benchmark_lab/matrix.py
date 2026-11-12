"""Benchmark matrix runner that wraps run_manifest.evaluate_profiled OR
evaluate without editing run_manifest.py.

Each matrix cell runs predict() over the same manifest with a fixed
configuration. Identical-clip repeats are used for stability (median/IQR);
fresh-clip repeats reuse the manifest clips in order and must not be called
target-device p95 unless hardware == actual target and conditions recorded.

Stages are measured via get_last_profile() when available; failures are
reported separately from successes and never silently dropped. Power is
NEVER inferred from CPU/thermal data and reported as NOT_MEASURED.
"""

from __future__ import annotations

import json
import math
import platform
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_manifest  # noqa: E402


@dataclass(frozen=True)
class MatrixCellConfig:
    """One cell of the benchmark matrix."""

    name: str
    predictor_kwargs: dict = field(default_factory=dict)
    repeats: int = 1
    warmup: int = 1
    same_clip_id: str | None = None  # if set, repeat this single clip
    frame_size: int | None = None  # documentation only; real value read from profile
    profile: bool = True
    threads: str = "runtime default; not explicitly configured"
    note: str = ""


def _nearest_rank(sorted_vals, frac):
    if not sorted_vals:
        return None
    rank = max(0, math.ceil(frac * len(sorted_vals)) - 1)
    return sorted_vals[rank]


def _timing_summary(values):
    if not values:
        return {"count": 0, "p50_ms": None, "p90_ms": None, "p95_ms": None, "p99_ms": None,
                "mean_ms": None, "median_ms": None, "min_ms": None, "max_ms": None}
    s = sorted(values)
    return {
        "count": len(values),
        "p50_ms": _nearest_rank(s, 0.50),
        "p90_ms": _nearest_rank(s, 0.90),
        "p95_ms": _nearest_rank(s, 0.95),
        "p99_ms": _nearest_rank(s, 0.99),
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
        "iqr_ms": s[math.ceil(0.75 * len(s)) - 1] - s[math.ceil(0.25 * len(s)) - 1] if len(s) >= 4 else None,
    }


def _split_cold_warm(wall_ms):
    """First successful wall time is cold; the rest warm (mirrors run_manifest)."""
    if not wall_ms:
        return [], []
    return wall_ms[:1], wall_ms[1:]


def _aggregate_stage_profiles(profiles):
    """Aggregate stage timings across successful profiles."""
    by_stage = defaultdict(list)
    extras = defaultdict(list)
    for p in profiles:
        stages = p.get("stages", {}) or {}
        for name, data in stages.items():
            if isinstance(data, dict) and "milliseconds" in data:
                by_stage[name].append(float(data["milliseconds"]))
                for k, v in data.items():
                    if k != "milliseconds":
                        extras[f"{name}.{k}"].append(v)
    agg = {}
    for name, vals in by_stage.items():
        agg[name] = _timing_summary(vals)
    return agg, dict(extras)


def _git_sha(root):
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def run_cell(items, predict, labels, config, predict_module_name=None):
    """Run one matrix cell. Returns a summary dict.

    Failures are collected per-call and reported explicitly; partial accuracy
    is not reported when any failure occurs in profiled mode (delegating to
    evaluate_profiled's own accuracy rules for label completeness).
    """
    wall_ms = []
    profiles = []
    failures = []
    predictions_all = []
    accuracy = {"clips_with_ground_truth": 0, "correct": 0, "top1": None,
                "representative_accuracy_measured": False}

    # Determine work list: same-clip repeats or full manifest
    if config.same_clip_id is not None:
        match = [it for it in items if it[0] == config.same_clip_id]
        if not match:
            raise ValueError(f"same_clip_id {config.same_clip_id!r} not in manifest")
        work_items = match * config.repeats
    else:
        # fresh-clip repeats: repeat the full manifest repeats times
        work_items = list(items) * max(1, config.repeats)

    label_map = {cid: truth for cid, _, truth, *_ in items if truth}
    if label_map and len(label_map) != len(items):
        raise ValueError("Supply ground_truth for every clip or none")

    # Warmup calls (discarded, not counted)
    for i in range(config.warmup):
        cid, cpath, _, *_ = work_items[i % len(work_items)]
        try:
            kwargs = dict(config.predictor_kwargs)
            if config.profile:
                kwargs["profile"] = True
            try:
                import inspect
                if inspect.signature(predict).parameters.get("clip_id") is not None and \
                        inspect.signature(predict).parameters["clip_id"].kind is inspect.Parameter.KEYWORD_ONLY:
                    kwargs["clip_id"] = cid
            except (TypeError, ValueError):
                pass
            predict(str(cpath), **kwargs)
        except Exception:
            pass  # warmup failures ignored

    correct = 0
    total_labelled = 0
    for cid, cpath, truth, *_ in work_items:
        started = time.perf_counter_ns()
        kwargs = dict(config.predictor_kwargs)
        if config.profile:
            kwargs["profile"] = True
        # pass clip_id if supported keyword-only (also for profiled path)
        try:
            import inspect
            sig = inspect.signature(predict)
            if "clip_id" in sig.parameters and sig.parameters["clip_id"].kind is inspect.Parameter.KEYWORD_ONLY:
                kwargs["clip_id"] = cid
        except (TypeError, ValueError):
            pass
        try:
            value = predict(str(cpath), **kwargs)
            validated = run_manifest.validate_prediction(value, cid, labels)
        except Exception as exc:
            failures.append({"clip_id": cid, "error_type": type(exc).__name__, "error": str(exc)})
            continue
        wall_ms.append((time.perf_counter_ns() - started) / 1e6)
        predictions_all.append(validated)
        if config.profile:
            getter = None
            if predict_module_name:
                getter = getattr(sys.modules.get(predict_module_name), "get_last_profile", None)
            if getter is None:
                getter = getattr(sys.modules.get(getattr(predict, "__module__", ""), None), "get_last_profile", None)
            if callable(getter):
                sp = getter()
                if isinstance(sp, dict) and "stages" in sp:
                    profiles.append({"clip_id": cid, **sp})
        if truth:
            total_labelled += 1
            correct += validated["prediction"] == truth

    cold, warm = _split_cold_warm(wall_ms)
    stage_agg, stage_extras = _aggregate_stage_profiles(profiles)
    rep = len(predictions_all) >= 20 and not failures and total_labelled > 0
    accuracy = {
        "clips_with_ground_truth": len(label_map),
        "correct": correct if total_labelled else 0,
        "top1": (correct / total_labelled) if total_labelled and not failures else None,
        "representative_accuracy_measured": rep,
        "qualification": ("Representative accuracy requires a documented validation split; "
                          "flag is only completeness/size check" if total_labelled else
                          "No ground truth supplied"),
        "note": f"Accuracy computed over {total_labelled} labelled successful calls in this cell",
    }
    return {
        "cell": config.name,
        "config": {
            "repeats": config.repeats, "warmup": config.warmup,
            "same_clip_id": config.same_clip_id, "profile": config.profile,
            "predictor_kwargs": config.predictor_kwargs,
            "frame_size_claimed": config.frame_size,
            "threads": config.threads, "note": config.note,
            "repeat_mode": "identical_clip" if config.same_clip_id else "fresh_manifest",
        },
        "clip_count": len(work_items),
        "successful_predictions": len(predictions_all),
        "failures": failures,
        "failed_predictions": failures,
        "accuracy": accuracy,
        "latency_all_calls": _timing_summary(wall_ms),
        "cold_call": _timing_summary(cold),
        "warm_calls": _timing_summary(warm),
        "stage_aggregates": stage_agg,
        "stage_extras": stage_extras,
        "profiles_count": len(profiles),
        "all_wall_ms": wall_ms,
    }


def build_summary(cells, *, manifest_path, artifact_paths, fixture_type="real",
                  split="validation", dataset="ASL Citizen (30-clip validation batch)",
                  hardware_note="Mac laptop (Apple M5 Pro, macOS 26.3 arm64) — NOT target device",
                  recorded_at=None, code_commit=None, env_extra=None):
    """Assemble the top-level benchmark summary with exact fields."""
    return {
        "status": "passed_schema" if all(not c["failures"] for c in cells) else "completed_with_failures",
        "representative_performance_measured": all(
            c["latency_all_calls"]["count"] >= 20 and not c["failures"] for c in cells) and fixture_type == "real",
        "performance_qualification": (
            "N>=20 successful real clips per cell with zero failures; still NOT target-device evidence "
            "unless hardware is the actual target device (see hardware fields)." if fixture_type == "real"
            else "Representative performance NOT MEASURED: fixture_type is synthetic or run incomplete."),
        "fixture_type": fixture_type,
        "split": split,
        "dataset": dataset,
        "hardware_is_target_device": False,
        "hardware_note": hardware_note,
        "cells": cells,
        "artifacts": artifact_paths,
        "manifest": {"path": str(manifest_path), "sha256": run_manifest.sha256(Path(manifest_path))},
        "code_commit": code_commit,
        "environment": env_extra or {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "device_model": platform.node(),
            "inference_runtime": run_manifest._runtime_version(),
            "delegate": "XNNPACK CPU (LiteRT)",
            "thread_count": "runtime default; not explicitly configured",
        },
        "power": "NOT_MEASURED; requires a documented external meter or qualified platform method — do not infer from CPU",
        "recorded_at": recorded_at or datetime.now(timezone.utc).isoformat(),
        "scope": "Runtime/interface measurement only unless complete ground_truth is supplied; not target-device evidence",
    }


def stage_ranking(summary):
    """Rank stages by mean ms for the first cell (or merged). Useful in reports."""
    if not summary.get("cells"):
        return []
    agg = summary["cells"][0].get("stage_aggregates", {})
    rows = [(k, v["mean_ms"]) for k, v in agg.items() if v["mean_ms"] is not None]
    return sorted([r for r in rows if r[1] is not None], key=lambda x: -x[1])
