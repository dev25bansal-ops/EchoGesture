"""Reproducible paired robustness evaluation for locally supplied labelled clips.

Each clip in the manifest is scored twice -- once clean, once with a
seeded corruption applied to every frame -- and the pair is recorded so
the clean/corrupted comparison is exact.  Per-clip corruption seeds are
derived from the run seed + clip position + clip_id
(``robustness_lab.config.derive_clip_seed``), so a run is reproducible
from (manifest, config) alone.

Representative marking: only a run that (a) is explicitly declared as a
labelled validation split (``--validation --split validation``), (b) has
>= 20 clips, (c) covers >= 2 distinct signers, and (d) has zero
clean/corrupted/decode failures may be marked representative.  Everything
else is a transform check and is labelled as such in every output.

Failure accounting: a clip whose clean prediction, corrupted prediction,
or decode fails is kept in the denominator, counted in
``failed_pair_count`` / ``failure_breakdown``, and listed with an error
string in ``pairs.jsonl``.  Nothing is silently dropped.

Outputs (in ``--output-dir``): ``robustness.json`` (summary + config +
artifact SHAs), ``pairs.jsonl``, ``pairs.csv``, ``report.md``.

Public export note: ``pairs.jsonl``/summary carry ``signer_id`` entries.
Sanitise with ``v8_privacy.sanitize_evidence.sanitize_pairs_jsonl`` /
``sanitize_robustness_summary`` before any public redistribution.
"""

import argparse
import json
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from corruptions import CORRUPTION_VERSION, CorruptionSpec, corrupt_frames, decode_video, write_video
import run_manifest
from robustness_lab import build_config, config_hash, derive_clip_seed
from robustness_lab.reporting import write_evidence
from robustness_lab.stats import aggregate_pairs


def _validate_labels(items):
    labels_by_id = {clip_id: truth for clip_id, _, truth, *_ in items if truth}
    if not labels_by_id:
        raise run_manifest.EvaluationError(
            "Robustness requires ground_truth for every clip; unlabelled clips cannot be scored"
        )
    if len(labels_by_id) != len(items):
        raise run_manifest.EvaluationError(
            "Robustness requires ground_truth for every clip; partial labelling is not scored"
        )
    return labels_by_id


def _signers(items):
    return sorted({(row[3] or "").strip() for row in items if len(row) > 3 and (row[3] or "").strip()})


def _is_representative(validation_flag, split, items, failures):
    """Single source of truth for the representative gate."""
    if not validation_flag or split != "validation":
        return False
    if len(items) < 20:
        return False
    if len(_signers(items)) < 2:
        return False
    if failures.get("clean_prediction_failed") or failures.get("corrupted_prediction_failed") \
            or failures.get("decode_error"):
        return False
    return True


def paired_evaluate(items, predict, labels, corruption, output_dir, representative=False):
    """Run clean vs corrupted prediction for every item; return summary + pairs.

    ``corruption`` is a ``CorruptionSpec``; its ``seed`` is the run seed
    from which per-clip seeds are derived.  ``output_dir/work`` is used
    for temporary corrupted clips and is always removed on exit, even
    when a clip fails.
    """
    _validate_labels(items)
    output_dir.mkdir(parents=True, exist_ok=True)
    workdir = output_dir / "work"
    workdir.mkdir(exist_ok=True)
    pairs = []
    failure_counts = Counter()
    try:
        for index, (clip_id, clip_path, truth, *rest) in enumerate(items):
            signer_id = (rest[0] if rest else "") or ""
            clip_seed = derive_clip_seed(corruption.seed, index, clip_id)
            per_clip = CorruptionSpec(corruption.kind, corruption.severity, clip_seed)

            clean_output = predict_clip(clip_id, clip_path, predict, labels)
            corrupted_output, decode_error = _predict_corrupted(
                clip_id, clip_path, per_clip, index, workdir, predict, labels
            )

            pair = {
                "clip_id": clip_id,
                "signer_id": signer_id,
                "ground_truth": truth,
                "corruption_seed": clip_seed,
                "clean_prediction": clean_output["prediction"] if clean_output else None,
                "corrupted_prediction": corrupted_output["prediction"] if corrupted_output else None,
                "clean_correct": bool(clean_output) and clean_output["prediction"] == truth,
                "corrupted_correct": bool(corrupted_output) and corrupted_output["prediction"] == truth,
                "clean_latency_ms": clean_output.get("latency_ms") if clean_output else None,
                "corrupted_latency_ms": corrupted_output.get("latency_ms") if corrupted_output else None,
            }
            if clean_output and clean_output.get("error"):
                pair["clean_error"] = clean_output["error"]
                failure_counts["clean_prediction_failed"] += 1
            if corrupted_output and corrupted_output.get("error"):
                if decode_error:
                    pair["decode_error"] = corrupted_output["error"]
                    failure_counts["decode_error"] += 1
                else:
                    failure_counts["corrupted_prediction_failed"] += 1
                    pair["corrupted_error"] = corrupted_output["error"]
            pairs.append(pair)
    finally:
        # Pinned (D1): corrupted derivative clips must never outlive
        # the run, even when a decode/write error aborts the loop.
        shutil.rmtree(workdir, ignore_errors=True)

    agg = aggregate_pairs(pairs)
    signer_count = agg["signer_count"]
    is_rep = bool(representative) and _is_representative(
        validation_flag=True, split="validation", items=items,
        failures=dict(failure_counts),
    )
    summary = {
        "status": "paired_evaluation_complete",
        "representative_robustness_measured": is_rep,
        "sample_count": agg["sample_count"],
        "signer_count": signer_count,
        "signer_ids": agg["signer_ids"],
        "affected_signer_count": signer_count if agg["signer_ids"] != ["unknown"] else None,
        "pairs": pairs,
        "clean_correct": agg["clean_correct"],
        "corrupted_correct": agg["corrupted_correct"],
        "clean_top1": agg["clean_top1"],
        "corrupted_top1": agg["corrupted_top1"],
        "accuracy_drop_percentage_points": agg["accuracy_drop_percentage_points"],
        "paired_flip_count": agg["paired_flip_count"],
        "paired_flip_rate": agg["paired_flip_rate"],
        "clean_correct_corrupted_wrong": agg["clean_correct_corrupted_wrong"],
        "clean_wrong_corrupted_correct": agg["clean_wrong_corrupted_correct"],
        "failed_pair_count": agg["failed_pair_count"],
        "failure_breakdown": dict(failure_counts),
        "clean_top1_ci95": agg["clean_top1_ci95"],
        "corrupted_top1_ci95": agg["corrupted_top1_ci95"],
        "per_signer": agg["per_signer"],
        "prediction_agreement_rate": agg["prediction_agreement_rate"],
        "corruption_type": corruption.kind,
        "corruption_severity": corruption.severity,
        "corruption_version": CORRUPTION_VERSION,
        "seed": corruption.seed,
        "note": (
            "Representative validation run: documented labelled validation split, "
            ">=20 clips, >=2 signers, zero failures."
            if is_rep else
            "Transform check only: not representative (requires --validation --split validation, "
            ">=20 labelled clips, >=2 signers, zero failures)."
        ),
    }
    return summary


def _predict_corrupted(clip_id, clip_path, spec, index, workdir, predict, labels):
    """Corrupt one clip's frames and predict; returns (output|None, is_decode_error)."""
    try:
        frames, fps = decode_video(clip_path)
    except (ValueError, OSError) as exc:
        return {"prediction": None, "error": f"decode: {type(exc).__name__}: {exc}"}, True
    corrupted_path = workdir / f"{index:08d}.mp4"
    try:
        write_video(corrupted_path, corrupt_frames(frames, spec), fps)
    except (ValueError, OSError) as exc:
        return {"prediction": None, "error": f"corrupt_write: {type(exc).__name__}: {exc}"}, True
    try:
        return predict_clip(clip_id, corrupted_path, predict, labels), False
    finally:
        corrupted_path.unlink(missing_ok=True)


def predict_clip(clip_id, clip_path, predict, labels):
    """Predict one clip; return the validated dict or an error dict (never raises)."""
    try:
        if run_manifest._accepts_clip_id(predict):
            value = predict(str(clip_path), clip_id=clip_id)
        else:
            value = predict(str(clip_path))
        return run_manifest.validate_prediction(value, clip_id, labels)
    except Exception as exc:
        return {"prediction": None, "error": f"{type(exc).__name__}: {exc}"}


def main(argv=None):
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--predictor", type=Path, default=root / "src/main.py")
    parser.add_argument("--labels", type=Path, default=root / "models/label100.json")
    parser.add_argument("--model", type=Path, default=root / "models/s3d_conv1d_qat_int8.tflite")
    parser.add_argument("--corruption", required=True,
                        choices=("occlusion", "low_light", "motion_blur", "crop", "frame_drop"))
    parser.add_argument("--severity", type=float, default=None,
                        help="Corruption severity in (0, 1]; defaults per corruption kind")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validation", action="store_true",
                        help="Declare this manifest a documented labelled validation split (not locked test data)")
    parser.add_argument("--split", default="unspecified")
    parser.add_argument("--min-signers", type=int, default=2,
                        help="Minimum distinct signers required for representative marking (default 2)")
    args = parser.parse_args(argv)
    try:
        manifest_path = args.manifest.resolve()
        items = run_manifest.read_manifest(manifest_path)
        labels = run_manifest.load_labels(args.labels)
        unknown = {truth for _, _, truth, *_ in items if truth} - labels
        if unknown:
            raise run_manifest.EvaluationError(f"Ground truth outside vocabulary: {sorted(unknown)}")
        for artifact in (args.predictor, args.labels, args.model):
            if not artifact.is_file():
                raise run_manifest.EvaluationError(f"Artifact not found: {artifact}")
        if args.validation and args.split != "validation":
            raise run_manifest.EvaluationError(
                "Representative robustness requires --split validation (with --validation) and at least 20 labelled clips"
            )
        if args.validation and len(items) < 20:
            raise run_manifest.EvaluationError(
                "Representative robustness requires at least 20 labelled clips"
            )
        model_sha = run_manifest.sha256(args.model)
        manifest_sha = run_manifest.sha256(manifest_path)
        config = build_config(
            args.corruption, severity=args.severity, seed=args.seed,
            split=args.split, validation=args.validation,
            manifest_sha256=manifest_sha, model_sha256=model_sha,
        )
        predict = run_manifest.load_predictor(args.predictor)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        summary = paired_evaluate(
            items, predict, labels,
            CorruptionSpec(config.corruption, config.severity, config.seed),
            args.output_dir,
            representative=args.validation,
        )
        # Representative also needs the signer-diversity leg, checked here
        # against the CLI --min-signers override.
        if summary["representative_robustness_measured"] and summary["signer_count"] < args.min_signers:
            summary["representative_robustness_measured"] = False
            summary["note"] = (
                f"Transform check only: signer coverage {summary['signer_count']} "
                f"< required {args.min_signers}."
            )
        summary["split"] = args.split
        summary["config"] = config.to_dict()
        summary["config_hash"] = config_hash(config)
        summary.update({
            "model_sha256": model_sha,
            "manifest_sha256": manifest_sha,
            "code_commit": run_manifest.git_sha(root),
            "artifacts": {
                name: {"sha256": run_manifest.sha256(path), "bytes": path.stat().st_size}
                for name, path in (("predictor", args.predictor), ("labels", args.labels), ("model", args.model))
            },
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        })
        pairs = summary.pop("pairs", [])
        write_evidence(args.output_dir, summary, pairs)
    except (run_manifest.EvaluationError, OSError, ValueError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
