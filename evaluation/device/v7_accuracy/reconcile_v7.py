#!/usr/bin/env python3
"""V7 reconciliation of all 346-clip accuracy evidence.

Recomputes from frozen predictions only (no model inference, no test split):
- primary 346-clip FP16/INT8 accuracy + Wilson 95% (failure = incorrect)
- conditional 345-clip sensitivity
- paired FP16/INT8: both/FP16-only/INT8-only/both-wrong, exact McNemar,
  conditional paired-difference CI, agreement, logit/prob divergence
- signer-level uncertainty: per-signer Wilson, signer-mean, leave-one-signer-out,
  signer-cluster bootstrap (deterministic seed, clips resampled within signer)
- top-5, confusions, per-class, no-hand failure record
- cross-checks vs v5 summaries (val_346, val_30 subset consistency) and v6 statistics
- model/label/manifest/split hash pinning
Deterministic: sorted iteration, fixed bootstrap seed. Standard library only.
Does not claim validation accuracy is official held-out (test) performance.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

EXPECTED_FP16 = "a012f2b5f7a1110961079f19eb0e28c9abacb95e4d0787352e4157ebca7044a4"
EXPECTED_INT8 = "9d24640a23244a3cd00c9fa62257733819f86ed1da5a10194c9b1cb64323700b"
EXPECTED_LABELS = "e78f5028e97e2eaf86aff65af403af65061eebef010f0fbd41012293e8019a4a"
SEED = 20261109

def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def artifact(p: Path, expected: Optional[str] = None) -> Dict[str, Any]:
    v = {"path": str(p), "sha256": sha256(p), "bytes": p.stat().st_size}
    if expected is not None:
        v["expected_sha256"] = expected
        v["sha256_matches_expected"] = v["sha256"] == expected
    return v

def wilson(k: int, n: int, z: float = 1.96) -> List[float]:
    if n == 0 or not (0 <= k <= n):
        return [0.0, 0.0]
    p = k / n; z2 = z * z; den = 1 + z2 / n
    c = (p + z2 / (2 * n)) / den
    hw = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / den
    return [max(0.0, c - hw), min(1.0, c + hw)]

def exact_mcnemar_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))

def metric(k: int, n: int) -> Dict[str, Any]:
    return {"correct": k, "total": n, "accuracy": (k / n if n else None),
            "wilson_95": wilson(k, n), "errors": n - k}

def conditional_diff_ci(b: int, c: int) -> Dict[str, Any]:
    n = b + c
    if n == 0:
        return {"lower": None, "upper": None, "n_discordant": 0,
                "method": "conditional Wilson interval for b/(b+c), linearly transformed to paired accuracy difference"}
    lo, hi = wilson(b, n)
    return {"lower": 2 * lo - 1, "upper": 2 * hi - 1, "n_discordant": n,
            "method": "conditional Wilson interval for b/(b+c), linearly transformed to paired accuracy difference"}

def per_group(records: List[Dict[str, Any]], field: str) -> Dict[str, Any]:
    out = {}
    for g in sorted({str(r[field]) for r in records}):
        sub = [r for r in records if str(r[field]) == g]
        n = len(sub)
        fp = sum(1 for r in sub if r["fp16_top1_correct"])
        i8 = sum(1 for r in sub if r["int8_top1_correct"])
        out[g] = {"n": n, "failed": sum(1 for r in sub if r.get("status") != "ok"),
                  "fp16": metric(fp, n), "int8": metric(i8, n),
                  "fp16_minus_int8_pp": (fp / n - i8 / n) * 100}
    return out

def confusions(records: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    pref = key.split("_")[0]
    cnt = Counter((r["ground_truth"], r.get(key, "<FAILED>")) for r in records
                  if not r[f"{pref}_top1_correct"])
    return [{"truth": t, "prediction": p, "count": c}
            for (t, p), c in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))]

def clip_bootstrap(successful: List[Dict[str, Any]], seed: int = SEED, reps: int = 20000):
    """Deterministic clip-level bootstrap for paired difference (FP16-INT8)."""
    rng = random.Random(seed)
    n = len(successful)
    diffs = []
    for _ in range(reps):
        fp = i8 = 0
        for _ in range(n):
            r = successful[rng.randrange(n)]
            fp += r["fp16_top1_correct"]; i8 += r["int8_top1_correct"]
        diffs.append((fp - i8) / n)
    diffs.sort()
    return {"method": "nonparametric clip bootstrap, sorted percentile interval",
            "reps": reps, "seed": seed,
            "lower_95": diffs[int(0.025 * reps)], "upper_95": diffs[int(0.975 * reps) - 1],
            "median": diffs[reps // 2]}

def signer_bootstrap(records: List[Dict[str, Any]], seed: int = SEED + 1, reps: int = 20000):
    """Signer-cluster uncertainty: resample clips within each signer (stratified)
    and report bootstrap distribution of overall and signer-mean accuracy.
    Also leave-one-signer-out (LOSO) range. With only 5 signers, outer
    resampling of signers is not informative; stratified within-signer
    resampling is reported instead, with explicit limitation."""
    rng = random.Random(seed)
    by_signer: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        by_signer.setdefault(r["signer_id"], []).append(r)
    signers = sorted(by_signer)
    overall_fp, overall_i8, signer_mean_fp, signer_mean_i8 = [], [], [], []
    for _ in range(reps):
        fp = i8 = n = 0; sf = []; si = []
        for s in signers:
            sub = by_signer[s]; m = len(sub)
            sfp = si8 = 0
            for _ in range(m):
                r = sub[rng.randrange(m)]
                sfp += r["fp16_top1_correct"]; si8 += r["int8_top1_correct"]
            sf.append(sfp / m); si.append(si8 / m); fp += sfp; i8 += si8; n += m
        overall_fp.append(fp / n); overall_i8.append(i8 / n)
        signer_mean_fp.append(sum(sf) / len(sf)); signer_mean_i8.append(sum(si) / len(si))
    def pct(a: List[float]) -> Dict[str, Any]:
        a = sorted(a)
        return {"lower_95": a[int(0.025 * reps)], "upper_95": a[int(0.975 * reps) - 1], "median": a[reps // 2]}
    loso = []
    for s in signers:
        sub = [r for r in records if r["signer_id"] != s]
        fp = sum(1 for r in sub if r["fp16_top1_correct"]); i8 = sum(1 for r in sub if r["int8_top1_correct"])
        loso.append({"dropped_signer": s, "n_remaining": len(sub),
                     "fp16_accuracy": fp / len(sub), "int8_accuracy": i8 / len(sub)})
    return {"method": "stratified within-signer clip bootstrap (signers fixed, 5 signers); outer signer resampling omitted as uninformative with k=5",
            "reps": reps, "seed": seed,
            "overall_fp16": pct(overall_fp), "overall_int8": pct(overall_i8),
            "signer_mean_fp16": pct(signer_mean_fp), "signer_mean_int8": pct(signer_mean_i8),
            "leave_one_signer_out": loso,
            "signer_count": len(signers),
            "limitation": "With only 5 validation signers, signer-population generalization cannot be estimated; bootstrap quantifies within-sample clip variation conditional on these 5 signers."}

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--predictions30", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--fp16-model", type=Path, required=True)
    ap.add_argument("--int8-model", type=Path, required=True)
    ap.add_argument("--summary346", type=Path, required=True)
    ap.add_argument("--summary30", type=Path, required=True)
    ap.add_argument("--v6-statistics", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    records = json.loads(args.predictions.read_text(encoding="utf-8"))
    records30 = json.loads(args.predictions30.read_text(encoding="utf-8"))
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    assert isinstance(records, list) and isinstance(records30, list)
    assert len(records) == 346 and len(records30) == 30, (len(records), len(records30))
    assert len(labels) == 100 and set(labels.values()) == set(range(100))
    ids = [r["clip_id"] for r in records]
    assert len(set(ids)) == 346, "duplicate clip IDs"
    with args.manifest.open(newline="", encoding="utf-8-sig") as f:
        manifest = list(csv.DictReader(f))
    assert {r["clip_id"] for r in manifest} == set(ids), "manifest/prediction IDs differ"
    assert len(manifest) == 346

    successful = [r for r in records if r.get("status") == "ok"]
    failed = [r for r in records if r.get("status") != "ok"]
    assert len(successful) == 345 and len(failed) == 1
    fr = failed[0]
    assert fr["clip_id"] == "5989539775823847-INTUITIVE" and fr["signer_id"] == "P5"
    assert fr["ground_truth"] == "INTUITIVE" and "No hand detected" in fr.get("error", "")
    # failed record must carry error metadata and count incorrect for both
    assert fr["fp16_top1_correct"] is False and fr["int8_top1_correct"] is False

    n = 346
    fp = sum(1 for r in records if r["fp16_top1_correct"])
    i8 = sum(1 for r in records if r["int8_top1_correct"])
    assert (fp, i8) == (272, 254), (fp, i8)
    fp_ok = sum(1 for r in successful if r["fp16_top1_correct"])
    i8_ok = sum(1 for r in successful if r["int8_top1_correct"])
    both = sum(1 for r in successful if r["fp16_top1_correct"] and r["int8_top1_correct"])
    fp_only = sum(1 for r in successful if r["fp16_top1_correct"] and not r["int8_top1_correct"])
    i8_only = sum(1 for r in successful if r["int8_top1_correct"] and not r["fp16_top1_correct"])
    both_wrong = sum(1 for r in successful if not r["fp16_top1_correct"] and not r["int8_top1_correct"])
    assert (both, fp_only, i8_only, both_wrong) == (242, 30, 12, 61), (both, fp_only, i8_only, both_wrong)
    assert both + fp_only + i8_only + both_wrong == 345
    p_exact = exact_mcnemar_p(fp_only, i8_only)
    assert abs(p_exact - 0.007915897334896727) < 1e-12, p_exact
    agree = sum(1 for r in successful if r["models_agree_top1"])
    assert agree == 281
    top5_fp = sum(1 for r in records if r["fp16_top5_correct"])
    top5_i8 = sum(1 for r in records if r["int8_top5_correct"])
    assert (top5_fp, top5_i8) == (309, 298)

    per_signer = per_group(records, "signer_id")
    per_class = per_group(records, "ground_truth")
    assert {k: v["n"] for k, v in per_signer.items()} == {"signer_9123b627": 24, "signer_8c1eb12a": 104, "signer_1393dac4": 101, "signer_feb2f4a2": 103, "signer_2f68a231": 14}
    assert len(per_class) == 100 and sum(v["n"] for v in per_class.values()) == 346

    # val_30 subset consistency: every val_30 clip must match its val_346 record
    by_id = {r["clip_id"]: r for r in records}
    assert {r["clip_id"] for r in records30} <= set(by_id)
    mism = []
    for r30 in records30:
        r = by_id[r30["clip_id"]]
        for k in ("fp16_prediction", "int8_prediction", "fp16_top1_correct", "int8_top1_correct",
                  "fp16_top5_correct", "int8_top5_correct", "ground_truth", "signer_id"):
            if r30.get(k) != r.get(k):
                mism.append((r30["clip_id"], k, r30.get(k), r.get(k)))
    assert not mism, mism[:5]
    s30_fp = sum(1 for r in records30 if r["fp16_top1_correct"])
    s30_i8 = sum(1 for r in records30 if r["int8_top1_correct"])
    assert (s30_fp, s30_i8) == (22, 21), (s30_fp, s30_i8)

    # cross-check v5 summaries and v6 statistics agree with recomputation
    s346 = json.loads(args.summary346.read_text(encoding="utf-8"))
    s30 = json.loads(args.summary30.read_text(encoding="utf-8"))
    v6 = json.loads(args.v6_statistics.read_text(encoding="utf-8"))
    assert s346["fp16"]["top1_correct"] == fp and s346["int8"]["top1_correct"] == i8
    assert s346["fp16"]["top5_correct"] == top5_fp and s346["int8"]["top5_correct"] == top5_i8
    assert s346["paired"]["both_correct"] == both and s346["paired"]["fp16_only_correct"] == fp_only
    assert s346["paired"]["int8_only_correct"] == i8_only and s346["paired"]["both_wrong"] == both_wrong
    assert s30["fp16"]["top1_correct"] == s30_fp and s30["int8"]["top1_correct"] == s30_i8
    assert v6["primary_all_346"]["fp16"]["correct"] == fp and v6["primary_all_346"]["int8"]["correct"] == i8
    assert v6["n_total"] == 346 and v6["n_failed"] == 1

    out = {
        "analysis": "V7 reconciliation of 346-clip validation accuracy evidence (frozen predictions, no inference, no test split)",
        "scope": "Official ASL Citizen validation split only (346 vocab-eligible clips, 5 signers). Test split untouched: no predictions, tuning, or model selection on test. Validation accuracy is NOT claimed as official held-out test performance.",
        "artifacts": {
            "predictions346": artifact(args.predictions),
            "predictions30": artifact(args.predictions30),
            "manifest": artifact(args.manifest),
            "labels": artifact(args.labels, EXPECTED_LABELS),
            "fp16_model": artifact(args.fp16_model, EXPECTED_FP16),
            "int8_model": artifact(args.int8_model, EXPECTED_INT8),
            "summary346": artifact(args.summary346),
            "summary30": artifact(args.summary30),
            "v6_statistics": artifact(args.v6_statistics),
        },
        "n_total": n, "n_successful": len(successful), "n_failed": len(failed),
        "n_signers": len(per_signer), "signers": sorted(per_signer), "n_classes": len(per_class),
        "primary_all_346": {
            "fp16": metric(fp, n), "int8": metric(i8, n),
            "paired_difference_pp": (fp / n - i8 / n) * 100,
            "paired_difference_proportion": fp / n - i8 / n,
        },
        "sensitivity_successful_345": {"fp16": metric(fp_ok, 345), "int8": metric(i8_ok, 345)},
        "top5": {"fp16": metric(top5_fp, n), "int8": metric(top5_i8, n)},
        "paired": {
            "successful_n": len(successful), "both_correct": both, "fp16_only_correct": fp_only,
            "int8_only_correct": i8_only, "both_wrong": both_wrong,
            "discordant": fp_only + i8_only,
            "argmax_agreements": agree, "argmax_agreement_rate": agree / len(successful),
            "exact_mcnemar_two_sided_p": p_exact,
            "exact_mcnemar_method": "Binomial(b+c,0.5), doubled one-sided tail, capped at 1",
            "conditional_paired_difference_95": conditional_diff_ci(fp_only, i8_only),
            "mean_logit_maxabs_diff": sum(r["logit_maxabs_diff"] for r in successful) / 345,
            "mean_logit_l2_diff": sum(r["logit_l2_diff"] for r in successful) / 345,
            "mean_prob_l1_diff": sum(r["prob_l1_diff"] for r in successful) / 345,
            "mean_fp16_top1_confidence": sum(r["fp16_confidence"] for r in successful) / 345,
            "mean_int8_top1_confidence": sum(r["int8_confidence"] for r in successful) / 345,
        },
        "clip_bootstrap_paired_difference": clip_bootstrap(successful),
        "signer_level_uncertainty": signer_bootstrap(records),
        "per_signer": per_signer, "per_class": per_class,
        "confusions": {"fp16": confusions(records, "fp16_prediction"), "int8": confusions(records, "int8_prediction")},
        "failures": [{"clip_id": fr["clip_id"], "signer_id": fr["signer_id"], "ground_truth": fr["ground_truth"],
                      "error_type": fr.get("error_type"), "error": fr.get("error"),
                      "counted_incorrect_for_both": True}],
        "val30_consistency": {"n": 30, "fp16_correct": s30_fp, "int8_correct": s30_i8,
                              "all_30_predictions_match_val346": True},
        "cross_checks": {"v5_summary346_matches": True, "v5_summary30_matches": True, "v6_statistics_matches": True,
                         "model_hashes_match_expected": all([
                             artifact(args.fp16_model, EXPECTED_FP16)["sha256_matches_expected"],
                             artifact(args.int8_model, EXPECTED_INT8)["sha256_matches_expected"],
                             artifact(args.labels, EXPECTED_LABELS)["sha256_matches_expected"]])},
        "limitations": [
            "Primary unit is the clip; clips are clustered within only 5 validation signers, so intervals do not support signer-population generalization.",
            "One no-hand preprocessing failure counts as incorrect for both models in the primary analysis.",
            "Test split untouched; no held-out test accuracy is reported or implied.",
            "Near-duplicate content across splits cannot be ruled out from manifest/prediction data alone (see split integrity audit).",
            "Per-class N is 2-6 clips; per-class rates are descriptive only.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": "PASS", "fp16": out["primary_all_346"]["fp16"],
                      "int8": out["primary_all_346"]["int8"], "exact_p": p_exact,
                      "output": str(args.output)}, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
