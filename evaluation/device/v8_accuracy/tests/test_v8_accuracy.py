"""Unit and end-to-end reproducibility tests for the V8 accuracy package.

Run from the checkout root:
python3 -I evaluation/device/v8_accuracy/tests/test_v8_accuracy.py
Standard library only; no installs; no nested agents; no test-split predictions.
All paths resolve relative to the repository checkout (no ephemeral /tmp paths).
"""
import json, math, subprocess, sys, tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = ROOT / "evaluation/device/v8_accuracy"
VAL = ROOT / "evaluation/device/v5_accuracy/splits/val.csv"
TEST = ROOT / "evaluation/device/v5_accuracy/splits/test.csv"
# Official train CSV is available in two existing copies; primary is the
# in-repo copy, cross-checked against the fixture copy below.
TRAIN = ROOT / "evaluation/device/v5_accuracy/splits/train.csv"
TRAIN_FIXTURE = Path("/tmp/echogesture-real-fixture/train.csv")
LABELS = ROOT / "models/landmark_labels.json"
FP16 = ROOT / "models/landmark_classifier_fp16.tflite"
INT8 = ROOT / "models/landmark_classifier_int8.tflite"
PRED346 = ROOT / "evaluation/device/v5_accuracy/results/val_346/predictions.json"
PRED30 = ROOT / "evaluation/device/v5_accuracy/results/val_30/predictions.json"
MANIFEST = ROOT / "evaluation/device/v5_accuracy/data/val_346/manifest.csv"
SUM346 = ROOT / "evaluation/device/v5_accuracy/results/val_346/summary.json"
SUM30 = ROOT / "evaluation/device/v5_accuracy/results/val_30/summary.json"
V6STATS = ROOT / "evaluation/device/v6_integrity/statistics.json"
V5AUDIT = ROOT / "evaluation/device/v5_accuracy/results/split_audit.json"
V6AUDIT = ROOT / "evaluation/device/v6_integrity/leakage_audit.json"

def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, (cmd, r.stdout, r.stderr)
    return r

def close(a, b, tol=1e-9):
    return abs(a - b) <= tol

def sha256(p: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def main():
    sys.path.insert(0, str(HERE))
    from reconcile_v8 import wilson, exact_mcnemar_p, conditional_diff_ci

    # Wilson oracles (match frozen summaries)
    lo, hi = wilson(272, 346)
    assert close(lo, 0.7399025495676037, 5e-9) and close(hi, 0.8260678781917024, 5e-9)
    lo2, hi2 = wilson(254, 346)
    assert close(lo2, 0.6851648107396099, 5e-9) and close(hi2, 0.7779019028816404, 5e-9)
    assert wilson(0, 346)[0] == 0.0 and wilson(346, 346)[1] > 0.999
    # McNemar oracles
    assert close(exact_mcnemar_p(30, 12), 0.007915897334896727, 1e-12)
    assert exact_mcnemar_p(5, 5) == 1.0 and exact_mcnemar_p(0, 0) == 1.0
    ci = conditional_diff_ci(30, 12)
    assert ci["n_discordant"] == 42 and ci["lower"] > 0

    rec = json.loads((HERE / "results/reconciliation.json").read_text())
    integ = json.loads((HERE / "results/split_integrity.json").read_text())

    assert rec["n_total"] == 346 and rec["n_successful"] == 345 and rec["n_failed"] == 1
    assert rec["primary_all_346"]["fp16"]["correct"] == 272
    assert rec["primary_all_346"]["int8"]["correct"] == 254
    assert close(rec["primary_all_346"]["fp16"]["accuracy"], 272 / 346)
    assert close(rec["primary_all_346"]["int8"]["accuracy"], 254 / 346)
    assert rec["primary_all_346"]["fp16"]["wilson_95"] == list(wilson(272, 346))
    assert rec["primary_all_346"]["int8"]["wilson_95"] == list(wilson(254, 346))
    assert close(rec["primary_all_346"]["paired_difference_pp"], (272 - 254) / 346 * 100)
    assert rec["sensitivity_successful_345"]["fp16"]["correct"] == 272
    assert rec["sensitivity_successful_345"]["int8"]["correct"] == 254
    assert rec["top5"]["fp16"]["correct"] == 309 and rec["top5"]["int8"]["correct"] == 298
    p = rec["paired"]
    assert (p["both_correct"], p["fp16_only_correct"], p["int8_only_correct"], p["both_wrong"]) == (242, 30, 12, 61)
    assert p["both_correct"] + p["fp16_only_correct"] + p["int8_only_correct"] + p["both_wrong"] == 345
    assert close(p["exact_mcnemar_two_sided_p"], 0.007915897334896727, 1e-12)
    assert p["argmax_agreements"] == 281
    assert close(p["mean_logit_maxabs_diff"], 1.7423469248025314, 1e-9)
    assert close(p["mean_prob_l1_diff"], 0.584138337360776, 1e-9)
    # per-signer / per-class partition
    assert sum(v["n"] for v in rec["per_signer"].values()) == 346
    assert sum(v["n"] for v in rec["per_class"].values()) == 346
    assert len(rec["per_class"]) == 100 and len(rec["per_signer"]) == 5
    assert rec["per_signer"]["P5"]["failed"] == 1 and rec["per_signer"]["P12"]["n"] == 24
    # confusions partition errors
    assert sum(c["count"] for c in rec["confusions"]["fp16"]) == 346 - 272
    assert sum(c["count"] for c in rec["confusions"]["int8"]) == 346 - 254
    assert {"truth": "BEE1", "prediction": "BREAKFAST1", "count": 2} in rec["confusions"]["fp16"]
    assert {"truth": "ZEBRA", "prediction": "JEWELRY", "count": 2} in rec["confusions"]["int8"]
    # failure record
    f = rec["failures"][0]
    assert f["clip_id"] == "5989539775823847-INTUITIVE" and f["signer_id"] == "P5"
    assert f["error_type"] == "ValueError" and "No hand detected" in f["error"]
    assert f["counted_incorrect_for_both"] is True
    # clip ledger counts every clip exactly once
    ledger = rec["clip_ledger"]
    assert len(ledger) == 346 and len({x["clip_id"] for x in ledger}) == 346
    assert Counter(x["outcome"] for x in ledger) == {"BOTH_CORRECT": 242, "FP16_ONLY": 30, "INT8_ONLY": 12, "BOTH_WRONG": 61, "FAILED": 1}
    # val30 consistency + hashes
    assert rec["val30_consistency"]["fp16_correct"] == 22 and rec["val30_consistency"]["int8_correct"] == 21
    assert rec["cross_checks"]["model_hashes_match_expected"] is True
    assert rec["artifacts"]["fp16_model"]["sha256"] == "a012f2b5f7a1110961079f19eb0e28c9abacb95e4d0787352e4157ebca7044a4"
    assert rec["artifacts"]["int8_model"]["sha256"] == "9d24640a23244a3cd00c9fa62257733819f86ed1da5a10194c9b1cb64323700b"
    assert rec["artifacts"]["labels"]["sha256_matches_expected"] is True
    # bootstrap: deterministic ranges contain point estimates qualitatively
    cb = rec["clip_bootstrap_paired_difference"]
    assert cb["seed"] == 20261109 and cb["reps"] == 20000 and cb["lower_95"] < cb["median"] < cb["upper_95"]
    sb = rec["signer_level_uncertainty"]
    assert sb["signer_count"] == 5 and len(sb["leave_one_signer_out"]) == 5
    for row in sb["leave_one_signer_out"]:
        assert 0.0 <= row["fp16_accuracy"] <= 1.0 and 0.0 <= row["int8_accuracy"] <= 1.0
    # split integrity
    assert integ["status"] == "PASS_WITH_LIMITATIONS" and all(integ["checks"].values())
    assert (integ["splits"]["train"]["eligible_rows"], integ["splits"]["val"]["eligible_rows"],
            integ["splits"]["test"]["eligible_rows"]) == (1789, 346, 1255)
    for pair in (integ["overlaps_full"], integ["overlaps_eligible"]):
        for v in pair.values():
            assert v["signer_overlap"] == [] and v["filename_overlap"] == [] and v["clip_id_overlap"] == []
    # train CSV: in-repo copy hash + fixture cross-check (both existing copies)
    assert integ["inputs"]["train_csv"]["sha256"] == "87a105649b98ea577182ebfce49d59fe9ca834a2489384f6fe7bc03f14966329"
    assert sha256(TRAIN) == "87a105649b98ea577182ebfce49d59fe9ca834a2489384f6fe7bc03f14966329"
    if TRAIN_FIXTURE.exists():
        assert sha256(TRAIN_FIXTURE) == sha256(TRAIN), "fixture train copy differs from in-repo copy"

    # reproducibility: reruns must reproduce exact primary numbers
    with tempfile.TemporaryDirectory() as tmp:
        r_out, i_out = Path(tmp) / "r.json", Path(tmp) / "i.json"
        run([sys.executable, "-I", str(HERE / "reconcile_v8.py"),
             "--predictions", str(PRED346), "--predictions30", str(PRED30),
             "--manifest", str(MANIFEST), "--labels", str(LABELS),
             "--fp16-model", str(FP16), "--int8-model", str(INT8),
             "--summary346", str(SUM346), "--summary30", str(SUM30),
             "--v6-statistics", str(V6STATS), "--output", str(r_out)])
        run([sys.executable, "-I", str(HERE / "split_integrity_v8.py"),
             "--train-csv", str(TRAIN), "--val-csv", str(VAL), "--test-csv", str(TEST),
             "--labels", str(LABELS), "--v5-audit", str(V5AUDIT), "--v6-audit", str(V6AUDIT),
             "--output", str(i_out)])
        r2, i2 = json.loads(r_out.read_text()), json.loads(i_out.read_text())
        assert r2["primary_all_346"] == rec["primary_all_346"] and r2["paired"] == rec["paired"]
        assert r2["per_signer"] == rec["per_signer"] and r2["per_class"] == rec["per_class"]
        assert r2["confusions"] == rec["confusions"] and r2["failures"] == rec["failures"]
        assert r2["clip_ledger"] == rec["clip_ledger"]
        assert i2["splits"] == integ["splits"] and i2["overlaps_full"] == integ["overlaps_full"]

    # predictions contain exactly the 346 manifest clip IDs, 5 signers, 100 classes
    records = json.loads(PRED346.read_text())
    import csv
    manifest_rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8-sig")))
    assert {r["clip_id"] for r in manifest_rows} == {r["clip_id"] for r in records}
    assert sorted({r["signer_id"] for r in records}) == ["P12", "P21", "P26", "P39", "P5"]
    assert Counter(r["status"] for r in records) == {"ok": 345, "failed": 1}
    # manifest equals vocab-eligible val rows exactly (independent count)
    labels = json.loads(LABELS.read_text())
    vocab = {k.strip().upper() for k in labels}
    val_rows = list(csv.DictReader(VAL.open(encoding="utf-8-sig")))
    elig_ids = {Path(r["Video file"]).stem for r in val_rows if r["Gloss"].strip().upper() in vocab}
    assert elig_ids == {r["clip_id"] for r in manifest_rows} == {r["clip_id"] for r in records}
    print("V8 accuracy tests: all passed")

if __name__ == "__main__":
    main()
