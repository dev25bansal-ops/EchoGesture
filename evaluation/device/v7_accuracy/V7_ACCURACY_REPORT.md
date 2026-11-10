# EchoGesture V7 Accuracy Reconciliation — 346-Clip Validation Evidence

Owner: Agent A07. Owned path only: `evaluation/device/v7_accuracy/` in the
isolated copy `/tmp/echogesture-v7-agents/A07`. No edits to the original
checkout, Dev model/training code, or any other agent's directory. Test split
untouched (counts/overlaps only; no predictions, tuning, or selection).

> **Read first:** results below are on the **official ASL Citizen validation
> split** (346 vocab-eligible clips, 5 signers). They are **not** official
> held-out (test) performance and no such claim is made.

## 1. Artifacts (hash-pinned, verified in code)

| Artifact | SHA-256 |
|---|---|
| FP16 `models/landmark_classifier_fp16.tflite` | `a012f2b5f7a1110961079f19eb0e28c9abacb95e4d0787352e4157ebca7044a4` |
| INT8 `models/landmark_classifier_int8.tflite` | `9d24640a23244a3cd00c9fa62257733819f86ed1da5a10194c9b1cb64323700b` |
| Labels `models/landmark_labels.json` (100 classes, 0–99 permutation) | `e78f5028e97e2eaf86aff65af403af65061eebef010f0fbd41012293e8019a4a` |
| Predictions val_346 `v5_accuracy/results/val_346/predictions.json` | `8d980fbe5ae41613b578f411855e0d1d96e4ac178ac5a84d9aae5f1a65546555` |
| Official train.csv `/tmp/echogesture-real-fixture/train.csv` | `87a105649b98ea577182ebfce49d59fe9ca834a2489384f6fe7bc03f14966329` |
| Official val.csv `v5_accuracy/splits/val.csv` | `6ac78c9d551dfcc81d225a5279d7637c2891a4490851234be6e4674666e21366` |
| Official test.csv `v5_accuracy/splits/test.csv` | `b1dee8f81204895ca1760f6bd582eec84cf390b9c97c4387e49b9cc6b906b21e` |
| val_346 manifest `v5_accuracy/data/val_346/manifest.csv` | `6b9a6a253a5d3edc55ca8bbb5ae138c832e8b0f5ac0f374fdc1df12085c3ab85` |

Analysis is recomputed from **frozen predictions only** — no model inference was
re-run, so paired FP16/INT8 comparisons use identical feature tensors by
construction. Detector mode: `image`.

## 2. Split integrity (recomputed + cross-checked vs v5/v6 audits)

| Split | Full rows | Signers (full) | Vocab-eligible | Eligible signers | Eligible glosses |
|---|---|---|---|---|---|
| train | 40,154 | 35 | **1,789** | 32 | 100 |
| val | 10,304 | 6 | **346** | 5 (P5/P12/P21/P26/P39) | 100 |
| test | 32,941 | 11 | **1,255** | 11 | 100 |

Exact overlaps (signer / filename / clip-ID stem), full and eligible subsets:
**all empty** across train↔val, train↔test, val↔test. No within-split duplicate
filenames or clip IDs. Each eligible split covers all 100 classes. This matches
`v5_accuracy/results/split_audit.json` and `v6_integrity/leakage_audit.json`
(status `PASS_WITH_LIMITATIONS`) — the earlier v5 `UNRESOLVED` training-split
question is resolved for exact manifest overlap by the official train.csv.
**Not verified:** near-duplicate/re-encoded content, signer aliases, whether
training actually consumed only train rows, test accuracy.

## 3. Primary result — all 346 clips (failure counted incorrect)

| Model | Top-1 | Wilson 95% | Top-5 | Wilson 95% |
|---|---|---|---|---|
| FP16 | 272/346 = **78.61%** | [73.99%, 82.61%] | 309/346 = 89.31% | [85.61%, 92.14%] |
| INT8 | 254/346 = **73.41%** | [68.52%, 77.79%] | 298/346 = 86.13% | [82.09%, 89.37%] |

FP16 − INT8 = **+5.20 pp** (+0.0520 proportion).

Sensitivity (345 successful clips only): FP16 272/345 = 78.84% [74.22%, 82.83%];
INT8 254/345 = 73.62% [68.73%, 77.99%]. The single excluded clip is the no-hand
failure below; it is never silently dropped (see §7).

## 4. Paired FP16 vs INT8 (345 successful clips)

| Outcome | Clips |
|---|---:|
| Both correct | 242 |
| FP16 only correct | **30** |
| INT8 only correct | **12** |
| Both wrong | 61 |

- Exact two-sided McNemar p = **0.007916** (Binomial(42, 0.5), doubled tail).
  At α=0.05 this rejects symmetric marginal correctness *in this fixed
  validation sample* — not a universal superiority or signer-generalization claim.
- Conditional paired-difference Wilson 95% (transformed): [0.129, 0.657] on the
  discordant-conditional scale.
- Clip bootstrap of paired difference (20,000 reps, seed 20261109): 95% percentile
  interval **[0.017, 0.090]**, median 0.052 — consistent with McNemar.
- Argmax agreement 281/345 = 81.4%; identical top-5 sets per v6 statistics.
- Divergence: mean |logit| diff 1.74, mean logit-L2 5.35, mean prob-L1 0.58.
  Mean top-1 confidence: FP16 0.47 vs INT8 0.26 (INT8 logits are compressed;
  confidence is not calibrated across precisions — do not compare as such).

Wilson overlap is not a paired test; the McNemar/bootstrap results above are the
paired evidence. Clip-level Wilson intervals also ignore within-signer clustering.

## 5. Signer-level uncertainty (only 5 signers — treat as descriptive)

| Signer | N | FP16 | Wilson 95% | INT8 | Wilson 95% |
|---|---|---|---|---|---|
| P12 | 24 | 13/24 = 54.2% | [35.1%, 72.1%] | 8/24 = 33.3% | [18.0%, 53.3%] |
| P21 | 104 | 79/104 = 76.0% | [66.9%, 83.2%] | 74/104 = 71.2% | [61.8%, 79.0%] |
| P26 | 101 | 82/101 = 81.2% | [72.5%, 87.6%] | 80/101 = 79.2% | [70.3%, 86.0%] |
| P39 | 103 | 92/103 = 89.3% | [81.9%, 93.9%] | 87/103 = 84.5% | [76.2%, 90.2%] |
| P5 | 14 | 6/14 = 42.9% | [21.4%, 67.4%] | 5/14 = 35.7% | [16.3%, 61.2%] |

- Stratified within-signer bootstrap (20,000 reps, seed 20261110, signers fixed):
  overall FP16 [74.3%, 82.7%], INT8 [69.1%, 77.7%]; signer-mean FP16
  [61.8%, 75.8%], INT8 [54.1%, 67.8%]. Outer signer resampling is omitted as
  uninformative with k=5; signer-population generalization cannot be estimated.
- Leave-one-signer-out FP16 accuracy: drop P12 → 80.4%, P21 → 79.8%,
  P26 → 77.6%, P39 → 74.1%, P5 → 80.1% (INT8: 76.4/74.4/71.0/68.7/75.0%).
  P39 (the strongest signer) drives the pooled rate; P12/P5 are the weak strata.
- The pooled 346-clip accuracy is N-weighted; the signer-mean (unweighted)
  accuracy is lower. Report both; do not present the pooled rate as a
  representative-signer estimate.

## 6. Confusions and per-class (N = 2–6 per class — descriptive only)

- FP16: only two count-2 confusions: BEE1→BREAKFAST1, ZEBRA→JEWELRY; all other
  error pairs are count 1. 48/100 classes at 100% (N=2–6).
- INT8: four count-2 confusions: APPLE→RAZOR2, CHEESEGRATER→DRAG1,
  ELEVATOR1→CHEW1, ZEBRA→JEWELRY. 43/100 classes at 100%.
- Worst FP16 classes (7 classes at 33.3%, N=3): ANYONE, FAIL, FLIP, KNIGHT1,
  MYSELF, PULLCONVINCE, SCROLLDOWN. Worst INT8: CHEEK 0/4, EMPTY2 0/3,
  ELEVATOR1 1/5, FINE1 1/4 — INT8 degradation concentrates in a few classes
  (notably ELEVATOR1, CHEEK, EMPTY2, FINE1, ADVERTISE, CAPTURE, BASEBALLCAP).
- Full ranked confusion lists and per-class tables: `results/reconciliation.json`.

## 7. No-hand failure

Exactly 1 clip: `5989539775823847-INTUITIVE` (P5, ground truth INTUITIVE),
`ValueError: No hand detected in any frame`. Primary analysis counts it incorrect
for both models; the 345-clip sensitivity is reported separately. No clip was
excluded from any reported denominator without disclosure.

## 8. val_30 consistency

All 30 val_30 predictions (predictions, correctness, top-5) match the same clips'
val_346 records exactly; val_30 totals 22/30 FP16, 21/30 INT8 agree with the
dedicated val_30 run. The 30-clip subset (26 glosses, unbalanced signers) remains
a measurement on that subset only.

## 9. Files and reproduction

- `reconcile_v7.py` — frozen-prediction reconciliation (paired/clip CIs,
  McNemar, signer bootstrap, confusions, cross-checks). Standard library only.
- `split_integrity_v7.py` — independent split-integrity recomputation.
- `results/reconciliation.json` — full structured output incl. per-class,
  ranked clip handling, bootstrap details, artifact hashes.
- `results/split_integrity.json` — split counts, overlaps, checks.
- `tests/test_v7_accuracy.py` — Wilson/McNemar oracles, exact-count assertions,
  confusion/failure partitions, temp-dir rerun reproducibility.

Run from the isolated checkout root (`/tmp/echogesture-v7-agents/A07`):

```bash
python3 -I evaluation/device/v7_accuracy/reconcile_v7.py \
  --predictions evaluation/device/v5_accuracy/results/val_346/predictions.json \
  --predictions30 evaluation/device/v5_accuracy/results/val_30/predictions.json \
  --manifest evaluation/device/v5_accuracy/data/val_346/manifest.csv \
  --labels models/landmark_labels.json \
  --fp16-model models/landmark_classifier_fp16.tflite \
  --int8-model models/landmark_classifier_int8.tflite \
  --summary346 evaluation/device/v5_accuracy/results/val_346/summary.json \
  --summary30 evaluation/device/v5_accuracy/results/val_30/summary.json \
  --v6-statistics evaluation/device/v6_integrity/statistics.json \
  --output evaluation/device/v7_accuracy/results/reconciliation.json

python3 -I evaluation/device/v7_accuracy/split_integrity_v7.py \
  --train-csv /tmp/echogesture-real-fixture/train.csv \
  --val-csv evaluation/device/v5_accuracy/splits/val.csv \
  --test-csv evaluation/device/v5_accuracy/splits/test.csv \
  --labels models/landmark_labels.json \
  --v5-audit evaluation/device/v5_accuracy/results/split_audit.json \
  --v6-audit evaluation/device/v6_integrity/leakage_audit.json \
  --output evaluation/device/v7_accuracy/results/split_integrity.json

python3 -I evaluation/device/v7_accuracy/tests/test_v7_accuracy.py
# -> V7 accuracy tests: all passed
```

No installs, no nested agents, no test tuning. Bootstraps are deterministic
(fixed seeds) — reruns reproduce identical intervals.
