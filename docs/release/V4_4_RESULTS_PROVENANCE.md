# V4.4 — results provenance

Every result this project quotes must be attributable to one exact model. This
file records what each number is, which model produced it, and how far it can be
trusted.

## The problem this fixes

Dev's branch carries results from two different model generations in files that
do not say which:

| Artifact | Says | Actually is |
| --- | --- | --- |
| `results/results.csv` | 77.02% test, 87.25% val | **nine-member masked** |
| `results/eval_summary_test.json` | 72.78% test, 87.51% top-5 | **four-member pre-mask** — the file's own `"ensemble"` field says `tcn21+tcnm61+tcnm62+rob71 (4x fp16 tflite)` |
| `results/robustness.json` | clean 73.66%, worst drop 6.65 | **four-member pre-mask** |
| `results.csv` robustness row | clean 76.94%, worst drop 4.40, "9-member masked" | claims the new model; **contradicts the JSON on disk** |

**None of the three JSON artifacts records a model hash.** That is the root
cause: an artifact cannot be attributed to a model, so a stale file looks as
authoritative as a current one.

Nothing was rewritten. Old numbers were not replaced with new ones, because that
would claim the old runs measured the new model.

## Versioned evidence

Created, each self-describing:

```
results/v4_2_baseline/results.json    HISTORICAL_MEASURED
results/v4_3_masked/results.json      MEASURED
results/v4_4_release/results.json     MEASURED
```

Each record carries: model family and version, git commit, combined model
SHA-256, member count, feature-width contract (255/381 and how many of each),
attention-mask status, dataset, split identifier, unique clip count, unique
signer count, source preprocessing version, device and runtime, measurement
method, execution date, and full metric definitions.

## The 184-evaluation correction

The V4.3 A/B/C report sums 184 evaluations. **That is not 184 independent sign
examples.** It is **72 unique source clips** from **37 signers**, some evaluated
several times under synthetic leading/trailing idle.

| | Count |
| --- | --- |
| Unique clips | 72 (32 train signers + 40 val clips over 5 signers) |
| Unique signers | 37 |
| Conditions per clip | up to 3 (clean, 2 s lead, 2 s each side) |
| Total evaluations | 184 |
| Independent sign examples | **72** |

**79.3% must never be quoted as independent accuracy on 184 videos.** The honest
statement is: *on 72 authorised clips, the nine-member ensemble scored 79.3%
across 184 evaluations including repeated idle-padding conditions.* The repeated
conditions are not independent samples of signer variation.

Paired changes per original clip (C vs A, full interval):

| Condition | A | C | Δ |
| --- | --: | --: | --: |
| train clean (n=32) | 29 | 31 | +2 |
| train 2 s lead (n=32) | 24 | 29 | +5 |
| val clean (n=40) | 31 | 31 | 0 |
| val 2 s lead (n=40) | 20 | 28 | +8 |
| val 2 s both (n=40) | 13 | 27 | +14 |

The gain concentrates where idle exists. On clean validation clips the paired
change is exactly zero.

**Uncertainty:** at n=32 and n=40 a 2-clip difference is inside noise. The
padded-condition gains (+8, +14 on n=40) are large enough to survive that, but no
confidence interval is claimed here — the paired structure (same clips, same
signers, one variable changed) is the stronger evidence, not the sample size.

## Splits — what each may be used for

| Split | Status | Permitted use |
| --- | --- | --- |
| ASL Citizen **train** | Used in model development | Pipeline correctness only |
| ASL Citizen **validation** | Used for V4.2/V4.3 mode and segmentation decisions | Already influenced choices; **not clean** |
| ASL Citizen **test** (1,249 clips) | Used in Dev's model development | **Not untouched.** Do not use to select thresholds or models |
| New independently collected signers | **Does not exist** | The only valid future generalisation evidence |

Dev's 77.02% is a historical development figure, not an independent estimate.

## Robustness — three different things, do not conflate

| Kind | Status |
| --- | --- |
| **A. Landmark-level corruption** | Dev reports worst 4.40 pp (`drop_hand_25pct`) on the nine-member ensemble per `results.csv`; `robustness.json` still holds the old 6.65. **Not reproduced here** — the checkpoints and harness state it used are not all available, and re-running against a different code state would produce a number that merely looks comparable. |
| **B. Pixel-level corruption** | **NOT MEASURED** in this cycle. |
| **C. Physical phone camera conditions** | **NOT TESTED.** |

The six conditions in `robustness.json` are all landmark-level: coordinate
jitter, dropped landmarks, dropped hand, halved frame rate, and a synthetic 25%
"camera further" that **scales landmark coordinates**. Scaling coordinates is not
moving a camera. **A pass on it is not evidence about real cameras.**

## Confidence calibration on the new model

The app's 0.20-score / 0.15-margin gate was validated on the **old four-member**
ensemble. Re-measured on the nine-member model, validation split, n=40:

| | Mean score | Mean margin |
| --- | --: | --: |
| Correct (n=30) | 0.775 | 0.721 |
| Incorrect (n=10) | 0.339 | 0.207 |

| Gate | Accepted | Coverage | Accuracy among accepted |
| --- | --: | --: | --: |
| **score ≥ 0.20 & margin ≥ 0.15 (current)** | **33/40** | **82.5%** | **30/33 (90.9%)** |
| score ≥ 0.20 only | 35/40 | 87.5% | 30/35 (85.7%) |
| margin ≥ 0.15 only | 33/40 | 82.5% | 30/33 (90.9%) |
| score ≥ 0.35 (old rule) | 32/40 | 80.0% | 28/32 (87.5%) |
| margin ≥ 0.30 | 29/40 | 72.5% | 26/29 (89.7%) |

**The current gate is the best of those tested on both coverage and accepted
accuracy, so it is unchanged.** The old 0.35 score rule would be strictly worse
under the new model. Selection was on validation only; the test split was not
consulted.

Caveat: n=40 with 10 errors is a small sample, and the differences between the
top three gates are one to two clips. The decision is "do not change it", which
is the safe direction when the evidence is thin.

## Latency — method labels

| Measurement | Value | Method |
| --- | --- | --- |
| Nine-member ensemble, one clip | 0.5 ms (members only) | laptop CPU |
| End-to-end `predict()`, warm | 245–322 ms | laptop CPU, includes MediaPipe over ~64–181 frames |
| First call (cold) | ~980 ms | laptop CPU, model init |
| Docker container, one clip | 1201 ms | container, includes decode + tracking |

**None of these is a phone measurement.** The old 276 ms figure is **not**
reused: it measured a different model and a different code path.

The organizer's ≤50 ms criterion — re-read as written — is not met by inference
alone on any hardware measured here, and capture interaction is seconds by
design. That remains **FAIL**, not "unmeasured".