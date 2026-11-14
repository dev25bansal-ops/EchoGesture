# A07 — Final QA + Submission Integration

**Auditor:** A07 (read-only on existing files; wrote only under `evaluation/device/qa_final3/`)
**Branch:** `hackathon/abhijeet-integration` · **HEAD:** `efaa3231110cfd9e7fb859925dac78cae75a620d`
**Interpreter:** `.venv/bin/python -I` (Python 3.11.18) · **Host:** macOS 26.3 arm64 (laptop, not target hardware)
**Window:** 2026-11-11 ~10:15–10:55 IST

> **The working tree was live during this audit.** Between the first and last check,
> `git status --porcelain -uall` grew 65 → 74 entries, six docs were rewritten at
> 10:42:33, `robustness_ensemble/results/*.json` were rewritten at 10:48:38, and
> `results/results.csv` changed twice (10:42:33, 10:51:11). HEAD did not move.
> Every figure below was **re-verified against the final state** after those writes.
> Two findings that were true at 10:20 were fixed by 10:42 and are recorded as
> resolved, not as open defects.

---

## 1. Regression

### Full discovery (the documented command)

```
.venv/bin/python -I -m unittest discover -s evaluation/device -p 'test_*.py'
```

| Run | Result | Exit |
| --- | --- | --- |
| Final (post-tree-change) | **Ran 281 tests — OK (skipped=1)** | 0 |
| 17 other runs | Ran 281 — OK (skipped=1) | 0 |
| 1 run (verbose, mid-audit) | Ran 281 — **FAILED (failures=1, skipped=1)** | 1 |

The single skip is `test_model_contract.ModelContractTest.test_real_dev_fp16_contract`
— `'Dev verification worktree not available'`.

**The one failure (1 in 18 runs), recorded in full:**

```
FAIL: test_fp16_sha_matches_expected_candidate
      (v7_demo.test_v7_demo.IdentityTest.test_fp16_sha_matches_expected_candidate)
  File ".../evaluation/device/v7_demo/test_v7_demo.py", line 236
    self.assertEqual(report["model_full_sha256"], EXPECTED_FP16_SHA256)
AssertionError: 'a012f2b5f7a1110961079f19eb0e28c9abacb95e4d0787352e4157ebca7044a4'
             != '0e4c392732e7da7e77632620dc72e4afce8b1a5aa00234dbf7a13a1049a7ad93'
```

Both values were identified independently:

* `a012f2b5…` = SHA-256 of `models/landmark_classifier_fp16.tflite` (832,888 B). The
  test hashed the correct file.
* `0e4c3927…` = SHA-256 of the four ensemble members concatenated **in meta order**
  (3,544,080 B). This is the ensemble's *combined identity*, not a classifier file.
  Verified by computing it directly.

So during that one run the compared *expected* constant was the ensemble identity
instead of the fp16 identity. The test and loader paths contain no such mutation,
and the failure did not reproduce in 17 later runs or in 2 standalone `v7_demo` runs.
**Mechanism not identified.** It ranks as an unexplained one-off flake in a test that
matters (model identity), not a reproducible defect. Flagged, not resolved.

### Per-suite runs (each named suite, run separately)

| Suite | Ran | Result | rc |
| --- | --- | --- | --- |
| performance_final | 42 | OK | 0 |
| ensemble_evidence | 11 | OK | 0 |
| robustness_ensemble | 13 | OK | 0 |
| release_final | 26 | OK (skipped=1) | 0 |
| release_ensemble | 10 | OK | 0 |
| **release_final2** | **0** | **OK (nothing ran)** | 0 |
| compliance_final | 53 | OK | 0 |
| quantization_final | 8 | OK | 0 |
| v7_demo | 16 | OK | 0 |
| v7_int8 | 12 | OK | 0 |
| v8_privacy | 56 | OK | 0 |
| privacy_lab | 7 | OK | 0 |
| v7_compliance | 12 | OK | 0 |
| v7_qa | 43 | OK | 0 |
| decoder_lab | 19 | OK | 0 |
| streaming_lab | 13 | OK (skipped=1) | 0 |
| v7_streaming | 18 | OK (skipped=1) | 0 |
| v5_latency | 4 | OK | 0 |
| v4_robustness | 4 | OK | 0 |

**No suite FAILED.** Two caveats, because "Ran 0 / OK" is not a pass:

* **`release_final2` contains no `test_*.py`** — only `echogesture_final2_ensemble.zip`.
  Its "Ran 0 … OK" is an empty collection, not evidence for that release.
* **The full-discovery command does not reach most suites.** 39 of the 59
  `test_*.py` files under `evaluation/device/` live in subdirectories with no
  `__init__.py` (e.g. `performance_final/`, `compliance_final/`, `v8_accuracy/tests/`).
  `unittest discover` recurses only into packages, so it collects the 10 top-level
  files plus `v7_demo`, `v7_qa`, `v8_privacy` = 281 tests. That is why the per-suite
  totals sum to **367**, not 281. The 281 figure is honest for the command as written;
  it is **not** the whole suite.

---

## 2. Ensemble end-to-end

**Reference clip:** `evaluation/device/v5_accuracy/data/val_30/0283886564670357-WHAT FOR.mp4`
**Command:** `.venv/bin/python -I src/main.py "<clip>" 2>/dev/null` → exit 0

```json
{
  "clip_id": "0283886564670357-WHAT FOR",
  "prediction": "WHATFOR1",
  "top5": ["WHATFOR1", "OFFEND", "KNIGHT1", "SCROLLDOWN", "DECIDE1"],
  "confidence": 0.8821,
  "latency_ms": 675.81
}
```

Matches the expected reference exactly (`WHATFOR1`, 0.8821). A second run at 10:51
returned the same label/confidence at 539.77 ms. `latency_ms` is wall-clock
clip-to-JSON on this laptop and varies run to run; it is not a p95.

**(a) Four interpreters.** Imported `src/main.py` as a module, called `_init_model()`:

* `len(_INTERPRETERS) == 4`
* `_MEMBER_NAMES` = `landmark_ens_tcn21_fp16.tflite`, `landmark_ens_tcnm61_fp16.tflite`,
  `landmark_ens_tcnm62_fp16.tflite`, `landmark_ens_rob71_fp16.tflite` (meta order)

| Member | in_idx | out_idx | input q | output q |
| --- | --- | --- | --- | --- |
| landmark_ens_tcn21_fp16 | 0 | 224 | (0.0, 0) | (0.0, 0) |
| landmark_ens_tcnm61_fp16 | 0 | 191 | (0.0, 0) | (0.0, 0) |
| landmark_ens_tcnm62_fp16 | 0 | 191 | (0.0, 0) | (0.0, 0) |
| landmark_ens_rob71_fp16 | 0 | 224 | (0.0, 0) | (0.0, 0) |

Per-member output indices differ (224 vs 191), which confirms the per-interpreter
handle resolution is real and not a shared-index shortcut. All members are float
(quantization 0.0/0), consistent with an FP16 ensemble.

**(b) Bad model selector.** `ECHOGESTURE_MODEL=bogus` →
`ValueError: ECHOGESTURE_MODEL must be 's3d' or 'landmark'`, process **exit 1**. Raised, not silently defaulted. ✅

**(c) Parallel tracker.** `ECHOGESTURE_TRACKER=parallel` → `WHATFOR1`, **0.8821**, exit 0 —
same label and confidence as the default path. `ECHOGESTURE_TRACKER=bogus` → exit 1. ✅

---

## 3. Evidence consistency

All arithmetic below was **recomputed by me** from the frozen artifacts
(`ensemble_evidence/recompute.py` against the official split CSVs), not copied from docs.

### 3.1 `results/results.csv` (final state)

| Metric | Measured | Pass |
| --- | --- | --- |
| top1_accuracy_test | 72.78 % | PASS |
| top5_accuracy_test | 87.51 % | NOT_MEASURED |
| latency_p95_full_clip | 274.2 ms (target <50) | **FAIL** |
| latency_p50_full_clip_parallel4 | 79.3 ms | NOT_MEASURED |
| model_size | 11.37 MB (target <20) | PASS |
| accuracy_drop_low_light | −3.33 pt | PASS |
| accuracy_drop_motion_blur | 0.0 pt | PASS |
| accuracy_drop_crop | 10.0 pt | **FAIL** |
| accuracy_drop_occlusion | 6.67 pt | **FAIL** (revised during audit; see 3.5) |
| accuracy_drop_frame_drop | 0.0 pt | PASS |
| avg_power / thermal_drift_10min | NOT_MEASURED | NOT_MEASURED |

### 3.2 Recomputation

| Population | Correct / scored | Top-1 | Top-5 |
| --- | --- | --- | --- |
| Test, scored denominator | 909 / 1249 | **72.78 %** | **87.51 %** |
| Test, all-eligible (6 unscored = wrong) | 909 / 1255 | 72.43 % | 87.09 % |
| Validation, scored | 260 / 345 | **75.36 %** | 87.25 % |
| Validation, all-eligible | 260 / 346 | 75.14 % | 86.99 % |

Test: 11 signers, per-signer mean/min/max 73.36 / 53.52 / 87.63. Validation: 5 signers.
These match `ensemble_evidence/RESULT.md` and `VAL_RESULT.md` exactly.

### 3.3 Latency

`performance_final/results_ensemble/benchmark.json` (`parity: true`, 12 clips, 0 disagreements):

| Config | p50 | p95 | model-inference stage mean | tracking stage mean | peak RSS |
| --- | --- | --- | --- | --- | --- |
| singleton | 231.78 ms | **274.18 ms** | 0.386 ms | 228.26 ms | 255.3 MiB |
| parallel_2 | 131.97 ms | 154.77 ms | 0.363 ms | 133.29 ms | 300.0 MiB |
| parallel_4 | 79.33 ms | **99.81 ms** | 0.365 ms | 83.01 ms | 393.8 MiB |

`results.csv` (274.2) and the demo docs (274.2 / 99.8) agree with the JSON to rounding.

### 3.4 Robustness (30 clips, 5 signers, seed 42)

| Condition | Severity | Correct / scored | Accuracy | Δ pp |
| --- | --- | --- | --- | --- |
| clean | — | 23 / 30 | 76.67 % | — |
| re-encode (control) | — | 23 / 30 | 76.67 % | 0.00 |
| low_light | 0.5 | 24 / 30 | 80.00 % | −3.33 |
| motion_blur | 0.5 | 23 / 30 | 76.67 % | 0.00 |
| frame_drop | 0.5 | 23 / 30 | 76.67 % | 0.00 |
| crop | 0.3 | 20 / 30 | 66.67 % | 10.00 |
| occlusion | 0.5 | 21 / 29 scored (1 no-hand) | 72.41 % scored / 70.00 % eligible | 6.67 |

`clean` 76.67 % reconciles with `results.csv`. These are a 30-clip `val_30` batch,
**not** the 1,249-clip test split — the two "clean" baselines are different
populations and must not be quoted interchangeably.

### 3.5 Changes observed mid-audit

* `accuracy_drop_occlusion` was **4.25 PASS** at 10:20 and is **6.67 FAIL** at 10:51.
  The artifact (`occlusion.json`: 21/29 scored, 1 no-hand) supports both readings;
  the row now uses the stricter all-eligible denominator and says so in its note.
  The stricter reading is the honest one. Cross-artifact state is now consistent.
* Six docs (including `app/android_final/README.md`, `models/README.md`,
  `results/results.csv`) carried **3,543,080** at 10:20 and now carry **3,544,080**.
  That is the correct fix (see 3.6).

### 3.6 The four specific flags requested

| Flag | Finding |
| --- | --- |
| Any place claiming **< 50 ms** | **None.** No artifact claims achievement. Every occurrence is a target, a FAIL, or an explicit "do not claim" (`SHOT_LIST_FINAL`, `SUBMISSION_DASHBOARD`, `ARCHITECTURE_R8`, `REQUIREMENTS_MATRIX` R12). `S3D_Conv1D_analysis.md:69` ("very likely under 50 ms on Snapdragon 8 Elite") is a hedged, clearly-labelled Cygnal proxy for the abandoned S3D baseline — not the shipped ensemble. **PASS** |
| Model-only latency conflated with **full-clip** | **None.** `benchmark.json` states timings are "full predict() clip-to-JSON with profile=True"; `results.csv` reports `latency_p95_full_clip` and names tracking as the bottleneck; `REQUIREMENTS_MATRIX` R12 explicitly separates "ensemble inference alone is 0.36 ms" from the full-clip number. **PASS** |
| **83.77 % validation** | **Unsupported, and no longer present.** It occurs zero times in the current tree and zero times in HEAD; it appears only in `RESULT.md`/`VAL_RESULT.md` as a flagged, rejected claim attributed to Dev's older `RESULTS.md`. Committed val predictions recompute to **75.36 %**. Treat any 83.77 % as withdrawn. |
| **model_size 3,543,080 vs 3,544,080** | **3,544,080 is correct.** Sum of the four member files on disk = 3,544,080 B (939,152 + 832,888 + 832,888 + 939,152); same sum inside the APK; same sum inside the release ZIP. Plus detector 7,819,105 B = 11,363,185 B. The 3,543,080 figure was 1,000 B low and has been corrected. **Residual imprecision:** 11,363,185 B is **11.36 MB** decimal, so the headline "11.37 MB" is 0.01 MB optimistic (immaterial to the 20 MB limit, but it is not the exact decimal value). |

### 3.7 Release artifact

`evaluation/device/release_final2/echogesture_final2_ensemble.zip`

* SHA-256 `48a062f27492ed30bd02cb00732401d9a751b5ae02b94f837deee3835213aff7`
* 9,743,727 bytes on disk, 17 entries
* Contains all four ensemble members + `landmark_classifier_fp16.tflite` + labels + meta,
  and `release.json` correctly declares `official_submission: false` with all four member SHAs.

---

## 4. APK

`app/android_final/app/build/outputs/apk/debug/app-debug.apk`

| Check | Result |
| --- | --- |
| Exists | Yes, mtime 10:33:45 |
| Size | **75,072,782 bytes** (71.6 MiB) |
| SHA-256 | `041e163fba2665bd33678a5463b0373c14d62b6b0c4cff28e67632aecc895b45` |
| `assets/hand_landmarker.task` | present — 7,819,105 B |
| `assets/landmark_ens_tcn21_fp16.tflite` | present — 939,152 B |
| `assets/landmark_ens_tcnm61_fp16.tflite` | present — 832,888 B |
| `assets/landmark_ens_tcnm62_fp16.tflite` | present — 832,888 B |
| `assets/landmark_ens_rob71_fp16.tflite` | present — 939,152 B |
| `assets/landmark_labels.json` | present — 1,492 B |
| `assets/landmark_model_meta.json` | present — 296 B |
| Member bytes inside APK | **3,544,080** — matches disk exactly |
| `apksigner verify` (JDK + build-tools 34.0.0) | **exit 0** |
| `adb devices` | **0 devices attached** |

All seven required assets are present and byte-identical to `models/`.

**Stale documentation:** `README.md:77` still says *"No Gradle/SDK on this machine —
no APK or on-device result claimed."* That is now false: the toolchain exists at
`ActionMesh-fork/android-toolchain/` and a signed debug APK is built. A public reader
would under-report what is done. (The absence of a *device* remains true.)

---

## 5. Clean-clone test

Two extractions, both run with `unittest discover -s evaluation/device`:

| Extraction | Contents | Result |
| --- | --- | --- |
| `git archive HEAD` + working-tree overlay (excluding `models/*.tflite`, `models/*.task`, `*.zip`, `build/`, `assets/`, `__pycache__`, `.venv`) | 381 files | **Ran 281 — OK (skipped=29)**, rc 0 |
| **`git archive HEAD` only** (what an actual `git clone` of this branch yields) | 332 files | **Ran 281 — OK (skipped=29)**, rc 0 |

**Zero failures in both.** 29 skips are all weight/data-dependent:

| Reason | Count |
| --- | --- |
| Real model artifacts not available | 14 |
| trained model weights are not committed (see NOTICE); supply them locally | 5 |
| frozen test predictions not present | 5 |
| trained weights are not committed (see NOTICE) | 4 |
| Dev verification worktree not available | 1 |

Note: `models/ptq_100_int8.tflite` and `models/s3d_conv1d_qat_int8.tflite` **are**
tracked and therefore appear in a public clone. The ensemble weights and the
MediaPipe detector are not, and no test wrongly passes without them.

A public clone therefore reproduces the same 281 green / 29 skipped result. The
281 number carries the coverage caveat in §1 (39 files in non-package dirs are
not collected by this command in any environment, clone or not).

---

## 6. Privacy scan

Scope: files added or modified this session (`git status --porcelain -uall`, 47
untracked + 17 modified; `qa_final3/` excluded as self-authored).

### 6.1 Real user paths — **FOUND**

`$REPO` appears in **6 session-added files**:

| File | Occurrences |
| --- | --- |
| `evaluation/device/robustness_ensemble/results/clean.json` | 30 (one `path` per clip) |
| `evaluation/device/robustness_ensemble/results/crop.json` | 2+ (same pattern) |
| `docs/submission/REQUIREMENTS_MATRIX.md` | 3 |
| `docs/demo/DEMO_RUNBOOK.md` | 3 |
| `docs/demo/DEMO_CHECKLIST.md` | 1 |
| `evaluation/device/performance_final/results_candidates/experiments.json` | 1 |

`REQUIREMENTS_MATRIX.md` also embeds the local `.venv` interpreter path. These are
real-account paths, not `/Users/example` placeholders. The repo already ships a
sanitizer for exactly this (`evaluation/device/v8_privacy/sanitize_evidence.py`,
which rewrites `/Users/…`, `/private/…`, `/var/folders/…`); it has **not** been
applied to `robustness_ensemble/results/` or `performance_final/results_candidates/`.

### 6.2 Keyed participant IDs — **FOUND** (raw IDs removed before commit)

`robustness_ensemble/results/{clean,crop,frame_drop,low_light,motion_blur,occlusion,reencode}.json`
carried `signer_id` values in the raw ASL Citizen participant namespace (30 per
file) plus a top-level `signers` list. Bundled with per-clip paths and per-clip
correctness, that is a linkable participant→clip→result record. `results.csv`
itself only says "5 signers" and is fine.

**Resolution:** all seven files were re-emitted with pseudonymized
`signer_<hash>` keys before their first commit, so the raw IDs were never
published. The docs that carried raw IDs in prose
(`robustness_ensemble/REPORT.md`, `ensemble_evidence/RESULT.md`, this report)
were rewritten to use anonymous S1–S11 labels.

By contrast, `ensemble_evidence/predictions_ensemble*.jsonl` are clean: `clip_id`,
`prediction`, `top5`, `confidence`, `latency_ms`, no signer or truth field.

### 6.3 Secrets / tokens / hostnames — **NONE**

No API keys, tokens, passwords, private keys, or `BEGIN …` blocks. No `.local`,
`.lan`, `.internal` hostnames. The only matches for `token`/`secret` are the
literal `NO_HAND_TOKEN = "No hand detected"` sentinel.

---

## 7. Summary of open items

| # | Item | Severity | Status |
| --- | --- | --- | --- |
| 1 | Keyed participant IDs (`signer_id`, raw `P…` values) in 7 committed-eligible robustness evidence files | **High** (submission privacy) | **Resolved** — pseudonymized before first commit; prose IDs replaced with S1–S11 |
| 2 | Real `$HOME` paths in 6 session-added files incl. submission docs | Medium | Open |
| 3 | Documented full-regression command collects 281 of 367 tests; 39 files in non-package dirs unreachable | Medium (coverage) | Open |
| 4 | `release_final2` is named as a suite but contains no tests | Low (labeling) | Open |
| 5 | One unexplained `v7_demo` identity flake (1/18 runs) | Low–Medium | Open, not reproduced |
| 6 | `README.md:77` still claims no SDK/APK on this machine | Low (accuracy) | Open |
| 7 | "11.37 MB" vs exact 11,363,185 B = 11.36 MB decimal | Low (rounding) | Open, immaterial |
| 8 | Power, thermal, target-device latency | — | **External**: no device attached |
| 9 | 3,543,080 → 3,544,080; occlusion 4.25 → 6.67 | — | Resolved during audit (10:42–10:51) |

### What was verified green

281 tests pass with 1 skip; all 19 named suites pass; the ensemble loads exactly 4
interpreters with distinct IO handles; `WHATFOR1`/0.8821 reproduces; bad env config
fails loudly; 72.78 % test and 75.36 % validation recompute exactly; the <50 ms
target is nowhere falsely claimed; model-only latency is never presented as
full-clip; model size arithmetic is correct at 3,544,080 B; the APK is well-formed,
signed, and carries all four members; a clean clone (both with and without the
working-tree overlay) reproduces 281 green with weight-dependent skips only.

---

## Verdict

PARTIALLY COMPLETE — SPECIFIC ENGINEERING DEFECTS

The assembled engineering is strong and every functional gate I executed passes.
The verdict is not "submission ready" because (a) keyed participant identifiers and
real account paths ship in session-added evidence files — a privacy defect the repo's
own sanitizer exists to prevent, and the single item that would most plausibly block
publication — and (b) the documented regression command silently covers only 76 % of
the test files, so "281 green" overstates what CI actually runs. Target-device power,
thermal and latency also remain unmeasured because no device is attached, but that is
an external gate, not an engineering defect.