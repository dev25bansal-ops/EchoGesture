# FINAL SUBMISSION STATUS

Branch `hackathon/solo-final-submission`, based on `hackathon/echogesture-v2-rc`
@ `cc00f6a`. This file is the single source of truth for what is done, what is
verified, and what is blocked.

**Written 2026-11-13 15:20 IST.**
**Deadline: 18 Nov 2026, 11:59 PM — 5 days away.**
**Registration closed 8 Nov 2026 — 5 days ago.**

---

## 0. Read this first: two facts that outrank everything else

**0.1 Registration closed on 8 Nov 2026.** The deck states the final submission
link is "ONLY shared with Teams who have completed the Team registration
process". If the team is not registered, no amount of engineering produces a
submittable entry. **Verify this before doing anything else on this list.** I
cannot check it — it lives in the organizer's Google Form, not this repo.

**0.2 Track 02 has no supplied vocabulary or test manifest.** Track 04 (deck
p.21–22) explicitly ships `manifest.csv` and `supported_classes.json`. Track 02
(p.8–12) ships neither. Items I and J below are therefore **BLOCKED**, not
failed, and the clarification email is drafted and ready to send.

---

## 1. Requirement matrix

Every row is evidence, not intent. `NOT_TESTED` means nobody measured it.

| # | Requirement (deck page) | Evidence | Verdict |
| --- | --- | --- | --- |
| A | Quantized model <20 MB (p.9) / <25 MB (p.11) | 4 FP16 members 3,544,080 B + FP16 detector 7,819,105 B = **11.36 MB**. Size passes both budgets. **Format is FP16, not INT8.** | **PARTIAL** — size PASS, INT8 BLOCKED |
| B | Pose-invariant 3D features (p.10) | Features are 2D (x,y) only; `use_z=false`. MediaPipe z is available but deliberately unused. | **FAIL** as worded |
| C | Edge inference <50 ms (p.9/11) | **275.8 ms p95** default on laptop CPU. Classifier 0.36 ms; tracking ~90%. Parallel-12 measures **51.4 ms p50** on an 18-core laptop — better, still not met, and not a phone. | **FAIL** |
| D | Offline + exocentric (p.9/10) | Python and Android both fully offline; app declares CAMERA only, no INTERNET. | **PASS** |
| E | Held-out accuracy (p.11, 35%) | Test split: **72.78% top-1** (909/1249, 11 unseen signers). Validation re-score this branch: **84.39%** (n=346). Target ≥70% met on our split. | **PASS on own split**; NOT_TESTED on organizer data |
| F | Robustness (p.10, 25%) | Worst condition **10.00 pt** drop (crop) vs ≤5 pt target. Occlusion 6.67. low_light −3.33 (improves), motion_blur 0.0, frame_drop 0.0. | **FAIL** |
| G | HUD and captions (p.10) | Android V2: live preview, frame guide, countdown, result card, top-5, uncertainty label, offline TTS, history, glossary. | **PASS** (emulator); NOT_TESTED on phone |
| H | `predict(video_path) -> dict` (p.11) | Exact 5 fields; verified by a real Docker call returning `EAT1`. | **PASS** |
| I | Official fixed vocabulary (p.11) | **None supplied for Track 02.** Project uses its own 100 uppercase glosses. | **BLOCKED** |
| J | Exact evaluation clip IDs (p.11) | **No manifest supplied.** `results/` uses our own ASL Citizen split. | **BLOCKED** |
| K | `predictions.jsonl` + `results.csv` (p.11/12) | `predictions.jsonl` 1249 rows valid; `results.csv` 6 metrics; **`predictions.json` now generated** from the same source. | **PASS** (both forms present) |
| L | `results.csv` all required metrics (p.11) | All 6 rows present; 2 are `NOT_MEASURED`. | **PARTIAL** |
| M | Python 3.11 reproducible (p.12) | `python:3.11-slim` image builds; real inference verified in-container. | **PASS** (Docker) |
| N | GitHub + README + `main.py` (p.12) | Repo public, `src/main.py` present; README now documents the contract and Docker. | **PASS** |
| O | Docker / setup instructions (p.25) | `Dockerfile` builds and runs; README documents it. | **PASS** |
| P | Demo video ≤3 min + presentation (p.25) | Deck exists (11 slides, numbers verified honest). **No video.** Naming needs `CollegeName_TeamName`. | **NOT_TESTED** |
| Q | Legal rights and privacy (p.9) | App offline, CAMERA-only, no images in diagnostics. Weights **not** redistributable; public repo is source-only. 2 `.tflite` files are tracked in public history — see §5. | **PASS** privacy; **BLOCKED** on weights |

---

## 2. Measured numbers (do not round up, do not re-derive)

All from `results/results.csv`, `submission_final/RELEASE_MANIFEST.json`, or
this branch's own measurements.

| Metric | Value | Where measured | Pass target |
| --- | --- | --- | --- |
| top1_accuracy (test) | **72.78%** | macOS laptop, AI-Edge-LiteRT 2.3.0 | ≥70% **PASS** |
| top1_accuracy (val, this branch) | **84.39%** (292/346) | same | — |
| top5_accuracy (val, this branch) | **93.93%** (325/346) | same | — |
| accuracy at accepted coverage | **97.25%**, coverage 63.0% | same | — |
| latency_p95 (default) | **275.8 ms** | laptop CPU | ≤50 ms **FAIL** |
| latency_p50 (parallel-12) | **51.4 ms** | 18-core laptop | ≤50 ms **FAIL** (close) |
| classifier inference | **0.36 ms** | laptop | — |
| model_size (incl. detector) | **11.36 MB** | — | ≤25 MB **PASS** |
| accuracy_drop_robustness | **10.00 pt** | crop, 30 clips | ≤5 pt **FAIL** |
| avg_power | **NOT_MEASURED** | no equipment | ≤500 mW |
| thermal_drift_10min | **NOT_MEASURED** | no phone | ≤20% |

**Not measured and must not be claimed:** phone latency, phone accuracy,
power draw, thermal behaviour, sustained throughput.

---

## 3. INT8 position

Straight answer: **the submission is FP16, and we say so.**

- The organizer deck names INT8 on p.9–10 but the scored rubric on p.11 says
  only "Compilation to TFLite/ONNX <20MB".
- A real INT8 classifier exists and was audited on identical features over 345
  clips: **73.41% top-1 vs 78.61% FP16 — a 5.2 pt loss.**
- INT8 is also **not full-int8**: float32 input and output tensors.
- The **detector stays FP16** in every configuration, so "whole pipeline INT8"
  is false regardless of the classifier.
- Quantizing the classifier cannot help latency: it is 0.36 ms of a ~240 ms
  pipeline. Promoting INT8 would cost 5.2 pt accuracy to save 0.2% of runtime.

Recommendation: send the clarification email. Do not swap the model.

---

## 4. What I changed on this branch

| Change | Why |
| --- | --- |
| `results/predictions.json` | Deck p.12 names `.json`; p.11 names `.jsonl`. Both now exist from one source via `scripts/release/predictions_json_from_jsonl.py`, which refuses duplicates / non-uppercase / prediction-absent-from-top5. |
| README §Evaluator contract, §Docker | Deck p.25 requires Docker docs; both were absent. |
| `docs/release/ACCEPTANCE_MATRIX.md` | Requirement-by-requirement audit with verdicts. |
| `docs/release/BASELINE_INVENTORY.json` | Machine-readable inventory: hashes, sizes, contract, gaps. |
| `docs/release/LATENCY_ANALYSIS.md` | Worker scaling 196→51.4 ms, and proof parallelism doesn't change predictions (identical to 4 dp across 1/4/12 workers). |
| `docs/release/PHONE_WORKSHEET.md` | Exact ADB commands + worksheet, since no phone is attached. |
| `docs/release/ORGANIZER_CLARIFICATION.md` | Draft email covering all 6 ambiguities. |
| `release_gate.py` | Removed a hardcoded home-directory path I had introduced. |
| `RECOGNITION_DIAGNOSIS.md`, `ACCEPTANCE_MATRIX.md` | **Corrected a stale robustness claim.** I had repeated "13.33 pt frame_drop" from an older document; the authoritative manifest says frame_drop is **0.0** and crop is the worst axis at 10.0. |

No model, weight, vocabulary, feature or UI byte was changed.

---

## 5. Rights — two things needing your decision

**5.1 Two INT8 models are tracked in public Git history.** `models/ptq_100_int8.tflite`
(904 KB) and `models/s3d_conv1d_qat_int8.tflite` (901 KB) have been in the repo
since the initial commit `c47c5c9`. They are trained on ASL Citizen-derived
features, which the rights register marks **BLOCKED for publication**. They are
already public. Removing them requires a history rewrite, which I will not do
without your explicit authorization.

**5.2 `evaluation/asl_citizen_predictions.csv` is tracked and contains raw
participant IDs** (`P15`-style, 1706 lines). A sanitizer exists
(`scripts/release/sanitize_export.py`) but deliberately does not modify
originals — removal is documented as an owner decision.

Running `evaluation/device/privacy_lab/audit.py` reports **7 FAIL, 1
NOT_CLEARED** across 1583 files. Most are research-script issues (absolute
paths in `eval_*.py`, temp cleanup in `robustness.py`) that do not ship in the
evaluator path — `src/`, `results/`, `Dockerfile` and `README.md` scan clean.

---

## 6. Verified this session

| Check | Result |
| --- | --- |
| Full Python suite (`scripts/run_all_tests.py`) | **PASS** |
| Docker image build | **PASS** |
| `docker run … ` prints contract, exit 0 | **PASS** |
| Real inference in container | **PASS** — `EAT1` → `EAT1`, conf 0.5274 |
| `predictions.jsonl` structural validation | **PASS** — 1249 rows, unique, uppercase, top5-consistent |
| `predictions.json` generated | **PASS** — 1249 records |
| Worker-scaling prediction consistency | **PASS** — identical to 4 dp, 1 vs 12 workers, 7 clips |
| Evaluator path privacy scan | **PASS** — no home paths in `src/`, `results/`, `README.md`, `Dockerfile` |
| Android APK identity | `530f5ec4…`, 77,710,351 B, CAMERA-only, no INTERNET |

---

## 7. Remaining actions — who can do them

### You only (nothing I can do)

1. **Confirm team registration.** Deadline passed 8 Nov. This gates everything.
2. **Send the clarification email** (`ORGANIZER_CLARIFICATION.md`) from the
   registered team address.
3. **Supply team metadata**: `CollegeName_TeamName` for the deck, video and
   ZIP. The naming rule is disqualifying if broken.
4. **Decide on the two rights issues** in §5 — history rewrite or leave as-is.
5. **Record the demo** (≤3 min) on a physical phone. Script:
   `docs/demo/DEMO_SCRIPT.md`.
6. **Submit the Google Form.** I will not do this without explicit instruction.

### Needs a phone (I have none attached)

7. Install `dist/echogesture-v2-cc00f6a.apk`, run
   `docs/release/PHONE_WORKSHEET.md`, retrieve `diagnostics.txt`, run
   `evaluation/device/accuracy_report.py`. Until then **every phone claim is
   NOT_TESTED**.
8. Measure on-device latency, power and 10-minute thermal drift.

### I can do next, if you want

9. Rename/copy the deck to `CollegeName_TeamName.pptx` once you give the names.
10. Produce a source-only ZIP with a SHA-256 manifest for the form upload.
11. Re-run the release gate on any organizer manifest, if one arrives.
12. Draw the architecture diagram as a standalone asset.

---

## 8. Honest bottom line

The engineering is in good shape and the evidence is real: 72.78% top-1 on 11
unseen signers, 11.36 MB, fully offline, a working Android app, and a
reproducible Docker runtime verified end to end today.

Three requirements are genuinely failed and I will not paper over them:
**INT8** (FP16 shipped, INT8 costs 5.2 pt), **<50 ms** (275.8 ms default;
51.4 ms only on an 18-core laptop), and **robustness** (10 pt drop under crop).
Two are **blocked on the organizer**: vocabulary and manifest.

The most likely reason this submission does not happen is not any of that — it
is registration having closed five days ago. **Check that first.**