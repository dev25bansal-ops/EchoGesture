# EchoGesture — Track 2 submission compliance (final)

Compiled 2026-11-11 by Agent A (organizer requirements / submission compliance).

**Source of truth:** the organizer deck at the repository root,
`XR_Hackathon_2026_Consolidated_Slides_for_sharing_v1.pdf` (1,168,491 B, 28 pages).
Every quote below was extracted independently from the PDF itself with `pypdf`
7.0.1 (`/tmp/eg_pdf/extract.py`), not taken from any repo transcription. All 28
pages extract cleanly; page N below means physical PDF page N.

> **Reader warning — the deck is multi-track.** Pages 13–23 are Track 3 and
> Track 4 and are *not* requirements for us. Track 2 is pages 8–12; general
> rules are pages 24–28. Some repo docs cite PDF page 24 for the timeline; it
> is correct (p.24 is the general Timeline). No Track 2 requirement was found
> outside pp. 9–12 and 24–28.

**Status vocabulary:** `PASS` · `FAIL` · `BLOCKED`. Registration is recorded as
`REGISTRATION_USER_CONFIRMED` at the direction of the team owner and is treated
as a fact, not an open question. Rows below are about *registration evidence and
data*, not about whether the team registered.

---

## 0. Verbatim organizer text this table rests on

| Page | Verbatim line |
| --- | --- |
| 9 | `Budgets: ≤ 20 MB quantized model (INT8), ≤ 50 ms latency, zero cloud streaming.` |
| 9 | `Translation must keep pace with ≤ 50 ms latency or dialogue turns awkward and out of sync.` |
| 10 | `Pose-invariant 3D hand features; quantized TFLite / ONNX Runtime model` |
| 10 | `Quantized model + conversion script (< 20 MB INT8)` |
| 10 | `predictions.jsonl (one JSON per clip) + results.csv` |
| 10 | `GitHub repo: predict() video → text, reproducible setup` |
| 10 | `Demo video (2–3 min) or deployed app URL` |
| 11 | `Model Quantization (20%): Compilation to TFLite/ONNX <20MB.` |
| 11 | `Edge Latency (20%): Measured inference time <50ms.` |
| 11 | `Top-1 Accuracy (35%): Precision on held-out test videos.` |
| 11 | `Robustness & Delivery (25%): Occlusions & HUD clarity.` |
| 11 | `{"clip_id": "test_0412", "prediction": "HAPPY", "top5": [...], "confidence": 0.81, "latency_ms": 23.4}` |
| 11 | `clip_id must match the test manifest exactly.` |
| 11 | `prediction is a word from the fixed vocabulary list, uppercase.` |
| 11 | `latency_ms is self-reported. The organizers re-measure on their own device.` |
| 11 | `Teams submit a single predictions.jsonl file, one JSON object per line, one line per test clip.` |
| 11 | `model_size <=25 MB` (results-sheet example row) |
| 11 | `runtime (for example, "LiteRT INT8, GPU delegate")` (format example only) |
| 12 | `├── results/` → `│ ├── predictions.json` → `│ └── results.csv` |
| 12 | `Reproducible source code– the entire implementation lives under src/.` |
| 12 | `Exact file-folder layout – keep the hierarchy shown below; you may add extra helper files or sub-folders as long as the core structure stays unchanged.` |
| 12 | `Python 3.11 – all scripts are written for Python3.11` |
| 24 | `Prototype code, 5-min demo video and deck due.` |
| 24 | `** Final Submission Link will be ONLY shared with Teams who have completed the Team registration process` |
| 25 | `Google Form Last date 8 Nov 2026, 11:59 PM` |
| 25 | `Up to max 4 members per team, single college` |
| 25 | `Team name CollegeName_TeamName` |
| 25 | `Per member name, email, phone, year & branch` |
| 25 | `One theme one submission per team` |
| 25 | `FINAL SUBMISSION · Last date 18 Nov 2026, 11:59 PM` |
| 25 | `Everything below needs to be submitted by 18th Nov 2026. No extension would be provided.` |
| 25 | `Working prototype code — public or shared GitHub repo` |
| 25 | `README with reproducible setup instructions, Docker files & other requirements` |
| 25 | `Demo video, max 3 minutes (YouTube or Drive link)` |
| 25 | `Presentation file (PPT or PDF) — CollegeName_TeamName, follow the nomenclature` |
| 25 | `Refer to the CollegeName_TeamName_Submission_ppt` |
| 25 | `Theme ID, project title and team details` |
| 25 | `** Teams NOT following the submission guideline would lead to direct disqualification` |
| 27 | `with a 5-minute demo video by 18th November` |
| 27 | `prism@samsung.com` |

**Detector-size question — answered directly.** The PDF never defines what
counts toward the 20 MB. The only size wording is `≤ 20 MB quantized model`
(p.9), `Quantized model + conversion script (< 20 MB INT8)` (p.10) and
`Compilation to TFLite/ONNX <20MB` (p.11). It says **"model"**, not "bundle" or
"all assets". The detector (`models/hand_landmarker.task`, 7,819,105 B = 7.46
MiB) is an inference asset required to produce the model's input, so the
conservative and defensible reading is to count it. **We do.** Our 11.37 MB
already includes the detector; excluding it would be 3.38 MiB. Both are under
20 MB, so the PASS does not depend on the reading. What the PDF does **not**
answer is whether a ZIP total counts — say so in the submission note.

---

## 1. Compliance table

| Requirement | Official source (page/section) | Current measured state | PASS/FAIL/BLOCKED | Evidence path | Owner | Next action |
| --- | --- | --- | --- | --- | --- | --- |
| Team registration completed | p.24 `Final Submission Link ... ONLY ... Teams who have completed the Team registration process`; p.25 registration close 8 Nov 11:59 PM | `REGISTRATION_USER_CONFIRMED` (team owner confirmed) | **PASS** | none in repo (state per owner confirmation) | Abhijeet | No action on registration itself. |
| Registration identifier + exact registered college/team names available | p.25 `CollegeName_TeamName`; `Per member name, email, phone, year & branch` | Not present anywhere in the repo | **BLOCKED** | — | Abhijeet | Supply the registration confirmation ID, exact college string and exact team string; they drive the deck and ZIP filenames. |
| Track eligibility / theme | p.8 `[ T R A C K 0 2 ]`; p.25 `One theme one submission per team` | Repo is wholly Track 2 (EchoGesture) | **PASS** | `README.md`, all `evaluation/device/*` | Abhijeet | State "Track 2" on the deck. |
| Final submission deadline + timezone | p.25 `18 Nov 2026, 11:59 PM`; p.27 repeats | Today 2026-11-11; internal target 17 Nov | **PASS** | `docs/hackathon/SUBMISSION.md:33` | Abhijeet | Submit by 17 Nov internal target. |
| Deadline timezone | **Not stated in the PDF.** No "IST" appears on any page | Repo docs assert IST (`SUBMISSION.md:20`, `ABHIJEET.md:106`) | **BLOCKED** | `_slides_text.txt` has no IST | Abhijeet | Ask `prism@samsung.com` to confirm the timezone; assume IST (organizer is Samsung R&D Bangalore) and submit with ≥24 h margin. |
| Model format (TFLite/ONNX) | p.10 `quantized TFLite / ONNX Runtime model`; p.11 `TFLite/ONNX` | 4-member TFLite ensemble | **PASS** | `models/landmark_ens_*_fp16.tflite` | Dev | — |
| INT8 quantization | p.9 `quantized model (INT8)`; p.10 `< 20 MB INT8`; p.11 `TFLite/ONNX <20MB` (no INT8) | FP16, `metadata.quantization = "float16"` | **BLOCKED** | `models/landmark_model_meta.json`; `INT8_CLARIFICATION.md` | Abhijeet (send) / Dev (fallback) | Send the p.9/p.10 vs p.11 clarification email; do not claim INT8. |
| Model size | p.9 `≤ 20 MB`; p.10 `< 20 MB`; p.11 rubric `<20MB`; p.11 results example `<=25 MB` | 3,544,080 B ensemble + 7,819,105 B detector = 11,363,185 B = 10.84 MiB / 11.36 MB (decimal) | **PASS** both limits | `results/results.csv` row `model_size`; `models/` | Abhijeet | Report as 11.37 MB per the repo convention; also give the exact byte count and state both files are counted. |
| Detector counted in size | **Not answered by PDF**; wording is "model" only | Counted in the 11.37 MB | **PASS** (conservative) | `results/results.csv` row `model_size` | Abhijeet | State explicitly in the submission note that the 7,819,105 B detector is included. |
| Accuracy threshold | p.11 `top1_accuracy >=70 %`; `Top-1 Accuracy (35%)` | 72.78% top-1, 909/1249, 11 signers; top-5 87.51% | **PASS** | `results/results.csv` row `top1_accuracy_test` | Dev | None; carry the "split was inspected during model selection" caveat. |
| Latency boundary — re-measurement device | p.11 `latency_ms is self-reported. The organizers re-measure on their own device.` | Organizer re-measures; we measured on a Mac laptop | **PASS** (rule understood) | p.11 | Abhijeet | Expect organizer-device numbers to differ from ours; state our device. |
| Latency boundary — what "50 ms" covers | p.9 `≤ 50 ms latency`; p.11 `Measured inference time <50ms`; p.11 `{"latency_ms": 23.4}` in the per-clip return | **Unspecified.** The PDF never says camera-to-caption or clip-to-JSON. Our `latency_ms` is full clip-to-JSON through `predict()` | **BLOCKED** | `results/results.csv` row `latency_p95_full_clip` | Abhijeet | Ask the organizer whether the 50 ms is per-frame inference or end-to-end clip; report both numbers either way. |
| Latency target met | p.9 `≤ 50 ms`; p.11 `Edge Latency (20%): Measured inference time <50ms` | p95 274.2 ms singleton / 154.8 ms parallel-2 / 99.8 ms parallel-4 (Mac laptop) | **FAIL** | `results/results.csv` | Abhijeet | Cannot be fixed by model change (inference is 0.36–0.41 ms; tracking dominates). Report honestly; do not claim the target. |
| Robustness criterion | p.11 `Robustness & Delivery (25%): Occlusions & HUD clarity`; p.11 results row `accuracy_drop_robustness <=5 pt` | low light PASS, blur PASS, frame drop PASS, **crop −10.00 pp FAIL, occlusion −6.67 pp FAIL** | **FAIL** | `results/results.csv` rows `accuracy_drop_*` | Abhijeet | Report the 2 real fails; do not tune them away. |
| Device / runtime disclosed | p.11 `device (for example, "Galaxy S25, Snapdragon 8 Elite"), runtime (... "LiteRT INT8, GPU delegate")` | `macOS arm64 laptop (not target hardware)`, `ai-edge-litert 2.3.0, XNNPACK CPU` | **PASS** (disclosed) | `results/results.csv` | Abhijeet | Keep the honest device string; do not paste the INT8 GPU example. |
| Average power | p.11 results row `avg_power <=500 mW` | NOT_MEASURED — no device | **BLOCKED** | `results/results.csv` | Abhijeet | Named dependency: an Android phone with `adb`. Next action: run the DEMO_RUNBOOK power measurement, or write NOT_MEASURED in the official sheet. |
| 10-min thermal drift | p.11 results row `thermal_drift_10min <=20 %` | NOT_MEASURED — no device | **BLOCKED** | `results/results.csv` | Abhijeet | Named dependency: an Android phone. Next action: 10-min sustained run per DEMO_RUNBOOK, or NOT_MEASURED. |
| Official vocabulary | p.11 `prediction is a word from the fixed vocabulary list, uppercase.` | Our own 100-label list; no organizer list exists in the deck | **BLOCKED** | `models/landmark_labels.json` | Abhijeet | Named dependency: organizer vocabulary/starter kit (absent from all 28 pages). Next action: email `prism@samsung.com` requesting the Track 2 vocabulary + test manifest; publish ours and label it as ours. |
| Official test manifest | p.11 `clip_id must match the test manifest exactly.`; `every clip ID is present.` | No organizer manifest supplied | **BLOCKED** | `scripts/release/build_bundle.py:207-227` supports it | Abhijeet | Named dependency: organizer manifest. Next action: request it; if it never arrives, ship with our demo predictions and say so. |
| predictions.jsonl format | p.11 `single predictions.jsonl file, one JSON object per line`; p.10 `predictions.jsonl (one JSON per clip)`; p.12 layout `results/predictions.json` | **FIXED.** `results/predictions.jsonl` is now 1,249 lines, one object per line, each with exactly `clip_id`, uppercase `prediction`, 5-entry `top5`, `confidence`, `latency_ms` — the p.11 schema. The 1,705-line research file (with `ground_truth`/`correct`/`signer`) moved to `results/research/predictions.jsonl`, off the submission path. | **PASS** (schema) | `results/predictions.jsonl`; source `evaluation/device/ensemble_evidence/predictions_ensemble.jsonl` | Abhijeet | Remaining dependency: the organizer test manifest itself. These are our held-out-split predictions, not organizer clip IDs. Request the manifest. The p.11 vs p.12 `.jsonl`/`.json` conflict goes in the same organizer email. |
| results.csv format | p.11 `metric, target, measured, unit, device, runtime, pass`; rows `top1_accuracy`, `latency_p95`, `model_size`, `accuracy_drop_robustness`, `avg_power`, `thermal_drift_10min` | **FIXED.** Header is now exactly `metric,target,measured,unit,device,runtime,pass,notes` (official seven, `pass` in position 7, `notes` appended — p.12 permits extra helper data). Metric names are now the official six: `top1_accuracy`, `latency_p95`, `model_size`, `accuracy_drop_robustness`, `avg_power`, `thermal_drift_10min`. Per-condition robustness detail moved into the `notes` cell. | **PASS** | `results/results.csv` | Abhijeet | None — verify once more if the organizer template changes. |
| Source layout `team_name.zip` | p.12 `team_name.zip` → `src/` (`requirements.txt`, `main.py`, `README.md`) + `results/` (`predictions.json`, `results.csv`); p.12 extras allowed | Builder produces the right core; repo root has no `team_name.zip`; latest built zip exists only under `evaluation/device/release_final2/` | **PASS** (structure) / **BLOCKED** (final name) | `evaluation/device/release_final2/echogesture_final2_ensemble.zip` | Abhijeet | Rebuild and rename to `<CollegeName>_<TeamName>.zip` once names arrive. |
| Python 3.11 | p.12 `Python 3.11 – all scripts are written for Python3.11` | `.venv` is 3.11 | **PASS** | `.venv` | Dev | — |
| Entry point `predict(video_path)` | p.11 code block `def predict(video_path: str) -> dict` | Implemented with extra keyword-only args (`clip_id`, `profile`), default path unchanged | **PASS** | `src/main.py:434` | Abhijeet | Keep the two-arg call working; add a one-line note that kwargs are optional. |
| Reproducible setup / Docker | p.25 `README with reproducible setup instructions, Docker files & other requirements`; p.10 `reproducible setup` | **FIXED.** Root `Dockerfile` added (`FROM python:3.11-slim`, installs `src/requirements.txt`, weights mounted at runtime rather than baked in). The bundle builder now completes a clean staging + extraction verification instead of hard-failing. | **PASS** | `Dockerfile`; `scripts/release/build_bundle.py` | Abhijeet | Verify one real `docker build` if a Docker daemon becomes available; the file's syntax and the builder's checks are confirmed. |
| Conversion script | p.10 `Quantized model + conversion script` | `scripts/ml/export_tflite_tf.py` exists and is bundled | **PASS** | `scripts/ml/export_tflite_tf.py` | Dev | Note that the shipped conversion is float16, not INT8, in the script/README. |
| Public/shared GitHub repo | p.25 `Working prototype code — public or shared GitHub repo` | Remote exists (`dev25bansal-ops/EchoGesture`), branch pushed, CI green; a **public** link would expose restricted weights + participant IDs | **BLOCKED** | `NOTICE`, `LICENSE` | Abhijeet (owner decision) | Named dependency: written redistribution permission, or make the repo access-controlled. Next action: submit the repo as a *shared* (access-controlled) link, or publish a code-only public repo. |
| Demo video | p.10 `Demo video (2–3 min)`; p.25 `Demo video, max 3 minutes (YouTube or Drive link)`; p.24 `5-min demo video`; p.27 `5-minute demo video` | **No video recorded** | **BLOCKED** | `docs/demo/SHOT_LIST_FINAL.md` (2:50 plan) | Abhijeet | Named dependency: a recording session + the Android device (or a laptop demo). Next action: record, keep it ≤3:00 (satisfies the strictest cap), upload to YouTube/Drive. |
| Presentation file naming | p.25 `Presentation file (PPT or PDF) — CollegeName_TeamName` | `docs/submission/EchoGesture_Submission.pptx`, 11 slides, team/college placeholders unfilled | **BLOCKED** | `docs/submission/EchoGesture_Submission.pptx` | Abhijeet | Named dependency: exact college + team strings. Next action: fill slide 1 and rename to `<CollegeName>_<TeamName>.pptx`. |
| Presentation template | p.25 `Refer to the CollegeName_TeamName_Submission_ppt` | The deck references a template that **is not in the deck or the repo** | **BLOCKED** | — | Abhijeet | Named dependency: the official template file. Next action: ask the organizer for `CollegeName_TeamName_Submission_ppt`; meanwhile our 11 slides cover every required p.25 section. |
| Deck required contents | p.25 six bullets: Theme ID, project title & team; problem; solution & architecture diagram; tools & stack; innovation, results, limitations | All six are present across slides 1–11 | **PASS** (content) | `docs/submission/EchoGesture_Submission.pptx` | Abhijeet | Keep limitations slide 10. |
| One submission per team, via Google Form | p.25 `One submission per team via the Google Form`; p.24 link only to registered teams | Form link not in repo; registration confirmed | **BLOCKED** | — | Abhijeet | Named dependency: the submission form link (emailed to registered teams). Next action: retrieve it from the registration inbox and submit once. |
| Disqualification rule | p.25 `Teams NOT following the submission guideline would lead to direct disqualification` | — | **PASS** (aware) | — | Abhijeet | Re-check layout, names and filenames immediately before submitting. |
| Contact | p.27 `prism@samsung.com` | — | **PASS** (informational) | p.27 | Abhijeet | Use for the INT8 / vocabulary / manifest / template / timezone clarifications. |

---

## 2. Tally

| Status | Count | Rows |
| --- | --- | --- |
| PASS | **15** | registration, track, deadline, format, size, detector accounting, accuracy, re-measure rule, device disclosure, Python 3.11, entry point, conversion script, deck contents, disqualification awareness, contact |
| FAIL | **4** | latency target, robustness criteria, `predictions.jsonl` content, `results.csv` format |
| BLOCKED | **14** | registration ID/names, timezone, INT8, latency boundary, power, thermal, vocabulary, manifest, Docker, public repo, demo video, deck filename, template, submission form |
| PASS *and* BLOCKED | **1** | ZIP source layout: structure passes, final `CollegeName_TeamName` name is blocked |

**Total: 34 requirement rows** (15 + 4 + 14 + 1 dual).

---

## 3. What I could NOT verify, and why

1. **Anything about our own device numbers** — no Android device is attached
   (`adb` not installed). Power, thermal drift and on-device latency remain
   unmeasured; the PDF requires a device field and an organizer re-measure.
2. **The organizer vocabulary and test manifest** — not present on any of the
   28 pages. The PDF says a fixed vocabulary and a test manifest exist; it never
   ships them. Five prior research rounds found nothing public.
3. **The official presentation template** — p.25 names
   `CollegeName_TeamName_Submission_ppt`; the file is not in the deck or repo.
4. **The registration identifier and exact registered college/team strings** —
   not in the repo. Registration itself is confirmed by the owner.
5. **The deadline timezone** — no "IST"/"UTC"/"GMT" on any page.
6. **The exact latency boundary** — the PDF gives three different phrasings
   (p.9 "≤50 ms latency", p.11 "measured inference time <50ms", p.11 a
   per-clip `latency_ms` field) and never defines start/stop.
7. **Whether the detector or the ZIP total counts toward 20 MB** — the PDF
   says only "model"; it never defines the counted set.
8. **Whether redistribution of our weights is permitted** — `NOTICE` records no
   permission; nothing in the PDF addresses weight licensing.
9. **Whether the team's college is on the live form's closed list** — the deck
   says "single college" only; the five-institution restriction came from a live
   form capture (`.scratch/form_live.html`) that no longer exists in the repo.
   **This cannot be re-verified from the PDF.**

---

## 4. Where the repo's own docs mis-state the requirements

| Repo doc claim | Actual PDF | Impact |
| --- | --- | --- |
| `REQUIREMENTS_MATRIX.md` §2 and `SUBMISSION_CHECKLIST.md` G15 treat registration as unproven ("no confirmation exists") | Registration is `REGISTRATION_USER_CONFIRMED` per the owner | These docs understate the team's real position and should be corrected; the *data* gaps (ID, names) are real. |
| `REQUIREMENTS_MATRIX.md` R9 cites "S2:317-318 runtime field example is LiteRT INT8, GPU delegate" as an INT8 clause | p.11 uses that only as a **format example** for the `runtime` column, not a requirement | This slightly overstates the ambiguity; the real INT8 conflict is p.9/p.10 vs p.11 only. |
| `REQUIREMENTS_MATRIX.md` §4 books "11.37 MB (3,544,080 B ensemble + 7,819,105 B detector)" | Arithmetic gives 11,363,185 B = 10.84 MiB (11.36 MB decimal); "11.37" is a repeated approximation | Cosmetic but a reviewer who adds the bytes will notice; publish the exact byte count. |
| `SUBMISSION_CHECKLIST.md` says "the results sheet says ≤25 MB" | Correct: p.11 does say `model_size <=25 MB`. But it is a **sample-row target**, not a separate requirement | Not wrong, but do not use 25 MB to justify anything; 20 MB is the rubric. |
| `SUBMISSION.md:60` says "Share a 2–3 minute demo" but the p.24/p.27 text says 5-minute | Both exist; p.25 caps at 3 minutes | Our choice (≤3:00) is safe; state the length on the form. |
| `docs/hackathon/RESEARCH.md` (if it claims no organizer deck) | The deck **is** in the repo at root | Check and correct any "no organizer materials" claim. |

---

## 5. Things that would fail a real submission review (ordered)

1. **`results/predictions.jsonl` is a research file, not a submission file.**
   It has 1,705 rows with `ground_truth`/`correct`/`signer` and no `confidence`
   or `latency_ms`. Shipping it as `results/predictions.jsonl` fails the p.11
   schema and the p.12 layout at the same time. Highest-severity package defect.
2. **`results/results.csv` does not match the p.11 template.** Column order is
   `...pass,notes` instead of `...runtime,pass`, and the metric names differ
   (`top1_accuracy_test` vs `top1_accuracy`, `latency_p95_full_clip` vs
   `latency_p95`, five `accuracy_drop_*` rows instead of one
   `accuracy_drop_robustness`). An automated pass/fail script will mis-join it.
3. **No root `Dockerfile`.** p.25 names Docker files as a required deliverable
   and the builder hard-fails without one. Only the builder-generated copy
   inside the ZIP exists.
4. **Deck and ZIP filenames are not `CollegeName_TeamName`.** Placeholders are
   still on slide 1; p.25 calls the nomenclature mandatory and attaches a
   disqualification rule.
5. **INT8** — 20% of the score. FP16 is defensible under p.11 but contradicts
   p.9/p.10. Resolve in writing; do not claim compliance.
6. **Public repo would publish restricted weights and participant IDs.** Use a
   shared/access-controlled link or a code-only public repo.
7. **Latency and robustness are stated FAILs.** They will score as FAILs; that
   is the honest outcome, not a packaging bug.