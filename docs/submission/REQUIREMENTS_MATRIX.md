# Track 2 organizer requirements — evidence matrix

Compiled 2026-11-11 by A04 (release/requirements role). Every row is grounded in a
file that exists in this repo. Where the organizer material is silent or the repo
contains no proof, the status is **UNKNOWN** — no organizer answer has been
invented here.

## Sources actually read

| ID | Source | Notes |
| --- | --- | --- |
| S1 | `XR_Hackathon_2026_Consolidated_Slides_for_sharing_v1.pdf` (repo root, 1,168,491 B) | Primary organizer material. Confirmed 28 `/Type /Page` objects and `/Count 28` in the raw PDF; `_slides_text.txt` has 28 `===== PAGE =====` markers, so the extraction is complete. Line citations below are to `_slides_text.txt`. |
| S2 | `_slides_text.txt` | Text extraction of S1 (no `pypdf`, no `pdftotext` available in this environment). |
| S3 | `docs/hackathon/SUBMISSION.md` | Team's checklist. Internal, not organizer text. |
| S4 | `docs/hackathon/ABHIJEET.md` | Team workflow. Internal. |
| S5 | `docs/hackathon/HANDOFF.md` | Team handoff contract. Internal. |
| S6 | `LICENSE`, `NOTICE` | Licensing record. |
| S7 | `results/results.csv`, `models/README.md` | Measured values and ensemble provenance. |

"Repo materials do not answer this" below means S1–S7 say nothing.

---

## 1. Requirements matrix

| # | Requirement | What the material actually says (file:line) | Our status | Evidence | Remaining action |
| --- | --- | --- | --- | --- | --- |
| R1 | Eligibility | S1/S2 p.2 defines Track 2 "EchoGesture: Sign Language Prediction on Edge Device — Fully on-device, exocentric sign language translation on wearable edge processors." No student-status, age, or geographic eligibility clause appears anywhere in S1. | **UNKNOWN** | Absence in S1. | Nothing to action unless the live form states criteria; capture the form text when reviewing. |
| R2 | Registration deadline | S2:667-668 "Google Form / Last date 8 Nov 2026, 11:59 PM"; S2:637-638 "8 NOV / Registration Closes / Final Submission Link Opens". | **REGISTRATION_USER_CONFIRMED** | The team owner confirmed registration on 2026-11-11. The deadline has passed and the team met it. | Registration is settled and is **not** an open blocker. The confirmation ID and exact registered college/team strings are held privately and are needed only to fill form fields and name the artifacts. |
| R3 | Registration requires the final-submission link | S2:659 "** Final Submission Link will be ONLY shared with Teams who have completed the Team registration process". | **REGISTRATION_USER_CONFIRMED** | Same as R2 — registration is confirmed. | The submission link is expected to reach the registered team; obtain it from the owner's inbox or from `prism@samsung.com`. |
| R4 | Team membership | S2:668-673 "Up to max 4 members per team, single college", "Per member name, email, phone, year & branch", "One theme one submission per team". | **UNKNOWN** | No member roster is recorded in the repo. `docs/release/DEV_HANDOFF.md:61` and `DEV_HANDOFF_PUBLISHED.md:51` list "Confirmation the meta `members` list is final" as still open — that is the *model* member list, not the team roster. | Record the team roster (names, emails, phones, year, branch) and the count before submitting. |
| R5 | Track selection | S2:39-41 "[ T R A C K 0 2 ]", S2:176-177 Track 02 title; S2:674 "One theme one submission per team". | **PASS** | This repo is entirely Track 2 (EchoGesture); README and every evaluation doc are Track 2. | State Theme/Track ID "Track 2" on the deck (S2:699). |
| R6 | Final submission deadline | S2:677 "FINAL SUBMISSION · Last date 18 Nov 2026, 11:59 PM"; S2:678 "Everything below needs to be submitted by 18th Nov 2026. No extension would be provided." | **PASS (on schedule)** | Internal target in `SUBMISSION.md:33` is 17 November; today is 11 November. | Submit by the internal 17 Nov target. |
| R7 | Finals timeline | S2:648-657: 23 Nov Top 15 announced; 27 Nov final demo; 2 Dec results. S2:760 "Tentative — subject to minor changes." | **PASS (informational)** | — | Plan demo readiness for 27 Nov if shortlisted. |
| R8 | Allowed model format | S2:217 "quantized TFLite / ONNX Runtime model"; S2:225-226 "Quantized model + conversion script (< 20 MB INT8)"; S2:252 "Quantized model file and conversion script"; S2:259-261 "Model Quantization (20%): Compilation to TFLite/ONNX <20MB." | **PASS (artifact legal), FAIL (as INT8-pipeline claim)** | Shipped artifacts are TFLite. But shipped weights are **FP16**, not INT8 (`models/README.md:11` "4-member FP16 ensemble"; `results.csv:6` "runtime=LiteRT FP16"). An INT8 landmark artifact exists (`models/landmark_classifier_int8.tflite`) but is not the shipped model. | Ship TFLite (done) and state FP16 honestly. If the organizer's rubric is read strictly as *INT8 required*, the current bundle does not satisfy it — see R9. |
| R9 | Is INT8 mandatory, or does "TFLite/ONNX under 20 MB" suffice? | **The slides say both, at different strengths.** S2:200 (challenge abstract): "Budgets: ≤ 20 MB quantized model **(INT8)**" — INT8 named in parentheses. S2:225-226 (required deliverables): "Quantized model + conversion script (**< 20 MB INT8**)". S2:259-261 (100-point rubric): "Model Quantization (20%): Compilation to **TFLite/ONNX <20MB**" — INT8 not mentioned, only the format and size. S2:317-318 (results-sheet example): runtime field example is "LiteRT **INT8**, GPU delegate". | **UNKNOWN (ambiguous in the source); currently FP16** | Exact wording above, quoted. The abstract and deliverables say INT8; the scored rubric says only TFLite/ONNX <20 MB. | **Do not claim INT8 compliance.** Get written organizer clarification, or benchmark and ship an actual INT8 candidate. `docs/release/SUBMISSION_DASHBOARD.md:81` already records "INT8 compliance — NOT established — selected classifier is FP16". Treat as an open scoring risk (20% weight), not a settled fail. |
| R10 | Model size limit | S2:200 "≤ 20 MB"; S2:259-261 "<20MB"; but S2:302-304 results-sheet target row "model_size <=25 MB". | **PASS** | `results.csv:6` measured 11.37 MB (3,544,080 B ensemble + 7,819,105 B detector); builder hard-fails at ≥20e6 bytes (`build_bundle.py:369-371`). | Fill the sheet's own field; report 11.37 MB and note the 20 vs 25 MB discrepancy. Count the detector in the total (we did). |
| R11 | What counts toward the size limit | Repo materials do not answer whether the hand-landmark detector counts. S1/S2 only says "model" / "model_size". | **UNKNOWN** | — | Conservative choice: we count the detector (included in the 11.37 MB). Confirm with organizer if possible. |
| R12 | Latency measurement boundary (per-frame vs per-clip) | S2:186-188 (motivation): "Translation must keep pace with ≤ 50 ms latency"; S2:200 "≤ 50 ms latency"; S2:263-264 "Edge Latency (20%): Measured inference time <50ms."; S2:287 "latency_ms is self-reported. The organizers re-measure on their own device."; S2:299-300 results row "latency_p95 <=50 ms". | **FAIL (on dev laptop, both readings)** | `results.csv:4` latency_p95_full_clip 274.2 ms → FAIL; `results.csv:5` parallel-4 p50 79.3 ms. Measured now from the clean extract: 535.86 ms and 597.80 ms on two runs for a single clip. Ensemble inference alone is 0.36 ms; MediaPipe tracking dominates (`results.csv:4` notes). | No target device has been measured (`results.csv:8-9`: avg_power and thermal_drift NOT_MEASURED, "No target device"). Latency is the largest unresolved scoring risk. The boundary is ambiguous in the slides ("inference time" vs full clip p95); report both, and get a phone measurement. |
| R13 | Required runtime | S2:217 "quantized TFLite / ONNX Runtime model"; S2:317-318 example runtime "LiteRT INT8, GPU delegate". On-device, no cloud: S2:190-194 "All visual processing stays on the device — zero raw bystander video uploaded to the cloud", S2:200 "zero cloud streaming". | **PASS (runtime), PASS (no cloud)** | `results.csv:2` runtime "ai-edge-litert 2.3.0, XNNPACK CPU"; predictor runs fully offline from the extracted zip. | Report the actual delegate (CPU/XNNPACK) in the sheet, not an INT8 GPU delegate we did not use. |
| R14 | Official vocabulary | S2:283-284 "prediction is a word from the fixed vocabulary list, uppercase." S2:231-236 datasets "How2Sign (80+ h ASL) & WLASL (2,000 words); starter baselines provided." No literal vocabulary list is present in S1. | **UNKNOWN — official list not obtained** | Our vocabulary is `models/landmark_labels.json` (100 labels, `models/README.md`). `SUBMISSION.md:25-31` records that no starter kit or vocabulary arrived and sets the fallback: define and publish our own. | Ship our fixed vocabulary in the bundle (done) and state its scope explicitly on the deck and README. Do not call it the organizer's list. |
| R15 | Test manifest | S2:281 "clip_id must match the test manifest exactly." S2:288-289 "The evaluator computes accuracy and checks that every clip ID is present." No manifest file is in S1. | **UNKNOWN — no organizer manifest supplied** | `build_bundle.py:207-227` supports `manifest.json` and enforces exact ID match; `--official` is hard-blocked without real organizer predictions (`build_bundle.py:475-478`). | If a manifest arrives, run it and regenerate predictions. Otherwise submit the provisional bundle with our own demo predictions and say so. |
| R16 | Predictions output format | S2:242-246 the evaluator calls one Python function `def predict(video_path: str) -> dict`; S2:276-279 return format `{ "clip_id", "prediction", "top5", "confidence", "latency_ms" }`; S2:288 "Teams submit a single predictions.jsonl file, one JSON object per line, one line per test clip." But S2:248 and S2:339-340 bundle diagrams say `predictions.json`. | **PASS (with ambiguity)** | `src/main.py` implements `predict(video_path) -> dict`; verified live below. Builder validates the same contract (`build_bundle.py:190-204`): required clip_id/prediction/confidence/latency_ms, uppercase label in vocabulary, confidence in [0,1], latency ≥ 0; top5 optional. | The `predictions.json` vs `predictions.jsonl` conflict is recorded in `SUBMISSION.md:74`. No organizer manifest exists, so no predictions file is shipped in this provisional build. Resolve against the final form/starter kit. |
| R17 | results.csv columns | S2:290-315: columns `metric, target, measured, unit, device, runtime, pass` and rows `top1_accuracy >=70 %`, `latency_p95 <=50 ms`, `model_size <=25 MB`, `accuracy_drop_robustness <=5 pt`, `avg_power <=500 mW`, `thermal_drift_10min <=20 %`. S2:314-320 "Teams fill in measured, device …, and runtime …". | **PASS (columns), partial (values)** | `results/results.csv` header is `metric,target,measured,unit,device,runtime,pass,notes` — superset of the template (adds `notes`, used for model SHA/evidence). `build_bundle.py:31,236-238` enforces a compatible column set. | Keep the extra `notes` column (additive). Fill robustness/avg_power/thermal rows — currently NOT_MEASURED. |
| R18 | Accuracy targets | S2:296-297 "top1_accuracy >=70 %"; S2:305-306 "accuracy_drop_robustness <=5 pt". | **PASS (top-1), UNKNOWN (robustness)** | `results.csv:2` top1 72.78% PASS, but its own note says "test split was previously inspected during model selection; not a pristine blind test" and 909/1249 scored. Robustness drop is not present as a measured row. | Do not overclaim: report 72.78% with the caveat, and measure the robustness drop or mark it NOT_MEASURED. |
| R19 | Power / thermal targets | S2:308-313 `avg_power <=500 mW`, `thermal_drift_10min <=20 %`. | **NOT_MEASURED** | `results.csv:8-9` both NOT_MEASURED ("No target device or power measurement method available"). | Measure on a phone, or mark NOT_MEASURED in the official sheet. |
| R20 | ZIP / folder layout | S2:327-340 `team_name.zip` → `src/` (`requirements.txt`, `main.py`, `README.md`), `results/` (`predictions.json`, `results.csv`). S2:343-347 "the entire implementation lives under `src/`", "keep the hierarchy shown below; you may add extra helper files or sub-folders as long as the core structure stays unchanged". S2:349 "Python3.11". | **PASS** | Built zip contains `src/{main.py,requirements.txt,README.md,features.py,landmark_frontend.py}`, `results/results.csv`, `models/*`, `scripts/ml/export_tflite_tf.py`, `Dockerfile`, `release.json`. Core structure matches; extras are permitted by S2:345-347. | `results/predictions.json` is absent because no organizer manifest exists (official build is blocked by design). Flag this gap explicitly in the submission note. |
| R21 | Source / reproducibility requirements | S2:222-223 "GitHub repo: predict() video → text, reproducible setup"; S2:343 "Reproducible source code – the entire implementation lives under `src/`"; S2:684 "Working prototype code — public or shared GitHub repo". | **PASS (structure), BLOCKED (public) for weights** | Source is under `src/` in the zip and in the repo. Public publication of the *weights* is blocked — see §3. | Publish code + a Docker build; supply weights only if the organizer needs a runnable artifact and licensing is resolved (§3). |
| R22 | Docker / install | S2:685-686 "README with reproducible setup instructions, Docker files & other requirements". | **PASS** | `Dockerfile` generated in the bundle (`build_bundle.py:527-530`: `FROM python:3.11-slim`, installs `src/requirements.txt`). `src/requirements.txt` pins 74 B of deps. | Verify the Docker build once on a clean machine before submitting. |
| R23 | Demo requirement | S2:227-228 "Demo video (2–3 min) or deployed app URL"; S2:254 "Demo video link (2–3 min) or deployed app URL"; S2:644 "Prototype code, 5-min demo video and deck due"; S2:689 "Demo video, max 3 minutes (YouTube or Drive link)". | **UNKNOWN (length conflict); format PASS** | `SUBMISSION.md:75` records the conflict and chooses 2–3 min, which satisfies the strictest of the four statements (max 3). | Record/attach the 2–3 min demo link before submitting. |
| R24 | PPT/PDF naming rule | S2:691-692 "Presentation file (PPT or PDF) — CollegeName_TeamName, follow the nomenclature"; S2:697 "Refer to the CollegeName_TeamName_Submission_ppt". | **BLOCKED** | Depends on the team's actual college and team name, which are not recorded anywhere in the repo (R4). | Obtain college + team name, then name the deck `CollegeName_TeamName`. |
| R25 | Deck contents | S2:699-707 "Theme ID, project title and team details; Problem statement in your own words; Solution and architecture diagram; Tools and tech stack used; Innovation highlights, results and limitations". | **PASS (content available)** | README, `ARCHITECTURE_R8.md`, `results/results.csv` cover these sections. | Assemble the deck against this list; state limitations (FP16, latency, unmeasured power/thermal) honestly. |
| R26 | Submission form | S2:694 "One submission per team via the Google Form"; S2:659 the link goes only to registered teams; S2:708 "Teams NOT following the submission guideline would lead to direct disqualification". | **WAITING_FOR_ORGANIZER** | Registration is confirmed; the form link has not been received in the repository. | Obtain the link from the owner's registered contact address, or request it from `prism@samsung.com`. |
| R27 | Disqualification rule | S2:708 "Teams NOT following the submission guideline would lead to direct disqualification". | **PASS (aware)** | — | Re-check the final layout and naming against S2 before submitting. |
| R28 | Contact | S2:745 "Queries prism@samsung.com". | **PASS (informational)** | — | Use this address for the registration and INT8 clarifications. |

---

## 2. Registration status — findings

**There is no registration, confirmation, receipt, team name, or college name
recorded anywhere in the repository.** Searched: every file under `docs/`,
`README.md`, all `.md`/`.txt`/`.json`/`.csv`, plus a filename scan for
`*regist*`, `*receipt*`, `*confirm*`, `*form*`.

What exists instead is only *instructions that one should happen*:

- `docs/hackathon/SUBMISSION.md:20-24` — "Register for Track 2 by **8 November
  2026, 11:59 PM IST** … Use `CollegeName_TeamName`."
- `docs/hackathon/README.md:30` — "Registration closes **8 Nov, 11:59 PM IST**."
- `docs/engineering/MULTI_AGENT_STATUS.md:3` — "Registration is an external human
  action due tonight".
- `docs/release/v7_submission/V7_FINAL_EVIDENCE.md:333` — "**No
  registration/submission/receipt.**"
- The live-form college restriction (`SUBMISSION.md:12-19`) was captured to
  `.scratch/form_live.html` on 6 November — **`.scratch/` does not exist now**, so
  that capture is gone.

**The 8 November deadline has passed (today is 11 November).** I am not assuming
the team registered.

**Top external blocker — steps Abhijeet must take:**

1. Check `prism@samsung.com` and the hackathon inbox for the registration
   confirmation from the Google Form (`https://forms.gle/oGBXsydqbhE3HS2o9`,
   `docs/hackathon/README.md:13`), and for the "Final Submission Link" email that
   S2:638 says opens when registration closes.
2. If a confirmation exists: save it to `docs/submission/registration/` (a PDF
   screenshot plus the confirmation text) so it is evidence in the repo, and
   record the exact team name and college used — they drive the R24 filename.
3. If no confirmation exists: email `prism@samsung.com` immediately, stating
   Track 2, the team members, the college, and the date the form was submitted;
   ask whether the team is registered and how to access the final submission
   form. Copy the reply into `docs/submission/registration/`.
4. If the team never registered: ask the organizer whether a late entry is
   possible. Without registration the submission link is not guaranteed
   (S2:659), which makes R2/R3/R26 the gating items for everything else.
5. Also confirm the college is one of the five accepted institutions
   (`SUBMISSION.md:13-19`: VIT Vellore, VIT Chennai, SRM, Thapar, Manipal), since
   the live form is a closed list the slides do not mention.

---

## 3. Licensing verdict

**Can we ship the ensemble weights publicly? No.** Not without written
permission from the rights holders. This is not a caution — `NOTICE` states it
directly.

Facts from `LICENSE` and `NOTICE`:

- **Repo-authored code is MIT** (`LICENSE:1-3`), but the grant is explicitly
  narrow: `LICENSE:29-42` says it covers "ONLY repo-authored source code and
  harnesses" and grants "NO redistribution rights over … model artifacts under
  `models/`".
- **ASL Citizen-derived (no redistribution permission):**
  `landmark_classifier_fp16.tflite` and `landmark_classifier_int8.tflite`, plus
  `landmark_labels.json`. `NOTICE:30-32`: "**NO confirmed redistribution
  permission. LOCAL/PRIVATE only pending written authorization**". The
  **4-member ensemble is ASL Citizen-derived too** — `models/README.md:20-23`
  marks every member "**no** (restricted)" — so the ensemble, our shipped model,
  is in the restricted set.
- **WLASL-derived (no redistribution permission):**
  `s3d_conv1d_qat_int8.tflite` and `ptq_100_int8.tflite`. `NOTICE:34-35`: "**NO
  confirmed redistribution permission for these specific weight files.** WLASL
  C-UDA (academic / non-commercial) restriction reattaches."
- **Apache-2.0 (shippable):** `models/hand_landmarker.task` (MediaPipe,
  `NOTICE:29`: "Apache-2.0 claimed. Re-verify current MediaPipe terms and carry
  the upstream attribution + source URL at submission.").

So of the bundle's shipped model assets, only the MediaPipe detector is clearly
redistributable. The four ensemble members and their 100-label vocabulary are not.

**Pre-existing public exposure — needs owner action, not a history rewrite:**

`origin/main` already contains, from commit `c47c5c9`:

- `evaluation/asl_citizen_predictions.csv` — 1,706 added lines carrying raw
  `P15`-style participant IDs (still present on `origin/main` today: it is the
  only `evaluation/*.csv` on that branch matching `P[0-9]+`).
- `models/ptq_100_int8.tflite` (904,256 B) and `models/s3d_conv1d_qat_int8.tflite`
  (901,200 B) — both still tracked on `origin/main` (`git ls-tree`).

`git branch -r --contains c47c5c9` shows it on `origin/main`,
`origin/hackathon/abhijeet-public`, `origin/hackathon/dev-model` and
`origin/hackathon/two-person-sprint`. This exposure predates this task. The
accurate response is an **owner decision** — contact the organizers/rights
holders, and decide whether to make the repository private or place a
redistributable subset in a separate public repo. **Do not rewrite published
history to hide it**; that is not a fix and destroys auditability. `NOTICE:68-77`
already flags that removing or hiding these tracked participant files is "a
separate owner decision, out of scope here".

Practical consequence for the submission: a GitHub link to this repository as-is
publishes restricted weights and participant IDs. Before handing a public link to
the organizer (`S2:684`), either make access controlled ("public or shared
GitHub repo", so a *shared private* repo is allowed), or publish a code-only
public repo. This needs the owner's decision and, ideally, organizer guidance.

---

## 4. Verified local release candidate

Built with the existing builder (no source, model, script, app or result file was
edited):

```
$REPO/.venv/bin/python -I \
  scripts/release/build_bundle.py build \
  --source . \
  --model models/landmark_classifier_fp16.tflite \
  --labels models/landmark_labels.json \
  --results results/results.csv \
  --output evaluation/device/release_final2/echogesture_final2_ensemble.zip \
  --extra scripts/ml/export_tflite_tf.py
```

The builder returned `"status": "structure_verified"` and hash-verified all four
ensemble members (it did not reject anything):

| Member | SHA-256 |
| --- | --- |
| `landmark_ens_tcn21_fp16.tflite` | `112279bc86090470b82b20b510746a946220322c25ec84cb3706a538c73dc22a` |
| `landmark_ens_tcnm61_fp16.tflite` | `01a261d8eb6c2c4d8b501f0ce1a36db338365d17fddabc63528a7941b1390f63` |
| `landmark_ens_tcnm62_fp16.tflite` | `27058a85e032bee3af444fb1538ff0bf32fd38fa2b01ce01a2f25c3a428d20b3` |
| `landmark_ens_rob71_fp16.tflite` | `08417e72da59150ac9adf749e6324c30fb001d4a401394e32f125d3276d1ab21` |

These match the fingerprints in `models/README.md:60-63`.

**Artifact**

| Field | Value |
| --- | --- |
| Zip path | `$REPO/evaluation/device/release_final2/echogesture_final2_ensemble.zip` |
| SHA-256 | `48a062f27492ed30bd02cb00732401d9a751b5ae02b94f837deee3835213aff7` |
| Size | 9,743,727 bytes (9.29 MiB) |
| Entries | 17 files (listed below) |
| Gitignored | **Yes** — `git check-ignore -v` → `.gitignore:27:*.zip` |

Contents: `Dockerfile`, `release.json`, `models/` (hand_landmarker.task,
landmark_classifier_fp16.tflite, the 4 ensemble members, landmark_labels.json,
landmark_model_meta.json), `results/results.csv`,
`scripts/ml/export_tflite_tf.py`, `src/` (main.py, requirements.txt, README.md,
features.py, landmark_frontend.py).

**Clean-room predictor run** — extracted to a fresh `mktemp -d`
(`/tmp/eg-final2-N0JCMw`), cwd = extract root, with
`$REPO/.venv/bin/python -I -c "import sys, json; sys.path.insert(0, 'src'); import main; print(json.dumps(main.predict(sys.argv[1])))"`
on `evaluation/device/v5_accuracy/data/val_30/0283886564670357-WHAT FOR.mp4`:

```json
{"clip_id": "0283886564670357-WHAT FOR", "prediction": "WHATFOR1", "top5": ["WHATFOR1", "OFFEND", "KNIGHT1", "SCROLLDOWN", "DECIDE1"], "confidence": 0.8821, "latency_ms": 535.86}
```

A second run from the same extract returned identical prediction/confidence and
`latency_ms: 597.8`, so the timing is a stable warm number, not cold-start noise.
The prediction `WHATFOR1` is the intended label for this clip
("WHAT FOR" → `WHATFOR1`), and it is from the shipped 100-label vocabulary.

Caveat to carry forward: `src/main.py` selects the landmark backend with
`models/landmark_classifier_fp16.tflite` as its base path and then loads the
`members` list from `models/landmark_model_meta.json`. The ensemble weights are
the ones actually executed for inference; the base file is also staged by the
builder. `results.csv:6` books 11.37 MB as 3,544,080 B ensemble + 7,819,105 B
detector, while the zip's `models/` directory also contains the 832,888 B base
file — every variant of the accounting is under 20 MB, but report which files
that number counts.

---

## 5. Open items ranked

1. **Registration proof / late entry** (R2, R3, R26) — blocks everything; §2.
2. **Public-repo licensing exposure** (R21, §3) — blocks a public GitHub link.
3. **Latency on a real device** (R12) — currently FAIL on a laptop; no phone run.
4. **INT8 ambiguity** (R9) — 20% of the rubric rests on it; get it in writing.
5. **Official vocabulary / manifest** (R14, R15) — absent; fallback in place.
6. **Team name + college** (R4, R24) — needed for the deck filename.
7. **Power / thermal / robustness** (R18, R19) — NOT_MEASURED.