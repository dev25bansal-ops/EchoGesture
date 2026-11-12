# Submission handoff pack — Track 2, Samsung XR Hackathon 2026

This is the one document a human uses to submit. Everything below is either
already done (and listed so you do not redo it) or is an action only you can
take. Compiled 2026-11-12 by Agent C after an independent re-extraction of the
organizer PDF and an audit of every artifact in the tree.

**Deadline: 18 November 2026, 11:59 PM.** The deck does not state a timezone
anywhere in its 28 pages (verified: `IST`, `timezone` and `Time zone` return
zero hits). Treat it as **IST** — the organizers are Samsung R&D Institute
India, Bangalore. Do not leave submission to the last hour.

---

## 1. The five actions, in order

Do these in this order. Each one has the exact text to send or paste.

| # | Action | Blocking? | Time |
| --: | --- | --- | --- |
| 1 | Send the email in §2 (INT8 + vocabulary + manifest, one thread) | no — send today | 5 min |
| 2 | Record the demo video per §4 | **yes** — p.25 requires a demo video, and the p.10 alternative ("deployed app URL") does not exist | 2–3 h |
| 3 | Fill the team/college placeholders and rename the deck per §5 | **yes** | 10 min |
| 4 | Assemble the submission package per §6 (attach vs link) | **yes** | 30 min |
| 5 | Submit once through the Google Form per §7 | **yes** | 15 min |

Action 1 is independent of everything else and should go out today. Actions
3–5 cannot complete without the exact registered strings, which only you hold.

---

## 2. Email to send — INT8 clarification

Send **from the address used for team registration**, so the organizer can
match the request to your registered team. Send it to `prism@samsung.com`
(the address printed on deck p.27 under "Queries").

**Subject:**

```
Track 2 (EchoGesture) — clarification: does FP16 TFLite under 20 MB satisfy the quantization criterion?
```

**Body — paste as-is, then fill the four bracketed fields:**

```
Hello,

We are a registered team in Track 2 (EchoGesture — sign language prediction on
an edge device) of the Samsung XR Hackathon 2026. We are preparing our final
submission for 18 November and would like one point confirmed in writing,
because the consolidated slides state the model budget in two different ways.

The deck says:

  Page 9 (Challenge Abstract):
    "Budgets: <= 20 MB quantized model (INT8), <= 50 ms latency, zero cloud
     streaming."

  Page 10 (Required Deliverables):
    "Quantized model + conversion script (< 20 MB INT8)"

  Page 11 (100-Point Jury Matrix, Model Quantization 20%):
    "Model Quantization (20%): Compilation to TFLite/ONNX <20MB."

Page 11 is the scored criterion and does not name INT8; pages 9 and 10 do. We
want to be scored on the artifact we actually have rather than on an
assumption, so we are asking before submitting.

Our current submission is a quantized (float16) TFLite ensemble of four
members. The deployed weights are 3,544,080 bytes, and the total including the
hand-landmark detector is 11,363,185 bytes (11.37 MB) — under 20 MB either way.
We use the CPU/XNNPACK delegate; there is no cloud path.

Could you please confirm which of the following is correct?

  1. An FP16 TFLite model under 20 MB satisfies the quantization criterion, or
  2. INT8 weights are mandatory, and an FP16 artifact would lose the
     quantization score.

The quantization criterion is worth 20 points of the 100-point matrix, so we
would rather ask than guess. If INT8 is mandatory, we would also be grateful to
know whether a corrected artifact can be accepted after 18 November, since we
would need to re-export and re-benchmark.

Thank you for your help.

[NAME]
[COLLEGE NAME] — [TEAM NAME]
Track 2 · EchoGesture
[EMAIL / PHONE]
```

### Why this wording

- It quotes the deck **verbatim with page numbers**, so there is nothing to
  dispute about what was said.
- It states our artifact honestly as FP16 at 11.37 MB, and does not claim INT8.
- It names the **20-point weight** so the answer arrives with the right
  priority and so a later dispute has a record of the stakes.
- It offers no fallback plan in the email. Do not promise an INT8 artifact we
  have not built.

### Record the answer

Save the reply as a PDF or screenshot plus plain text under
`docs/submission/organizer/`. If the answer is "FP16 is fine", cite it on the
INT8 row of `docs/submission/COMPLIANCE_MATRIX_PDF.md` and in the submission
note. If the answer is "INT8 mandatory", **that is a model change and Dev owns
it** — see `docs/submission/INT8_CLARIFICATION.md` §4 for the accepted
fallback path. Do not swap in the 62.61% single-model INT8 artifact to chase
20 points while quietly keeping the 72.78% headline; that would be a false
measurement claim.

---

## 3. Second email — vocabulary and manifest (same thread)

Send this **as a reply in the same thread**, not as a new message. Reply in
the same thread keeps the organizer's context in one place.

**Body:**

```
Thank you. Two further items, in the same thread so you have everything
together.

1. The fixed vocabulary list and the test manifest.
   Page 11 says "prediction is a word from the fixed vocabulary list,
   uppercase" and "clip_id must match the test manifest exactly". We have not
   been able to locate either file. Could you point us to the vocabulary list
   and the test manifest, or confirm that no organizer-supplied manifest or
   vocabulary exists for Track 2?

   We need this for two reasons: our predictions file uses our own clip IDs
   derived from the ASL Citizen filenames, which will not match an organizer
   manifest we have never seen; and our model predicts from a 100-gloss
   vocabulary, which may not be the organizer's list.

   If there is no organizer manifest, please confirm that a self-consistent
   predictions.jsonl is acceptable.

2. Filename for the predictions file.
   Page 10 and page 11 both say "predictions.jsonl", but the page 11 package
   bullet says "predictions.json and results.csv in results folder" and the
   page 12 bundle diagram shows "results/predictions.json". We currently ship
   both names with identical JSON-Lines content. If only one is read by your
   evaluator, please tell us which.

3. Deadline timezone.
   The slides state 18 Nov 2026, 11:59 PM without a timezone. We are assuming
   IST; please correct us if that is wrong.

Thank you.
```

If no reply arrives before the deadline: ship both filenames (already staged),
state plainly in the submission note that no organizer manifest was ever
supplied, and ship `results.csv` with `NOT_MEASURED` where we could not measure.
Truthful and complete beats invented.

---

## 4. Demo video — shot list summary

**Full instructions: [`docs/demo/DEMO_RECORDING_GUIDE.md`](../demo/DEMO_RECORDING_GUIDE.md).**
Read that file before recording; it has exact commands, exact narration and
verified expected outputs.

### Hard cap

**≤ 3 minutes.** The deck gives three different lengths:

| Page | Wording |
| --- | --- |
| 10 | `Demo video (2–3 min) or deployed app URL` |
| 24 | `Prototype code, 5-min demo video and deck due` |
| 25 | `Demo video, max 3 minutes (YouTube or Drive link)` |
| 27 | `submit a working prototype with a 5-minute demo video` |

The **strictest** reading is 3 minutes, and the page 25 submission rules carry
the disqualification warning. **Record a 2:50 cut.** It satisfies all four
readings. Do not pad to 5:00.

### Structure (2:50)

| Time | Shot |
| --- | --- |
| 0:00–0:20 | The problem. Title card, one line: `100 isolated ASL signs · not continuous translation · not an interpreter replacement` |
| 0:20–0:40 | The approach. Pipeline diagram: clip → landmarks → 24×255 features → 4 FP16 models → mean logits → softmax → top-5 |
| 0:40–1:10 | **Live recognition (offline file path).** Run `app/offline_demo.py` on a real clip; show the JSON and the HUD |
| 1:10–1:40 | Offline operation. Show the five model files and their byte counts; state 11,363,185 B |
| 1:40–2:05 | Ensemble and architecture detail |
| 2:05–2:30 | Accuracy, size, runtime table with N and device on every row |
| 2:30–2:50 | Limitations and what is next |

### What must NOT be claimed

Say none of these. If a shot would imply one, cut the shot.

- **No 50 ms.** The number is **not met**. State p95 **275.8 ms singleton /
  152.5 parallel-2 / 93.9 parallel-4**, on a **macOS arm64 laptop that is not
  the target hardware**. (An earlier run measured 274.2 / 154.8 / 99.8; the
  2026-11-12 run supersedes it. Either is honest, but use the current one and
  say which.)
- **No continuous translation.** This is a 100-gloss *isolated-sign*
  recogniser. It is not an interpreter replacement. Say so on screen.
- **No INT8.** The artifact is FP16. Do not let a slide or a caption say INT8.
- **No 3D features.** The features are **2D (x/y only)**; `src/features.py`
  discards the MediaPipe z channel with `use_z=False`. Do not say "3D hand
  features" or show it in the pipeline diagram.
- **No on-device camera, no phone footage, no power or thermal numbers.**
  No physical device has ever been attached. Everything is the offline file
  path on a laptop. Do not show a camera preview, a phone, or the word "live".
- **No fabricated HUD.** Use real runs only.
- **No controlled-improvement claim** between 62.61% (the older single model)
  and 72.78% (the shipped ensemble) — different models, different runs.
- **No per-member accuracy figures** — none are recorded in the repo.
- **No "83.77% validation"** — that number has no supporting artifact; the
  recorded validation figure is 75.36% (260/345) and it is optimistic relative
  to test because members were selected on it.

### On screen, throughout

```
OFFLINE FILE DEMO · not live camera · not the target hardware
```

Do not crop out the HUD's own `OFFLINE | integration not release-certified`
badge.

### Where to host it

The deck accepts **YouTube or Drive link** (p.25). Upload unlisted/public on
YouTube, or share a Drive link with "anyone with the link can view". Verify the
link opens in a private browser window before pasting it into the form.

---

## 5. Deck renaming and placeholders

### The file

```
CollegeName_TeamName.pptx
```

Deck p.25: `Presentation file (PPT or PDF) — CollegeName_TeamName, follow the
nomenclature`. Use the **exact registered strings**, with no spaces. If your
college is registered as `Example Institute of Technology` and your team as
`EchoGesture`, the file is `ExampleInstituteOfTechnology_EchoGesture.pptx` —
match whatever the registration form recorded. A PDF export is also accepted;
`.pptx` is preferred so the reviewer can see the notes.

Current file: `docs/submission/EchoGesture_Submission.pptx` (11 slides).

### Placeholders that must be filled

There is exactly **one** text box with placeholders, on **slide 1**:

```
Team: <TEAM NAME — confirm before submission>  ·  <COLLEGE NAME — confirm>
```

Replace both bracketed strings with the registered values. Nothing else in the
deck is a placeholder — verified by scanning every text frame on all 11 slides.

### But also add, because p.25 requires it

Deck p.25 lists what the presentation must cover. Compare against the 11
slides:

| p.25 requirement | Slide |
| --- | --- |
| Theme ID, project title and team details | Slide 1 has title and team, **but no Theme ID** |
| Problem statement in your own words | Slide 2 ✓ |
| Solution and architecture diagram | Slide 3 is text-only — **no diagram image** |
| Tools and tech stack used | Partly (slide 3, slide 9) |
| Innovation highlights, results and limitations | Slides 5–8, 10 ✓ |

Two gaps a reviewer will notice:

1. **Theme ID is missing.** Add `Track 2` or the exact theme ID from the
   registration form to slide 1.
2. **The architecture "diagram" is a text arrow chain, not an image.** This is
   acceptable — the text is accurate and readable — but if you have 20 minutes,
   a simple boxed diagram reads better to a jury. Do not spend longer than that
   on it; the content is already correct.

The deck contains **no images at all** (verified: zero picture shapes on all 11
slides). That is a style choice, not a compliance failure.

---

## 6. What to attach vs what to link

This is the most consequential decision in the pack. Get it wrong and the
outcome ranges from a rejected form to a licence problem.

### Attach / upload

| Artifact | Why it can be attached |
| --- | --- |
| `CollegeName_TeamName.pptx` (or PDF) | Repo-authored, MIT code and our own text. No third-party content. |
| Demo video link (not the file) | p.25 asks for a link. Do not upload a large MP4 to a form; link it. |

### Link

| Artifact | How |
| --- | --- |
| Source repository | `https://github.com/dev25bansal-ops/EchoGesture` — **must be public or shared** (p.25: `Working prototype code — public or shared GitHub repo`) |

The public branch is `hackathon/abhijeet-public`. The **shipped** model is
absent from it: no `landmark_ens_*_fp16.tflite`, no `hand_landmarker.task`, no
APK, no datasets, no per-signer evidence. `landmark_model_meta.json`,
`landmark_labels.json` and `label100.json` are present.

One caveat you should know before linking it: **two older research weight files
are still tracked on that branch** — `models/ptq_100_int8.tflite` (904,256 B)
and `models/s3d_conv1d_qat_int8.tflite` (901,200 B). They are not used by the
shipped runtime and do not produce any number we report, but `NOTICE` §2 marks
them as having no redistribution permission. Full analysis and the owner
decision are in `docs/submission/LICENSING_REVIEW.md` §2b. This does not block
submission; it is recorded so it is not lost.

**If the repository is private**, the organizer cannot read it. Either make it
public before submitting, or add the reviewer account as a collaborator and say
so in the submission note. A private repo with no granted access is a
non-submission.

### The private ZIP with weights — **do not attach to a public repo**

The bundle `evaluation/device/release_final2/echogesture_final2_ensemble.zip`
(9,007,144 B) contains the four FP16 weight files and their vocabulary. Per
`NOTICE` §2 and `docs/release/ARTIFACT_RIGHTS.md`, those weights are derived
from ASL Citizen (Microsoft Research licence, non-commercial, no
redistribution) and WLASL (C-UDA, academic/non-commercial) and carry **no
documentary redistribution permission**. The same is true of the APK, which
embeds them.

**Recommendation — access-controlled delivery:**

1. Put the ZIP in a private cloud folder (Drive / OneDrive / S3) and share it
   with **only** the organizer address (`prism@samsung.com`) or the submission
   form's file field. Do not set "anyone with the link".
2. In the submission note, say exactly what it is and under what terms:
   *"Weights are provided to the organizers for evaluation only. They are
   derived from ASL Citizen and WLASL and are not cleared for redistribution;
   please do not republish."*
3. **Never** commit the ZIP, the APK, or any `.tflite`/`.task` to the public
   repository, and never attach them to a public issue or PR.
4. If the form accepts no file upload and offers only a link field, use a
   link that requires the organizer to request access, and say so in the note.

Handing restricted weights to competition organizers for evaluation is a
different act from publishing them. Whether the ASL Citizen licence permits
even that is a question for the rights holder — the repo cannot settle it, and
this pack does not assert that it is permitted. It asserts only: **do not
publish, and say what the terms are.**

### What NOT to attach

- Any APK (embeds the restricted weights).
- Any `.tflite`, `.task`, or the release ZIP on a public surface.
- Any raw ASL Citizen / WLASL clip.
- `evaluation/asl_citizen_predictions.csv` — it carries raw `P15`-style
  participant IDs. A publish-safe copy exists at
  `evaluation/device/compliance_final/publish/asl_citizen_predictions.csv`
  (participant column dropped). Use only the publish-safe copy, and only if a
  reviewer asks for per-clip evidence.
- Any local path, token or `.env`.

---

## 7. The form

Deck p.25 and p.27 give the channel: **one submission per team via the Google
Form**. The registration link printed on p.27 is
`https://forms.gle/oGBXsydqbhE3HS2o9`. The **submission** link is stated to be
shared only with teams that completed registration (`** Final Submission Link
will be ONLY shared with Teams who have completed the Team registration
process`, p.24) — use the link you were sent, not the registration link.

Fill, per p.25:

- Theme: **Track 2 — EchoGesture: Sign Language Prediction on Edge Device**
  (deck p.2 wording).
- Team name in `CollegeName_TeamName` form.
- Per-member: name, email, phone, year, branch.
- Repository link (public or access granted).
- README / Docker note (the repo root has both; say so).
- Demo video link (≤3 min).
- Presentation file named `CollegeName_TeamName`.

Take a screenshot of the confirmation page and save it under
`docs/submission/organizer/`. This is your receipt.

---

## 8. Already verified — do not redo

Every item below was checked directly against an artifact in the tree on
2026-11-12. Re-running them wastes time you need for the demo video.

| Item | Verified value | Evidence |
| --- | --- | --- |
| Ensemble | 4 × FP16 TFLite, 3,544,080 B | `models/landmark_model_meta.json`; SHA-256 of all four match `models/README.md` |
| Detector | `hand_landmarker.task`, 7,819,105 B | `stat`, SHA `fbc2a300…` |
| Total size | 11,363,185 B = 11.37 MB, under both the 20 MB and 25 MB readings | arithmetic on the above |
| Quantization | `"quantization": "float16"` — **not INT8**, and labelled as such everywhere | `models/landmark_model_meta.json`; `RELEASE_MANIFEST.json` → `is_int8: false` |
| Top-1 | 909/1249 = 72.78%; 6 further eligible clips produced no hand; all-eligible 72.43% | `results/predictions.jsonl` (1,249 rows, exact 5-key schema, no empty predictions) |
| Top-5 | 87.51% | `evaluation/device/ensemble_evidence/RESULT.md` |
| Latency p95 | 275.8 / 152.5 / 93.9 ms, singleton / parallel-2 / parallel-4 (2026-11-12 run; earlier run 274.2 / 154.8 / 99.8) | `RELEASE_MANIFEST.json` → `release_candidate.latency_remeasured`; older run recomputable from `performance_final/results_ensemble/benchmark.json` |
| 50 ms | **NOT met** | same |
| Ensemble inference cost | 0.41 ms; hand tracking is 90% of runtime (217 ms of ~240 ms) | `performance_final/profile_final/PROFILE.md` |
| Robustness | low light +3.33, blur 0.00, frame drop 0.00 PASS; occlusion −6.67, crop −10.00 FAIL | `robustness_ensemble/RESULTS.md` |
| Power / thermal | **NOT MEASURED** — no device | `results/results.csv` rows 5–6 |
| Results sheet | exact p.11 template: 8 columns, 6 metric rows, correct order | `results/results.csv` |
| Predictions file | 1,249 rows, p.11 schema, both `.jsonl` and `.json` names | `results/predictions.jsonl` |
| `predict()` | returns exactly `clip_id, prediction, top5, confidence, latency_ms` | `src/main.py:552` |
| Vocabulary | 100 glosses, all uppercase | `models/landmark_labels.json` |
| Offline | merged Android manifest has CAMERA only; INTERNET and ACCESS_NETWORK_STATE removed with `tools:node="remove"` | `merged_manifest/debug/.../AndroidManifest.xml` read back |
| Camera | prefers `DEFAULT_BACK_CAMERA`, front fallback (exocentric) | `MainActivity.kt:281-284` |
| Android tests | 20 JVM unit tests, 4 instrumentation tests passing on an emulator | `android_four_model_evidence.md` |
| Physical device | **never attached** — no install, camera, power or thermal measurement | `android_apk_evidence.md` |
| Python | 3.11.18 | `.venv/bin/python -V`; `Dockerfile` is `python:3.11-slim` |
| Conversion script | present in the built ZIP at `scripts/ml/export_tflite_tf.py` | ZIP listing |
| Docker | fresh build runs `--help` **and** a real prediction (`WHATFOR1`, 0.8818) | `DOCKER_ACCEPTANCE.md`, `COMPLIANCE_MATRIX_PDF.md` §9.1 |
| Deck | 11 slides, placeholders present, no 3D / INT8 / 50 ms-met / continuous claim | python-pptx scan |
| CI | source-only GitHub Actions green | `RELEASE_MANIFEST.json` |

### Two things to know about the package

1. **Docker works, and I verified it.** A fresh build of the committed
   `Dockerfile` prints `--help` and runs a real prediction inside the container:
   `WHATFOR1`, confidence 0.8818. Two older images in the local Docker daemon
   (`echogesture:rc`, `echogesture:rc2`) are stale and fail on `import cv2` —
   do not cite them; rebuild from the `Dockerfile`. Evidence:
   `evaluation/device/DOCKER_ACCEPTANCE.md` and `COMPLIANCE_MATRIX_PDF.md` §9.1.
2. **Confirm the release ZIP is current before you hand it over.** An older
   `evaluation/device/release_final2/echogesture_final2_ensemble.zip` predates
   the current `src/main.py` and lacks `src/parallel_tracker.py`. If the
   packaging owner has rebuilt it, use the new one; if not, ask them to rebuild
   rather than shipping the stale archive.

---

## 9. Submission note — paste this with the form

If the form has a free-text field, or if you send a covering email, use this.
It is deliberately short and every sentence is verifiable.

```
EchoGesture — Track 2. 100-gloss isolated ASL recognition, fully offline.

What it is: MediaPipe hand landmarks -> 24x255 per-frame features -> a
four-member FP16 TFLite ensemble (mean logits, one softmax) -> top-5 gloss.
11,363,185 B total including the hand detector.

Measured on a macOS arm64 laptop (ai-edge-litert 2.3.0, XNNPACK CPU), which is
not the target hardware:
  - Top-1 72.78% (909/1249) on the ASL Citizen signer-disjoint test split,
    11 unseen signers; top-5 87.51%. The test split was inspected during model
    selection, so this is not a pristine blind evaluation.
  - Full clip-to-JSON p95 275.8 ms singleton, 152.5 ms parallel-2, 93.9 ms
    parallel-4. The 50 ms target is not met.
  - Robustness: low light, motion blur and frame drop pass; crop (-10.00 pp)
    and occlusion (-6.67 pp) exceed the 5 pp budget.

What we do not claim: no INT8 (our artifact is FP16; see our clarification
request), no 3D features (our features are x/y only), no on-device camera or
power/thermal numbers (no physical device was available), and no continuous
translation.

Weights are derived from ASL Citizen and WLASL and are supplied for evaluation
only; they are not cleared for redistribution. Source is MIT.
```

---

## 10. Final pre-submit check

- [ ] Clarification email sent; reply saved or noted as pending
- [ ] Demo video recorded, ≤3:00, uploaded, link opens in a private window
- [ ] Deck slide 1 placeholders replaced with registered strings
- [ ] Deck renamed `CollegeName_TeamName.pptx`
- [ ] Repository public, or organizer access granted and stated
- [ ] ZIP/APK/weights **not** on any public surface
- [ ] Submission note text pasted
- [ ] One submission only — no duplicate
- [ ] Confirmation screenshot saved under `docs/submission/organizer/`