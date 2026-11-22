# Final delivery checklist

Date 2026-11-21 (latest revision). Deadline **2026-11-22 23:59 IST** — per the organizer's V3-2 deck, slide 24 (the 18 Nov date the repo carried is superseded).

Two states per line: **DONE** means it is finished and verified here. **HUMAN**
means it cannot be finished by me — it needs a person, a device, or an
authorisation I do not have.

---

## A. Artifacts

| # | Item | State | Detail |
| --: | --- | --- | --- |
| 1 | Old APK (superseded) | **HISTORICAL** | `echogesture-p1.apk`, `150ffeeb…cb7a`, 83,350,522 B. Replaced 2026-11-21 by the build below; kept only so old references resolve. |
| 2 | Submission ZIP | **REBUILT 2026-11-22 (the earlier hash was stale)** | `EchoGesture_Submission.zip`, `5d310cec…8e25`, 13,365,184 B. **The previous row claimed `7691190b…6cca` and no such file existed anywhere on disk** — that was my own documentation claiming an artifact I had not produced. Rebuilt now by `scripts/release/build_bundle.py`, which stages and self-verifies: 25 entries, 9 FP16 members, `src/main.py`, `results/results.csv`, `results/predictions.jsonl`. Clean-room extracted to an empty directory and `predict()` run **from the extracted copy only**: `NOON_at_07` → `NOON1` at 0.955, returning exactly the five required keys. Gitignored (`*.zip`), so it is a local delivery artifact — upload it with the form, do not expect it in the repo. |
| 3 | PPTX | **DONE** | `VIT_Chennai_Sky_Grazers.pptx`, `9837b222…eeab`, 573,845 B, 9 slides, built **inside the organizer's own template** (`Submission_Final_forGithub.pptx`), so the section list and header geometry are the template's rather than our own invention. Carries the corrected claims (85.45% not 77.02%, 150 tests not 110, latency FAIL on the shipping build, one-tap flow, the gate's test-split accuracy). **Byte count in both manifests was stale (413,539) and has been corrected to the measured value on 2026-11-22** — the hash had been updated but the count had not. **Rebuilt again the same evening to carry the demo video link on slide 7**, as a 6.5 pt URL with a clickable hyperlink; that rebuild is the `9837b222…` above and it supersedes the `6a2ea2bc…` the manifest records as `replaced_sha256`. |
| 4 | PDF | **DONE** | `VIT_Chennai_Sky_Grazers.pdf`, `97000bee…bd5c`, 403,123 B, 9 pages. Native Microsoft PowerPoint export, not the PIL rasteriser (which was only ever a layout check): text is selectable and fonts are embedded. Every figure was re-extracted from the PDF's text layer to confirm it carries the corrected numbers. **There was no PDF entry in either manifest at all** — it existed on disk and was cited in prose but was not in the hash list. Added as `presentation_pdf` on 2026-11-22. Re-exported after the slide-7 demo link was added; slide 7 now carries **9 clickable link annotations** in the PDF (verified by extracting the `/URI` actions), and the last line of text on that slide ends 17.9 pt above the 405 pt page edge, so the URL does not overflow. |
| 6 | Release manifest | **DONE — repaired 2026-11-22** | Both `RELEASE_MANIFEST.json` files now resolve completely: **12 entries verified against disk, 0 problems**. Three defects were fixed: (a) the deck's byte count was stale; (b) the PDF had no entry; (c) `submission_zip.path` pointed at `submission/EchoGesture_Submission.zip`, which does not exist (the zip is at the repo root), and `submission_zip_2026_11_22` pointed at a root path with a mismatched hash. A twelfth entry, `demo_video_hosted_2026_11_22`, records the public Drive URL, its `drive_file_id`, and the hash of the bytes an anonymous download returns. The three historical APK entries (`apk_p1`, `apk_v4_2_rollback`, `apk_v4_3`) either no longer exist or were rebuilt; each is now marked `present_on_disk` with its path cleared rather than left pointing somewhere wrong, and `apk_v4_3` is re-hashed from its surviving copy under `.scratch/apkcheck/`. |
| 5 | APK — **ship this** | **DONE (Auto 6.2, published 2026-11-22)** | `EchoGesture-Auto-6.2.apk`, `b1c0e671…2dcf8`, 81,281,198 B, package `org.echogesture.finalapp.auto` vc9/6.2. **Signed, non-debuggable RELEASE build** — `android:debuggable` absent, signature `CN=measurement` (cert `6573b6fd…1bb97`, identical to the 00:29 build it replaces, so this is a rebuild not a re-key). 150/150 unit tests pass on the release variant. Carries all 10 model assets; its `landmark_model_meta.json` reads `ensemble_test_top1: 0.8545`, matching the deck. **Rebuilt on deadline morning to add the per-frame p95 line** (`result_profiler_p95`), because the release build has no other readable channel for the scored metric (`allowBackup="false"`, non-debuggable, no `run-as`). Driven on the same Vivo V2404 after install: `MECHANIC1` 89.3% and `DARK1` 73.4%, both matching the debug build on the same clips, with HUD p95 **detect 51 ms / convert 4 ms** and **detect 50 ms / convert 3 ms**. Those are real p95s on the shipping artifact — but off the **file-pick** path, not the live camera with a hand at the device, so the `latency_p95` row stays FAIL. |
| 5c | APK — superseded build | **HISTORICAL** | `a56ba20f…8ada7`, 80,815,032 B — the 00:29 release build, same models and key, without the p95 HUD line. Replaced by row 5; kept only so older references resolve. |
| 5b | APK — historical | **HISTORICAL** | `EchoGesture-2026-11-21.apk`, `96f00458…`, 82,894,510 B, DEBUGGABLE. Installed and driven on a physical Vivo V2404; every latency, power and thermal figure in results.csv came from this build, so those are upper bounds on the release APK above. |
| 5 | Deck filename | **DONE 2026-11-22** | Now `VIT_Chennai_Sky_Grazers.pptx` / `.pdf` (college VIT Chennai, team Sky Grazers). Rebuilt from the organizer's template, so slide 1 carries the team string and the full member names — Dev Datya Pratap Bansal, Miskin Abhijeet Laxman — and slide 9 carries the team string. Slide 5's latency row was corrected in `build_template_deck.py`, the source that generates the deck, after a rebuild silently reverted an earlier edit to the built file. |
| 6 | Release manifest | **DONE** | `submission/RELEASE_MANIFEST.json`, validated JSON, hashes re-measured 2026-11-21 against the shipped APK, deck and PDF. |
| 6b | Demo video | **RECORDED AND HOSTED 2026-11-22** | **The submitted demo is `EchoGesture_demo_2026-11-22.mp4`: 6,729,172 B, 640×368, 30 fps, 1:20, narration, sha256 `0c55189b…d4a2`.** Hosted publicly at **[https://drive.google.com/file/d/1vojMAB5cpxQc-LkbCZlxoBeoqUklFqsE/view](https://drive.google.com/file/d/1vojMAB5cpxQc-LkbCZlxoBeoqUklFqsE/view)** and linked from slide 7 of the deck as a clickable hyperlink. Access verified **anonymously** — an unauthenticated `GET` to the direct-download endpoint returns the full 6,729,172 bytes as `video/mp4`, `isItemTrashed: false`, and the SHA-256 of the downloaded bytes is identical to the local file. Content: a real person on camera introducing the project, then four captured signs on the shipping app with the matching result card shown beside each (`DEAF1` 71.0%, `VOICE` 94.5%, `COMB2` 81.3%, `APPLE` 70.4%), a four-step pipeline slide, and an end card. **Known gaps, stated rather than hidden:** the video is **1:20 against the deck's ask of 2–3 min**; it is 640×368; and the narration uses a project name the unbiased ASR renders as "Acogaster"/"Accu-Gesture" while every title card reads **ECHO GESTURE** — the team should confirm by ear which word is actually spoken. Gitignored (`*.mp4`) and deliberately not committed: ASL Citizen footage and a person's face on a public repo. |
| 6d | Earlier generated walkthrough | **HISTORICAL** | `docs/submission/EchoGesture_demo_2026-11-22.mp4`, `dd4ec5fa…5b98`, 28,154,151 B, 2:55.0, built by `scripts/media/build_demo_video.py` from `.scratch/thresh/probs_test.npz`. This was a narrated walkthrough generated from model output, not a filmed demo, and it is **superseded by row 6b** — the team's own recording is what was uploaded and linked. Kept only so older references resolve; no claim in the submission cites it any more. |
| 6c | Refusal path re-confirmed on the shipping build | **DONE 2026-11-22 14:31 IST** | Re-ran the no-hands test from a cold idle state on `EchoGesture-Auto-6.2.apk`: tapped **Start Recognition** with the camera pointed at an empty room, and after the auto-segmentation window closed the app showed the amber panel **"No hands in view"** over *"No sign started. Tap Start Recognition and perform one sign."* The refusal appeared within **8 seconds** and held. **History was then opened and contained six entries (85–95% each, all from earlier real recognitions) and no entry from the refusal** — confirming that a declined capture is not written as a recognised sign. Fresh screenshot: `docs/evidence/auto62_07_refusal_reproduced_2026-11-22.png`. This resolves an ambiguity: an earlier tap in the same session produced a 52.9% "POSSIBLE MATCH" reading, which was a **mis-timed tap** (the button had not yet entered the waiting state, so a previously buffered clip was scored) rather than the no-hands path leaking a guess. The documented behaviour is correct as written. |

---

## B. Verification actually run

| # | Check | State | Result |
| --: | --- | --- | --- |
| 7 | Android JVM tests | **DONE** | 150 / 150 pass on the merged `submission/final` tree (was 110 before the camera-fix tests landed) |
| 7b | Docker image builds and runs | **DONE — was BROKEN, fixed 2026-11-22** | `docker build -t echogesture .` failed before today: `.dockerignore` excluded `models/` and `app/`, both of which the Dockerfile `COPY`s, so the build died at `failed to compute cache key: "/app/smoke_test.py": not found`. A stale comment in the same file claimed the weights were "mounted at runtime", which the Dockerfile contradicts — it bakes them in on purpose. Fixed by un-excluding `models/` and narrowing `app/` to `app/android_final/` + `app/offline_demo.py`. Verified on Docker 29.9.3, linux/x86_64: build succeeds (`model init OK: 9 members`), `docker run --rm echogesture` exits 0 printing `9 interpreters invoked`, and a corpus clip through the mount returns `{"prediction": "SAIL2", "confidence": 0.6893, "latency_ms": 621.54}` — the same gloss and confidence `predict()` gives on the host. Build-time model rejection confirmed: `_init_model()` raises `FileNotFoundError` when the ensemble is absent, so a missing model fails `docker build`, not `docker run`. |
| 7c | The APK's source branch is identified honestly | **DONE 2026-11-22 — was a false claim** | The README said the shipping APK "rebuilds from this tree". It does not. `codex/android-auto-6.2` @ `33e82b5` is **not merged** into `submission/final` — 29 commits behind and 125 files apart under `app/android_final/` — and no `applicationIdSuffix = ".auto"` exists anywhere on `submission/final`, so this tree **cannot** produce `org.echogesture.finalapp.auto` at all. The two are genuinely different apps: AutoSegmenter vs AutoFinishDetector, SequentialClipDecoder vs SequentialVideoDecoder, one-tap vs record-then-finish. The README now names the branch and the checkout command, and carries a "Two Android apps, and which one ships" table. **The branch is pushed to `origin`, so the artifact stays reproducible — it is the prose that was wrong.** |
| 8 | Android instrumentation | **DONE** | 16 pass, 0 fail, 1 skip (emulator, API 34) |
| 9 | Python suite | **DONE — not green** | 433 / 502 pass, 3 fail, 23 error, 43 skip. All 26 traced to gitignored or retired artifacts; classified individually in `TEST_TRIAGE.md`. Not suppressed. |
| 10 | `predict()` from the shipped tree | **DONE — re-verified 2026-11-22** | Run from `src/main.py` on a real ASL Citizen clip in the current tree: returned exactly the five required keys (`clip_id`, `prediction`, `top5`, `confidence`, `latency_ms`), `clip_id` echoed from the filename, and recognised `SAIL2` on a SAIL clip at confidence **0.689**. This is the native-geometry path the Android app uses, not the 256-squash path the old ZIP ran. |
| 10b | `predictions.jsonl` reconciles with the accuracy claim | **DONE — verified 2026-11-22** | 1,251 records, schema `clip_id / prediction / top5 / confidence / latency_ms / latency_scope`, and every `clip_id` echoes its filename stem. Re-scored against the **authoritative** `models/landmark_labels.json` labels carried in each cached clip's `gloss` field (not parsed from filenames, which is wrong — 79.6% that way): **1,069 / 1,251 = 85.45%**, reproducing `results.csv`, the deck and the video exactly. The 4 clips that are skipped are the cache's own no-detection clips, the same four `eval_tflite_cache.py` skips. |
| 10c | Official Track-2 items present | **DONE** | All four resolve from the repo root: `results/predictions.jsonl` (262,166 B), `results/results.csv` (17,543 B, six scored rows plus supporting rows), `src/main.py` (19,325 B, the `predict()` implementation), `README.md` (21,748 B). |
| 11 | Geometry contract | **DONE** | 9 / 9. Default and explicit 256 compare byte-identical on tensors, not just shape. |
| 12 | Visual slide inspection | **DONE** | All 9 slides rasterised and looked at. No clipped text, collisions, or missing glyphs. Slide 5's six-row results table verified legible row by row. |

The 150 / 150 figure is **JVM only**. It does not cover the app on a phone.

---

## C. Model and claims

| # | Item | State | Detail |
| --: | --- | --- | --- |
| 13 | Geometry contract | **DONE** | `predict()` runs **native 640×480**, matching the Android app — `clip_to_landmarks(..., size=None)` at `src/main.py:336`. The 256-squash default and its `ECHOGESTURE_FRAME_SIZE` override were removed in `8740f27`; all docs now say native. A rewritten contract test (`evaluation/device/test_geometry_contract.py`) asserts the live call site passes `size=None` and that the deleted env var has not crept back. **The previous version of this row said the opposite and its test called a deleted function — 5 tests were failing with AttributeError.** |
| 14 | Ensemble identity | **DONE** | Nine members, `995cc8d0…b4bb`, FP16, 8.51 MB. With detector 16.33 MB — under the 20 MB budget. |
| 15 | Robustness ledger corrected | **DONE** | The 4.40 pp landmark PASS is **withdrawn** (wrong aggregator). Shipped aggregator on the fixed native cache gives **5.76 pp — FAIL**; at the app's margin gate, **2.39 pp** at 86%/80% coverage. Native *pixel* robustness is **unmeasured**; the 4.48 pp pixel figure is 512-letterbox geometry, not the app's. |
| 15b | HUD-clarity evidence (half of the 25% criterion) | **DONE (screenshots added 2026-11-22)** | `docs/HUD_AND_OCCLUSION_EVIDENCE_2026-11-22.md` — every refusal string and the `ResultPolicy` gate, with the on-device observation that a no-hands capture is declined **and excluded from History**. States the *declined* vs *uncertain* distinction the rubric is likely to check, and lists what is not claimed. **Plus five unedited screenshots in `docs/evidence/`** taken on the shipping release build: the offline badge and ensemble SHA, the glossary, the no-hands refusal, the too-long guidance, and a correct `MECHANIC1` reading at 89.3% with its alternatives and the p95 line. `docs/evidence/README.md` maps each shot to the claim it supports and to what it does not show. |
| 15c | HUD now reports a per-frame p95 | **DONE (new 2026-11-22)** | `result_profiler_p95` shows detect/convert p95 on the result card, only when the profile has ≥20 frames. Previously the card showed **only** `analysisP50` — a median of convert+detect — which can never stand in for the scored p95, and the release build has no other readable channel (`allowBackup="false"`, non-debuggable, no `run-as`). This makes `latency_p95` measurable on the artifact that ships. Display-only; compiles clean; unit tests re-run. |
| 16 | Latency claim retracted | **DONE** | The 15.7 ms/frame figure had no artifact behind it. On the shipping **Auto 6.2** build, GPU delegate, **live camera with a hand in frame**, 19 sessions gave detect p95 **72–100 ms (median 85)**, over the 50 ms target in 19 of 19 — `docs/evidence/LATENCY_AUTO62_LIVE_CAMERA_2026-11-22.md`. **FAIL**, analysed: the detector is the bottleneck and the classifier is 54–211 ms once per clip. (The DEBUG-build 46–54 empty-frame / 68–70 with-a-hand numbers are historical and are not the shipping configuration.) |
| 17 | Conflicting documents reconciled | **DONE** | `V4_5_CANONICAL_SUMMARY.md` is now the single source. Old evidence docs carry a SUPERSEDED banner. |

---

## D. Privacy and legal

| # | Item | State | Detail |
| --: | --- | --- | --- |
| 18 | Participant IDs, working tree | **DONE** | 0 raw IDs in the five V4.5 docs and the three data files. Compliance suite 49/49. Numbers unchanged, only labels. |
| 19 | Participant IDs, Git history | **HUMAN** | Eleven pre-existing files, back to the initial commit `c47c5c9`. Needs a coordinated rewrite plus a Dev re-clone. **Not performed.** |
| 20 | Model redistribution rights | **HUMAN** | No confirmed permission. The APK and ZIP contain the weights. |
| 21 | INT8 wording | **HUMAN** | FP16 ships. Organizer clarification drafted, unsent. **Never claim INT8 compliance.** |

---

## E. Submission mechanics

| # | Item | State | Detail |
| --: | --- | --- | --- |
| 22 | Registration | **HUMAN — unverified** | No confirmation record exists in the repo. Do not assume it went through. |
| 23 | Team string | **DONE 2026-11-22** | `VIT_Chennai_Sky_Grazers`, supplied by the team lead. Unblocks #5. |
| 24 | Demo video | **DONE — hosted and linked** | See row 6b. Recorded by the team, uploaded to Google Drive, verified publicly downloadable without sign-in, and linked from slide 7 as a clickable URL. **1:20 against a 2–3 min ask** — see row 6b's stated gaps. |
| 25 | Phone test | **DONE on one physical device, 2026-11-21, 2026-11-22** | Vivo V2404 (Android 16, SM7550). **Auto 6.2, the shipping build, driven 2026-11-22 11:53-12:09 IST:** live-camera recognition verified end to end (VOICE 62.4%, COMB2 75.1%, DEAF1 66.7%, VOICE 94.9%), the one-tap Start -> auto-segment -> result flow worked without a Finish tap, the **Speak** button is present, **Export diagnostics...** produced a 23-record file, History persisted across a completed recognition (*'Saved on this device'*, VOICE 85% 11:52 AM), the refusal state declined correctly with *'No hands in view - No sign started'*, and Settings showed the ensemble SHA-256 995cc8d0...b4bb matching `results.csv`. A 19-session live-camera latency export was taken in the same run and is committed as `docs/evidence/auto6.2_diagnostics_live_camera_2026-11-22.txt`. The same clip reproduces MECHANIC1 at 89.3% on the superseded release build; the debug instrumentation changed nothing about the prediction. **No phone accuracy was measured** - one device, one signer. |
| 26 | Google Form | **HUMAN — blocked on authorisation** | I will not submit it. Requires explicit approval, and I will not report "submitted" without an actual receipt. |

---

## The four things that could still cost this a disqualification

1. **Registration unverified.** If it did not go through, no other work matters.
2. **Rights.** The APK and ZIP contain model weights with no confirmed redistribution permission.
3. **Deck filename.** A submission named with a placeholder team string is a paperwork failure.
4. **Phone accuracy.** There is none. Every accuracy number in this package is a laptop, on a corpus, in a test split. The one phone run confirms the pipeline works and that the clip path is correct; it is **not** a measured phone accuracy. Any slide or sentence implying phone performance is wrong.

---

## What is genuinely ready

The engineering is finished and the claims are defensible. Nine models, a mask
that was missing from every prior export, byte-identical Python/Android parity,
a documented offline posture with INTERNET removed from the manifest, and a
results table that states its own failures in the same font as its successes.

What remains is paperwork, a phone, and three decisions that are the owner's.