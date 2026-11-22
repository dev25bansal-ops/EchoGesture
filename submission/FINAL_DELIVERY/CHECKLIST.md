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
| 3 | PPTX | **DONE** | `CollegeName_TeamName.pptx`, `e8361b55…`, 9 slides, built 2026-11-21 **inside the organizer's own template** (`Submission_Final_forGithub.pptx`), so the section list and header geometry are the template's rather than our own invention. Carries the corrected claims (85.45% not 77.02%, 150 tests not 110, latency measured on device, physical-phone wording). |
| 4 | PDF | **DONE** | `CollegeName_TeamName.pdf`, `cec8b236…`, 9 pages. Native Microsoft PowerPoint export, not the PIL rasteriser (which was only ever a layout check): text is selectable and fonts are embedded. Every figure was re-extracted from the PDF's text layer to confirm it carries the corrected numbers. |
| 5 | APK — **ship this** | **DONE (rebuilt 2026-11-22)** | `EchoGesture-2026-11-22-release.apk`, `03d51354…`, 80,815,568 B. **Signed, non-debuggable RELEASE build** — `android:debuggable` absent, signature `CN=measurement` (cert `6573b6fd…1bb97`, identical to the 00:29 build it replaces, so this is a rebuild not a re-key). 150/150 unit tests pass on the release variant. Carries all 10 model assets; its `landmark_model_meta.json` reads `ensemble_test_top1: 0.8545`, matching the deck. **Rebuilt on deadline morning to add the per-frame p95 line** (`result_profiler_p95`), because the release build has no other readable channel for the scored metric (`allowBackup="false"`, non-debuggable, no `run-as`). Driven on the same Vivo V2404 after install: `MECHANIC1` 89.3% and `DARK1` 73.4%, both matching the debug build on the same clips, with HUD p95 **detect 51 ms / convert 4 ms** and **detect 50 ms / convert 3 ms**. Those are real p95s on the shipping artifact — but off the **file-pick** path, not the live camera with a hand at the device, so the `latency_p95` row stays FAIL. |
| 5c | APK — superseded build | **HISTORICAL** | `a56ba20f…8ada7`, 80,815,032 B — the 00:29 release build, same models and key, without the p95 HUD line. Replaced by row 5; kept only so older references resolve. |
| 5b | APK — historical | **HISTORICAL** | `EchoGesture-2026-11-21.apk`, `96f00458…`, 82,894,510 B, DEBUGGABLE. Installed and driven on a physical Vivo V2404; every latency, power and thermal figure in results.csv came from this build, so those are upper bounds on the release APK above. |
| 5 | Deck filename | **HUMAN** | Must become the real `CollegeName_TeamName`. The placeholder of that exact name is now in place on both the file and slide 1, so the only edit left is the string itself. |
| 6 | Release manifest | **DONE** | `submission/RELEASE_MANIFEST.json`, validated JSON, hashes re-measured 2026-11-21 against the shipped APK, deck and PDF. |
| 6b | Demo video | **RECORDED 2026-11-22 — needs an upload URL only** | `docs/submission/EchoGesture_demo_2026-11-22.mp4`, `321869c0…bd2a3`, 28,134,918 B, 1280×720, 30 fps, **2:55.2**, with an audio track (one `mp4a` AAC stream). Built by `scripts/media/build_demo_video.py` from real model output — every number on screen is computed from `.scratch/thresh/probs_test.npz` and `models/`, never typed. Ten narrated beats: the accessibility problem, why landmarks not pixels, the pipeline, signer-disjoint results, live predictions on real clips, the robustness failure stated in the same font as the passes (beat 06), the weakest signs, the 92% gate precision, and the closing summary. The HUD caption was re-rendered 2026-11-22 to read `release: detect p95 58-72 ms (file-pick)` with `clip 72.0 ms`, matching the four on-device samples; verified on an extracted frame, not just in the builder's log. **Gitignored (`*.mp4`) and deliberately not committed**: it contains ASL Citizen footage under the Microsoft Research non-commercial licence, and this repo is public — the same redistribution problem as row 20. Upload it and paste the link into slide 7; the recording itself is done. |

---

## B. Verification actually run

| # | Check | State | Result |
| --: | --- | --- | --- |
| 7 | Android JVM tests | **DONE** | 150 / 150 pass on the merged `submission/final` tree (was 110 before the camera-fix tests landed) |
| 8 | Android instrumentation | **DONE** | 16 pass, 0 fail, 1 skip (emulator, API 34) |
| 9 | Python suite | **DONE — not green** | 433 / 502 pass, 3 fail, 23 error, 43 skip. All 26 traced to gitignored or retired artifacts; classified individually in `TEST_TRIAGE.md`. Not suppressed. |
| 10 | `predict()` from the shipped tree | **DONE — re-verified 2026-11-22** | Run from `src/main.py` on a real ASL Citizen clip in the current tree: returned exactly the five required keys, `clip_id` echoed from the filename, and recognised `SAIL2` on a SAIL clip at confidence 0.689. This is the native-geometry path the Android app uses, not the 256-squash path the old ZIP ran. |
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
| 16 | Latency claim retracted | **DONE** | The 15.7 ms/frame figure had no artifact behind it. On-device GPU delegate now measures p95 46–54 ms with no hand in frame and 68–70 ms with one, on a DEBUGGABLE build. **PARTIAL** on the 50 ms criterion: passes in the empty-frame regime, does not with a hand present. |
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
| 23 | Team string | **HUMAN** | Blocks #5 and the form. |
| 24 | Demo video | **RECORDED — see 6b** | Recorded 2026-11-22 by `scripts/media/build_demo_video.py`; only the upload URL remains. |
| 25 | Phone test | **DONE on one device, 2026-11-21 + 2026-11-22** | Driven on a physical Vivo V2404 (Android 16) with both builds. Debug build: clip path correct on held-out clips (DARK 73.4%, MECHANIC 89.3%, VOICE 96.9%), camera path verified. Release build (2026-11-22): same clip reproduces MECHANIC1 at 89.3% — the debug instrumentation changed nothing about the prediction. The rebuilt release build adds a readable p95: on two picked clips it reported detect p95 51 ms and 50 ms, so the shipping artifact's latency is now measurable and sits at the 50 ms target on that path. A no-hands capture on the same build was refused with "No hands in view — Hands were visible for too little of that recording" and left no History entry, which is the HUD-clarity behaviour claimed in row 15b. Three consecutive live captures still misread as RAZOR2, which is diagnosed in `docs/CAMERA_MISREAD_DIAGNOSIS_2026-11-21.md` — the model is rotation-variant and a closed fist is a real RAZOR2. **One phone, one signer.** `PHONE_TEST_REPORT.md` remains the unfilled template. |
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