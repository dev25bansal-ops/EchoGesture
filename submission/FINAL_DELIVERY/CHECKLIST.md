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
| 2 | Submission ZIP | **DONE** | `EchoGesture_Submission.zip`, `7691190b…6cca`, 13,379,265 B. Clean-room extracted; `predict()` ran from the extracted copy only; 30/30 clips, 0 errors. |
| 3 | PPTX | **DONE** | `CollegeName_TeamName.pptx`, `9c7558b1…`, 9 slides, built 2026-11-21 **inside the organizer's own template** (`Submission_Final_forGithub.pptx`), so the section list and header geometry are the template's rather than our own invention. Carries the corrected claims (85.45% not 77.02%, 150 tests not 110, latency measured on device, physical-phone wording). |
| 4 | PDF | **DONE** | `CollegeName_TeamName.pdf`, `df032430…`, 9 pages. Native Microsoft PowerPoint export, not the PIL rasteriser (which was only ever a layout check): text is selectable and fonts are embedded. Every figure was re-extracted from the PDF's text layer to confirm it carries the corrected numbers. |
| 5 | APK | **DONE** | `EchoGesture-2026-11-21.apk`, `96f00458…`, 82,894,510 B, built from `submission/final` at `29c48b6`. Installed and driven on a physical Vivo V2404. Its own `landmark_model_meta.json` reads `ensemble_test_top1: 0.8545`, matching the deck. |
| 5 | Deck filename | **HUMAN** | Must become the real `CollegeName_TeamName`. The placeholder of that exact name is now in place on both the file and slide 1, so the only edit left is the string itself. |
| 6 | Release manifest | **DONE** | `submission/RELEASE_MANIFEST.json`, validated JSON, hashes re-measured 2026-11-21 against the shipped APK, deck and PDF. |
| 6b | Demo video link | **HUMAN** | Slide 7 reads `DEMO VIDEO: <paste the 2-3 min YouTube or Drive link here>` in red. Not recorded. Script is in `submission/DEMO_SCRIPT.md`. |

---

## B. Verification actually run

| # | Check | State | Result |
| --: | --- | --- | --- |
| 7 | Android JVM tests | **DONE** | 150 / 150 pass on the merged `submission/final` tree (was 110 before the camera-fix tests landed) |
| 8 | Android instrumentation | **DONE** | 16 pass, 0 fail, 1 skip (emulator, API 34) |
| 9 | Python suite | **DONE — not green** | 433 / 502 pass, 3 fail, 23 error, 43 skip. All 26 traced to gitignored or retired artifacts; classified individually in `TEST_TRIAGE.md`. Not suppressed. |
| 10 | Clean-extraction `predict()` | **DONE** | 30 / 30 clips, 0 errors, 76.67% top-1 |
| 11 | Geometry contract | **DONE** | 9 / 9. Default and explicit 256 compare byte-identical on tensors, not just shape. |
| 12 | Visual slide inspection | **DONE** | All 9 slides rasterised and looked at. No clipped text, collisions, or missing glyphs. Slide 5's six-row results table verified legible row by row. |

The 150 / 150 figure is **JVM only**. It does not cover the app on a phone.

---

## C. Model and claims

| # | Item | State | Detail |
| --: | --- | --- | --- |
| 13 | Geometry contract | **DONE** | `predict()` runs **native 640×480**, matching the Android app — `clip_to_landmarks(..., size=None)` at `src/main.py:336`. The 256-squash default and its `ECHOGESTURE_FRAME_SIZE` override were removed in `8740f27`; all docs now say native. A rewritten contract test (`evaluation/device/test_geometry_contract.py`) asserts the live call site passes `size=None` and that the deleted env var has not crept back. **The previous version of this row said the opposite and its test called a deleted function — 5 tests were failing with AttributeError.** |
| 14 | Ensemble identity | **DONE** | Nine members, `995cc8d0…b4bb`, FP16, 8.51 MB. With detector 16.33 MB — under the 20 MB budget. |
| 15 | Robustness ledger corrected | **DONE** | The 4.40 pp landmark PASS is **withdrawn**. Shipped aggregator gives 5.36 pp — FAIL. Native pixel robustness is **unmeasured**. |
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
| 24 | Demo video | **HUMAN** | Not recorded. Script is in `submission/DEMO_SCRIPT.md`. |
| 25 | Phone test | **DONE on one device, 2026-11-21** | Driven on a physical Vivo V2404 (Android 16) with the shipped build: clip path correct on held-out clips (DARK 73.4%, MECHANIC 89.3%, VOICE 96.9%), camera path verified. Three consecutive live captures still misread as RAZOR2, which is diagnosed in `docs/CAMERA_MISREAD_DIAGNOSIS_2026-11-21.md` — the model is rotation-variant and a closed fist is a real RAZOR2. **One phone, one signer.** `PHONE_TEST_REPORT.md` remains the unfilled template. |
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