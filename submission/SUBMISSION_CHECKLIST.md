# Submission checklist — Track 02

Status as of **2026-11-14**. Internal target 17 Nov. Deadline 18 Nov 23:59 IST.

Statuses: **DONE** · **FAILED** · **BLOCKED** · **NOT TESTED** · **NEEDS HUMAN**

---

## Gate 0 — Competition entry

| Item | Owner | Priority | Status | Evidence | Next action |
| --- | --- | --- | --- | --- | --- |
| Registration completed | Abhijeet | **P0** | **NEEDS HUMAN — UNVERIFIED** | No confirmation record in the repo. `docs/submission/organizer/` says "Nothing received." | Produce the Google Form confirmation email. Without it the final submission link is never sent. |
| Exact `CollegeName_TeamName` | Abhijeet | **P0** | **NEEDS HUMAN** | The string is not in the repo by design | Copy it verbatim from the registration entry |
| Both members from one of the five eligible colleges | Abhijeet | **P0** | **NEEDS HUMAN** | College field is a closed list of 5 institutions | Confirm against the live form |
| Final submission form link | Abhijeet | **P0** | **BLOCKED** | Only sent to registered teams | Arrives after registration is confirmed |

**If registration did not happen, nothing else can be submitted.** This is the
first thing to check, before any further engineering.

---

## Gate A — Application

| Item | Owner | Status | Evidence |
| --- | --- | --- | --- |
| APK builds | Abhijeet | **DONE** | `assembleDebug` BUILD SUCCESSFUL |
| APK installs | Abhijeet | **DONE (emulator)** | `adb install -r` → `Success` |
| App opens, no crash/ANR | Abhijeet | **DONE (emulator)** | PID alive, empty crash buffer |
| Nine members embedded | Abhijeet | **DONE** | 9 × `.tflite` in APK assets |
| Footer shows correct model | Abhijeet | **DONE** | Screenshot: `9-model ensemble · … SHA-256 995cc8d0…` |
| Camera permission only | Abhijeet | **DONE** | `aapt2 dump permissions` → CAMERA only |
| No INTERNET permission | Abhijeet | **DONE** | Removed in manifest; absent from APK |
| Record → recognise workflow | Abhijeet | **NOT TESTED** | No physical device |
| Live recognition on a phone | Abhijeet | **NOT TESTED** | — |

---

## Gate B — Evaluator

| Item | Owner | Status | Evidence |
| --- | --- | --- | --- |
| `predict(video_path)` runs | Abhijeet | **DONE** | Correct result from clean extraction |
| Nine models load | Abhijeet | **DONE** | `use_z` = `[F,F,F,F,T,T,T,T,T]` |
| Both feature widths | Abhijeet | **DONE** | 255 and 381 members both invoked |
| Attention mask fed | Abhijeet | **DONE** | Mask matches on all clips |
| Hash pinning enforced | Abhijeet | **DONE** | Unpinned/substituted members raise |
| No machine-specific paths | Abhijeet | **DONE** | Resolves `../models` relative to `src/` |
| 30/30 clips, 0 errors | Abhijeet | **DONE** | 23 correct, 76.67% |

---

## Gate C — Submission package

| Item | Owner | Status | Evidence |
| --- | --- | --- | --- |
| `src/` layout per PDF p12 | Abhijeet | **DONE** | main, features, frontend, tracker, requirements, README |
| `results/predictions.json` | Abhijeet | **DONE** | 30 entries |
| `results/results.csv` | Abhijeet | **DONE** | One row per metric, pass/fail, notes |
| Conversion script included | Abhijeet | **DONE** | `src/export_ensemble.py` |
| ZIP extracted and run clean | Abhijeet | **DONE** | `main.__file__` under `/tmp/final_room2` |
| ZIP SHA-256 recorded | Abhijeet | **DONE** | `35b902df…` |
| **Official test-manifest predictions** | Abhijeet | **BLOCKED** | No organizer manifest exists. Never fabricate IDs. |

---

## Gate D — Evidence

| Item | Status | Evidence |
| --- | --- | --- |
| No false accuracy claim | **DONE** | 77.02% labelled corpus; phone marked NOT TESTED |
| No false latency claim | **DONE** | 314.9 ms reported as FAIL |
| No false quantisation claim | **DONE** | FP16 stated everywhere |
| No false robustness claim | **DONE** | Both harnesses reported separately |
| Presentation matches shipped model | **DONE** | Rebuilt; 8 slides; stale 4-model deck replaced |
| Deck claims verified against source | **DONE** | 8 numeric claims checked programmatically |

---

## Gate E — Presentation and demo

| Item | Owner | Status | Evidence |
| --- | --- | --- | --- |
| Deck built | Abhijeet | **DONE** | `submission/presentation/EchoGesture_Track02.pptx`, 8 slides |
| Deck content covers PDF p25 list | Abhijeet | **DONE** | theme, title, team, problem, solution, architecture, tools, innovation, results, limits |
| Deck renamed to `CollegeName_TeamName` | Abhijeet | **NEEDS HUMAN** | Depends on the registration string |
| Real screenshots in deck | Abhijeet | **DONE** | 3 captured from the shipped build |
| Demo script | Abhijeet | **DONE** | `submission/DEMO_SCRIPT.md` |
| Demo video recorded | Abhijeet | **BLOCKED** | Needs a phone and a camera |
| Demo link | Abhijeet | **BLOCKED** | Follows the recording |

---

## Gate F — Rights and integrity

| Item | Status | Evidence |
| --- | --- | --- |
| Weights excluded from source branches | **DONE** | `*.tflite` gitignored; verified absent |
| APK/ZIP excluded from source branch | **DONE** | Both gitignored |
| Model redistribution permission | **BLOCKED** | `NOTICE` records none. Owner decision. |
| Weights already public elsewhere | **KNOWN** | Private on other branches; a rewrite is the owner's call, not performed |

---

## Gate G — Collaboration

| Item | Status | Evidence |
| --- | --- | --- |
| Shared branch pushed and verified | **DONE** | `shared/v4.4-dev-integration` @ `facae8f` |
| Build instructions reproducible | **DONE** | `docs/release/BUILD_INSTRUCTIONS_SHARED.md` |
| Dev can reproduce | **NEEDS HUMAN** | Dev has not run it |

---

## The five things that actually block submission

1. **Registration unverified** — no confirmation record exists.
2. **Team/college string** — cannot name the PPT, ZIP or APK without it.
3. **Demo video** — needs a phone and a camera.
4. **Rights** — weights have no confirmed redistribution permission.
5. **Google Form** — needs explicit human authorization and produces the receipt.

Everything a machine can do is done. These five need you.