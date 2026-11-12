# EchoGesture — final submission state

**Competition:** Samsung XR Hackathon 2026, Track 2
**Deadline:** 18 Nov 2026, 23:59 (timezone not stated in the deck; repo assumes IST)
**Internal target:** 17 Nov 2026
**Updated:** 11 Nov 2026

This table is the single source of truth. Each row has a state, an owner and one
next action. No row says "pending".

## States

`DONE_VERIFIED` · `DONE_PUBLISHED` · `READY_FOR_USER` · `WAITING_FOR_DEV` ·
`WAITING_FOR_ORGANIZER` · `WAITING_FOR_HARDWARE` · `WAITING_FOR_PERMISSION` ·
`FAILED_WITH_CAUSE`

---

## Deliverables

| # | Deliverable | State | Evidence / path | Owner | Next action |
| --: | --- | --- | --- | --- | --- |
| 1 | Four-model FP16 ensemble | `DONE_VERIFIED` | `predict()` → `WHATFOR1` @ 0.8821 from a clean extract | — | none |
| 2 | `predict(video_path) -> dict` contract | `DONE_VERIFIED` | `src/main.py`; returns clip_id, uppercase prediction, top5, confidence, latency_ms | — | none |
| 3 | `results/predictions.jsonl` in the p.11 schema | `DONE_VERIFIED` | 1,249 lines, exact five keys, uppercase labels, top5 length 5 | Abhijeet | replace with organizer-manifest IDs when they arrive |
| 4 | `results/results.csv` in the p.11 template | `DONE_VERIFIED` | 6 rows, exact column order, official metric names | Abhijeet | none |
| 5 | Root `Dockerfile` | `DONE_VERIFIED` | `Dockerfile` at repo root; builder now ships it instead of a stub | Abhijeet | optional: `docker build` if a daemon is available |
| 6 | Release bundle | `DONE_VERIFIED` | `evaluation/device/release_final2/echogesture_final2_ensemble.zip`, 9,007,144 B, SHA-256 `34a0e667634548bed0d0339622f9b15618699e0a99672a98c53ff59977a50c08` | — | rename to `<CollegeName>_<TeamName>.zip` once names are known |
| 7 | Test suite | `DONE_VERIFIED` | 572 run / 568 pass / 0 fail / 0 import-blocked / 4 skipped | — | rerun after any source change |
| 8 | Source-only CI | `DONE_PUBLISHED` | GitHub Actions green on `26b2990` and later | — | none |
| 9 | Android APK | `DONE_VERIFIED` | `app/android_final/app/build/outputs/apk/debug/app-debug.apk`, 75,072,782 B, signed, 4 members embedded | Abhijeet | install when a phone is available |
| 10 | Architecture document | `DONE_VERIFIED` | `submission_final/architecture/ARCHITECTURE.md` | — | none |
| 11 | Release manifest | `DONE_VERIFIED` | `submission_final/RELEASE_MANIFEST.json` | — | none |
| 12 | Presentation deck | `READY_FOR_USER` | `docs/submission/EchoGesture_Submission.pptx`, 11 slides | Abhijeet | fill team/college, rename to the required nomenclature |
| 13 | Demo video | `READY_FOR_USER` | recording guide prepared; **no video recorded** | Abhijeet | record ≤3:00, upload, put the link in the form |
| 14 | INT8 clarification email | `READY_FOR_USER` | `docs/submission/INT8_CLARIFICATION.md` | Abhijeet | send to `prism@samsung.com` |
| 15 | Dev sign-off | `WAITING_FOR_DEV` | `docs/release/DEV_SIGNOFF_REQUEST.md`; PR #1 review requested | Dev | answer the two metric discrepancies and confirm the hashes |
| 16 | On-device measurements | `WAITING_FOR_HARDWARE` | none | Abhijeet | with a phone + adb: latency, power, 10-min thermal |
| 17 | Weight redistribution rights | `WAITING_FOR_PERMISSION` | weights gitignored; `NOTICE` records status | Dev | state permission, or confirm there is none |
| 18 | Official vocabulary and test manifest | `WAITING_FOR_ORGANIZER` | not present in any of the deck's 28 pages | Abhijeet | request both in the same email |
| 19 | Final submission | `WAITING_FOR_USER` | — | Abhijeet | submit after 12, 13 content and approvals are in place |

---

## Measured results, frozen

| Metric | Target | Measured | Verdict |
| --- | --- | --- | --- |
| Top-1 accuracy | ≥70% | **72.78%** (909/1249 scored; 72.43% of 1255 eligible) | PASS |
| Top-5 accuracy | — | 87.51% (1093/1249) | — |
| Latency p95, full clip | ≤50 ms | **274.2 ms** singleton / 154.8 ms parallel-2 / **99.8 ms** parallel-4 | **FAIL** |
| Model size | <20 MB | **11.37 MB** (3,544,080 B ensemble + 7,819,105 B detector = 11,363,185 B) | PASS |
| Robustness drop | ≤5 pp | low_light −3.33, motion_blur 0.00, frame_drop 0.00 → PASS; **crop 10.00, occlusion 6.67 → FAIL** | **FAIL** |
| Average power | ≤500 mW | NOT MEASURED | — |
| 10-min thermal drift | ≤20% | NOT MEASURED | — |

All measured on a macOS arm64 laptop with ai-edge-litert 2.3.0 (XNNPACK CPU).
Not target hardware.

---

## Defects found and fixed this sprint

1. **`results/predictions.jsonl` was the research file** — 1,705 lines with
   `ground_truth`/`correct`/`signer`, no `confidence`/`latency_ms`. Violated the
   p.11 schema and the p.12 layout at once. Replaced with the 1,249-line
   `predictions_ensemble.jsonl`, which already conformed; research file moved to
   `results/research/`.
2. **`results/results.csv` used non-template metric names and column order** — an
   automated pass script would have mis-joined. Rewritten to the official six
   rows and exact columns.
3. **No root `Dockerfile`** — `build_bundle.py` hard-failed without one (p.25
   requires Docker files). Added, and the builder now ships the repository's
   Dockerfile instead of overwriting it with a stub.
4. **Builder never staged `predictions.jsonl`** — now stages it when a manifest
   exists to validate against, and refuses to when it does not.

---

## The honest limitations

Stated here so they are never quietly dropped from a submission or a demo:

- The 50 ms target is **not met**, and cannot be met by changing the model —
  inference is 0.36 ms, tracking is 90% of runtime.
- Crop and occlusion exceed the robustness budget.
- The 1,249-clip test split was inspected during model selection. It is a
  held-out split, not a pristine blind evaluation.
- No physical Android device was available. On-device latency, power and thermal
  are unmeasured, not estimated.
- The model performs isolated 100-gloss recognition, not continuous
  full-sentence translation.
- The shipped artifact is FP16, not INT8.
- Weights have no confirmed redistribution permission.

---

## What is genuinely left

Four things, and only the first is on the critical path:

1. **Team and college strings.** They drive the deck filename, the ZIP filename
   and the submission form. Nothing else in the package needs inventing.
2. **The demo video.** Recording guide is ready; the recording is not.
3. **Two emails.** INT8/FP16 clarification and the vocabulary/manifest request —
   both drafted, one send.
4. **Dev's answers** on the 83.77% and 920-vs-909 discrepancies. If they cannot
   be explained, the numbers are simply not cited, and the package is still
   submittable.