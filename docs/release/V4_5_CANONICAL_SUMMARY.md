# Canonical release summary — Track 02

> **SUPERSEDED 2026-11-21.** This file is dated 2026-11-16 and carries the 18 Nov
> deadline, the pre-cache-fix 85.51% row, and the 8-slide deck that has since been
> replaced. The current canonical summary is
> **`submission/FINAL_DELIVERY/RESULTS.md`** (2026-11-21: deadline 22 Nov, 85.45%
> on 1,251 clips at native geometry, 9-slide template deck). Read that one. This
> file is kept only for its historical claim ledger.

**This is the one file to read for current numbers.** Every other results
document is either historical or a detail sheet behind this one.

Date 2026-11-16. Deadline 2026-11-18 23:59 IST.

---

## The single most important table

These are **different experiments**. They are never combined, averaged, or
ranked against each other.

| # | System / population | Top-1 | Status |
| --: | --- | --: | --- |
| 1 | Single classifier, 1,249 test clips | **62.61%** | **HISTORICAL** — superseded baseline, not shipped |
| 2 | Nine-member ensemble, cached landmarks, 1,249 test | **77.02%** | **HISTORICAL** — old-cache provenance; cache presence is wrong on 29% of test clips |
| 3 | Nine-member ensemble, fresh extraction, 256 squash, 1,249 test | **82.47%** | **Dev-reported**, not re-run here |
| 4 | Nine-member ensemble, fresh extraction, native 640×480, 1,249 test | **85.51%** | **Dev-reported**, not re-run here |
| 5a | Nine-member ensemble, **346-clip validation**, 256 squash | **82.37%** | **Measured here**, default geometry |
| 5b | Nine-member ensemble, **346-clip validation**, native 640×480 | **85.84%** | **Measured here**, opt-in geometry |

> **Two numbers exist for the 346-clip validation split and they are not in
> conflict.** `results.csv` reports **89.31%** (`top1_accuracy_val`), from
> `results/eval_summary_val.json` — the nine members scored against
> **pre-extracted native landmarks** in `.scratch/landmarks_native`, the same
> cache the test split uses. Row 5b's **85.84%** is the same nine members on the
> same 346 clips taken through the **full `evaluation/device/geometry_paired.py`
> path**, which patches `read_frames` and runs real `predict()` — a different
> decode and normalisation route. Landmark-cache evaluation is higher than
> video-path evaluation because MediaPipe runs on already-decoded frames rather
> than on the decoded stream; the gap is 12 clips.
>
> Which to quote: **89.31% if describing the corpus landmark evaluation** (that
> is what `results.csv` and the deck use), **85.84% if describing the end-to-end
> video path**. Never put the two side by side as if one were wrong. The test
> split has the same split — 85.45% on the native cache (`results.csv`) versus
> 85.51% reported on fresh extraction.
| 6 | Submission ZIP, clean extraction, 30 clips | **76.67%** | **Measured here** — the packaged artifact |
| 7 | Physical Android phone | — | **NOT TESTED** |

### Which number belongs to the submission

**The submission ZIP runs at 256 squash** (`ECHOGESTURE_FRAME_SIZE` unset). Its
measured result is row 6: **76.67% on 30 clean-extraction clips**. The corpus
figure behind it is row 2's 77.02%.

> **Stale as of `8740f27`.** Both the squash default and `ECHOGESTURE_FRAME_SIZE`
> were deleted; `predict()` now runs native geometry unconditionally. Everything
> in this paragraph describes the older ZIP, not the shipped code.

Rows 3, 4 and 5b are **not** the submission's number. Row 5b is the closest
thing to a headline for the nine-model release, and it must always be stated as
*"346-clip validation split, native geometry, opt-in"* — never as a bare
percentage, and never as phone accuracy.

---

## Model identity

| | |
| --- | --- |
| Members | 9 — four 255-wide (x,y), five 381-wide (x,y,z) |
| Precision | **float16** — not INT8 |
| Ordered ensemble SHA-256 | `995cc8d0b6f6ac25abb9f7ddca01c3ce71edf8bc2d34e34f4813a2b99dd3b4bb` |
| Detector SHA-256 | `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1` |
| Ensemble size | 8,508,832 B (8.51 MB) |
| With detector | 16,327,937 B (16.33 MB) — under the 20 MB budget either way |
| Vocabulary | 100 fixed classes |

---

## Robustness — corrected, and one PASS withdrawn

| Experiment | Worst drop | Verdict |
| --- | --: | --- |
| Landmark, 256 squash, 1,249 test, **shipped aggregator** | **5.36 pp** | **FAIL** |
| Landmark, native, 1,249 test | **5.76 pp** | **FAIL** |
| Pixel, 512 letterbox, 1,249 test (Dev) | 4.48 pp | PASS — **512 letterbox only** |
| Pixel, 256 squash, 4-model, 30 clips (ours) | 10.00 pp | FAIL — older model |
| Pixel, **native 640×480**, nine models | — | **NOT MEASURED** |

**The previously published 4.40 pp landmark PASS is withdrawn.** It was computed
by averaging per-member softmax probabilities; the shipped code averages raw
logits then applies one softmax. Under the real aggregator the criterion fails.

**Dev's 4.48 PASS does not apply to the Android app.** It is 512-letterbox
geometry; the app is 640×480 native. Native-camera robustness is unmeasured.

---

## Latency — one claim retracted

| Claim | Status |
| --- | --- |
| 15.7 ms/frame (MediaPipe) | **RETRACTED** — no supporting artifact ever existed |
| Re-measured per-frame median | **~142–151 ms** — already above the 50 ms budget |
| Clip end-to-end, warm p50 | 236 ms (256 squash) / 242 ms (native) |
| Clip p95 warm | 314.9 ms |

**The ≤50 ms criterion is FAIL** on every configuration measured. No phone
measurement exists.

---

## Tests

| Suite | Result |
| --- | --- |
| Android JVM | **110 / 110 pass** |
| Android instrumentation | **16 pass, 0 fail, 1 skip** |
| Python `evaluation/device` | 433 pass, 3 fail, 23 error, 43 skip |
| Clean-extraction `predict()` | 30 / 30 clips, 0 errors |

The 26 Python failures are **identical on the untouched frozen baseline** and
all trace to three gitignored artifacts (see `V4_5_TEST_TRIAGE.md`).

---

## Artifacts

| Artifact | SHA-256 |
| --- | --- |
| APK `echogesture-p1.apk` | `150ffeeb0eaaa8914db362e2bcd41de26c93608193ebcee8457433dc5ea5cb7a` |
| ZIP `EchoGesture_Submission.zip` | `7691190b8ce045684a6117ab6a2b97b7b79327563147e9699f69cec939646cca` |
| Deck `EchoGesture_Track02.pptx` | `951d5177c435c02c4a0cce763c82945e6b4a5a7dfe3c5aad05aa435112350e66` |

---

## Claims that must not be made

| Never say | Because |
| --- | --- |
| "77% / 82% / 85% accurate on a phone" | No phone measurement exists |
| "85.84% is our submission's accuracy" | That is opt-in native geometry on validation |
| "184 independent signs" | 184 evaluations over 72 unique clips |
| "INT8 quantised" | FP16 ships |
| "Meets the 50 ms target" | 314.9 ms warm p95 |
| "Robustness criteria pass" | Landmark FAILs 5.36; native pixel unmeasured |
| "Android and Python pipelines are identical" | VIDEO-mode vs IMAGE-mode detect |
| "15.7 ms per frame" | Retracted |

---

## Blocked on humans

Registration unverified · exact `CollegeName_TeamName` · demo video · model
redistribution rights · Google Form approval.

---

## Known metadata discrepancy — recorded, not resolved

`assets/landmark_model_meta.json` ships inside the APK and the ZIP with
`ensemble_test_top1: 0.7702`. Dev's branch tip
(`hackathon/dev-model` @ `73bff0d`) has changed that same field to `0.8545`,
measured on the rebuilt native-geometry cache.

| | value | provenance |
| --- | --: | --- |
| Our shipped artifacts | **0.7702** | cached-landmark run; presence mask wrong on 29% of test clips |
| Dev's current branch | **0.8545** | native-cache rebuild, re-verified at `fd4a13f` |

**The field is never read or displayed by the app** — it is metadata only;
`ModelContract.kt` uses the file for member order and widths, not for accuracy.
So this is not a live display defect.

It is a *claim* discrepancy: an organizer who unzips the APK would read 77.02%
and could compare it against Dev's repository and find 85.45%. Both numbers are
internally consistent — ours is the cached-landmark population, Dev's is the
native-cache population — but a reader without this table cannot tell.

**Not fixed here, deliberately.** Adopting 85.45% would mean shipping a number
whose run I have not reproduced, and rebuilding the APK two days before the
deadline to change a field nobody reads. The figure quoted in
`RESULTS.md`, the deck and `RELEASE_MANIFEST.json` is 77.02%, which matches the
shipped artifact and is labelled with its population everywhere it appears.

**Action if this matters to the owner:** rebuild the APK from Dev's
`models/landmark_model_meta.json`, re-run the 30-clip clean-extraction gate, and
update the three claim sites together. That is a few hours plus a re-verification,
not a one-line edit.