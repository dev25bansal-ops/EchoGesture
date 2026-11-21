# Canonical release summary — Track 02

**This is the one file to read for current numbers.** Every other results
document is either historical or a detail sheet behind this one.

Date 2026-11-21 (latest revision). Deadline **2026-11-22 23:59 IST** — per the organizer's V3-2 deck, slide 24 (the 18 Nov date the repo carried is superseded).

---

## The single most important table

These are **different experiments**. They are never combined, averaged, or
ranked against each other.

| # | System / population | Top-1 | Status |
| --: | --- | --: | --- |
| 1 | Single classifier, 1,249 test clips | **62.61%** | **HISTORICAL** — superseded baseline, not shipped |
| 2 | Nine-member ensemble, cached landmarks, 1,249 test | **77.02%** | **HISTORICAL** — old-cache provenance; presence mask wrong on 29% of test clips |
| 3 | Nine-member ensemble, fresh extraction, 256 squash, 1,249 test | **82.47%** | **Dev-reported**, not re-run here |
| 4 | Nine-member ensemble, fresh extraction, native 640×480, **1,251** test | **85.45%** | **Re-measured here** — the shipped ensemble; 1,069/1,251 |
| 5a | Nine-member ensemble, **346-clip validation**, 256 squash | **82.37%** | **Measured here**, historical squash geometry |
| 5b | Nine-member ensemble, **346-clip validation**, native 640×480 | **85.84%** | **Measured here**, opt-in native geometry |
| 6 | Submission ZIP (`7691190b…`), clean extraction, 30 clips | **76.67%** | **HISTORICAL** — that ZIP predates `8740f27` and runs 256 squash; it is not in the tree and is not what `predict()` does now |
| 7 | Physical Android phone | — | **Run on one device 2026-11-21**, pipeline verified; **no accuracy measured** |

### Which number belongs to the submission

The deck quotes **85.45% on 1,251 test clips at native 640×480** (row 4). That
is the geometry the Android app feeds the model, and it is now measured here on
the full test split rather than taken on report. It is a **corpus** figure under
MediaPipe CPU in Python — not phone accuracy, and not a GPU-delegate result.

The **Python entry point** runs the same **native** geometry as the app —
`predict()` calls `clip_to_landmarks(..., size=None)` at `src/main.py:336`, with
no squash default and no environment override (both were removed in `8740f27`).
So Python and Android agree on geometry, which is why 85.45% describes both.

The 82.47% in row 3 is the historical 256-squash measurement, kept as history.
Nothing in the shipped tree runs that path any more; it is not a description of
`predict()`. (An earlier revision of this file said the Python entry point
"keeps 256 squash as its default" — that was true before `8740f27` and is false
since.)

Row 2's 77.02% is retained only as history. It was produced by a cache whose
presence mask was wrong on 29% of clips; the evaluation changed, not the model.
Any document still quoting 77.02% as current is stale.

Row 5b must always be stated as *"346-clip validation split, native geometry,
opt-in"* — never as a bare percentage, and never as phone accuracy.

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

## Latency — retracted, then re-measured on device

| Claim | Status |
| --- | --- |
| 15.7 ms/frame (MediaPipe) | **RETRACTED** — no supporting artifact ever existed |
| Desktop x86 per-frame, 256 squash | ~142–151 ms p50 — above the 50 ms budget |
| On-device **CPU** delegate, 640×480 VIDEO | **95 ms p50 / 166 ms p95** |
| On-device **GPU** delegate, no hand in frame | **46–54 ms p95**, 19–25 analysed fps |
| On-device **GPU** delegate, hand in frame | **68–70 ms p95**, 16–18 analysed fps |

**The ≤50 ms criterion is PARTIAL.** On the GPU delegate the detector sits at or
just inside 50 ms when nothing is in frame, and 18–20 ms over it once a hand is
present. Both are DEBUGGABLE builds, so they are upper bounds. The gap matters
because single-member TFLite classification adds ~40 ms warm on top.

Sources: `docs/ON_DEVICE_LATENCY_2026-11-19.md` (CPU),
`docs/GPU_DELEGATE_2026-11-19.md` (GPU, no hand),
`docs/CAMERA_MISREAD_DIAGNOSIS_2026-11-21.md` (GPU, with hand).

---

## Tests

| Suite | Result |
| --- | --- |
| Android JVM, `submission/final` @ `29c48b6` | **150 / 150 pass** |
| Android instrumentation | **16 pass, 0 fail, 1 skip** (emulator, API 34) |
| Python `evaluation/device` | 433 pass, 3 fail, 23 error, 43 skip |
| Clean-extraction `predict()` | 30 / 30 clips, 0 errors |

The 26 Python failures are **identical on the untouched frozen baseline** and
all trace to three gitignored artifacts (see `TEST_TRIAGE.md`).

---

## Artifacts

| Artifact | SHA-256 |
| --- | --- |
| APK `EchoGesture-2026-11-21.apk` | `96f00458f54de715c7a6f489f9ae92425bc73a089043a91ccbe0ae6d4f060035` |
| ZIP `EchoGesture_Submission.zip` | `7691190b8ce045684a6117ab6a2b97b7b79327563147e9699f69cec939646cca` |
| Deck `CollegeName_TeamName.pptx` | `9c7558b155d7ed1bdd3c06baf5504f6d1465d144930f1c3600df8ce7293666de` |
| Deck `CollegeName_TeamName.pdf` | `df032430e653558af9b443e5dc560e6c9355461c1b7f2f94037e7afad1c048b8` |

---

## Claims that must not be made

| Never say | Because |
| --- | --- |
| "77% / 82% / 85% accurate on a phone" | No phone accuracy has been measured; one phone run only proves the pipeline works |
| "85.84% is our submission's accuracy" | That is opt-in native geometry on validation |
| "184 independent signs" | 184 evaluations over 72 unique clips |
| "INT8 quantised" | FP16 ships |
| "Meets the 50 ms target unconditionally" | GPU p95 is 46–54 ms empty, 68–70 ms with a hand in frame, on a debug build |
| "Robustness criteria pass" | Landmark FAILs 5.36; native pixel unmeasured |
| "Android and Python pipelines are identical" | VIDEO-mode vs IMAGE-mode detect |
| "15.7 ms per frame" | Retracted |
| "The live camera is as reliable as a clip" | Same sign: 0.969 from a file, 0.246 live in the same session |

---

## Blocked on humans

Registration unverified · exact `CollegeName_TeamName` · demo video · model
redistribution rights · Google Form approval.

---

## Metadata discrepancy — resolved 2026-11-21

`assets/landmark_model_meta.json` ships inside the APK with
`ensemble_test_top1`. It previously read `0.7702`, the cached-landmark figure,
while the deck quoted 85.45%. **That is no longer true of the shipped APK.**

Verified by extracting the field from `EchoGesture-2026-11-21.apk` itself:

| | value | provenance |
| --- | --: | --- |
| `EchoGesture-2026-11-21.apk` | **0.8545** | native-cache rebuild, matches the deck |
| Historical `echogesture-p1.apk` | 0.7702 | cached-landmark run; presence mask wrong on 29% of test clips |

The field is metadata only — `ModelContract.kt` reads the file for member order
and widths, never for accuracy — so nothing on screen changes. It is included
here because an organizer who unzips the APK reads it, and it should agree with
the deck. It now does.