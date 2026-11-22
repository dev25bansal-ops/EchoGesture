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
| 6 | Submission ZIP (`5d310cec…`), clean extraction, 30 clips | **76.67%** | **HISTORICAL** — that ZIP predates `8740f27` and runs 256 squash; it is not in the tree and is not what `predict()` does now |
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
| On-device **GPU**, DEBUG, no hand in frame (superseded) | 46–54 ms p95 — not the shipping configuration |
| On-device **GPU**, DEBUG, hand in frame (superseded) | 68–70 ms p95 — diagnostics on, not the shipping configuration |
| SIGNED RELEASE, file-pick (*superseded build*) | 58 / 65 / 67 / 72 ms p95 across four classifications |
| **On-device GPU, SIGNED AUTO 6.2, LIVE CAMERA** | **72–100 ms p95, median 85**, across 19 sessions with a hand in frame |

**The ≤50 ms criterion FAILS on every measured sample.** On the shipping Auto 6.2
build, all 19 live-camera sessions were over the target (72–100 ms p95, median
85). The superseded release build's four file-pick samples were each 8–22 ms
over; p50 passes at 34–45 ms, so the failure is in the tail. The last two of
those are the same clip, which makes the 5 ms gap between them device run-to-run
variance rather than a content difference. An earlier version of this table
reported only DEBUG-build figures (46–54 ms empty-frame, 68–70 ms with a hand)
and called the criterion PARTIAL; those are superseded and are retained here
only for provenance. Classification adds a further ~40 ms warm on top of the
detector.

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
| APK `EchoGesture-Auto-6.2.apk` (**ship this**) | `b1c0e6710e24352dc24a9ebe930527cb7667f67abca905915ea8c49a9ac2dcf8` |
| APK `EchoGesture-2026-11-22-release.apk` (*superseded*) | `9930894280967c0809b52595b103e0a36b93097768f7f61e401b7e50a3b4b397` |
| APK `EchoGesture-2026-11-21.apk` (debuggable; source of the power figure) | `96f00458f54de715c7a6f489f9ae92425bc73a089043a91ccbe0ae6d4f060035` |
| ZIP `EchoGesture_Submission.zip` | `5d310cecf755f1593a3bc9b9b4b56d38e7ca58962d8da0f0325218c746da8e25` |
| Deck `VIT_Chennai_Sky_Grazers.pptx` | `ce577e46d00542aabb2be604e0199979099a50547e93491b730d7c48311a67a7` |
| Deck `VIT_Chennai_Sky_Grazers.pdf` | `c995831a3ed580ee9c5335402955edc1944ad9ed7e0966785f8e0d39cf62325a` |
| Demo video `EchoGesture_demo_2026-11-22.mp4` (gitignored; upload separately) | `321869c07afaf1b20993c2159d8a07d9e312f718a108956108fb679badebd2a3` |

All hashes above were re-measured from the files on disk on 2026-11-22. The
previous table listed a ZIP hash (`5d310cec…`) for a file that did not exist,
and deck hashes from before the team-name rename; both are replaced.
