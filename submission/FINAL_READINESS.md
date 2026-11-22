# Final readiness — Track 02

**Status: BLOCKED.** Not on engineering. On five human items listed at the end.

Date 2026-11-14. Deadline 2026-11-18 23:59 IST. Internal target 2026-11-17.

> **Superseded 2026-11-21 — kept as a dated record.** Three things in this file
> have since changed: a physical phone *was* reached and the shipped build driven
> on it; the APK to ship is `EchoGesture-2026-11-22-release.apk` (`9930894…4b397`),
> with the debuggable `EchoGesture-2026-11-21.apk` (`96f00458…0035`) behind every
> latency, power and thermal number; and the deck quotes 85.45%. For current numbers read
> [`FINAL_DELIVERY/RESULTS.md`](FINAL_DELIVERY/RESULTS.md) and
> [`FINAL_DELIVERY/CHECKLIST.md`](FINAL_DELIVERY/CHECKLIST.md).

---

## Gate A — Application

**PASS on an emulator. NOT TESTED on a phone.**

The APK installs, launches and loads nine members on Android 14
(`sdk_gphone64_arm64`). No crash, no ANR. The footer reads
`9-model ensemble · FP16 TFLite · SHA-256 995cc8d0…`, confirming the running
model by hash rather than by filename.

A P1 defect was found and fixed this cycle: the shipped build displayed
**"Four-model ensemble"** while running nine. The count is now derived from the
loaded member list, so it cannot go stale again. 110/110 JVM tests pass after
the change.

**What is not proven:** that a person signing into a real camera gets correct
captions. That requires a phone.

---

## Gate B — Evaluator

**PASS.**

`predict(video_path)` runs from a freshly extracted copy of the submission ZIP
with no imports from the working tree. Verified: nine members load, both feature
widths (255/381) are exercised, the attention mask is fed, hashes are pinned,
and the output has exactly the five required keys.

30 of 30 clips processed with zero errors on the packaged copy; 23 correct.

---

## Gate C — Integrity

**PASS.**

| | |
| --- | --- |
| Ordered ensemble SHA-256 | `995cc8d0b6f6ac25abb9f7ddca01c3ce71edf8bc2d34e34f4813a2b99dd3b4bb` |
| Members | 9, hashes pinned in both Python and Kotlin |
| Submission ZIP | `35b902dfc3d9a72c1b0180098430596b5e5a71d3fd99414c2dc22c38ce2217e6` |
| P1 APK | `150ffeeb0eaaa8914db362e2bcd41de26c93608193ebcee8457433dc5ea5cb7a` |

Weights, APK and ZIP are gitignored on the source branch — verified, not
assumed.

---

## Gate D — Evidence

**PASS.**

Every figure carries its population. 85.45% is labelled a corpus result at
native geometry, and 82.47% is given separately for the 256-squash entry point.
Latency is labelled a laptop measurement and is **not** claimed to meet the
50 ms criterion. Quantisation is stated as **FP16**, not INT8. Robustness is
reported as two separate harnesses with different verdicts, never merged.

The stale four-model deck was replaced because it would have described a model
we do not ship.

---

## Gate E — Competition requirements

Re-read from the organizer deck, not from internal notes.

| Requirement (PDF page) | Status |
| --- | --- |
| `predict()` in `src/main.py`, Python 3.11 (p11, p12, p17) | **PASS** |
| `src/` holds the whole implementation (p12) | **PASS** |
| `results/predictions.json` + `results.csv` (p11, p12) | **PASS (provisional)** — no organizer manifest exists, so these are our own clip IDs, clearly not organizer IDs |
| Quantised model + conversion script (p10, p11) | **PARTIAL** — conversion script included; model is FP16, budget met by size (16.33 MB < 20 MB) |
| Demo video 2–3 min (p10, p25) | **RECORDED 2026-11-22** — `docs/submission/EchoGesture_demo_2026-11-22.mp4`, 2:55, needs an upload URL on slide 7 |
| Presentation, named `CollegeName_TeamName` (p25) | **DONE** — `VIT_Chennai_Sky_Grazers.pptx` / `.pdf` |
| Public or shared GitHub repo (p25) | **PASS** — repo is public |
| One submission per team via Google Form (p25) | **BLOCKED** — needs authorization; I will not submit or report it submitted without a receipt |

### The INT8 question

Page 9 says "≤ 20 MB quantized model (**INT8**)". Page 11's scoring line says
"Compilation to TFLite/ONNX <20MB" without naming INT8, and its results sheet
gives the size target as `<=25 MB`. The three places disagree.

We ship FP16 at 16.33 MB including the detector. **We do not claim INT8.** The
clarification email is drafted and **unsent**; until it is answered this is
recorded as **ORGANIZER CLARIFICATION REQUIRED**, not as a pass.

---

## Gate F — Submission

**BLOCKED.** See below.

---

## Gate G — Collaboration

**PASS on our side.**

`shared/v4.4-dev-integration` is pushed and remote-verified. Dev's branch was
never modified, merged or force-pushed. Build instructions reproduce the
pinned hashes from either branch.

Dev has not yet run them.

---

## What was done this cycle

| | |
| --- | --- |
| P1 fix | Model-count string derived from loaded members; rebuilt APK; 110/110 tests |
| Submission ZIP | Built to the official layout, clean-extraction tested, 30/30 clips |
| Presentation | Rebuilt for the shipped nine-model system, 8 slides, 3 real screenshots |
| Demo script | Full 3-minute storyboard with narration and a "never say this" table |
| Evidence | Manifest, results summary, checklist, phone worksheet |
| Verification | Artifact hashes, permissions, cleanup guarantees |

## What is still blocked

1. **Registration is unverified.** No confirmation record exists anywhere in the
   repo. If the team did not register by 8 Nov, the final submission link was
   never sent, and nothing else in this package can be submitted.
2. **The `CollegeName_TeamName` string is resolved** — `VIT_Chennai_Sky_Grazers`,
   applied to the deck and PDF. The ZIP and APK keep their product names, which
   the naming rule does not govern.
3. **The demo video is recorded**; it needs an upload and a link on slide 7.
4. **Rights.** The weights are ASL Citizen-derived with no confirmed
   redistribution permission. Submitting them is the owner's decision.
5. **The Google Form** requires explicit authorization and produces the receipt
   that would let this file say SUBMITTED.

**This is not a submission.** It is everything that can be built and verified
before one.

---

## V4.5 update — 2026-11-16

### What changed this cycle

| Item | Change |
| --- | --- |
| Evaluator | Frame geometry is **native 640×480, unconditionally** (`8740f27` removed the 256-squash default and the `ECHOGESTURE_FRAME_SIZE` override). This row previously described the opt-in that was then current. |
| Evaluator | `latency_ms` no longer includes cold start; MediaPipe graph is built during `_init_model()` |
| Tests | +9 geometry-contract tests; 433 passed (was 424), same 3 failed / 23 errors / 43 skipped |
| Docs | `FINAL_EVIDENCE.md` per-frame row corrected from a pass to **RETRACTED / FAIL** |
| Deck | Robustness row corrected to **5.76 FAIL** (was 4.40 PASS with the wrong aggregator, then 5.36 on the old cache) |
| ZIP | Rebuilt with corrected results.csv; **behaviour identical** — 23/30, 0 errors |

### The geometry finding

Feeding the same nine models native 640×480 instead of a 256 square squash,
measured through our own frozen `predict()` on the **346-clip validation split**:

| geometry | top-1 |
| --- | --: |
| 256 squash (former default, now removed) | 82.37% |
| native 640×480 (**the shipped path**) | **85.84%** |

Paired n=345: +12 clips, McNemar exact **p = 0.0501** (22 gained, 10 lost).
4 of 5 signers improve; 16 classes better, 76 unchanged, 8 worse by one clip.

**Not adopted as the default.** p=0.05 is exactly at the threshold, the Android
app already runs at native, and re-verifying every published number days before
the deadline is a worse trade than it looks. Full reasoning in
`docs/release/V4_5_RELEASE_DECISION.md`.

### Corrections that make the submission more honest, not less impressive

- Landmark robustness is **FAIL 5.76**, not the previously reported 4.40 PASS.
  The 4.40 used per-member softmax averaging; the shipped code averages raw
  logits and then applies one softmax. The 5.36 figure that briefly replaced it
  came from the old 256-squash cache. On the native cache the worst corruption
  (hand dropout, both coordinates and presence zeroed) costs **5.76 pp**, and
  the criterion genuinely fails.
- The 15.7 ms/frame figure is **retracted**. Re-measured per-frame median is
  ~142–151 ms, already above the 50 ms budget before any Android work.
- Dev's pixel robustness PASS at 4.48 is at **512 letterbox**, which the app does
  not use. Native-camera robustness is **NOT MEASURED**.

### Unchanged

APK `150ffeeb…` — byte-identical, no Android source touched, 110/110 JVM tests.
Registration, team string, demo video, rights and the Google Form are still the
five human blockers.
