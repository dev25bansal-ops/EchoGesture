# Final readiness — Track 02

**Status: BLOCKED.** Not on engineering. On five human items listed at the end.

Date 2026-11-14. Deadline 2026-11-18 23:59 IST. Internal target 2026-11-17.

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

Every figure carries its population. 77.02% is labelled a corpus result.
Latency is labelled a laptop measurement and reported as **FAIL** against the
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
| Demo video 2–3 min (p10, p25) | **BLOCKED** — not recorded |
| Presentation, named `CollegeName_TeamName` (p25) | **DONE, needs renaming** |
| Public or shared GitHub repo (p25) | **PASS** — repo is public |
| One submission per team via Google Form (p25) | **BLOCKED** — needs authorization |

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
2. **The `CollegeName_TeamName` string** is needed to name the PPT, the ZIP and
   the APK. It is deliberately not in the repo.
3. **The demo video** needs a phone and a camera.
4. **Rights.** The weights are ASL Citizen-derived with no confirmed
   redistribution permission. Submitting them is the owner's decision.
5. **The Google Form** requires explicit authorization and produces the receipt
   that would let this file say SUBMITTED.

**This is not a submission.** It is everything that can be built and verified
before one.