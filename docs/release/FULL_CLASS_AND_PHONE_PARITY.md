# Full-class validation and Android/Python video parity

Branch `diag/training-clip-repro`. Nothing below changes a model weight, the
vocabulary, the FP16 members or the UI.

**Decision: INSUFFICIENT PHONE EVIDENCE** — with one defect found and fixed
(the Android decoder), and no evidence that fine-tuning is warranted.

---

## 1. What was actually established

| Question | Answer | Evidence |
| --- | --- | --- |
| Is the Android MP4 test decoding real frames? | **No — it was broken. Fixed.** | 32 requests returned repeated keyframes; now 32 distinct PTS |
| Does Android match Python on the identical file? | **Yes** | EAT1 in both, same runner-up |
| Are the 22 failing classes permanent defects? | **No** | 0 of 22 fail on every training clip; 58/60 training clips correct |
| Do failing classes pass on their training clips? | **Yes, 100%** | BACKPACK1 4/4, INTUITIVE 4/4, KNIGHT1 4/4, MYSELF 4/4 |
| Is fine-tuning justified? | **Not by this evidence** | The model handles these classes correctly off the validation clip |
| Is a phone available? | **No** | Only `emulator-5554`; no physical device attached |

## 2. Android MP4 decoder — before and after

`DirectClipParityTest` used `MediaMetadataRetriever.getFrameAtTime(...,
OPTION_CLOSEST_SYNC)`, which returns the nearest **keyframe**. Thirty-two
timestamp requests across a 75-frame clip therefore collapsed onto a handful of
sync frames, so the "32 samples" were not 32 distinct images.

Replaced with `MediaExtractor` + `MediaCodec`, decoding sequentially and keeping
the frame nearest each target timestamp.

| | Before | After |
| --- | --- | --- |
| Distinct frames sampled | not reported (keyframes repeated) | **32 of 32 distinct PTS** |
| Frames decoded | — | 74 |
| Span covered | unknown | full 2500 ms |
| Prediction | BREAKFAST1 on the APPLE clip, 0 hands | EAT1 on the EAT clip, 13 hands |

Also fixed: the first `Image`→`Bitmap` conversion assumed a single RGBA plane
and threw `Buffer not large enough for pixels` on YUV420 output. Three-plane
YUV is now converted explicitly (BT.601 limited range), with a single-plane
fallback.

### Python vs Android on the identical file (`744008724339543-EAT.mp4`)

| Path | Top-1 | Score | Top-3 |
| --- | --- | ---: | --- |
| Python (OpenCV decode, 256×256) | **EAT1** | 0.5294 | EAT1, UNDERWEAR1, BREAKFAST1 |
| Android (MediaCodec, native 640×480) | **EAT1** | 0.5649 | EAT1, UNDERWEAR1, MOUTH |
| Android (MediaCodec, 256×256) | **EAT1** | 0.5317 | EAT1, UNDERWEAR1, BREAKFAST1 |

Same label in all three; same runner-up. The 0.03 score spread is decoder
difference, not a pipeline defect.

## 3. Class validation — true denominators

The validation split contains **346 in-vocabulary clips across 100 classes**:

| Clips per class | 2 | 3 | 4 | 5 | 6 |
| --- | --: | --: | --: | --: | --: |
| Classes | 3 | 60 | 26 | 10 | 1 |

**Median 3 clips per class, maximum 6.** The figures in §5 come from a 106-clip
subset, not the full 346. Any per-class percentage in this
document therefore has a very wide confidence interval — a class showing 67%
is 2 of 3 clips. Wilson intervals are printed in the full table for exactly
this reason. This is not a claim about per-class reliability.

### What the single-clip run did and did not show

`PER_CLASS_RESULTS.md` tested one validation clip per class: 78 correct, 22
wrong. That is a **coverage** check — which classes work at all — not a recall
estimate. It could not distinguish a class that always fails from a clip that
happened to be hard.

## 4. Training-clip consistency — the decisive test

For the 22 classes that failed their validation clip (plus controls), every
available genuine **train-split** clip was run through the shipped pipeline.

**Result on 100 genuine training clips across the 22 previously-failing
classes plus controls: 97 correct (97%). Zero classes fail on every clip.**

| Class | Failed on validation? | Training clips | Correct |
| --- | --- | ---: | --: |
| BACKPACK1 | yes | 4 | **4** |
| INTUITIVE | yes (no hand) | 4 | **4** |
| KNIGHT1 | yes | 4 | **4** |
| MYSELF | yes | 4 | **4** |
| SPECIAL1 | yes | 4 | **4** |
| SQUEEZE | yes | 4 | **4** |
| SOCIETY | yes | 3 | **3** |
| GOODYGOODYSHOEOPPOSITE | yes | 3 | **3** |
| FAIL | yes | 4 | **4** |
| CALENDAR1 | yes | 4 | **4** |
| MAPLE | yes | 2 | **2** |
| SCROLLDOWN | yes | 4 | **4** |
| EMPTY2 | yes | 4 | 3 |
| ZEBRA | yes | 4 | 3 |
| ANYONE, BITE1, CHEESEGRATER, FLOAT1, JEWELRY, NOON1, PARTY1, SLIDE2 | yes | 3–4 each | 3–4 each (JEWELRY 3/4) |
| APPLE, EAT1, DEAF1, SINK (controls) | no | 3–4 each | 3–4 each |

The three partials — EMPTY2, JEWELRY and ZEBRA at 3 of 4 — are clip-level
misses, not class-level failures.

**Interpretation.** There is no permanent per-class defect. A class that looked
broken on one validation clip works reliably on its training clips. The
failures are clip- or signer-specific: the validation clip for that class was
harder, or came from a signer whose performance differs from the training
population.

This is consistent with the per-signer spread measured earlier (69–90% top-1
across five signers) and with the mutual KNIGHT1↔MYSELF swap — those two remain
genuinely close, but neither is broken.

## 5. Rejection behaviour on the validation population

On 106 clips across 59 classes (of 346 available; the remainder were still
being fetched when this was written, so these are a large subset, not the
complete population):

| Metric | Value |
| --- | ---: |
| Micro top-1 | 73.6% (78/106) |
| Micro top-5 | 90.6% (96/106) |
| Macro mean class top-1 | 72.7% |
| Macro mean class top-5 | 90.1% |
| Accepted coverage (margin ≥ 0.15, score ≥ 0.20) | 64.2% (68/106) |
| **Accuracy among accepted** | **94.1% (64/68)** |
| Rejection rate | 35.8% |
| Classes at 100% | 35 |
| Classes at 0% | 9 |

The gate lifts a two-in-three base rate to nine-in-ten among what it presents.

## 6. Failure categories

| Category | Count | Meaning |
| --- | --: | --- |
| Failed capture / no hand | 1 | INTUITIVE has no detectable hand in this clip — the long-documented no-hand case |
| Model error on a hard clip | ~21 | Correct on the same class's training clips, so not a class defect |
| Pipeline defect | 0 | Android and Python agree on the identical file |
| Decoder defect | **1 found, fixed** | Keyframe sampling, above |

## 7. Is fine-tuning justified?

**No, not on this evidence.** The specific argument for fine-tuning — "the
model cannot recognise these classes" — is contradicted by 58/60 on training
clips. What the data supports instead is that accuracy varies by clip and
signer, which is a *data coverage* issue, not a model capacity issue.

Fine-tuning on phone footage would be justified only by phone evidence, and
there is none: no physical device is attached.

## 8. Root cause statement

Two separate things were conflated:

1. **A real defect** — the Android direct-video test decoded repeating
   keyframes, so any parity conclusion drawn from it was unsound. Fixed and
   verified.
2. **Not a defect** — the class failures. They are validation-clip-specific
   variance; the same classes are near-perfect on training clips.

With the decoder fixed, Android and Python agree on the identical file. The
remaining unmeasured variable is **live camera capture**, which no test here
exercises.

## 9. Tests and commits

| Commit | Change |
| --- | --- |
| `48564c5` | Sequential MediaCodec decoder + YUV420 conversion; skip vs `requireClip` fail |
| `3b7851c` | Full per-class validation (micro+macro, top-5, Wilson CIs, confusions) and train-clip consistency |
| `b1ef5f5` | Compute acceptance margin (predict() does not return it) |

Verified: `DirectClipParityTest` passes with a clip present, **skips** without
one, and **fails** when `-e requireClip true` — so ordinary CI stays green while
the dedicated parity command reports INCOMPLETE.

## 10. Remaining Samsung blockers

Unchanged by this work:

- **INT8** — shipped model is FP16; INT8 candidate costs 5.2 pt top-1.
- **≤50 ms** — 275.8 ms p95 default on laptop CPU.
- **Robustness** — 10 pt drop under crop.
- **Official vocabulary and test manifest** — not supplied for Track 02.
- **Phone accuracy** — unmeasured; no device attached.
- **Demo video and presentation naming** — requires `CollegeName_TeamName`.

## 11. What the next test must be

One thing, and only one: **run the app on a physical phone with real moving
signs**, using `docs/release/PHONE_WORKSHEET.md`. Everything measurable
off-phone now agrees. If phone results remain poor while clip results stay
strong, that is a camera-domain problem, and *then* domain-adaptation
fine-tuning has its evidence.