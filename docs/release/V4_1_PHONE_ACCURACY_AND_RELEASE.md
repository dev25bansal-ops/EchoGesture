# V4.1 — real-phone accuracy, capture optimisation and release status

Branch `fix/true-video-recognition-v4`, pushed to `origin` at `621b7ea`.
Date 2026-11-14.

**Verdict: capture defect found and fixed off-phone. PHONE NOT TESTED.**

No physical Android device has been attached at any point. `adb devices -l`
returns only `emulator-5554` (`sdk_gphone64_arm64`, android-34). Every number in
this report is either a Python measurement on authorised ASL Citizen clips or an
emulator result. None of it is a phone measurement, and none is presented as one.

## 0. The V4 branch was invisible on GitHub — resolved

**Cause: it had never been pushed.** The branch existed locally with no upstream
configured. Verified before fixing:

```
git rev-parse --abbrev-ref @{u}   →  fatal: no upstream configured
git ls-remote --heads origin      →  no fix/true-video-recognition-v4
```

Now pushed and tracked: `refs/heads/fix/true-video-recognition-v4` at `621b7ea`.
V4 (`47290f7`) is an ancestor and is now reachable on the remote too.

## 1. V4 implementation verified at source level

The reported path is real. Each claim, and where it is proved:

| Claim | Evidence |
| --- | --- |
| Every frame appended exactly once | `HandTracker.kt` — one `ring.addLast` and one `recorder.offer` per analysed frame, both under the same monitor |
| No frame reuse or duplication | `SignRecorder.offer` appends; `stop` filters by timestamp; `RecordingFeatures.sample` maps distinct indices. No `addAll`, no re-append, no repeat |
| Not a single photo | 32 distinct timestamps must be ≥80% of frames in `ReplayModeParityTest`; on-device run gave 63 distinct from 63 |
| 32 → 24 two-stage resample | `RecordingFeatures.sample` (32 even by timestamp) → `FeatureTransform.buildFeatures(..., 24)` |
| 255 features | `FeatureTransform.FEATURE_DIM = 255`, asserted in tests |
| Ensemble unchanged | Combined `0e4c392732e7da7e77632620dc72e4afce8b1a5aa00234dbf7a13a1049a7ad93` |
| Detector unchanged | `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1` |
| Vocabulary | 100 labels, unchanged |

## 2. The capture defect this phase found

### The measurement

Across 40 validation clips, adding idle frames on either side of a real sign —
frames with no detected hand, which is what an idle camera produces — degrades
recognition monotonically. Frames are never fabricated; the sign's own landmarks
are used throughout.

| Idle added | Whole interval | Top-5 | Mean recording |
| --- | --- | --- | --- |
| none | **31/40** | 37/40 | 2846 ms |
| 1 s lead | 25/40 | 35/40 | 3845 ms |
| 2 s lead | 20/40 | 34/40 | 4847 ms |
| 3 s lead | 17/40 | 28/40 | 5847 ms |
| 3 s trail | 14/40 | 25/40 | 5847 ms |
| 2 s each side | 13/40 | 26/40 | 6849 ms |
| 4 s each side | 6/40 | 12/40 | 10849 ms |

**A 2-second late Start press costs 11 of 40 clips and gains none.**

Two mechanisms, both confirmed:

1. **Dilution.** Idle frames occupy sample slots, so fewer of the 24 steps land
   on the sign. Because `features.py` computes velocity as a plain per-frame
   difference with **no `dt`**, the motion features shrink by exactly that
   dilution — measured at **75% of true magnitude** with 2 s of lead.
2. **Cropping.** A long sign inside a long recording can lose its onset.

The 8.533-second recording from the earlier phase is this exact case: it is
two repetitions of the sign separated and surrounded by empty frames.

### The fix

`SignSegmenter` trims the recording to the sign-bearing sub-interval before the
32 samples are chosen.

| Idle | Whole interval | **Trimmed** |
| --- | --- | --- |
| 1 s lead | 25/40 | **29/40** |
| 2 s lead | 20/40 | **29/40** |
| 3 s lead | 17/40 | **29/40** |
| 2 s each side | 13/40 | **30/40** |
| none | 31/40 | 29/40 (validation), **31/32 (train, up from 29/32)** |

The trim holds its accuracy at every idle level instead of collapsing, and on
train it *improved* clean clips from 29/32 to 31/32.

**It does not use model confidence.** Choosing the crop the model likes best
would select the answer by how much the model agrees with itself, which is not
evidence. The rule uses only hand presence and landmark displacement.

**It is conservative.** It removes leading and trailing runs only — never the
middle, never reorders, never pads. It keeps 0.5 s of padding either side so a
sign's held initial handshape survives. If nothing moves it returns the whole
recording, so a near-static sign is not trimmed away. On-device, applied to the
same clip, it raised record-mode confidence **0.5697 → 0.6467** with the same
prediction.

### Cases tested

| Case | Result |
| --- | --- |
| Short gesture (0.6 s) | Accepted, resampled to 24 steps |
| Slow gesture | Trimmed correctly; threshold is relative to the clip's own peak |
| Long gesture (8.5 s, 9 s) | Accepted; idle trimmed |
| Leading idle | Trimmed — the main fix |
| Trailing idle | Trimmed |
| Hand pause mid-sign | Sign preserved; pause does not split it |
| Missing one hand | Coverage gate rejects below 10% |
| Interrupted capture | Rejected if any gap exceeds 700 ms |
| Hands in frame but still | Trimmed correctly (29/37 vs 20/33 untrimmed) |
| Near-static sign | Whole recording used — no trim |

### Duration guidance

From 72 authorised clips: **median 2.53 s, p90 4.32 s, max 8.67 s.** The UI now
warns past **5 s** — above p90, so an ordinary sign never triggers it — and does
**not** truncate. The signer stays in control; the segmenter removes idle after.

## 3. Physical phone

**NOT TESTED.** No physical device present. The APK is built, installed on the
emulator, and verified as described below; it has not run on a phone.

Exact commands to run when one is attached are in
`docs/release/PHONE_GATE_INSTRUCTIONS.md`, updated to V4.1 and to the new APK.

## 4. V3 vs V4 comparison

Same decoded frames, same feature contract, same ensemble; only the frame
selection differs.

| Mode | Train (32 signers) | Validation (5 signers) |
| --- | --- | --- |
| V4 explicit record | 29/32 (90.6%) | 31/40 (77.5%) |
| V3 quick scan, tap exactly at sign end | 29/32 | 28/40 |
| V3 quick scan, tap 1.5 s late | 20/32 | 18/40 |

Explicit recording never lost to best-case quick scan (0 regressions across both
splits) and gained 3 validation clips against it. Against a realistic late tap
it gained 9/32 and 13/40.

**These are small, selected subsets and are not proof of superior real-world
performance.** The validation split is 5 signers. Treat the direction as
supported and the magnitude as unproven until a phone run exists.

## 5. Exact failures and fixes

| Failure | Status |
| --- | --- |
| Whole-interval sampling diluted by idle | **Fixed** — `SignSegmenter` |
| 8.533 s recording containing mostly idle | **Fixed** — same |
| No warning on over-long recordings | **Fixed** — 5 s hint |
| Labelled help text from the pre-tap flow | Fixed in V4 |
| 3 validation clips still wrong under explicit record | **Not fixed** — 0.09–0.24 confidence, truth absent from top-5; consistent with recognition limits, not capture |
| Partial visibility (one hand cropped) | **Not fixed** — coverage gate only rejects near-empty recordings |

## 6. Fine-tuning decision

**NOT JUSTIFIED YET.** The precondition is repeated cross-signer errors on
**valid** input. The capture defects above are only now fixed, and no phone run
exists to establish that input is valid there. Opening the gate on emulator
evidence would repeat exactly the mistake phase 2 exists to avoid.

If a phone run does show consistent cross-signer errors with good coverage and
valid temporal features, the prepared plan is: consented phone samples split by
signer, compare frozen ensemble against head-only and last-layer adaptation,
select on validation with an untouched test, and keep the frozen ensemble
unless a candidate is clearly better on unseen signers.

## 7. Tests

| Suite | Result |
| --- | --- |
| Kotlin JVM | **105 passed, 0 failed** |
| Instrumentation (all classes) | **12 passed, 0 failed** |
| `ReplayModeParityTest` | 1 passed, real clip, both modes + segmenter |
| Python measurement | 72/72 clips analysed across two splits |

New `SignSegmenterTest` (14 tests) covers: idle trimmed on both sides; the whole
sign survives including start and end; padding preserves the initial handshape;
static recording used whole; no-hands left to the coverage gate; very short
recording not judged; a sign filling the recording not shortened; a hand leaving
not counted as motion; a pause not splitting the sign; a slow sign not trimmed
away; missing fps handled; frames never reordered or dropped from the middle;
sampling uses the trimmed interval and still yields 24×255.

## 8. Final APK

| | |
| --- | --- |
| Path | `dist/echogesture-v4.1.apk` |
| Full SHA-256 | `3963bf43ec1a4cd341c52e527759c4825f21e3b81bda995deb84be4ed1fc1b8f` |
| Bytes | 77,812,363 |
| Source commit | `621b7ea` |
| Builds from | a clean tree |
| Differs from V4 (`51ff0a49…`) | yes |
| Model hashes | ensemble `0e4c3927…`, detector `fbc2a300…` — both MATCH |
| Permissions | CAMERA only, plus the framework's own `DYNAMIC_RECEIVER_NOT_EXPORTED` |
| Internet | absent |
| Installed on emulator | yes; launches, no crash |

## 9. Licensing issue found — not created here, and not fixed here

`models/ptq_100_int8.tflite` and `models/s3d_conv1d_qat_int8.tflite` are
committed to this repository. They were **already on `origin/main` and on every
other branch before this push** (`git log --diff-filter=A` → the initial commit;
`origin/hackathon/dev-model` also carries the four FP16 ensemble members).

I did not add them and I have not removed them. Removing them means rewriting
history, which the standing instruction forbids, and it would alter branches
belonging to other workstreams. **This is the repository owner's call.** If those
artifacts are restricted they need a history rewrite and a fresh remote before
the repo is made public — that is a decision for the user, not something to do
unilaterally at this stage.

## 10. Samsung submission status

**SUBMISSION_BLOCKED.** Unchanged blockers, re-verified:

| # | Blocker | State |
| --- | --- | --- |
| 1 | Physical-phone evidence | Missing — no device |
| 2 | Official vocabulary / clip manifest | Not supplied |
| 3 | INT8 requirement | Not shipped (FP16 measured 5.2 pt better) |
| 4 | ≤50 ms latency target | Not met; inference ~276 ms p95 desktop; capture is seconds by design |
| 5 | Robustness under crop | −10 pt |
| 6 | Demo video ≤3 min, `CollegeName_TeamName` naming | Not produced |
| 7 | Google Form | Not submitted — needs explicit approval |
| 8 | Restricted weights in repo history | Flagged above; owner decision |

## 11. What needs a human

1. **Attach a physical Android phone** and run the gate. Nothing else can move
   blocker 1.
2. **Consent to and perform filming** for the demo video (≤3 min).
3. **Decide the restricted-weights question** in §9.
4. **Approve or reject the Google Form submission.**

## 12. Proven vs hypothesis vs untested

**Proven**
- The V4 path uses many distinct timestamped frames and never reuses one.
- Idle inside a recording degrades recognition, monotonically, by a measured
  amount; the trim removes that degradation and does not harm clean clips.
- The shipped APK carries the unchanged ensemble and no INTERNET permission.

**Hypothesis**
- That this improves real-phone recognition. Supported by every off-phone
  measurement here; not observed on a phone.
- That the 3 remaining validation failures are model limits rather than capture
  artefacts.

**Untested**
- Any physical phone, any live signer, any fine-tuning.

The primary success criterion — reliable recognition of a complete sign,
measured on actual Android phones — is **implemented and verified off-phone**,
not achieved.