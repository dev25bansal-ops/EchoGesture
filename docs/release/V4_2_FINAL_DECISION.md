# V4.2 — final decision

Branch `fix/true-video-recognition-v4`, commit `71fca9b`.
Date 2026-11-14. Deadline 2026-11-18.

**Verdict: the V4.1 segmentation claim was overstated and is corrected. A real
jitter defect was found and fixed. No physical phone was tested.**

## 1. Did trimming harm clean recordings?

**Yes, on the validation split. The V4.1 report claimed it did not. That claim
was wrong.**

The user was right. Reproduced and quantified on the same 40 validation clips:

| Condition | Whole interval | Motion trim | Helped | Harmed |
| --- | --- | --- | --- | --- |
| Clean (no idle) | **31/40** | 29/40 | 2 | **4** |

The V4.1 comment cited "31/32 on train, up from 29/32" as evidence that trimming
never hurts. That was a different split measured under a different condition,
presented without validation's 29/40 — the same condition, opposite direction.
Both numbers are now stated together everywhere the decision is discussed.

Measured across both splits:

| Condition | Train | Validation |
| --- | --- | --- |
| Clean, whole | 29/32 | 31/40 |
| Clean, trimmed | 31/32 | 29/40 |
| 2 s lead, whole | 24/32 | 20/40 |
| 2 s lead, trimmed | 31/32 | 29/40 |

The clean-recording effect flips sign between splits and is not distinguishable
from noise at n=32 and n=40. The padded-recording benefit is large and has the
same sign on both. **The padded benefit is the established result; the clean
effect is not.**

Corrections applied to: `V4_1_PHONE_ACCURACY_AND_RELEASE.md` (marked inline, not
rewritten, so the record shows what was claimed), and the `SignSegmenter`
doc comment.

## 2. What safety changes were made

### 2a. A demonstrated defect: one jitter spike captured the threshold

`choose()` derived its motion threshold from `d.max()`. A single noisy frame
therefore raised the threshold above every genuinely moving frame, and the
trimmed window collapsed onto the spike. This was **not a hypothesis — it was
reproduced as a failing test**: one spiked frame in a 183-frame recording cut
the window from 91 to 32 frames.

**Fix:** the reference is now the 95th percentile of moving-frame displacement
(`SignSegmenter.robustPeak`), which a handful of outliers cannot move. The
regression test `anIsolatedDetectorSpikeInsideIdleDoesNotDefineTheSign` pins it.

### 2b. Degenerate-floor guard

A trim keeping under 10% of the recording is treated as a misfire and the whole
recording is used. Measured to cost nothing — identical accuracy to the
unguarded trim across 360 evaluations — while preventing a brief blip in a long
recording from becoming "the sign".

The floor is deliberately far below any real trim (measured median kept fraction
0.64 clean, 0.27 heavily padded). A floor high enough to prevent the clean-clip
regression would also disable exactly the trims that remove real padding.

### 2c. Two alternatives implemented, measured and rejected

| Candidate | Aggregate correct (360 evaluations, both splits, 5 idle conditions) |
| --- | --- |
| A whole interval, no trim | 219/360 (60.8%) |
| **B/E motion trim with percentile + floor (shipped)** | **289/360 (80.3%)** |
| C conservative gated trim | 242/360 (67.2%) |
| D motion trim with a 45% keep floor | 222/360 (61.7%) |

**Why the gate lost:** clean clips have leading near-zero runs up to **4.07 s**
(p90 2.77 s) and trailing up to 1.87 s. That overlaps the idle the gate is meant
to detect, so no deterministic rule can separate "the camera was pointed at
nothing" from "the signer held a starting handshape". At a 0.8 s trigger it fired
on 33/40 clean clips; raised above the overlap to 2.0 s it still harmed clean
recordings (27/40) and recovered far less of the padded gain than the
unconditional trim.

**Why the floor lost:** the clean harms were trims keeping 48–50% of the clip.
A floor at 45% prevented them and also prevented the padded cases from being
trimmed at all — restoring the untrimmed result.

**No classifier confidence was used to choose any crop, at any point.** Every
rule uses only hand presence and landmark displacement.

## 3. Does the new strategy improve validation performance?

**On padded recordings, yes, consistently and by a large margin. On clean
recordings, it is a small and unresolved loss.**

The shipped configuration is not a new strategy so much as the same trim made
robust: percentile reference (fixes a demonstrated defect) plus a degenerate
floor (costs nothing). Neither improves accuracy on clean clips. What they do is
remove the failure mode where one noisy frame corrupts the boundary.

## 4. Does the full sign reach the temporal model?

Yes, with one honest caveat.

- Every analysed frame is appended exactly once (`HandTracker` — one
  `ring.addLast` and one `recorder.offer` per frame, same monitor).
- `SignRecorder` refuses frames outside `[Start, Finish]` and re-checks at
  `stop()`.
- Sampling picks 32 distinct frames by real timestamp, then resamples to 24.
  On-device: **63 distinct timestamps from 63 sampled frames**.
- Features are 24×255, asserted in tests.

**Caveat:** the trim can remove frames the signer intended as part of the sign —
that is precisely the clean-recording harm measured above. The full *recording*
is captured; what reaches the model is the trimmed interval.

## 5. Was a physical Android phone tested?

**No.** `adb devices -l` returned only `emulator-5554` (`sdk_gphone64_arm64`,
android-34) throughout.

The V4.2 APK was installed and exercised on the emulator: it launches, Start →
Finish → an empty capture produces an honest "No hands in view" rejection with
no fabricated prediction, and the process stays alive. **Emulator behaviour is
not phone evidence.**

## 6. Is targeted fine-tuning justified?

**No.** The precondition is repeated cross-signer errors on *valid* input. The
capture defects were only fully understood this phase, and no phone run exists to
establish that input is valid there. Training now would fit a model to a
pipeline whose capture side is still unmeasured on the target hardware.

## 7. Final APK identity

| | |
| --- | --- |
| Path | `dist/echogesture-v4.2.apk` |
| Full SHA-256 | `5490e02908159e24dac1b45dca7b21afbbe77909fb62ceab4348828e8d939dfc` |
| Bytes | 77,908,968 |
| Source commit | `71fca9b` |
| Differs from V4.1 (`3963bf43…`) | yes |
| Ensemble | `0e4c392732e7da7e77632620dc72e4afce8b1a5aa00234dbf7a13a1049a7ad93` — MATCH |
| Detector | `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1` — MATCH |
| Vocabulary | 100 labels, unchanged |
| Permissions | CAMERA only (plus the framework's own dynamic-receiver permission) |
| Internet | absent |
| Installed and launched | yes (emulator) |

## 8. Real vs simulated measurements

| Kind | What |
| --- | --- |
| **Simulated** (Python, authorised clips) | Every accuracy figure in this and prior reports; the idle-padding table; the A/B/C/D/E comparison; the boundary-idleness measurement |
| **Emulator, real models** | Replay parity (63 distinct frames, both modes, segmenter); on-device classification confidence; record-flow UI behaviour |
| **Physical phone** | **Nothing. No device.** |

All Python figures add idle as *frames with no detected hand*, which is what an
idle camera produces. No landmarks are fabricated and no frame is duplicated.

## 9. Samsung requirements

| Requirement | Status |
| --- | --- |
| Reproducible `predict(video_path)` | PASS — unchanged, still callable |
| Fixed vocabulary + output format | PASS — 100 labels, 5-key dict |
| Python 3.11 | PASS |
| Results / model evidence files | PASS — present in `evaluation/` |
| **Quantization requirement** | **FAIL** — FP16 shipped, INT8 measured 5.2 pt worse |
| **≤50 ms edge latency** | **FAIL** — inference ~276 ms p95 desktop; capture is seconds by design |
| **Robustness** | **PARTIAL** — −10 pt under crop (frame-drop 0.0) |
| README / source package | PASS |
| Docker / setup | PASS |
| **3-minute demo video** | **BLOCKED** — needs filming |
| Presentation | BLOCKED — needs the video |
| **Team/college naming** | **BLOCKED** — exact string not supplied |
| **Rights and licences** | **BLOCKED** — see §10 |
| **Final submission form** | **BLOCKED** — needs explicit approval |

## 10. A rights issue that is not mine to resolve

`models/ptq_100_int8.tflite` and `models/s3d_conv1d_qat_int8.tflite` are
committed to this repository, as are the four FP16 members on
`hackathon/dev-model`. They were there **before** this work: `git log
--diff-filter=A` points at the initial commit, and they are present on
`origin/main` and every other remote branch.

I have not added them and I have not removed them. Removing them means rewriting
history, which my instructions forbid, and it would disturb branches owned by
other workstreams. **If these are restricted artifacts, the repository must not
be made public until the owner rewrites that history.** Flagged, not acted on.

## 11. Tests

| Suite | Result |
| --- | --- |
| Kotlin JVM | **110 passed, 0 failed** |
| Instrumentation | **13 passed, 0 failed** |
| Replay parity (real clip) | PASS — 63/63 distinct, both modes, segmenter |
| Python segmentation study | 360 evaluations across 2 splits × 5 conditions |

Note: batching many instrumentation classes into one `am instrument` call
crashed the test process with a native SIGSEGV inside MediaPipe. Each class
passes when run on its own (13/13). The crash is a test-harness limitation, not
an app defect — but it does show a native detector fault is not catchable by
Kotlin `try/catch`, so a real device fault of that kind would kill the process
rather than surface an error state. That gap is **known and unfixed**.

## 12. Exact human actions remaining

1. **Attach a physical Android phone** and run `PHONE_GATE_INSTRUCTIONS.md`.
   Nothing else moves the primary criterion.
2. **Film the ≤3-minute demo video** (consent from any person appearing).
3. **Decide the restricted-weights question** in §10.
4. **Supply the exact `CollegeName_TeamName`** string.
5. **Approve or reject** the Google Form submission.

## 13. Proven / hypothesis / untested

**Proven**
- Trimming helps padded recordings substantially on both splits.
- Trimming does *not* demonstrably help clean recordings, and the honest reading
  of 31/40 vs 29/40 is that it may hurt them slightly.
- A single jitter spike captured the old threshold; the percentile reference
  fixes it and a test pins it.
- The whole-sign path captures a genuine multi-frame sequence and never reuses a
  frame.

**Hypothesis**
- That V4.2 improves real-phone recognition. Every measurement supports the
  direction; none of it was taken on a phone.
- That the clean-recording loss is noise. It is within noise at these sizes, but
  "within noise" is not "zero".

**Untested**
- Any physical phone, any live signer, any fine-tuning.
- Behaviour under real camera shake (simulated jitter only).
- Partial occlusion of one hand.

The primary success criterion — reliable recognition of a complete sign measured
on actual Android phones — remains **unmet**, and cannot be met without hardware.