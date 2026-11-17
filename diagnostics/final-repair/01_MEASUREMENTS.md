# 01 — What was actually measured

Every figure here was produced in this session. Nothing is carried over from an
earlier report, and nothing is a prediction.

## A. The model works. Confirmed on 345 real clips.

The shipped nine-member ensemble was run through `src/main.py::predict()` on all
346 authorised validation clips (5 unseen signers, all 100 vocabulary classes).

| Metric | Value |
| --- | --- |
| Scored clips | 345 |
| Correct | **302** |
| Accuracy | **87.5%** |
| 95% CI (Wilson) | ±3.5 pp |

Per signer:

| Signer | Correct | Accuracy |
| --- | --- | --- |
| P12 | 20 / 24 | 83.3% |
| P21 | 90 / 104 | 86.5% |
| P26 | 85 / 101 | 84.2% |
| P39 | 96 / 103 | 93.2% |
| P5 | 11 / 13 | 84.6% |

Every signer lands in an 83–93% band. There is no signer the model collapses on.
The spread is what an 87.5% engine looks like across small samples, not evidence
of a signer-generalization failure.

**One clip** (`5989539775823847-INTUITIVE.mp4`) produces no hand detection in any
frame and is excluded rather than counted as wrong.

## B. The same pipeline on the device agrees.

`ClipReplayAccuracyTest` on `eg34`, 30 labelled clips through the actual Android
detector, feature transform and classifier:

| Geometry | Correct | Accuracy |
| --- | --- | --- |
| native 640×480 | **26 / 30** | **86.7%** |
| 256×256 squash | 24 / 30 | 80.0% |

Android 86.7% against Python 87.5% on the same engineer's corpus. **The two
implementations agree, and both work.**

## C. Acceptance policy is calibrated and it works.

Measured on the 345 clips, with top-1 and top-2 probabilities captured from the
actual ensemble output:

| Rule | Coverage | Accuracy of accepted |
| --- | --- | --- |
| No gate | 100% | 87.5% |
| **margin ≥ 0.15, score ≥ 20%** | **86.1%** | **96.0%** |
| margin ≥ 0.20, score ≥ 15% | 84.1% | 96.6% |
| margin ≥ 0.25, score ≥ 15% | 80.6% | 97.8% |

Separation at the shipped rule:

| Group | Share | Accuracy |
| --- | --- | --- |
| Accepted | 86.1% | **96.0%** |
| Declined | 13.9% | **35.4%** |

The gate is doing exactly what it claims: accepted readings are 96% correct,
declined readings are near chance. **A user who saw a sign presented as
recognised was, on this population, right 96 times in 100.**

This is why "the app misrecognises everything" cannot be explained by result
handling. Something downstream of the model must be degrading the input before
the gate ever sees it.

## D. The build on the release branch was broken. Fixed.

See `00_ROOT_CAUSE.md`. Summarised: the branch shipped **4 old model binaries**
(`112279bc…`) while `ModelContract` requires **9** (`af031d89…` … ) plus a
`members_detail` block the branch's meta file did not contain. Any APK built
from the branch would have thrown during model load.

Fixed by replacing the branch assets and `models/` with the nine verified
members taken from the delivered APK. All nine match the pinned hashes exactly.
The branch now builds an APK that loads `995cc8d0…` and reaches the Camera tab
with the nine-model footer visible.

## E. What could NOT be measured, and why that bounds the conclusions

| Quantity | Status | Why |
| --- | --- | --- |
| Real CameraX analysis rate on a phone | **NOT MEASURED** | No physical device attached |
| Live hand-detection rate | **NOT MEASURED** | Same |
| Hand-detection continuity | **NOT MEASURED** | Same |
| Live segment duration | **NOT MEASURED** | Same |
| Path C (phone filming a screen) | **NOT TESTED** | Same |
| Path D (a person signing into the phone) | **NOT TESTED** | Same |

`adb devices` showed only `emulator-5554`. `system_profiler` found no USB
Android device. No wireless ADB target responded.

The emulator's camera is a synthetic colour pattern that never contains a hand.
It can prove the UI binds a camera; it can never prove recognition.

**Every statement in this packet about live recognition is therefore limited to
"the pipeline is correct and the model is accurate on captured video". The step
from "a person signs into a phone" to "the app reads it" is untested.**

## F. The live-only risk I could quantify without a phone

`RecordingFeatures.sample()` passes a recording through untouched when it holds
≤ 32 frames. Above 32 it picks frames by timestamp; below 32 there is no
timestamp resampling at all, and the index-based 32→24 step then samples
irregularly-timed frames as if they were evenly spaced.

Whether a real phone falls below that threshold depends entirely on the analyser
rate, which is the quantity I could not measure. Using the real clip durations:

| Clip duration | At 10 fps | At 15 fps | At 25 fps |
| --- | --- | --- | --- |
| median 2.17 s | **20 frames** | 33 frames | 54 frames |
| % of sample below 33-frame threshold | **100%** | 50% | 0% |

At 10 fps every one of the ten sampled clips would take the unresampled path; at
25 fps none would. **This is a live hypothesis, not a finding.** It is stated in
`04_LIVE_RISK.md` with the experiment that would settle it.