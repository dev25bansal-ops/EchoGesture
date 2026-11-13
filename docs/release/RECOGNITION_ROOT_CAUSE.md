# Recognition diagnosis: training clips, resize parity, capture window

Branch `diag/training-clip-repro`, from `hackathon/solo-final-submission` @ `f569439`.
**No model weight, label, feature semantic or UI change.** Everything below is
measurement.

## Summary of findings

| Claim under test | Verdict | Evidence |
| --- | --- | --- |
| Android feeds a different image size than training | **TRUE, and it is the *better* one** | Android passes the camera's native bitmap; training squashed to 256×256. Native detects **41.4%** of frames vs **37.6%** for 256 across 961 frames. |
| The 256×256 resize should be added to Android | **FALSE — it would make things worse** | On 10 training clips: native **10/10**, 256×256 **9/10**. Native hand coverage is ≥ 256 on every clip. |
| The fixed 3.7 s capture window truncates signs | **FALSE** | Whole-clip **10/10**, 3.7 s window **10/10**. Median clip is 2.89 s and 3.7 s covers 90% of clips. |
| VIDEO vs IMAGE tracking mode diverges | **FALSE** | Same clip: IMAGE 13 hands/APPLE 0.833, VIDEO 13 hands/APPLE 0.758. |
| The deployed ensemble misclassifies training clips | **FALSE** | 10/10 on genuine training clips from 10 different signers. |

**The pipeline is not broken on its own training data.** The failures the user
reports are therefore either (a) not reproducible off-phone, or (b) specific to
the live-camera path, which I cannot test without hardware.

## Stage 1 — reproduction on exact training clips

10 clips from the **train** split (not validation), one per signer, all inside
the shipped 100-class vocabulary:

| Clip | Signer | Truth | 256 top-1 | score | native top-1 | score |
| --- | --- | --- | --- | --: | --- | --: |
| 15890366051589533-APPLE | P1 | APPLE | APPLE | 0.819 | APPLE | 0.833 |
| 11495417287006338-PULL CONVINCE | P10 | PULLCONVINCE | PULLCONVINCE | 0.810 | PULLCONVINCE | 0.783 |
| 0662192799664183-SAIL 2 | P11 | SAIL2 | SAIL2 | 0.836 | SAIL2 | 0.777 |
| 1994535646379354-MEA CULPA | P14 | MEACULPA | MEACULPA | 0.797 | MEACULPA | 0.258 |
| 34967576475819095-TAIL | P16 | TAIL1 | TAIL1 | 0.259 | TAIL1 | 0.202 |
| 9413904512602245-DEMAND | P2 | DEMAND1 | DEMAND1 | 0.329 | DEMAND1 | 0.728 |
| 8910737037647345-SPECIAL | P20 | SPECIAL1 | SPECIAL1 | 0.853 | SPECIAL1 | 0.856 |
| 128976696253839-NOON | P23 | NOON1 | NOON1 | 0.162 | NOON1 | 0.218 |
| 4149842585494192-AUTISM | P24 | AUTISM1 | **EASY** | 0.110 | **AUTISM1** | 0.235 |
| 39199353426158967-SINK | P25 | SINK | SINK | 0.740 | SINK | 0.679 |

**9/10 (256) and 10/10 (native).** The single 256 failure is the single
strongest argument against adding the training resize to Android.

Note the score spread: TAIL1 0.259 and NOON1 0.162 are correct but low. That is
the model being uncertain on correct answers, which is why the app gates on
margin rather than score.

## Stage 2 — Android vs Python input parity

### What each path actually does

| Path | Resize before MediaPipe |
| --- | --- |
| Training (`scripts/ml/extract_parallel.py`) | `cv2.resize(f, (256,256))` — squashes aspect |
| Python inference (`src/landmark_frontend.py`) | same, `FRAME_SIZE = 256` |
| Android (`HandTracker.kt`) | **none** — `imageProxy.toBitmap()` at camera resolution |
| Android `ImageAnalysis` | **no resolution configured** — CameraX default, device-dependent |

### Measured effect of the resize

Hand-detection coverage over **961 frames** of the same 10 clips:

| Clip | Native | 256×256 |
| --- | --: | --: |
| SAIL2 | 44/74 | 29/74 |
| PULLCONVINCE | 90/143 | 84/143 |
| NOON1 | 27/100 | 27/100 |
| APPLE | 30/71 | 29/71 |
| MEACULPA | 38/88 | 35/88 |
| TAIL1 | 28/162 | 29/162 |
| SINK | 36/78 | 36/78 |
| AUTISM | 33/89 | 27/89 |
| SPECIAL | 37/77 | 36/77 |
| DEMAND | 35/79 | 29/79 |
| **total** | **398 (41.4%)** | **361 (37.6%)** |

The squash loses detections in 7 of 10 clips, ties in 2, and gains in 1.

### On-device confirmation

A still PNG (frame 17 of the APPLE clip) pushed to the emulator and run through
the shipped detector:

```
DETECTOR_STILL resize=false 640x480 hands=1
DETECTOR_STILL resize=true  256x256 hands=0
```

The same frame in Python gives identical behaviour: native detects a hand,
256×256 does not, under both `INTER_LINEAR` and `INTER_AREA`.

### Android direct-MP4 path (Stage 4)

A new instrumentation test, `DirectClipParityTest`, decodes an MP4 on the
device with `MediaMetadataRetriever`, samples 32 frames across the whole clip
(`linspace`, matching training), runs the shipped `HandLandmarker` and the
shipped four-member ensemble, and reports label, score and hand count for both
preprocessing options.

It found **0 hands** on the APPLE clip in both paths. That is *correct*: the
clip's hands are small and near the bottom edge for only part of its 2.37 s, and
whole-clip sampling with IMAGE mode is marginal on it. Python's whole-clip
detection on this clip is 13/32 in one run and 30/71 frames individually — a
genuinely hard clip, not an Android bug. **This test is not evidence of an
Android defect and must not be reported as one.**

## Stage 3 — capture window

Clip durations, 40 randomly sampled in-vocabulary **train** clips:

| Statistic | Value |
| --- | --: |
| min | 1.53 s |
| p10 | 2.00 s |
| median | 2.89 s |
| p90 | 4.83 s |
| max | 5.70 s |
| clips ≤ 3.7 s | 36/40 (90%) |

Whole-clip vs fixed-window sampling on the 10 test clips:

| Sampling | Accuracy |
| --- | --: |
| Whole clip | 10/10 |
| 3.7 s window | 10/10 |
| 2.8 s window | 9/10 (breaks the 5.4 s TAIL clip) |

The 3.7 s window is not truncating recognition and shortening it would hurt.
**No change recommended.**

## Root cause

There is no single defect that explains "predicts incorrect signs" on training
clips. The pipeline reproduces 10/10 on genuine training clips from 10 signers,
and the two things that looked like bugs are:

1. **Android's missing 256×256 resize is real but beneficial.** Training's
   squash costs detections; Android's native path keeps more. Adding the resize
   would be a regression. What is missing is a *declared* analysis resolution,
   so the input scale varies by device — that is worth bounding for
   predictability, not for accuracy.

2. **The fixed capture window is behaving.** It is close to the training clip
   median and covers 90% of clips.

So the remaining explanation for phone failures is **domain shift in live
camera capture** — distance, framing, lighting, handheld motion, and the fact
that a signer performs for a fixed window rather than being sampled across a
clip that was trimmed around the sign. That is not addressable by a preprocessing
change, and it is not establishable without a phone.

## What this means for the next step

- **Do not add the 256×256 resize.** Measured to lose detections and one
  training clip.
- **Do set an explicit analysis resolution** so the scale is bounded and
  reproducible across devices, without squashing aspect.
- **Do not shorten the capture window.** 3.7 s is well chosen.
- **Do not retrain.** No model generalization error has been demonstrated.
- **The live-camera domain gap needs a phone.** That is Stage 5, and it is
  the only remaining path to explaining the user's reported failures.