# V4 — true video recognition

Branch `fix/true-video-recognition-v4`. Base `12feeef`. Date 2026-11-14.

**Verdict: PIPELINE FIXED AND MEASURED. Physical-phone evidence still absent.**

The user's observation was correct in substance and wrong in one detail, and
the difference matters. The app was **not** scanning a single photo — but the
interval it did analyse was often not the sign. That is now fixed, and the fix
is measured on 72 authorised clips from 37 different signers.

## 1. The original problem

Repeated incorrect predictions on a physical Android phone. The user asked:
*"The training dataset contains sign-language VIDEO clips, but the Android
application seems to scan a picture when I tap Recognise."*

## 2. One photo, or multiple frames?

**Multiple frames, and it always was.** The old path delivered 32 distinct
analysed frames resampled to 24 model steps with real per-frame timestamps.
There was no single-image path in production.

So the literal hypothesis is false. But it pointed at something real, which is
what the next section measures.

## 3. What the old pipeline actually did

`PERFORM SIGN → TAP → classify the previous 3.7 s`. Concretely:

1. CameraX delivers a frame to `ImageAnalysis` (`STRATEGY_KEEP_ONLY_LATEST`, 640×480).
2. `HandTracker.analyse()` runs MediaPipe in VIDEO mode and appends the frame's
   hand landmarks and arrival timestamp to a 200-frame rolling ring.
3. On tap, `selectCapture()` filters to frames at or before the tap, takes the
   last 3700 ms, picks 32 evenly spaced by timestamp, and resamples to 24.
4. `FeatureTransform.buildFeatures()` produces 24×255.
5. Four FP16 members each produce logits; logits are averaged; one softmax; top-5.

The defect was never the tensor or the model. It was **which frames the tensor
was built from.** A fixed trailing window is correct only if the sign happens to
sit inside it. Three things break that:

- **The tap comes late.** The window then holds the sign's tail plus idle time.
- **The sign is longer than 3.7 s.** The window truncates its onset.
- **The signer starts before the window opens.** The onset is already gone.

Measured on a training clip: the hand occupies frames 23–54 of 75. A window
that *started* at the tap contained zero sign frames — the original defect,
fixed in `12feeef`. A window *ending* at a late tap contains only the sign's
tail.

## 4. What the new pipeline does

`START → perform one sign → FINISH`, then classify exactly that interval.

- `SignRecorder` holds a bounded, timestamped buffer that exists **only**
  between the two presses. It refuses frames offered outside the interval, and
  `stop()` re-checks, so a frame from before Start or after Finish cannot enter.
- `RecordingFeatures` reproduces **both** training stages: 32 evenly spaced
  samples across the interval (`read_frames`), then 32→24 (`build_features`).
  Going straight from N frames to 24 would look equivalent and would silently
  move every model step.
- Quick scan is **preserved unchanged** behind the same frozen ensemble, so the
  two modes are directly comparable.
- Raw video is never stored. Only landmarks, in memory. No INTERNET permission.

### Design decisions and what limits them

| Concern | Behaviour |
| --- | --- |
| A. Signer starts before history exists | No longer possible — recording starts at the press |
| B. Sign finishes, tap too late | No longer possible — Finish ends the interval at the press |
| C. Sign much shorter than 3.7 s | Accepted; resampled to 24 steps (min 400 ms) |
| D. Sign longer than 3.7 s | Accepted up to 12 s, then auto-stop |
| E. Pause during sign | Accepted — coverage, not motion, is the gate |
| F. Only part of motion visible | **Unfixed.** Coverage gate rejects <10%; a partly-visible sign can still score |
| G. Irregular analysis FPS | Handled — selection is by real timestamp |
| H. Hands present, wrong sign | Model question, not capture. See §10 |
| I. Front/rear switch | Ends the recording; sensors are never spliced |
| J. Stale frames after a stall | Rejected if any gap > 700 ms |
| K. Buffer shorter than advertised | Duration and coverage reflect only real analysed frames |
| L. Signing shown on a screen | Works the same — the model reads landmarks, not people |

Bounds are **capture** bounds, not model bounds: 400 ms–12 s, then resampled to
24 steps. Both a 1.5 s and a 9 s sign reach the model as 24 steps, exactly as a
short and a long training clip both became 32 samples.

## 5. Frame and timestamp evidence

From `ReplayModeParityTest` on the emulator with a real clip
(`SCROLL DOWN`, 8.5 s, 260 decoded frames):

```
REPLAY decoded=260 frames=63 distinct=63 rec_dur=8533ms rec_hand=23
       rec_top1=SCROLLDOWN rec_score=0.5697
       quick_top1=SCROLLDOWN quick_score=0.5161
REPLAY span rec=8533ms quick=3700ms
```

- 63 distinct timestamps from 63 sampled frames — no repeated keyframes.
- Record mode covers **8533 ms**; quick scan covers **3700 ms** of the same clip.
- Both modes reach the same answer here, with record mode more confident.

## 6. Before/after on identical clips

Both modes were simulated over the **same decoded frames** of the **same clips**,
using `src/features.py` and `src/main.py`'s own `_run_ensemble` — not a
re-implementation.

| Mode | Top-1 | Top-5 | Accepted | Accuracy among accepted |
| --- | --- | --- | --- | --- |
| **Train, 32 clips / 32 signers** | | | | |
| Explicit record | **29/32 (90.6%)** | 32/32 (100%) | 29 (90.6%) | 29/29 (100%) |
| Quick scan, tap at clip end (best case) | 29/32 (90.6%) | 32/32 (100%) | 29 (90.6%) | 29/29 (100%) |
| Quick scan, tap 1.5 s late | 20/32 (62.5%) | 27/32 (84.4%) | 16 (50%) | 14/16 (87.5%) |
| **Validation, 40 clips / 5 signers** | | | | |
| Explicit record | **31/40 (77.5%)** | 37/40 (92.5%) | 28 (70%) | 25/28 (89.3%) |
| Quick scan, tap at clip end (best case) | 28/40 (70.0%) | 36/40 (90.0%) | 27 (67.5%) | 24/27 (88.9%) |
| Quick scan, tap 1.5 s late | 18/40 (45.0%) | 26/40 (65.0%) | 14 (35%) | 11/14 (78.6%) |

"Accepted" = top-two margin ≥ 0.15 and score ≥ 0.20, the app's own gate.

**What this proves and what it does not.**

- Explicit recording **loses nothing** to best-case quick scan: 0 clips in
  either split are correct under best-case quick scan and wrong under explicit.
- On validation it gains **3 clips**, all of which the best-case quick scan
  misses — so the gain is not only about tap timing.
- Against a **late tap**, explicit recording gains **9/32** and **13/40**. This
  is the realistic comparison: a real user does not tap precisely at clip end.
- Best-case quick scan is generous — it assumes the tap lands exactly when the
  sign ends, which in a live recording it does not.

## 7. Android/Python parity

`ReplayModeParityTest` decodes a real clip on-device and pushes the resulting
landmarks through the **production** `SignRecorder`, `RecordingFeatures`,
`HandTracker.selectCapture`, `FeatureTransform` and `LandmarkClassifier`. No
test-only feature maths anywhere. It asserts monotonic timestamps, ≥80%
distinct timestamps, a valid 24×255 tensor, no invented frames, sample
endpoints matching the interval, and a deterministic prediction.

Both modes produced `SCROLLDOWN` for a `SCROLL DOWN` clip.

## 8. Real-phone results

**None. No physical device was attached.** `adb devices -l` returned only
`emulator-5554`. Every number above is either a Python simulation on authorised
clips or an emulator result. None of it is a phone measurement.

## 9. Accuracy across multiple signers

The validation table is 5 signers × 40 clips; the train table is 32 signers ×
32 clips, one clip each. Per-signer breakdown is in
`evaluation/device/v4_mode_comparison/results_val.json`. Five signers is too
few to claim generalisation across signers; it is enough to show the mode
ordering holds on data that was not used to develop the mode.

## 10. Errors fixed, and errors remaining

**Fixed**

1. Capture window could contain only post-sign idle time (V3, `12feeef`).
2. A thin buffer triggered a waiting path that re-introduced (1) (V3, `87efd29`).
3. A stalled analyser could be classified from the whole stale ring (V3).
4. **The model now receives the signer's own start-to-finish interval** (V4).
5. Stale help text told users to "tap Recognise sign" after a flow that now
   starts with Start sign.

**Remaining**

1. **No phone evidence.** The dominant unknown.
2. **Partial visibility** (case F) is only partly handled — the coverage gate
   rejects almost-empty recordings, not a signer whose second hand is cropped.
3. **Three validation failures** remain under explicit record. All are
   low-confidence (0.09–0.24) and low-coverage, and none has the truth in top-5
   — they look like recognition limits on those clips, not capture faults.
4. **Static/near-static signs** are accepted by design. The mode does not try to
   judge whether sufficient *motion* occurred, because some real signs have
   little.

## 11. Model changes

**NO.** Four FP16 members unchanged. Verified inside the shipped APK:

| Artifact | SHA-256 (prefix) |
| --- | --- |
| `landmark_ens_tcn21_fp16.tflite` | `112279bc86090470` |
| `landmark_ens_tcnm61_fp16.tflite` | `01a261d8eb6c2c4d` |
| `landmark_ens_tcnm62_fp16.tflite` | `27058a85e032bee3` |
| `landmark_ens_rob71_fp16.tflite` | `08417e72da59150a` |
| **Combined** | **`0e4c392732e7da7e77632620dc72e4afce8b1a5aa00234dbf7a13a1049a7ad93`** |
| Detector | `fbc2a30080c3c557…` |

Vocabulary 100 labels, unchanged.

## 12. Is fine-tuning justified?

**Not yet.** Fine-tuning requires repeated cross-signer errors on **valid**
input. Explicit recording removes the capture-fault explanation; until it has
run on real signers, "valid input" is unproven on a phone. The three remaining
validation failures are consistent with normal recognition limits at 77.5%
top-1 on unseen signers, not with a broken pipeline.

## 13. Tests

| Suite | Result |
| --- | --- |
| Kotlin JVM (`SignRecorderTest`, `PreTapCaptureContractTest`, others) | **91 passed, 0 failed** |
| Instrumentation `ReplayModeParityTest` | **1 passed** (real clip, both modes) |
| Instrumentation `RecordModeOnDeviceTest` | **4 passed** |
| Instrumentation regression (`CameraFeaturePath`, render, pre-tap, ensemble) | **8 passed** |
| Python simulation | 72/72 clips analysed |
| Lint | no new findings |

Key JVM cases: a duplicated static frame is not a dynamic sequence; sampling
spans the whole interval; Start/Stop select exactly their interval; no pre-start
or post-stop contamination; no stale reuse; monotonic timestamps; low coverage;
no hands; brief sign; long sign; pause; camera interruption and switch;
repeated attempts; 24×255 shape; held static pose accepted; no-hand intervals
preserved.

## 14. APK

`dist/echogesture-video-v4.apk`
SHA-256 `b4b04fbc0ec41c82f4417b42855c1267aa974f5c88ed1523bfc70c684f5d4cf8`,
77,808,835 bytes, commit `411cee7`.

Differs from the previous candidate `ce83cf4c…` — this is not a rebuild of the
same tree.

## 15. Samsung submission blockers

Unchanged from V3, and still real: no physical-phone evidence, INT8 not shipped
(FP16 costs 5.2 pt less), ≤50 ms latency target unmet (inference ~276 ms p95 on
desktop; capture interaction is several seconds by design), robustness −10 pt
under crop, official vocabulary/manifest not supplied, demo video and
`CollegeName_TeamName` naming not produced, Google Form not submitted.

## 16. Proven vs hypothesis

**Proven**
- The old app used multiple timestamped frames, not one photo.
- A fixed trailing window can and does contain the wrong interval.
- Explicit recording never selected a worse interval than best-case quick scan
  on 72 authorised clips.
- The shipped APK carries the unchanged frozen ensemble and no INTERNET.

**Hypothesis, not proven**
- That explicit recording will improve real-phone recognition. It is strongly
  suggested by the 9/32 and 13/40 late-tap gains, but no phone has run it.
- That the remaining 3 validation failures are model limits rather than capture
  artefacts.

**Not tested**
- Any physical phone.
- Multiple live signers.
- Signer-domain fine-tuning.

## 17. The exact test that closes the phone gap

```bash
adb devices -l
adb install -r dist/echogesture-video-v4.apk
```

Settings → Capture diagnostics → on. For each of 10 glosses, 5 attempts across
the conditions in `docs/release/PHONE_GATE_INSTRUCTIONS.md`: tap **Start sign**,
perform the whole sign, tap **Finish sign**. Record failures too.

```bash
adb shell run-as org.echogesture.finalapp cat files/diagnostics.txt > diagnostics.txt
python3 evaluation/device/accuracy_report.py diagnostics.txt annotations.txt
```

Until that exists, the honest status is **PHONE_EVIDENCE_BLOCKED**. The final
success criterion — a real person signing start-to-finish and getting a
reproducible measured result — is **implemented and verified off-phone**, not
**achieved**.