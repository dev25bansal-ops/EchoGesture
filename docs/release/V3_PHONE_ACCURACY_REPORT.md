# V3 phone accuracy report

**Status: PHONE_EVIDENCE_BLOCKED.** No physical Android device was attached at
any point during this work. `adb devices -l` returned exactly one entry,
`emulator-5554` (`sdk_gphone64_arm64`, android-34) — an emulator, never a phone.
Nothing below is a phone measurement, and nothing here is presented as one.

What this report *does* contain is a real production defect found, fixed and
verified off-phone, and the exact reason the phone number is still missing.

## 1. What was investigated

Two defects were reported in the capture path. Both were real; both are fixed.

### Defect A — the waiting path re-introduced the post-tap bug

`beginCapture(onComplete: () -> Unit)` cleared the ring and then checked whether
≥ 32 frames with sufficient history were buffered. If they were, it classified
immediately. **If they were not, it set `capturing = true` and waited** for the
next 3.7 s of camera frames before completing.

That waiting branch is the original defect wearing a different hat. The signer
performs the sign and *then* taps, so the frames describing it are already
behind the tap. Waiting after the tap records the empty seconds that follow.

Measured on an authorized ASL Citizen training clip: the hand occupies frames
23–54 of 75. A window starting at the tap contains **zero** sign frames. The
confidence on the post-sign tail degrades as coverage falls — 0.565 → 0.486 →
0.361 at 13 → 10 → 8 hand frames.

### Defect B — the stale-ring fallback

`captureSnapshot()` filtered the ring to the trailing 3.7 s and then:

```kotlin
val raw = ring.filter { it.capturedAtMs >= cutoff }.ifEmpty { ring.toList() }
```

When the trailing filter came back empty — a stalled analyser, a paused camera,
a slow device — it fell back to the **entire ring**, including frames far older
than the span. A dead camera would be classified as though it had just
delivered.

## 2. The fix: a strict pre-tap capture contract

`HandTracker.beginCapture()` is now synchronous and total. It has no waiting
path and no fallback. Every outcome is one of:

| Outcome | Meaning |
| --- | --- |
| `Ready(capture)` | A window of real frames, all at or before the tap |
| `Insufficient(NO_FRAMES)` | Nothing buffered, or nothing at or before the tap |
| `Insufficient(TOO_SHORT, spanMs)` | History exists but covers < 1200 ms |
| `Insufficient(STALE)` | Newest frame is > 700 ms older than the tap |
| `Insufficient(REPEATED, gapMs)` | Another sign was read < 1200 ms ago |

The decision lives in `HandTracker.selectCapture(frames, tappedAt, previous)`,
a pure companion function. Production calls it; the tests call the same
function. There is no test-only re-implementation of the logic.

Behaviour the contract guarantees:

- **Tap timestamp recorded first**, and only frames with `capturedAtMs <= tappedAt`
  are eligible. A frame that arrives while the tap is being handled belongs to
  the *next* sign and can never enter the window.
- **Window is `CAPTURE_SPAN_MS` (3700 ms) ending at the tap**, resampled to 32
  points by real timestamp — the `linspace(0, n-1, 32)` step training used
  (~115 ms), not the camera's 33 ms.
- **No fallback.** A window shorter than the span is reported at its real
  duration; the model never sees frames from outside the span.
- **Genuine handless frames survive.** Onset and offset of a real sign are
  empty; they stay in the window so coverage is measured honestly.
- **Camera switch clears the buffer and the repeat guard** (`clear()` resets
  both), so a rebind is a new context rather than a rapid re-tap.
- **Ring operations are under one monitor**; the decision holds it and the
  analyser appends under it.
- **Nothing blocks the UI thread.** `beginCapture()` is a filter over a bounded
  (≤ 200 frame) deque; inference still runs on `inferExecutor`.

### UI consequences

`UiState.CAPTURING` no longer exists. A tap goes straight to `PROCESSING` or to
a message. Each rejection has its own text:

| Reason | Shown |
| --- | --- |
| `NO_FRAMES` / `TOO_SHORT` | "Not enough of the sign was recorded yet. Perform the whole sign, then tap straight away." |
| `STALE` | "The camera stopped sending frames during that sign. Hold still for a moment, then sign again." |
| `REPEATED` | "That sign was just read. Pause a moment, perform the next sign, then tap again." |

## 3. What was verified, and how

### JVM unit tests — 64 passed, 0 failed

`./gradlew :app:testDebugUnitTest`

`PreTapCaptureContractTest` has 22 behavioural tests driving the real
`selectCapture` with injected timestamps and deterministic synthetic buffers:

| Scenario | Test |
| --- | --- |
| Full history | `fullHistoryClassifiesTheWindowEndingAtTheTap`, `framesOlderThanTheSpanAreExcludedNotStretched` |
| History not warmed | `historyShorterThanTheMinimumSpanIsRejectedWithItsRealSpan`, `noFramesAtAllIsReportedSeparatelyFromTooShort` |
| No hands | `aHandlessWindowIsStillAValidCaptureSoCoverageCanBeJudged`, `genuineHandlessFramesInsideTheSignArePreserved` |
| Camera paused | `aStalledAnalyserIsRejectedRatherThanClassifyingAStaleWindow`, `aBriefFrameGapJustUnderTheStaleLimitIsStillClassifiable` |
| Timestamp gaps | `irregularFrameTimingSelectsRealFramesNotTheFirstOnes`, `perFrameTimestampsDecideDurationAndFps` |
| Camera switch | `framesFromThePreviousCameraCannotEnterTheNewCapture` |
| Very short sign | `aShortSignStillYieldsAFullWindowOfItsSurroundingFrames` |
| Delayed tap | `aTapLongAfterTheSignSeesAnEmptyWindowAndIsNotRewardedWithTheOldSign` |
| Rapid repeated taps | `aSecondTapOnTheSameSignIsRefusedInsteadOfRepeatingTheAnswer`, `aTapAfterTheRepeatGapIsAcceptedAgain`, `theRepeatGuardIsOnlySetByAnAcceptedCapture` |
| Concurrent arrival during tap | `aFrameThatArrivesAfterTheTapIsExcludedEvenIfItIsInTheBuffer`, `postTapFramesAloneAreNeverEnoughToClassify` |

Plus sampling invariants: `theSampledWindowSpansTheSameTemporalStepAsTraining`,
`aThinBufferIsPassedThroughWithoutInventingFrames`.

### Instrumentation — 5 passed, 0 failed, on `emulator-5554`

- `CameraFeaturePathTest` — 2 passed
- `ProcessingStateRenderTest` — 1 passed
- `ResultCardRenderTest` — 1 passed
- `PreTapOnDeviceTest` — 2 passed (new): the tap decision returns in < 100 ms
  on-device, an empty ring is refused, and `clear()` forgets the repeat guard

`EnsembleInferenceTest` — 2 passed, confirming the four-member ensemble still
loads and runs.

### Model identity — unchanged

| Member | SHA-256 prefix |
| --- | --- |
| `landmark_ens_tcn21_fp16.tflite` | `112279bc86090470` |
| `landmark_ens_tcnm61_fp16.tflite` | `01a261d8eb6c2c4d` |
| `landmark_ens_tcnm62_fp16.tflite` | `27058a85e032bee3` |
| `landmark_ens_rob71_fp16.tflite` | `08417e72da59150a` |
| **Combined, meta order** | **`0e4c392732e7da7e77632620dc72e4afce8b1a5aa00234dbf7a13a1049a7ad93`** |

Matches `ModelContract.ENSEMBLE_SHA256` byte for byte. The vocabulary
(`models/landmark_labels.json`, 100 labels) is untouched.

### Lint — no new findings

`./gradlew :app:lintDebug` produced only the pre-existing advisory categories
(`DefaultLocale`, `GradleDependency`, `LockedOrientationActivity`, …). Nothing
from this change.

## 4. What is NOT verified

- **Recognition accuracy on a real phone.** Not measured at all. There is no
  device.
- **Tap-to-result latency on a real phone.** The 275.8 ms p95 in
  `LATENCY_ANALYSIS.md` is from warm desktop evaluation, not a phone. The new
  synchronous path removes the up-to-3.7 s wait that the old waiting branch
  could add, but the remaining inference time is unchanged and unmeasured on
  phone hardware.
- **Whether the pre-tap fix produces correct signs for a human signer.** It is
  verified to select the right *frames*; whether the model then classifies them
  correctly is the open question the phone test exists to answer.
- **Camera framing, lighting and distance effects.** Unmeasured on a phone.

## 5. The exact test that closes this

`docs/release/PHONE_GATE_INSTRUCTIONS.md` has the full sequence. In short:

```bash
adb devices -l
adb install -r dist/echogesture-v3-87efd29.apk
```

then Settings → Capture diagnostics → on, perform 10 signs × 5 conditions from
authoritative demonstrations, and:

```bash
adb shell run-as org.echogesture.finalapp cat files/diagnostics.txt > diagnostics.txt
python3 evaluation/device/accuracy_report.py diagnostics.txt annotations.txt
```

Until that produces a file, the honest verdict is **PHONE_EVIDENCE_BLOCKED**.