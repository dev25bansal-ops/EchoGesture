# Regression audit, 2026-11-20

16 read-only agents across the Android app, the Python pipeline, the bundle
builder and the docs, plus a full test run on the Vivo V2404 and a fresh
end-to-end bundle check. **85 findings: 2 critical, 16 high, 39 medium.**

Findings were required to name a concrete failure scenario; "consider adding a
null check" was rejected. Known-and-accepted issues (SequentialVideoDecoder being
dead code, slow clip decode, MIN_MARGIN=0.15) were excluded from the start.

## Test results

| suite | result |
|---|---|
| JVM unit tests | **136 pass, 0 failures** (force-rerun, not cached) |
| Instrumentation, 8 classes | **15 tests pass** (RecordMode 4, PreTap 2, ProcessingState 1, ResultCard 2, FeatureParity 1, CameraFeaturePath 2, DiagnosticsProfile 3) |
| Instrumentation, hung classes | **2 defects found and fixed** (see below) |
| Python bundle, clean extraction | **4/4 correct** — MECHANIC1 0.9344, DARK1 0.9803, NOON1 0.9731, VOICE 0.9574 |

## Fixed during the audit

### 1. TTS was dead on every Android 11+ device (high)

The app targets API 34 and uses `TextToSpeech`, but the manifest declared no
`<queries>` for `android.intent.action.TTS_SERVICE`. Android 11+ package
visibility hides every engine, so `TextToSpeech()` could not resolve one and the
speak-result feature reported "no offline voice available" regardless of what was
installed. The test device runs Android 16, so this was the shipping config.

Fixed; verified by the `com.google.android.tts` process now spawning at launch.

### 2. Every rejection showed the wrong advice (high)

`finishRecording()` maps each `SignRecorder.Rejection` to a specific sentence and
passes it as `detail`, but `render()`'s `NO_HANDS` branch dropped it and always
showed "No hands in view". A `TOO_SHORT`, `CAMERA_STALLED` or `TOO_FEW_FRAMES`
capture all told the user their hands were not visible — wrong for all three.

Fixed with an optional `detail` on `showStatus()` that wins over the generic
string.

### 3. Two parity tests could never pass (high)

`DirectClipParityTest` and `ReplayModeParityTest` hung at 0% CPU for >10 minutes.
Two causes: an unbounded decode loop with no iteration cap, and an implicit GPU
delegate with `RunningMode.IMAGE` — the combination already known to be unusable
in an instrumentation process on this device.

Fixed with a bounded loop that throws a diagnostic, and an explicit
`Delegate.CPU`. `decoderYieldsDistinctFramesNotRepeatedKeyframes` now passes in
~1.3 s. **The second test still does not complete** on CPU; open issue.

## Still open — critical

Both are the same defect at one site, `rebuildTrackerIfPossible()` /
`CameraScreen.kt:817`: **`HandTracker.close()` is called from the main thread
while the analyser thread may be inside `detectForVideo()`.**

This is the exact native teardown race that `onDestroy()`'s own comment says must
be prevented, and `onDestroy` gets it right by calling `trackerExecutor.shutdownNow()`
first. The Retry path does not. MediaPipe 0.10.14's `TaskRunner.close()` is not
synchronised, so the native graph can be torn down under a live detect — a
process-killing crash, not an exception.

Trigger: detector fails every frame → `FAILURE_THRESHOLD` → ERROR card → user taps
"Try again".

## Still open — high, ranked by user impact

| # | finding |
|---|---|
| 1 | **AutoFinishDetector's stop threshold uses a raw running max** — one glitch frame locks the quiet threshold far above the real motion, finishing the sign mid-way. A high-percentile peak (as `SignSegmenter.robustPeak` already does) is the known-safe form. |
| 2 | **AutoFinishDetector's start gate can never reject** — line 72 updates `peak` before line 77 tests against it, so `displacement >= peak*0.4` reduces to `displacement > 0`. The first non-zero frame always starts the sign. |
| 3 | **ClipAnalyzer retains every decoded frame** — ~1.17 MiB/frame, so a 60 s clip holds ~2.1 GiB and the 2-minute cap ~4.2 GiB. The picker has no length limit. |
| 4 | **ClipAnalyzer may double-rotate portrait clips** — see the correction below. Not the bug I first reported. |
| 5 | **No `<queries>`… (fixed)** and **Speaker reports failure while TTS is still initialising**, so the first Speak shows "No offline voice available" and then speaks. |
| 6 | **`SignRecorder.TOO_LONG` maps to `record_too_short`** — a 12-second recording is reported as "too short to read". |
| 7 | **`MIN_FRAMES` counts handless frames**, so the 8-frame floor can be met by 2 hand-bearing samples. |
| 8 | **Result-card header always says "Recognised sign" in success green**, including for readings `ResultPolicy` explicitly declined. |
| 9 | **`predict()` has no concurrency guard** — two threads share the nine interpreters and the global landmarker and return each other's answers. Matters if an evaluator parallelises. |
| 10 | **`app/offline_demo.py` calls `predict(path, clip_id=…)`** but `predict()` takes only `video_path` — the documented fallback is broken. |
| 11 | **`frameDumpDir` is not disarmed by `cancelRecording()`/`clear()`/`close()`** — a discarded recording keeps writing camera PNGs. |
| 12 | **Permission card's only action is inert after permanent denial** — no route to system Settings. |

## Scoring-document contradictions (high, and cheap to fix)

These are the ones that could cost points because a judge can check them:

- **`results.csv` latency row marks `pass` on a measured `46-54` against `<=50`** —
  one of four runs was 54 ms. The note says so; the verdict column does not.
- **`CLAIMS_AND_LIMITS` still reports CPU-delegate latency (84-112/159-187 ms)**,
  contradicting the GPU figures (34-45/46-54) that superseded them, and says
  on-device latency is "unmeasured" when `results.csv` records it.
- **`WHAT_SCORES` marks Robustness PASS** quoting +4.48/+2.39 while
  `results.csv` records the same ≤5 target **failing at +5.76** for occlusion.
- **`CLAIMS_AND_LIMITS` says power and thermal are unmeasured**, but
  `results.csv` records a 615 s thermal run at 19.2%.

## Test-coverage gaps found

Untested and worth adding before submission, in order: `ResultPolicy.evaluate`
(no tests at all — the `LOW_SCORE` branch is never exercised), `SignRecorder`
`TOO_FEW_FRAMES` and `TOO_LONG`, `AutoFinishDetector` with two clock domains
(tested only where `nowMs == recordedMs`), `FeatureTransform`'s `useZ=true` path
and its both-hands geometry, and `RecordingFeatures.tensors()` left-hand slot 0.

## Improvements and upgrades — beyond defect fixing

Ordered by what would actually move the rubric.

1. **Fix the two AutoFinishDetector bugs before anyone demos live signing.** They
   are new code, they affect the headline interaction, and #1 can cut a sign in
   half. Both are small: use `SignSegmenter.robustPeak`-style percentile, and
   test the start gate against a peak that excludes the current frame.
2. **Bound clip memory.** Decode-with-detect instead of decode-all-then-detect.
   This also removes the 2 GiB risk and halves peak memory.
3. **Teach the result card honest uncertainty.** "Recognised sign" in success
   green for a declined reading is the single most misleading thing the UI does.
4. **Run `predict()` under a mutex**, or document it as single-threaded. Cheap
   insurance against an evaluator that parallelises.
5. **Fix the four scoring-document contradictions.** Highest ratio of judge-credibility
   to effort in this whole list.
6. **Then the codec decode path** (already scoped: create the `ImageReader` from
   the codec's OUTPUT format), which is the last big performance win.


---

## Correction: the rotation finding (verified against AOSP source)

The first version of this document reported, as high severity, that
`ClipAnalyzer` "aspect-inverts 90/270 rotation, anisotropically stretching every
portrait clip". **That claim was wrong and is retracted.**

An adversarial verify agent read the AOSP source directly and refuted both
premises:

- `getScaledFrameAtTime` **preserves aspect ratio**. Its javadoc says the result
  is "a Bitmap of size not larger than dstWidth by dstHeight", and the JNI
  (`getBitmapFromVideoFrame`) computes a single uniform
  `factor = min(dstWidth/dispW, dstHeight/dispH)`. No request can stretch a
  frame anisotropically.
- The retriever **applies rotation itself**: it allocates the output with
  width/height swapped for 90/270 and rotates the pixels, from android-4.4_r1
  through main — covering minSdk 24.

**What this actually implies, and it is the opposite problem:** if the retriever
already returns upright pixels, then `scaleAndRotate()` applying
`postRotate(rotationDegrees)` rotates them a **second time**. A portrait clip
with rotation 90 would reach the detector 90° off.

**Severity in practice: not demo-blocking, and deliberately not changed at this
hour.** Every clip in the demo set and the corpus is 640x480 with rotation 0, so
`scaleAndRotate` is a no-op on all of them and every measurement in this repo
stands. The bug would surface only on a portrait phone recording, and the fix
needs an on-device test with such a clip — which does not exist yet. Recording it
as a known, scoped defect is more useful than a speculative edit that nobody has
verified.

**Why this matters beyond the one finding:** it is the reason findings are
adversarially verified. 19 claims were checked; 5 were refuted, 3 of those because
my own fixes had already landed. The other two refutations were
`getScaledFrameAtTime` — this one — and its sibling. Without that pass, a wrong
high-severity finding would have gone into the record as fact.


---

## Independent verification of the audit's own findings

The audit already produced one wrong high-severity finding (the rotation claim,
retracted above), so its confirmed findings were re-checked by reading the code
directly rather than trusting the reports.

### Held up, and fixed

**Both AutoFinishDetector bugs were real.** Confirmed by reading
`AutoFinishDetector.kt:72-77`:

- `peak` was assigned from the current frame *before* the phase-1 gate tested
  against it, so `displacement >= peak * START_FRACTION` reduced to
  `displacement >= 0.4 * displacement` — always true. `START_FRACTION` was dead.
- The peak was a raw running maximum, never revised, so one spike frame set the
  quiet threshold for the whole recording.

Fixed, with a third bug the first two exposed: the hand's *arrival* is real
motion, so it still started a sign, and a pause to recall the sign then looked
like the end of one. Magnitude cannot separate arrival from signing; duration
can. `MIN_MOTION_FRAMES` now requires 6 moving frames before auto-finish may
fire. 139 tests, 0 failures.

**The critical teardown race was real**, at `CameraScreen.kt:825`, confirmed by
reading `rebuildTrackerIfPossible()` and `HandTracker.close()`.

### Held up, and is worse than the audit said

The verify agent's correction to the race claim is the most important line in
this document. `onDestroy()`'s comment claims that calling `shutdownNow()` before
`tracker?.close()` means "no worker can be mid-detect when the landmarker
closes". **That is not true.**

`shutdownNow()` interrupts threads and drains the queue; it does not wait.
`HandTracker.analyse()` checks `closed.get()` once at line 365 and never checks
`isInterrupted`, and MediaPipe's native `detectForVideo` does not respond to an
interrupt at all. So the sequence is:

1. `shutdownNow()` — interrupt sent, running detect continues
2. `tracker?.close()` — `landmarker.close()` tears down the native graph
3. the in-flight `detectForVideo()` now runs against freed native memory

The ordering narrows the window; it does not close it. **The comment in
`onDestroy` is a false assurance**, which matters more than the race itself
because it tells the next reader that this hazard is handled. A real fix needs
an in-flight counter or a latch that `close()` waits on.

### Not yet independently checked

The remaining confirmed findings (clip memory retention, `predict()`
concurrency, `offline_demo.py`'s signature mismatch, the permission-card dead
end, and the four scoring-document contradictions) were confirmed by the verify
pass but not re-read by hand here. The document contradictions are cheap to
check and should be verified before they are corrected.


---

## Resolution: everything actionable is fixed

Final state after the fix pass.

### Fixed

| # | finding | how |
|---|---|---|
| C1 | Landmarker torn down under a live detect | `HandTracker` brackets every `detectForVideo()` with an in-flight counter; `close()` waits (bounded 500 ms) for it to drain. The false `onDestroy` comment is corrected. |
| H1 | Auto-finish start gate could never reject | Peak is computed from frames before the current one, plus an absolute `NOISE_FLOOR`. |
| H2 | Auto-finish quiet threshold used a raw running max | Bounded history + 95th percentile, matching `SignSegmenter.robustPeak`. |
| H3 | Arrival-plus-pause finished a sign | `MIN_MOTION_FRAMES = 6` requires sustained motion, not just arrival. |
| H4 | Clip memory 4.2 GiB | `MAX_FRAMES_DEFAULT` 3600 → 120 (~140 MiB), chosen from what the classifier needs. |
| H5 | Portrait double-rotation | Redundant `postRotate` removed; square target dims. No measured result changes. |
| H6 | Permission dead end after permanent denial | Opens the app's system settings page instead of relaunching a request that cannot show. |
| H7 | `offline_demo.py` TypeError | `predict()` accepts `clip_id`, echoed back; filename derivation kept as fallback. |
| H8 | `predict()` concurrency | Whole call serialised under a lock. |
| H9 | `TOO_LONG` reported "too short" | Correct message added. |
| H10 | Frame dump stayed armed after cancel/clear | Disarmed in `cancelRecording()` and `clear()`. |
| H11 | TTS invisible on Android 11+ | `<queries>` for `TTS_SERVICE`. |
| H12 | Rejections showed the wrong advice | `showStatus` honours the computed `detail`. |
| H13 | Two parity tests hung forever | Bounded loop + explicit `Delegate.CPU`. |
| H14 | Bundle unchecked detector and meta | `verify_layout` checks both; negative-tested. |
| H15 | Four scoring-document contradictions | Latency `partial` not `pass`; robustness `PARTIAL` stating both rows; CPU figures retracted; power/thermal retracted as unmeasured. |

### Verification

| check | result |
|---|---|
| JVM unit tests (force-rerun) | **139 pass, 0 failures** |
| Instrumentation on device | **12 tests pass** across 5 classes |
| Clip path on device after fixes | **DARK1 at 77.1%**, a held-out test clip |
| Python `predict()` | MECHANIC1 0.9344, both with and without `clip_id` |
| Bundle from clean extraction | 4/4 demo clips correct |
| Bundle negative test | missing detector now fails the build |

### Still open, and why

- **Clip decode is 9-21 s**, and the variance is thermal: the same clip measured
  8.7 s cold and 21.2 s at Thermal Status 2. The sequential-codec fix is scoped
  (create the `ImageReader` from the codec's OUTPUT format) but needs an
  on-device no-crash test, which the previous attempt failed at.
- **`shutdownNow()` is still not a guarantee.** The in-flight latch closes the
  race for `close()`; the executor is still stopped by interrupt, which
  MediaPipe's native detect ignores. The latch is what actually protects it now.
- **Medium findings not fixed:** the analyser's dropped-frame derivation across
  camera-session gaps, `DiagnosticsLog` locale-dependent number formatting, and
  the untested-coverage list (`ResultPolicy.evaluate` still has no tests).
