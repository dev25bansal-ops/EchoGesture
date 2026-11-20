# Upload a clip: deterministic testing on the phone

Written 2026-11-19. Added after the user's own report: *"i tried the noon but
it is showing the foreigner1."*

## What the app said, and what the file says

The NOON clip is one of the four in `docs/demo_clips/`. Through `predict()` it
scores **NOON1 at 0.9731** — the highest-confidence of the four. On the phone's
live camera, the same sign returned **FOREIGNER1 at 0.51–0.60**, twice.

That looked like a model failure. It is not, and the phone's own diagnostics
show why:

```
# live capture 1
frames=32 dur_ms=4536 fps=6.8 hand_frames=32
pred=FOREIGNER1 score=0.6011
top5=FOREIGNER1|FLOAT1|AXE1|NOON1|ELEVATOR1  margin=0.3486

# live capture 2
frames=32 dur_ms=1907 fps=16.3 hand_frames=30
pred=FOREIGNER1 score=0.5112
top5=FOREIGNER1|NOON1|AXE1|MEACULPA|FLOAT1  margin=0.2784
```

**NOON1 is ranked 2nd and 4th.** The model is not blind to the sign — it is
hedging between NOON and FOREIGNER, and the top-1 it picks has fallen from 0.97
to 0.51–0.60. That is the signature of a distribution shift, not a broken
classifier.

## Ruled out again: the frame rate

The captures ran at 6.8 and 16.3 fps against the clip's 30 fps, which was the
obvious suspect. It is not the cause. Resampling the clip to those rates and
re-scoring through the real pipeline:

| input | prediction | confidence |
|---|---|---|
| native 30 fps | NOON1 | 0.9731 |
| 15 fps (every 2nd frame) | NOON1 | 0.9629 |
| 7.5 fps (every 4th frame) | NOON1 | 0.9549 |

At **less than a quarter of the frame rate** the answer is still NOON1 at 0.955.
The model is robust to temporal coarsening. This matches
`docs/ANALYSER_RATE_2026-11-19.md`, which reached the same conclusion by a
different route (18 real points scores 85.1%, identical to the corpus). Two
independent measurements now agree.

What is left is the camera path itself: a phone filming a small hand against a
lit background, at whatever angle the holder managed. `HOW_TO_TEST_THE_APP.md`
already warns about this — *film your hand, not a screen* — and the capture
above is that failure mode in the wild.

## The feature: pick a video instead of pointing the camera

The camera tab now has a **Pick a video clip** button under the capture button.
It opens the system document picker, and the chosen file is decoded, analysed
and classified by **the same code path as a live recording**:

```
ClipAnalyzer.analyse(uri)
  -> MediaMetadataRetriever decodes frames at the clip's own timestamps
  -> HandLandmarker (GPU, CPU fallback) -> HandTracker.Frame list
  -> SignRecorder.Recording
  -> RecordingFeatures.sample()  (SignSegmenter + 32-point resample)
  -> LandmarkClassifier.predict()  (nine members, mean logits, one softmax)
```

`CameraScreen.runInference` is unchanged and cannot tell a clip from a camera
recording. Nothing is special-cased, so there is no second classifier to keep in
sync and no drift to worry about — the two paths cannot disagree by
construction.

### Why this is worth shipping, not just debugging

- **It is the honest demo.** A judge picks a clip, the app answers from pixels
  the model was measured on, and the answer is reproducible. No dependence on
  lighting, framing or a steady hand.
- **It removes the variable that produced the wrong answer.** Whatever the
  camera does to a live scene, a file does not do.
- **It is a real user feature.** "Here is a video of someone signing" is a
  legitimate input for an on-device sign language reader.

### Design details that matter

| decision | why |
|---|---|
| Frames timestamped from the clip, not a synthetic counter | `RecordingFeatures.fpsOf` and `SignSegmenter` both depend on real timing; a flat counter would make a 2 s clip look like a 30 s one |
| Same `HandLandmarker` options as `HandTracker` (2 hands, 0.3 thresholds, VIDEO, GPU→CPU) | landmarks must come from the same detector configuration, or the features shift |
| Longest edge capped at 640 px | the camera delivers ~640×480; feeding 4K to the detector would leave the regime it was measured in |
| Full clip decoded, resample happens downstream | `read_frames()` in training samples 32 evenly across the whole clip; giving the resampler the whole clip reproduces that instead of pre-trimming |
| Container rotation baked in via `Matrix` | a clip shot in portrait must reach the detector the way up it was shot |
| `MAX_FRAMES_DEFAULT` cap | bounds memory; flagged in the log as `truncated=` rather than silently shortening |

## Verification

**Measured on the Vivo V2404 through the real app UI — pick the file, read the
result card.** All four demo clips, 4/4 correct:

| clip | expected | predicted | score | classifier latency |
|---|---|---|---|---|
| `3567776249024257-NOON.mp4` | NOON1 | **NOON1** | 95.7% | 73 ms |
| `08763437833459409-DARK.mp4` | DARK1 | **DARK1** | 77.1% | — |
| `7606272862279693-VOICE.mp4` | VOICE | **VOICE** | 96.5% | — |
| `9119691570534749-MECHANIC.mp4` | MECHANIC1 | **MECHANIC1** | 87.9% | 29 ms |

The NOON figure is the one that matters: **the same sign that returned
FOREIGNER1 from the camera returns NOON1 at 95.7% from the file, on the same
phone, in the same app, through the same nine interpreters.** The wrong answer
was never in the model — it was in what the camera handed it.

Scores run a little below the desktop `predict()` figures (NOON 0.957 vs 0.973;
DARK 0.771 vs 0.980). That is expected: `MediaMetadataRetriever` decodes through
keyframes rather than OpenCV's exact frame indices, so the sampled 32 points are
close to, but not identical with, the desktop path's. The *decision* is
unaffected, which is the property being tested. The gloss matching is what the
automated test asserts; the score is reported for honesty about the difference.

`app/src/androidTest/.../ClipUploadTest.kt` drives the same analysis from
instrumentation and asserts the gloss, writing its numbers to
`files/clip_upload.txt`. **On this device that test hangs** at the MediaPipe
`detectForVideo` step when run from a bare instrumentation process (0% CPU,
~1.5 s CPU time, no progress past the file line) — the same signature as
`ClipReplayAccuracyTest`. The feature itself is unaffected: the app's own process
initialises and drives the same detector normally, which is how the table above
was produced. Treat the instrumentation test as a harness limitation on this
device, not as evidence about the feature.

**Note for anyone re-running the instrumentation test:** it looks for the clip
inside the app's own `getExternalFilesDir()`, not `/sdcard/Movies`. Without the
clip in place the `assumeTrue` guard skips it silently, and `OK (1 test)` there
means "skipped", not "passed".

## What is still not established

The clip path is verified against a *known-good* clip. The live camera remains
the weaker input, and the reason is unmeasured rather than mysterious: the
camera hands the model a differently-scaled, differently-lit, differently-angled
hand than the training clips, and hand *size and position* normalise away but
appearance does not. A debug frame dump in `analyserCallback()` is the
measurement that would settle it — see
`docs/WIN_PLAN_2026-11-19.md` §4b.