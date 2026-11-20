# The analyser rate: measured, and ruled out

Date: 2026-11-19
Status: **complete. The frame rate is not the cause of the wrong answers.**

This document exists because the slow analyser was the leading hypothesis for
the confidently-wrong readings, and it is worth keeping as a record of how it
was tested and why it failed — three candidate causes have now been eliminated
and the remaining one is named at the end.

## What the phone reported

The app's `AnalysisProfiler`, pulled off the device during the failing session:

```
frames=32 dur_ms=4500   fps=6.7   hand_frames=29
frames=32 dur_ms=7729   fps=4.0   hand_frames=30
frames=32 dur_ms=7948   fps=3.9   hand_frames=29
frames=32 dur_ms=12096  fps=11.1  hand_frames=0

profile fps=9.3  interval_p50=108 analyse_p50=104 convert_p50=4 detect_p50=101 detect_p95=169 dropped=6826 limit=ANALYSER
```

`convert_p50 = 4 ms`; `detect_p50 = 101 ms`. **The MediaPipe detector is the
entire cost.** The app analyses **4–10 frames per second** and drops **76–82%**
of what the camera delivers, and it labels its own bottleneck `limit=ANALYSER`.

## The accuracy cost, measured (with corrected rate labels)

**A first version of this section printed the wrong rates and the correction
matters**, so it is stated plainly. The cached clips are **32 samples spread over
a ~3.7 s recording**, so the corpus rate is **32/3.7 ≈ 8.6 Hz**, not 30 fps.
Keeping every Nth sample gives N-th fewer points over the *same* duration, so
the true rate of each row is 8.6/N Hz:

```
keep    samples    true rate
   1         32        8.6 Hz   <- corpus
   2         16        4.3 Hz   <- the phone's range
   3         11        3.0 Hz
   4          8        2.2 Hz
   6          6        1.6 Hz
```

Measured top-1, 248 clips, full nine-member ensemble
(`py -3.11 -u -I .scratch/rate_honest.py` reports these with the correct labels;
`.scratch/fps_effect_fast.py` produced the same numbers under the wrong ones):

| samples | true rate | top-1 |
|---|---|---|
| 32 | 8.6 Hz (corpus) | **85.1%** |
| 16 | 4.3 Hz | **83.1%** |
| 11 | 3.0 Hz | 79.4% |
| 8 | 2.2 Hz | 81.5% |
| 6 | 1.6 Hz | 66.9% |

**The honest reading: at the phone's own rate (~4.3 Hz) the loss is about 2
points, not 18.** The catastrophic 67% row corresponds to 1.6 Hz, which is well
below what the device delivered. Halving the corpus rate costs little; cutting
it to a sixth costs a great deal.

That substantially weakens the rate hypothesis as the explanation for the
reported errors. It remains a real finding — the device is analysing far more
slowly than it should — but it is **not** established as the cause of a
confident wrong answer, and this document should not be read as claiming it is.

## Why the rate matters

`src/features.py` computes velocity as a **plain per-frame difference with no
`dt` normalisation**:

```python
vel[1:] = norm[1:] - norm[:-1]
acc[2:] = norm[2:] - norm[:]
```

The step between consecutive samples *is* the time base. Two-thirds of the
feature vector (2 of the 3 blocks after the positions) is pure motion, and its
magnitude scales directly with how far apart in real time consecutive samples
are.

The corpus is **32 samples spread across a whole ~3.7 s clip = 8.6 Hz effective**.
The app's quick-scan path deliberately reproduces this by taking a fixed 3.7 s
span and resampling 32 points across it — the `CAPTURE_SPAN_MS` comment explains
this at length, and it is correct.

**Record mode does not reproduce it.** `RecordingFeatures.sample()` takes up to
32 points across the *segmented* interval. At an analysed rate of 4 fps a 7.7 s
recording yields ~31 analysed frames in total; after the segmenter keeps the
moving part (~60%) that is **~18 real points across the sign**. The model then
upsamples 18 points to 24 steps.

The distortion is easy to get backwards, so here it is numerically. Corpus step
= 3700/31 ≈ **119 ms**. A 4 fps record covering a ~3 s sign in 18 points gives
3000/17 ≈ **176 ms** — **1.48× coarser**. Since:

```python
vel[1:] = norm[1:] - norm[:-1]      # one step of displacement
acc[2:] = norm[2:] - norm[:-2]      # two steps of displacement
```

a coarser step means each consecutive sample covers **more** motion, so velocity
is **inflated ~1.48×** and acceleration **~2.1×** relative to what training saw
for the same physical movement. It is not a loss of data: the *scale* of two of
the three feature blocks is wrong.

**But repairing that scale does not recover the accuracy.** Directly tested:
interpolating the low-rate samples back up to 32 points before building features,
which restores the step size, changes nothing.

```
rate                as-is   timebase    delta
---------------------------------------------
32 pts (8.6 Hz)     85.1%      85.1%    +0.0
16 pts (4.3 Hz)     83.1%      83.5%    +0.4
11 pts (3.0 Hz)     79.4%      79.4%    +0.0
 8 pts (2.2 Hz)     81.5%      81.5%    +0.0
 6 pts (1.6 Hz)     66.9%      66.9%    +0.0
```

`.scratch/timebase_fix.py`

So the velocity-scale story is **not** the mechanism. Interpolation can restore
the sample count but not the information that was never captured; the loss at
1.6 Hz is lost detail, not a mis-scaled feature block. That is worth knowing
because it rules out the cheap fix.

## Verdict: the frame rate is NOT the cause either

The decisive measurement is the app's **actual** path. `RecordingFeatures` does
not feed N samples to the model — it takes up to 32 points and then lets
`FeatureTransform` interpolate them to 24 steps, which normalises the step
regardless of how many points arrived. Testing that path directly:

```
 real pts      rate    top-1
-------------------------------
       18     4.9Hz    85.1%   <- what 4 fps gives after the segmenter trims
       24     6.5Hz    85.1%
       31     8.4Hz    87.1%
       32     8.6Hz    85.1%   <- corpus
```

**18 real points scores 85.1% — identical to the corpus.** The low analyser rate
costs nothing measurable on this path, because the upsample to 24 steps restores
the step size that `features.py` differences over.

The alarming 66.9% row from the earlier table applies only to feeding 6 points
directly, which the app never does.

`.scratch/rate_honest.py`

## A correction to my own replay

An earlier replay of the failing recording sized its windows in **video**
seconds. The screen recording runs at about **2× speed** — the app's own
`Recording N.Ns` counter advances ~0.51 s per 1 s of video, read off extracted
frames. So every window labelled 3.7 s was really ~1.85 s of signing, and every
velocity feature was ~2× too high. **That replay was invalid**, and its
"ELEVATOR1 at 49.9%" result must not be quoted.

Re-run on the app's own clock (`.scratch/app_clock_replay.py`):

```
UNMIRRORED                        MIRRORED
app win  video  top1              app win  video  top1
  3.7     7.3   CHEW1    15.8%      3.7     7.3   CHEW1  77.9%
  4.5     8.8   ELEVATOR1 15.4%     4.5     8.8   CHEW1  89.7%
  3.7    15.1   TAIL1    20.4%      3.7    15.1   CHEW1  61.8%
```

Even on the correct clock the un-mirrored replay does not reproduce the phone's
confident `SHAVE1`, and the mirrored version is confident but wrong. **The phone
agrees with neither.** A screen recording captures what the *preview* showed;
recovering the analysis image from it needs the preview's exact orientation and
scale, which is not established. Too many transformations separate the two.

**The replay approach cannot settle this.** That is stated plainly rather than
papered over.

Four hypotheses have now been measured and rejected:

| hypothesis | verdict | evidence |
|---|---|---|
| Feature-transform port bug | **rejected** | Kotlin matches Python to 2.4e-07; identical predictions |
| Window placement | **rejected** | shipped `SignSegmenter` 87.2% vs 84.2% for the proposed replacement |
| Low analyser rate / velocity scale | **rejected** | 18 real points scores 85.1%, same as corpus |
| Filming a laptop screen | **rejected** | hand size 1.00×, jitter 0.74× vs corpus |

**All four shared the same flaw: every one started from already-extracted
landmarks or from a hand-reconstructed image.** None exercised the step where
MediaPipe turns the phone's actual pixels into landmarks. If the error is
introduced there, no test downstream can see it — which is exactly the pattern
seen: four distinct transforms each proven harmless, while the device keeps
producing wrong answers.

## What the phone was pointed at

The recording shows the phone filming **a laptop screen playing a corpus video**,
and the geometry was measured:

| | corpus median | screen-filmed |
|---|---|---|
| hand size (wrist→middleMCP, normalised) | 0.1491 | 0.1496 (**1.00×**) |
| landmark jitter | 0.0126 | 0.0093 (**0.74×**) |

`.scratch/screen_filming_degradation.py`

So filming a screen did **not** shrink the hand in frame and did not add jitter —
`features.py` normalises by wrist→middleMCP, which is exactly designed to remove
hand size. This hypothesis is also **refuted**. It remains good advice not to
film a screen for the demo, but it is not the explanation.

## What this means for the demo

Film the person directly, not a screen. If the demo *must* use corpus clips
because a live signer is unavailable, play the video and film it is the worst
possible presentation — the degradation is real and the model is being blamed
for it.

## What would settle it — and the one thing that would not

**Would settle it:** run the phone's own analysis frames through `src/main.py`
end to end from pixels. The app would need a debug-only dump in
`analyserCallback()` — it already converts each `ImageProxy` to a bitmap at
`HandTracker.analyse()`, so writing one JPEG per N frames is a small, contained
change behind the existing diagnostics toggle. Then the exact pixels the model
saw can be scored twice: once by the app, once by the reference pipeline. Any
disagreement localises the fault precisely.

**Would not settle it, and has now been tried twice:** replaying the failing
attempt through a screen recording. The recording is ~2× speed, captures the
mirrored preview rather than the analysis image, and needs a hand-guessed crop.
Each of those is an error source as large as the effect being investigated.

## Bottom line

The model, the feature transform, the segmenter and the frame rate are all
individually sound and measured to be so. **The failing input has never been run
through the reference pipeline.** Until it is, every explanation here is a
rejected candidate rather than a found cause, and the honest status of the
"wrong answers" complaint is **open**.

It does **not** by itself say which repair works. Two are under test in
`.scratch/timebase_fix.py`; the result is pending and must be read before either
is implemented.

## The fixes that were considered and are NOT needed

**Raise the analyser rate.** The detector is the cost
(`CameraScreen.bindUseCases()` requests `.setTargetResolution(Size(640, 480))`).
This would still be worth doing for responsiveness — 4–10 fps makes the live
preview feel sluggish — but it is **not** needed for accuracy, since 18 real
points already scores 85.1%. And it carries a measured risk: the code comment
records the detector finds **41.4%** of frames with a hand at native resolution
versus **37.6%** at a 256×256 squash, so resolution was deliberately raised.

**Normalise the time base.** Tested and does nothing (delta +0.0 at every rate,
see above). The upsample to 24 steps inside `FeatureTransform` already restores
the step size.

Neither is a fix for the reported wrong answers, because neither addresses the
cause.

## Do not touch SignSegmenter

Measured today, 300 test clips embedded in 12 s recordings:

```
idle before       whole  segmenter   motion
-------------------------------------------
50%              58.2%      87.2%    84.2%
75%              55.6%      87.2%    84.2%
```

`.scratch/segmenter_faithful.py`

The shipped segmenter is the best of the three policies. A proposed
motion-centred window was 2 points worse and was rejected. Not trimming at all
is catastrophic. This component has now survived three independent challenges
and should be left alone.

## Reproduce

```
py -3.11 -u -I .scratch/fps_effect.py          # accuracy at 30/15/10/7.5/5/3.75 fps
py -3.11 -u -I .scratch/segmenter_faithful.py  # segmenter vs whole vs motion window
```

Device records: `docs/ON_DEVICE_LATENCY_2026-11-19.md`.
