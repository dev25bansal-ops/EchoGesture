# The camera is not the problem. The capture window is.

Measured 2026-11-20 on the Vivo V2404 with a real camera frame dump, in answer
to *"why does the clip work but the camera not?"* and *"sometimes correct,
often wrong, and low score when correct."*

**The headline result overturns the framing of the question.** The camera's
pixels classify correctly. What fails is how many of them reach the model.

## 1. The dumped camera frames classify at 0.97

Five real frames captured through the app's own `HandTracker.analyse()` path
(diagnostics on, one Start/Finish recording), run through the shipped
nine-member pipeline:

```
dumped frames: 5, 960x960
frames with a hand: 2 / 5
hand bbox: x[0.46,0.63] y[0.51,0.74]  (17% x 23% of frame)

prediction:  VOICE   conf=0.9678   margin=0.9622
agreement:   9/9
gate:        ACCEPT
```

The scene was a hand in front of a laptop screen — the exact "do not film a
screen" case the test guide warns about — and it still scored **0.9678 with all
nine members agreeing.**

**So image quality, background, angle and hand size were never the cause.** Every
hypothesis in `WHY_CAMERA_MISSES_2026-11-19.md` that pointed at the pixels is
superseded by this.

## 2. What actually fails: frames per capture

The same session's own diagnostics:

| capture | frames | duration | fps | hands seen |
|---|---|---|---|---|
| first | 32 | 5697 ms | 5.4 | 11/32 (34%) |
| second | **4** | 1540 ms | **1.9** | 2/4 |

The second capture gave the model **four frames**. The analyser also reported
`delivered=689 dropped=279` — 29% of frames dropped.

That single fact produces both reported symptoms:

- **Correct but low score.** Enough frames land for the right answer, but with a
  thin margin, because the model sees a sparse sample of the motion.
- **Wrong answer.** So few frames land that the window holds a fragment of the
  sign, or mostly idle, and the model names whatever that fragment resembles.

`presence=101111000000000000000000011111000` in the first capture shows the
second half: the hand is detected, disappears for two-thirds of the window, then
returns. `SignSegmenter` trims to a span that may not contain the whole sign.

## 3. Why the clip path never hits this

`ClipAnalyzer` decodes every frame of the file at a fixed rate. It cannot drop
frames and cannot run at 1.9 fps. **That is the entire asymmetry** — not the
pixels, not the model, not the geometry.

## 4. A concrete cause, and the fix

`CameraScreen` asked for `setTargetResolution(640, 480)`. The dumped frames came
back **960 x 960** — a square, 1.5x larger on each axis.

`setTargetResolution` is a **hint**, not a constraint: Android's `ImageAnalysis`
picks the closest supported size, and this device offers a square one that
satisfies "at least 640". A square frame is also the wrong *shape*: the corpus
is 640x480 (4:3 landscape), and the detector letterboxes what it is given.

The cost is direct. 960x960 is **2.25x the pixels** of 640x480, paid twice per
frame — once in `toBitmap()` and once in `detectForVideo()`. Per-frame detect
cost is the currency the analyser is short of, so this is a plausible
contributor to running at 1.9 fps.

Replaced with a `ResolutionSelector` carrying an explicit 4:3 aspect-ratio
strategy and a `ResolutionStrategy` at 640x480 with
`FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER`, which pins the shape rather than
leaving it to the device.

**Not yet verified on hardware** in this document: the dump that would confirm
640x480 is now delivered needs one more recording. The build installs and runs.

## 5. What this means for the earlier documents

- `WHY_CAMERA_MISSES_2026-11-19.md` refuted frame *rate*, frame *count*,
  timestamps and tracking mode on **clip pixels**, and concluded the remaining
  cause was domain shift in the image. That conclusion is now **wrong**: the
  camera's own pixels classify at 0.97. The refutations of timestamps and
  tracking mode still stand; the domain-shift conclusion does not.
- `NON_SIGNING_HANDS_2026-11-19.md` (a still hand gets a gloss 43% of the time)
  is unaffected and is a separate real defect.

## 6. What to do about it

1. **Verify the resolution fix** with one more dump — frames must come back
   640x480, and the fps in `profile` should rise.
2. **Raise the analyser rate.** With 2.25x fewer pixels per frame there is
   headroom; if fps is still below ~10 on this device, the next lever is
   dropping `NUM_HANDS` to 1 (the demo signs are one-handed) or capping the
   analysis size further.
3. **Reject short captures.** A capture with 4 frames over 1.5 s should be
   refused as unusable, not classified. `SignRecorder` already has `TOO_SHORT`
   and `NO_FRAMES`; what is missing is a **minimum frame count** independent of
   duration, which is what actually failed here.

## Reproduce

```
# arm: Settings -> diagnostics ON (already set)
# in the app: Start sign -> perform -> Finish sign
adb exec-out run-as org.echogesture.finalapp tar -c files/framedump > dump.tar
```
