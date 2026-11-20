# Window placement: investigated and cleared

Date: 2026-11-19
Status: **closed. Window placement is not the cause, and the shipped segmenter
should not be changed.**

> **Correction, first.** An earlier version of this document claimed the user's
> correctly-signed ELEVATOR clip "returns ELEVATOR1 at 49.9% with 100% hand
> coverage when windowed correctly", and concluded the app had analysed the wrong
> part of the recording. **That replay was invalid.** The screen recording runs at
> about 2× speed (the app's own `Recording N.Ns` counter advances 0.51 s per 1 s
> of video), so every window was half the duration it was labelled, and every
> velocity feature was ~2× too high. Re-run on the app's own clock, the replay
> does not reproduce the phone's answer in either orientation.
>
> The conclusion below — that the shipped segmenter is the best of the policies
> tested — rests on a separate, valid measurement and stands.

## The invalid evidence, kept for the record

The replay that originally supported this document, **now known to be wrong**.
The windows below are labelled in video seconds; the recording runs at ~2× speed,
so each is really about half as long in the app's own clock, and the velocity
features are ~2× too high:

```
window            top1            score  margin   cov  runner-up
------------------------------------------------------------------
  2.9-6.6  s     CEMETERY        34.3%   0.061   19%  TAIL1:28%
  4.4-8.1  s     CHEW1           18.6%   0.066   59%  ELEVATOR1:12%
  5.9-9.6  s     ELEVATOR1       39.2%   0.272   97%  CHEW1:12%
  7.4-11.1 s     ELEVATOR1       49.9%   0.216  100%  CHEW1:28%
  8.8-12.5 s     ELEVATOR1       22.3%   0.004   78%  CHEW1:22%
 10.3-14.0 s     CHEW1            9.8%   0.043   38%  SLIDE2:6%
 11.8-15.5 s     HURDLE/TRIP1    24.1%   0.048    6%  PARTY1:19%

best: ELEVATOR1 at 7.4-11.1s (49.9%, margin 0.216, cov 100%)
```

`C:/Python314/python.exe .scratch/user_video_scan.py`

All of the above is **superseded** — see the correction at the top. It is kept
because it is the exact reasoning that led to the wrong conclusion, and the
lesson is in how it was wrong: a screen recording of a mirrored preview,
transcoded at an unknown speed, re-cropped by hand, is not a faithful
reconstruction of the `ImageProxy` the model reads.

## Why the app's window was wrong

This section's conclusion is also **withdrawn**. The recording durations below are
real (they come from the app's own diagnostics), but nothing establishes that the
window landed beside the sign — the replay that appeared to show it does not hold
up.

The phone's own diagnostics for that session:

```
frames=32 dur_ms=4500   pred=SHAVE1
frames=32 dur_ms=7729   pred=SHAVE1
frames=32 dur_ms=7948   pred=SHAVE1
frames=32 dur_ms=12096  hand_frames=0
```

A 7.7 s recording, and the sign occupies roughly 5.9–11.1 s of a 15.9 s clip.
`HandTracker.beginCapture()` takes `CAPTURE_SPAN_MS = 3700` of real time
**ending at the tap**. If the tap comes well after the sign, or the recording
started well before it, the fixed trailing window lands beside the sign rather
than on it.

`SignSegmenter` exists to fix exactly this, and it does fire — but it is
unconditional-trim-only: `SignSegmenter.kt` records the reasoning, and the
measured alternatives (216 evaluations) were 131/216 for no trim, 180/216 for
the trim it uses, 133/216 for a kept-fraction floor, 146/216 for a gated trim.
It removes leading and trailing near-static runs. It does **not** recentre a
window around the largest motion, and it cannot recover if the tap comes long
after the sign has finished and the intervening frames hold other movement.

## The second effect: mirroring

The same window, mirrored (i.e. as the user sees it in the preview), gives a
completely different and much more confident answer:

```
  4.4-8.1  s     CHEW1           89.0%   0.870   59%  ELEVATOR1:2%
  5.9-9.6  s     CHEW1           87.1%   0.824   97%  ELEVATOR1:5%
  7.4-11.1 s     CHEW1           55.7%   0.165  100%  ELEVATOR1:39%
```

A mirrored frame is a valid but *different* input. `src/landmark_frontend.py`
states the reference never swaps handedness and the live demo does not mirror;
`HandTracker.toHands()` passes MediaPipe's handedness label straight through.
So the analysis path is un-mirrored and correct — but this means **a user
watching the mirrored preview is seeing a different image from the one being
classified**, and if they adjust their hand to match the preview they are
adjusting to the mirror. That is worth knowing for the demo script, even though
the pipeline itself is doing the right thing.

## What this does not explain

The CHEEK → RAZOR2 report is not explained by this alone, and this document
should not be read as claiming it is. That case came in at 80.7% top-1 with a
margin of at least 0.614; a misplaced window usually produces the *low*-coverage,
low-margin profile above, not a confident wrong answer. The two reported
failures may have different causes. What is now established is that at least
the ELEVATOR case is window placement, and that the user's signing and the
model both behave correctly once the window is right.

## Proposed fix — MEASURED AND REJECTED

The proposal was to replace the fixed trailing span with a motion-centred one.
It was measured and **the shipped code already does better**, so nothing changes.

Three policies, 300 test clips, each embedded in a 120-frame (~12 s) recording
with the sign surrounded by idle. `segmenter` is a faithful port of the shipped
`SignSegmenter.choose()`; `motion` is the proposed fixed-length motion window.

```
idle before       whole  segmenter   motion
-------------------------------------------
50%              58.2%      87.2%    84.2%
75%              55.6%      87.2%    84.2%
```

`.scratch/segmenter_faithful.py`

- The shipped segmenter is the **best of the three**, at 87.2% in both
  placements.
- The proposed motion window is **2 points worse** and would have been a
  regression.
- Not trimming at all is catastrophic (≈56%), which independently confirms
  trimming is worth having.

Note the difference from the earlier table in this document: that one compared
a plain *trailing* window against a motion window, and the trailing window is
**not what the app does**. Record mode calls `RecordingFeatures.sample()` →
`SignSegmenter.choose()`, which keeps `[first moving frame, last moving frame]`
plus 0.5 s of pad. Comparing against a trailing window flattered the proposal.

**Conclusion: leave `SignSegmenter` alone.** Two earlier attempts to improve
this area were also measured and rejected (the `>=45%` keep-floor and the gated
trim; see the header of `SignSegmenter.kt`). The component has now survived
three independent challenges and should not be touched again without a
measurement that beats 87.2%.

## Also worth telling the user for the demo

Tap **Finish** promptly after the sign ends. The app's own guidance says so
("Tap Finish as soon as the sign ends"), and the recording above is the case
that guidance is about: 15.9 s for a sign that takes about 3 s.

## Reproduction

```
C:/Python314/python.exe .scratch/user_video_cropped.py   # crop + mirror comparison
C:/Python314/python.exe .scratch/user_video_scan.py      # the sliding window table
```

Input: `C:\Users\dev25\Downloads\WhatsApp Video 2026-11-19 at 4.43.55 AM.mp4`.
The preview crop is derived from the in-app bounds `[44,270][1036,1262]` on a
1080-wide screen, scaled by 720/1080 to this recording.
