# AUTO_CAPTURE_TEST_RESULTS

## §13 scenario coverage

| # | Scenario | Test | Result |
| --- | --- | --- | --- |
| 1 | Normal movement → auto stop → one classification | `oneTapProducesAnOutcomeWithNoSecondTap`, `a completed sign auto-stops after the quiet period` | PASS |
| 2 | No hands → timeout → retry, no classifier call | `a sign that never goes quiet times out`, no-hand path returns NO_HANDS before `runInference` | PASS |
| 3 | Static sign | `a held static pose also starts a sign`, `a static sign still yields a usable segment` | PASS |
| 4 | Two-stage sign, must not stop at internal pause | `a pause inside a two-part sign does not stop it` | PASS |
| 5 | Brief dropout, must not stop early | `a hand leaving the frame ends an interrupted sign` | PASS |
| 6 | Low FPS, irregular timestamps | `slow 5 fps input still segments without a frame-count trigger` | PASS |
| 7 | Camera switch | `cameraSwitchDiscardsTheAttemptRatherThanSplicing` | PASS |
| 8 | App backgrounding | `backgroundingSuspendsTheAttemptSafely` | PASS |
| 9 | Rapid repeated starts | `rapidRepeatedStartsKeepOneAttempt`, `arming twice does not splice two attempts` | PASS |
| 10 | Uncertain result shows retry, not a gloss | `lowMarginResultIsShownButLabelledUncertain` (rewritten) | PASS |
| 11 | Max duration, bounded memory | `and a sign that never goes quiet times out` + `MAX_BUFFERED_FRAMES` cap | PASS |
| 12 | Repeated valid signs, no stale frames | `reArmingDiscardsThePreviousSignsFrames`, `consecutive signs do not share frames`, `completedOutcomeIsConsumedOnce` | PASS |

## Totals

| Suite | Count | Result |
| --- | --- | --- |
| JVM (`AutoSegmenterTest`, `RecordingFeaturesTest`, others) | 146 | 0 failures |
| Instrumentation (`OneTapCaptureTest` alone) | 9 | 0 failures |

## Verified on the device (manual UI walkthrough)

| Action | Result |
| --- | --- |
| Launch | "Searching for hands" / **Start Recognition** |
| Tap Start Recognition | **"Ready — perform one sign"** / Stop waiting |
| Tap Stop waiting | back to idle |
| Process alive, crash buffer | clean |

## Two bugs found during this work

**1. Armed state had no exit.** The button read "Stop waiting" and the handler
was `Unit`. A user who tapped Start by mistake was trapped until they left the
tab. Found by tapping the button on the device, not by reading the code. Fixed
and pinned by `armThenCancelReturnsToIdle`.

**2. Unbounded capture buffer.** See `AUTO_CAPTURE_IMPLEMENTATION.md`.

## A third finding: I was wrong about the double trim

I changed the one-tap path to skip the motion trim, on the strength of a Python
stand-in. Measured on the real Kotlin segmenter the trim is **better**
(0.867 vs 0.833). The change was reverted.

## What these tests do NOT prove

That a real person signing into a real phone camera crosses these thresholds
correctly. Every test above either drives synthetic frames through the real
tracker or walks the UI on an emulator whose camera never sees a hand.
