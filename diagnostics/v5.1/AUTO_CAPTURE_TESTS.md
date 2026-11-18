# AUTO_CAPTURE_TESTS

## What was implemented

A one-tap capture state machine, `AutoSegmenter`, ported and wired into the live
analyser path.

```
IDLE -> ARMED -> SIGNING -> PROCESSING -> RESULT/RETRY
```

The signer taps **Start Recognition** once. The segmenter watches the analysed
frames, decides where the sign began and ended, and the outcome is classified
automatically. No second tap is needed.

**Manual fallback is retained.** While armed the button reads "Stop waiting" and
cancels; while signing it reads "Finish now" and ends the attempt immediately.

## Design constraints that came from measurements, not taste

- **Time, not frame counts.** The brief's numbers are milliseconds, and the
  analyser rate varies per device, so every threshold is milliseconds.
- **Motion is relative to a noise floor** measured while the signer waits, not a
  fixed constant. Landmarks are normalised by hand size, so a fixed threshold
  means different physical motion at different distances.
- **Static signs are handled.** A stable held pose arms a sign, so a handshape
  held still is not ignored for lack of movement.
- **Internal pauses do not end a sign.** The end test requires quiet
  *continuously*; any motion restarts the timer. This protects two-part signs.

## Tests

`AutoSegmenterTest` — 15 tests, all passing (JVM):

| Test | What it pins |
| --- | --- |
| does not arm a sign from frames before the tap | no pre-tap leakage |
| a lone detection does not start a sign | acquisition confirmation |
| sustained motion starts a sign | the normal onset |
| a held static pose also starts a sign | static-sign support |
| a static sign still yields a usable segment | it survives to classification |
| a sign does not start before the arming delay | settling time |
| a completed sign auto-stops after the quiet period | **the core one-tap promise** |
| a pause inside a two-part sign does not stop it | two-stage signs |
| the sign eventually stops after a long pause | no infinite capture |
| a sign that never goes quiet times out | bounded attempt |
| a hand leaving the frame ends an interrupted sign | dropout handling |
| consecutive signs do not share frames | no stale contamination |
| slow 5 fps input still segments without a frame-count trigger | time-based logic |
| manual finish works even while a sign is running | **the required fallback** |

`OneTapCaptureTest` — 6 tests, all passing (instrumentation, real `HandTracker`):

| Test | What it pins |
| --- | --- |
| oneTapProducesAnOutcomeWithNoSecondTap | the whole promise, end to end |
| completedOutcomeIsConsumedOnce | a sign is never classified twice |
| reArmingDiscardsThePreviousSignsFrames | no frame sharing |
| disarmProducesNoOutcome | cancel is clean |
| armThenCancelReturnsToIdle | **regression: armed state had no exit** |
| manualFinishWorksDuringASign | fallback through the tracker |

## The bug found while testing this

While armed, the primary button's handler was `UiState.AUTO_ARMED -> Unit`. The
button read **"Stop waiting"** but did nothing. A user who tapped Start by
mistake was trapped: the only way out was to leave the tab.

Found by actually tapping the button on the emulator, not by reading the code.
Fixed in `cancelAutoCapture()` and pinned by `armThenCancelReturnsToIdle`.

## Verified UI transitions on the device

| Action | Screen |
| --- | --- |
| launch | "Searching for hands" / **Start Recognition** |
| tap Start Recognition | **"Ready — perform one sign"** / Stop waiting |
| tap Stop waiting | back to "Searching for hands" / Start Recognition |

No crash, process alive throughout.

## What is NOT verified

**A real sign completing the automatic stop on a real phone.** The emulator's
camera never contains a hand, so the automatic end could only be driven by
synthetic frames through the tracker — which is what `OneTapCaptureTest` does.
That proves the state machine and its wiring. It does not prove that a person
signing into a phone camera crosses those thresholds.
