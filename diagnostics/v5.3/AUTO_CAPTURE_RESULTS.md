# AUTO_CAPTURE_RESULTS

## Status: unchanged from V5.2, still passing

No Android source was modified this session, so the one-tap implementation is the
V5.2 build byte-for-byte. Its evidence carries forward unchanged:

| Suite | Count | Result |
| --- | --- | --- |
| `AutoSegmenterTest` (JVM) | 15 | pass |
| `OneTapCaptureTest` (instrumentation) | 9 | pass |
| Full JVM suite | 146 | 0 failures |
| Full device suite (V5.2 run) | 35 | 0 failures |

## Verified UI transitions

| Action | Screen |
| --- | --- |
| launch | "Searching for hands" / **Start Recognition** |
| tap once | **"Ready — perform one sign"** / Stop waiting |
| tap Stop waiting | back to idle |
| auto-complete | PROCESSING → RESULT or RETRY, no second tap |

Manual Finish remains available while signing.

## §13 scenario coverage

All twelve scenarios have a passing test: normal movement, no hands, static sign,
two-stage sign, brief dropout, low FPS, camera switch, backgrounding, rapid
repeated starts, uncertain result, max duration, repeated valid signs.

## The double-trim question, settled

§5.3 of the brief notes that V5.2 measured the current second trim as better
(26/30 vs 25/30). Independently re-confirmed this session through the shipped
Kotlin samplers on 30 clips: **trim 0.867, no-trim 0.833**. The existing path is
kept. `sampleWithoutTrim` remains only as the control arm for that measurement.

## What is still not verified

**A real person signing into a real phone.** The emulator camera never contains a
hand, so the automatic end has only ever fired on synthetic frames driven through
the real tracker. The state machine is proven; the thresholds have never met a
live signer.
