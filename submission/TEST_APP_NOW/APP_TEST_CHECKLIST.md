# EchoGesture — app test checklist

Print this or keep it on screen. One column is what I could verify without a
phone; the other is what only your phone can answer.

**APK:** `EchoGesture-Test-Ready.apk` · `6653160e…d37c`

---

## Part 1 — What is already verified (emulator, Android 14)

These were exercised on the rebuilt APK by tapping the live UI. You do not need
to re-test them unless something looks different on your hardware.

| # | Item | Verified |
| --- | --- | --- |
| 1 | APK installs | ✓ |
| 2 | App cold-launches to the camera screen | ✓ 2876 ms |
| 3 | All nine models load without error | ✓ |
| 4 | Footer reads `9-model ensemble · FP16 TFLite · SHA-256 995cc8d0b6f6…` | ✓ |
| 5 | Camera preview renders live | ✓ |
| 6 | Camera permission prompt appears | ✓ |
| 7 | Denying it shows a clear card with a retry button | ✓ |
| 8 | Re-requesting after denial works and resumes the preview | ✓ |
| 9 | Start sign begins a recording | ✓ |
| 10 | A recording with no hands is rejected with "No hands in view" | ✓ |
| 11 | It does not guess when there is nothing to read | ✓ |
| 12 | Glossary shows `100 of 100 signs` | ✓ |
| 13 | Glossary search filters the list | ✓ |
| 14 | Glossary entry opens with Speak / Copy | ✓ |
| 15 | History shows the correct empty state | ✓ |
| 16 | Settings renders all six controls | ✓ |
| 17 | Framing guide toggle visibly changes the camera screen | ✓ |
| 18 | All four tabs reachable; Back returns to Camera | ✓ |
| 19 | App survives backgrounding and returns | ✓ |
| 20 | App survives landscape→portrait rotation | ✓ |
| 21 | Zero crashes in the crash buffer | ✓ |
| 22 | No INTERNET permission in the compiled manifest | ✓ |

---

## Part 2 — What only your phone can answer

Mark each. This is the part that matters.

### Install and launch

```
[ ] Installs without error
[ ] Opens to the camera screen (not a blank screen)
[ ] Camera permission asked; "While using the app" works
[ ] Preview appears within a few seconds
[ ] Footer reads "9-model ensemble"
```

### Recognition — the core of the test

Try at least 5 signs. **Record every attempt, including the wrong ones.**

```
Sign 1: ____________   Got: ____________   Score: ____%   [ ] correct [ ] wrong
Sign 2: ____________   Got: ____________   Score: ____%   [ ] correct [ ] wrong
Sign 3: ____________   Got: ____________   Score: ____%   [ ] correct [ ] wrong
Sign 4: ____________   Got: ____________   Score: ____%   [ ] correct [ ] wrong
Sign 5: ____________   Got: ____________   Score: ____%   [ ] correct [ ] wrong
```

```
[ ] Start sign changes to Finish sign and shows a Recording banner
[ ] Finish sign produces a result card
[ ] Card shows a sign name, score, and "Classifier took N ms"
[ ] Four alternatives appear under the result
[ ] "Recognise again" starts a fresh recording
[ ] "Clear" empties the result
[ ] A second sign does not reuse the first result
[ ] Hands leaving frame mid-sign gives "No hands in view", not a guess
[ ] A very short recording is rejected with a message
```

### Latency

```
Classifier time shown on the card: ______ ms
```

That number is the classifier only. The full clip-to-result time is longer.
The brief's target is ≤50 ms; the laptop measured ~236 ms for the whole
pipeline. Whatever your phone shows is the real number — please note it.

### Other screens

```
[ ] History shows the recognitions from this session
[ ] Settings: all six controls respond
[ ] Framing guide off → overlay gone on Camera; on → it returns
[ ] Result text size changes the size of the recognised sign
[ ] Haptic feedback off → no vibration on capture
[ ] Speak results on → sign is read aloud
[ ] Backgrounding the app and returning keeps it working
[ ] Rotation does not crash the app
```

### Stability

```
[ ] No crash during the whole session
[ ] Preview does not freeze
[ ] No "app not responding" dialog
```

---

## Part 3 — If something goes wrong

Report it like this:

```
Device:
Android version:
APK hash: 6653160ea3321c35ef19b45da3f69801557e3d598739ff3f4cb526171a93d37c
Screen:
Action performed:
Expected:
Actual:
Screenshot/video:
Can reproduce (always / sometimes / once):
Severity (blocker / major / minor):
```

Crash log (phone connected by USB):

```bash
adb logcat -d > ~/Desktop/echogesture-crash.txt
```

---

## What would make this a failed test

Being honest about the bar:

- **A crash on a real phone** — the emulator never crashed, so a crash there is a
  real finding.
- **The camera never opening on real hardware.** The emulator logs a CameraX
  `CameraIdListIncorrectException` because it has no real camera; if that also
  appears on your phone, the app is not usable and it is a blocker.
- **A result card that never appears** even with a clean hand-in-frame recording.
- **A wrong sign at high confidence** — that is useful, not a blocker, and is
  exactly the kind of thing that needs reporting.

## What would not be a failed test

- Low confidence scores. The model knows 100 signs out of a much larger
  vocabulary; anything outside those 100 maps to the nearest one it knows.
- Wrong answers. This is a research prototype with a measured laptop accuracy of
  77.02% on the corpus test split. Getting things wrong is expected.
- "No hands in view" when your hands are half out of frame. That is the app
  working.
