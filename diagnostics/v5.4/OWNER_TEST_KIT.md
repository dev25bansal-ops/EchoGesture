# OWNER TEST KIT

No physical phone was attached to the build machine during this session, so
every live-camera claim remains unverified. This kit exists so the test can be
done without a cable, and so the result is precise enough to act on.

## Option A — no ADB at all (preferred)

1. Install `EchoGesture-V5.4.apk` on the phone.
2. Open EchoGesture → allow camera.
3. **Settings → Capture diagnostics → on.**
4. Go to Camera. Tap **Start Recognition** once. Perform one sign.
5. Note what the screen showed.
6. Repeat for 5 signs, 3 attempts each. Fill in `PHONE_TRIALS.csv`.
7. **Settings → Export diagnostics…** → choose Mail, Drive, or "Copy to
   clipboard", and send the file back.

The export contains capture statistics only: frame counts, timings, presence
flags, predicted glosses. No images, no landmarks, no identifiers. It leaves the
phone only because you chose a destination.

## Option B — with ADB

```bash
sh collect-phone-diagnostics.sh
```

It refuses to run against an emulator, prints the model and camera-permission
state, and writes the record beside itself.

## What the exported record tells me

Each attempt is one line plus a `profile` line:

```
frames=32 dur_ms=2100 fps=15.2 hand_frames=18 pred=BEE1 score=0.9421
  margin=0.8100 end_reason=QUIET seg_ms=1800 mask_steps=20/24
  both_hands=14 longest_dropout=2 slot_flips=0
profile fps=29.4 interval_p50=34 analyse_p50=13 delivered=684 dropped=0 limit=CAMERA
```

Reading it:

| Symptom | Field | Category |
| --- | --- | --- |
| `hand_frames` low, `longest_dropout` high | detector lost the hands | A — camera/detector |
| `seg_ms` far shorter than the sign, `end_reason=HAND_LOST` | segmenter cut it | B — boundary |
| `slot_flips` high | left/right assignment is unstable | A or C |
| `limit=ANALYSER`, low `fps` | the app is the bottleneck | A |
| healthy capture, confident wrong gloss | model generalisation | D |
| confident wrong gloss that Android and Python disagree on | mapping | E |

**One session of 15 attempts with this file is enough to name the category.**
Without it, any fix would be a guess.

## Privacy

- Nothing is uploaded automatically. The app has no INTERNET permission.
- The export is a deliberate tap.
- No camera frames, no landmark arrays, no identifying information are written
  at any point.
