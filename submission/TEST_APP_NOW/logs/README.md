# Logs

## `crash_buffer.txt` — 0 bytes, and that is the result

This is a verbatim `adb logcat -d -b crash` capture, taken **after** the full
manual session against the rebuilt APK `6653160e…d37c` — launch, recording
attempt, all four tabs, settings toggle, backgrounding, and rotation.

It is empty because the buffer contained nothing. Grepping it for
`echogesture` also returns 0. **Zero app crashes.**

An empty file here is not a missing capture. It is the finding.

## `rebuilt_launch_logcat.txt`

The app's own process log (`adb logcat -d --pid=<app pid>`), 373 lines, covering
the cold launch. What it shows:

- `Initialized TensorFlow Lite runtime` and `Created TensorFlow Lite XNNPACK delegate`
- Per-member delegate replacement: `Replacing 272 out of 272 node(s)` and
  `Replacing 165 out of 165 node(s)` — the two member widths (255 and 381)
- `CameraX … CameraIdListIncorrectException: Expected camera missing from device`

### About that CameraX exception

It is **an emulator artifact, not an app fault.** This AVD was booted with
`-camera-back emulated -camera-front none`, so CameraX cannot enumerate a
matching camera id and logs the exception on open. The preview still runs, and
every screen works. Real hardware enumerates its cameras normally, so this line
is not expected on your phone — but it is recorded here rather than hidden,
because if you do see a camera that never opens on real hardware, that is a
blocker and I want the evidence trail to be honest about what the emulator
already reported.
