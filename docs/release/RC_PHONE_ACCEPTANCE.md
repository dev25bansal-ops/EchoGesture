# Release-candidate phone acceptance

Branch: `hackathon/echogesture-v2-rc` (from `abhijeet-integration` @ `d67d468`,
plus the four PR #2 corrections).

## Status: NOT EXECUTED — no physical device was attached

At the time of this build the only attached device was `emulator-5554`
(`sdk_gphone64_arm64`, Android 14 / API 34, 1080×2400, density 420). No physical
Android phone was visible to `adb devices`. Everything below that requires a real
camera therefore remains **unverified**, and this document must not be read as a
pass.

What the emulator *did* establish is listed in step 1 so the phone run can
concentrate on what the emulator cannot do.

## Why the emulator cannot close this

The emulator's virtual camera is a synthetic scene with no hands. Every
`connectedDebugAndroidTest` run exercises the camera plumbing — rotation
reporting, tracker acceptance, missing-hand frame positions — but no test ever
sees a real hand in a real frame. A passing instrumentation suite is not
evidence of a working live-camera prediction.

## Prerequisites

- The debug APK from this branch, built with the complete pinned asset set.
  It contains restricted weights; sideload it, do not redistribute it.
- USB debugging on, phone unlocked, and `adb devices` shows one physical
  device next to (or instead of) the emulator.
- Good indoor light. The model was trained on recorded clips; low light and
  motion blur are the most likely causes of a low score.

### Connecting the phone from a Mac

No driver is needed on macOS. The phone is the only side that has to be
configured, and the order matters — USB debugging must be switched on *before*
the Mac can see the device.

1. **Unlock the phone** and open **Settings → About phone**. Tap **Build number**
   seven times; Android replies that developer options are unlocked.
2. **Settings → System → Developer options** → turn on **USB debugging**.
   On Samsung it is **Settings → About phone → Software information → Build
   number**, then **Developer options** under Settings. On Xiaomi it is
   **Settings → About phone → MIUI version** seven times, then **Additional
   settings → Developer options → USB debugging**, and also **Install via USB**.
3. Connect the phone to the Mac **with a data cable**. Charge-only cables are a
   common cause of "nothing appears" — try a different cable before anything else.
4. **Pull down the notification shade on the phone** and tap the USB notification.
   Choose **File transfer / MTP** (not "Charging only").
5. A dialog appears on the phone: **Allow USB debugging?** with the Mac's RSA
   fingerprint. Tick **Always allow from this computer** and tap **Allow**.
   If no dialog appears, tap **Developer options → Revoke USB debugging
   authorisations**, unplug, and reconnect.
6. Confirm from the Mac:

```bash
adb devices -l
```

A physical device appears as a serial that is **not** `emulator-5554`, with
`device` (not `unauthorized` or `offline`). `unauthorized` means step 5 was not
accepted. `offline` usually means the cable or the USB mode.

7. Record the identity this run is made against:

```bash
adb shell getprop ro.product.model
adb shell getprop ro.build.version.release
adb shell getprop ro.build.version.sdk
adb shell getprop ro.product.cpu.abi
adb shell pm list features | grep -E "camera|flash|autofocus"
```

If several devices are attached, prefix every command with `-s <serial>` so the
emulator cannot receive the install.

Install:

```bash
adb install -r app/android_final/app/build/outputs/apk/debug/app-debug.apk
```

## Checks

Run in order and record the outcome of each. A failure at any step stops the
run; report the step number and the on-screen text.

### 1. Already covered on the emulator — re-check only that it still holds

| # | Check | Emulator result |
| --- | --- | --- |
| 1.1 | App launches to the Camera tab, no crash | pass |
| 1.2 | All four tabs switch: Camera, Glossary, History, Settings | pass |
| 1.3 | Glossary lists 100 signs, search filters, detail sheet opens | pass |
| 1.4 | Settings renders every control; toggles respond | pass |
| 1.5 | History shows the empty state and its explanation | pass |
| 1.6 | Only CAMERA is requested; no internet permission | pass (installed manifest) |
| 1.7 | Instrumentation suite: 8 tests, 0 failed, 0 skipped | pass |
| 1.8 | JVM suite: 30 tests, 0 failed, 0 skipped | pass |

None of the above involved a hand in a real frame. They establish that the app
starts, renders and navigates; they establish nothing about recognition.

### 2. Camera and permissions

| # | Check | Expected |
| --- | --- | --- |
| 2.1 | First launch shows the system camera permission prompt | prompt appears; no other permission is requested |
| 2.2 | Denying permission leaves the app usable | app stays open, states clearly that the camera is off; no crash |
| 2.3 | Granting permission shows a live preview | preview fills the frame guide, upright and correctly rotated |
| 2.4 | Rotating the phone | preview stays upright; no stretched or mirrored image |
| 2.5 | Tap the flip button | preview switches between front and rear; the choice survives a relaunch |
| 2.6 | Switch back to the original camera | preview returns; hands are still detected |
| 2.7 | Hold one hand in the guide, then both | "Hands detected" then "Ready to recognise"; the pill tracks reality |
| 2.8 | Tap the torch button on the rear camera | torch toggles; no crash on a device without a torch |

2.5 matters for the model, not just the UI: a camera rebind starts from a
cleared frame ring, so a frame from the previous camera must never reach the
ensemble.

### 3. Live recognition

Perform each sign held steady inside the frame guide for a full capture window.

| # | Check | Expected |
| --- | --- | --- |
| 3.1 | No hands in frame, then capture | "No hands" state, not a fabricated prediction |
| 3.2 | Recognise a sign from the glossary | one gloss shown, confidence badge, top-5 alternatives, latency line |
| 3.3 | Recognise a different sign | a different gloss; recognition is not stuck on one class |
| 3.4 | Move your hands quickly out of frame mid-capture | no prediction from an empty window |
| 3.5 | Repeat 3.2 three times | same gloss each time, confidence varies within a plausible band |
| 3.6 | **Time capture → visible caption with a stopwatch, 3 runs** | record each in milliseconds; compare with the on-screen latency line |
| 3.7 | Tap **Copy** after a prediction | "Copied" toast; pasting elsewhere yields the gloss |
| 3.8 | Tap **Recognise again** | a new capture starts; the previous result is replaced |
| 3.9 | After a "No hands" result, return your hands and capture | recovery to a normal result with no relaunch |
| 3.10 | Recognise at least 5 different supported glosses | record each: signed vs predicted vs confidence |

3.6 measures the product-level number the user experiences. The on-screen
latency line reports inference time only, so the stopwatch value will be higher —
record both rather than substituting one for the other.

3.10 is the false-recognition test. Every mismatch is recorded as a failure and
reported; a plausible wrong gloss is a model question for Dev, not an app defect,
but it is still a failure of the feature.

### 4. History and privacy

| # | Check | Expected |
| --- | --- | --- |
| 4.1 | After a recognition, open History | the gloss, its score and the time are listed |
| 4.2 | Navigate away and back | history survives within the session |
| 4.3 | Clear saved history from Settings | list empties; the empty state returns |
| 4.4 | Force-stop and relaunch the app | history is empty — nothing persisted to disk and no video was saved |
| 4.5 | Inspect the app's files directory | contains no captured frames or video |

`adb shell ls -R /sdcard/Android/data/org.echogesture.finalapp/files/` should show
no image or video files after a recognition.

### 5. Speech

| # | Check | Expected |
| --- | --- | --- |
| 5.1 | Turn "speak results" on and recognise a sign | the gloss is spoken aloud, offline |
| 5.2 | Turn it off and recognise again | nothing spoken |
| 5.3 | Airplane mode on, recognise a sign | recognition and speech still work |

5.3 is the offline proof. If speech fails in airplane mode the device has no
offline voice installed — record that as a device limitation, not an app
failure, and confirm the text path still works.

### 6. Settings persistence

| # | Check | Expected |
| --- | --- | --- |
| 6.1 | Change result text size, haptics and speech | each takes effect immediately |
| 6.2 | Force-stop and relaunch | the three choices are remembered |
| 6.3 | Change the preferred camera in Settings, relaunch | the camera you picked is the one that opens |

### 7. Glossary

| # | Check | Expected |
| --- | --- | --- |
| 7.1 | Search for a gloss by its first letters | the list filters to matches; the count reads "N of 100 signs" |
| 7.2 | Search for something not in the 100 | "No sign matches “…”" with a Clear search action, not a crash |
| 7.3 | Tap Clear search | the full 100 return |
| 7.4 | Open a gloss detail sheet | the gloss, "One of the 100 output classes" and the Speak/Copy actions render |
| 7.5 | Tap Copy in the sheet | "Copied" toast; the gloss pastes elsewhere |

### 8. Stability and resources

| # | Check | Expected |
| --- | --- | --- |
| 8.1 | Roughly 20 consecutive recognitions | no crash, no ANR, latency does not climb |
| 8.2 | Switch tabs during processing | no crash; the result is not attributed to the wrong tab |
| 8.3 | Background the app mid-capture and return | camera resumes; no crash |
| 8.4 | Record `dumpsys meminfo` before and after 20 recognitions | memory does not climb run over run |
| 8.5 | Leave the app idle on the camera screen for 5 minutes | no thermal throttle, no crash, battery drain is not abnormal |

For 8.4:

```bash
adb shell dumpsys meminfo org.echogesture.finalapp | grep -E "TOTAL PSS|TOTAL RSS"
```

Run it once before the first recognition and again after the twentieth. The two
totals belong in the report.

## Recording the result

For each numbered check record pass / fail / not run, and for any failure the
exact on-screen text plus `adb logcat -d -s EchoGesture:* AndroidRuntime:E`.
Leave the outcome and the evidence in the handoff notes. Until every row in
sections 2–7 is recorded as a pass, the verdict for this branch is
`PHYSICAL_CAMERA_VERIFIED` = **pending**, regardless of the emulator results.