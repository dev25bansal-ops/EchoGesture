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
| 1.6 | Only CAMERA is requested; no internet permission | pass (manifest) |

### 2. Camera and permissions

| # | Check | Expected |
| --- | --- | --- |
| 2.1 | First launch shows the system camera permission prompt | prompt appears; no other permission is requested |
| 2.2 | Denying permission leaves the app usable | app stays open, states clearly that the camera is off; no crash |
| 2.3 | Granting permission shows a live preview | preview fills the frame guide, upright and correctly rotated |
| 2.4 | Rotating the phone | preview stays upright; no stretched or mirrored image |

### 3. Live recognition

Perform each sign held steady inside the frame guide for a full capture window.

| # | Check | Expected |
| --- | --- | --- |
| 3.1 | No hands in frame, then capture | "No hands" state, not a fabricated prediction |
| 3.2 | Recognise a sign from the glossary | one gloss shown, confidence badge, top-5 alternatives, latency line |
| 3.3 | Recognise a different sign | a different gloss; recognition is not stuck on one class |
| 3.4 | Move your hands quickly out of frame mid-capture | no prediction from an empty window |
| 3.5 | Repeat 3.2 three times | same gloss each time, confidence varies within a plausible band |

Record the gloss shown, the confidence percentage and the reported latency for
3.2 and 3.3. If a gloss is wrong, record what it predicted and what you signed —
a wrong-but-plausible gloss is a model question for Dev, not an app defect.

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

### 7. Stability

| # | Check | Expected |
| --- | --- | --- |
| 7.1 | Roughly 20 consecutive recognitions | no crash, no ANR, latency does not climb |
| 7.2 | Switch tabs during processing | no crash; the result is not attributed to the wrong tab |
| 7.3 | Background the app mid-capture and return | camera resumes; no crash |

## Recording the result

For each numbered check record pass / fail / not run, and for any failure the
exact on-screen text plus `adb logcat -d -s EchoGesture:* AndroidRuntime:E`.
Leave the outcome and the evidence in the handoff notes. Until every row in
sections 2–7 is recorded as a pass, the verdict for this branch is
`PHYSICAL_CAMERA_VERIFIED` = **pending**, regardless of the emulator results.