# How to install and test EchoGesture on your phone

Written for you — no Android Studio needed.

> **This is a debug test build, not a store release.** It is not distribution-signed
> and will not auto-update. Install it only on a device you control, and only the
> file matching the checksum below.

**APK:** `EchoGesture-Test-Ready.apk`
**SHA-256:** `6653160ea3321c35ef19b45da3f69801557e3d598739ff3f4cb526171a93d37c`
**Size:** 86,851,453 bytes

---

## 1. The file is on this Mac already

Reveal it:

```bash
open -R ~/Desktop/EchoGesture-Test-Ready.apk
```

Verify it before you move it anywhere:

```bash
shasum -a 256 ~/Desktop/EchoGesture-Test-Ready.apk
```

It must print exactly:

```
6653160ea3321c35ef19b45da3f69801557e3d598739ff3f4cb526171a93d37c
```

**If it does not match, stop.** Do not install a file whose checksum you could not
confirm. Tell me and I will rebuild it.

A second copy lives at
`~/AppData/EchoGesture/submission/TEST_APP_NOW/EchoGesture-Test-Ready.apk`.

The APK is deliberately **not** on GitHub: it is 83 MB and contains model weights
whose redistribution rights are unresolved. The folder is the delivery.

### What is actually inside it

| Component | Size |
| --- | --- |
| Nine FP16 ensemble members | **8.51 MB** |
| MediaPipe hand detector (`hand_landmarker.task`) | **7.82 MB** |
| Native libs for four ABIs, app code, resources | remainder |
| **Whole APK** | **86.85 MB** |

The APK is large mainly because it carries TFLite and MediaPipe native libraries
for four CPU architectures (`arm64-v8a`, `armeabi-v7a`, `x86`, `x86_64`). Your
phone uses one of them.

---

## 2. Install it — use ADB, not Finder

**Do not rely on the phone appearing in macOS Finder.** Many Android phones do not
mount as a writable disk over USB on modern macOS. Use ADB.

### 2a. Check whether ADB is already on your Mac

```bash
which adb || echo "ADB NOT INSTALLED"
```

If that prints `ADB NOT INSTALLED`, this project already has a copy:

```bash
export PATH="$HOME/AppData/ActionMesh-fork/android-toolchain/sdk/platform-tools:$PATH"
adb version
```

### 2b. Enable USB debugging on the phone

1. **Settings → About phone**
2. Tap **Build number** seven times — *"You are now a developer"*
3. **Settings → System → Developer options**
4. Turn on **USB debugging**
5. Connect by USB. The phone asks *"Allow USB debugging?"* — tick
   **Always allow from this computer**, tap **Allow**.

### 2c. Confirm the phone is authorised

```bash
adb devices -l
```

You need the word `device`, not `unauthorized` or `offline`:

```
List of devices attached
R58XXXXXXXX     device usb:3-1 product:... model:SM_G991B ...
```

| What you see | Meaning | Fix |
| --- | --- | --- |
| `unauthorized` | Prompt not accepted on the phone | Unlock the phone, find *Allow USB debugging?*, tap **Allow** |
| `offline` | Connection wedged | `adb kill-server`, then `adb devices -l` again; try another cable or port |
| Nothing listed | Charge-only cable, or debugging off | Use the phone's original cable; recheck 2b |
| `no permissions` | Linux udev rules | Not applicable on macOS |

### 2d. Install

```bash
adb install -r ~/Desktop/EchoGesture-Test-Ready.apk
```

Expected: `Success`

| Error | Meaning | Fix |
| --- | --- | --- |
| `INSTALL_FAILED_UPDATE_INCOMPATIBLE` | A different build with the same package name is installed | `adb uninstall org.echogesture.finalapp`, then install again |
| `INSTALL_FAILED_INSUFFICIENT_STORAGE` | Needs ~90 MB free | Free space, retry |
| `INSTALL_FAILED_OLDER_SDK` | Below Android 7.0 (API 24) | Unsupported |
| `adb: failed to read ...` | Wrong path | Re-run the `shasum` in §1 and use the exact path it printed |

### 2e. Launch it

```bash
adb shell am start -n org.echogesture.finalapp/.MainActivity
```

Or tap the **EchoGesture** icon in the app drawer.

### 2f. Confirm the installed file is the one you hashed

```bash
adb shell pm path org.echogesture.finalapp
```

Then hash the installed copy:

```bash
adb shell sha256sum "$(adb shell pm path org.echogesture.finalapp | sed 's/package://' | tr -d '\r')"
```

It must match the same checksum. This proves the phone is running the exact file
that was tested, not a leftover from an earlier install.

---

## 3. If you would rather not use a cable

Emailing `.apk` files is often blocked and most cloud drives are awkward. If you
do use one:

- **Google Drive** — upload from the Mac, open Drive on the phone, download, open
  from **Files → Downloads**.
- Grant install-from-this-source **only** to the file manager or Drive app you are
  actually using, and only for this file.

When Android shows *"For your security, your phone is not allowed to install
unknown apps from this source"*:

1. Tap **Settings** on that prompt
2. Enable **Allow from this source** for that one app
3. Go back and tap **Install**

If **Play Protect** warns you, **read the warning first**. It appears because the
APK is not distributed through the Play Store, which is expected here. Confirm the
§1 checksum matches before choosing **Install anyway**. Do not dismiss a warning
you have not read.

---

## 4. Grant the camera

Camera is the **only** permission this app requests. It has no internet permission
— confirm in **Settings → Apps → EchoGesture → Permissions**, where Internet will
not be listed.

On first launch, tap **While using the app**.

If you tap **Don't allow**, the app shows a "Camera permission needed" card with an
**Allow camera access** button. Tap it. If the dialog does not reappear, open
**Settings → Apps → EchoGesture → Permissions → Camera → Allow**, then return.

---

## 5. Wait for the models

Nine models load on a cold start. You will see **"Preparing models"**, then
**"Searching for hands"**.

Check the footer on the Camera screen. It must read:

> **9-model ensemble · FP16 TFLite · SHA-256 995cc8d0b6f6…**

If it says anything other than **9-model**, stop and tell me — that would be a real
bug.

---

## 6. Learn what it can recognise

Tap **Glossary**. You should see **100 of 100 signs**.

Type in the search box to filter. Tap a sign for a sheet with **Speak** and
**Copy**.

These are the model's *output class names* — they tell you which sign was
recognised, not what it means.

### Five signs to start with

All are in the vocabulary and are one-handed, which holds steadier:

```
APPLE      BOOK      CANDY      COFFEE      PHONE
```

Also available: `BANANA  CHAIR  COMPUTER  DOG1  DRINK  FISH  FLOWER  HAT  HOUSE  KEY  LIGHT  MILK  MONEY  PIZZA`

---

## 7. Recognise a sign

1. Tap **Camera**.
2. Hold the phone so your **hand is fully inside the guide box** and your upper body
   is visible. Even light; do not point at a bright window.
3. Tap **Start sign**. The button becomes **Finish sign** and a red **Recording**
   banner appears.
4. **Perform one complete sign**, start to finish.
5. Tap **Finish sign** as soon as the sign ends.
6. Wait about a second.

### What a successful result looks like

A result card appears with:

- the recognised sign in large text
- a **model score** percentage
- **"Classifier took N ms on this device"**
- how many frames were captured, over how long
- four **alternatives**
- **Copy**, **Speak**, **Recognise again**, **Clear**

### How to tell a sign was REJECTED

A rejection is a *different* card: an amber warning triangle and a short
instruction, with **no sign name**. Examples:

| Message | Cause |
| --- | --- |
| **No hands in view** — "Keep both hands inside the frame guide and try again." | MediaPipe found no hand in the recorded frames |
| A too-short message | Recording was under the minimum duration |
| A too-long warning | Recording ran past the expected sign length |

A rejection means **the app did not guess**. That is correct behaviour, not a bug.

### What the score means

The score ranks the 100 known signs. It is **not** a calibrated probability that
the reading is correct. A 97% score does not mean it is right 97% of the time.

**Record every attempt, including the wrong ones.** Failures are more useful to me
than successes.

---

## 8. History and Settings

**History** keeps recognitions in memory for the session only, unless you enable
persistence in Settings.

**Settings** has six controls:

| Control | What should happen |
| --- | --- |
| Preferred camera | Switching Front/Back changes which camera opens |
| Framing guide | Turning off removes the hand-placement box from the preview |
| Result text size | Small / Medium / Large changes the result sign size |
| Haptic feedback | Turning off stops the vibration on capture |
| Speak results | Turning on reads each recognised sign aloud |
| Keep history on this device | Turning on makes History survive an app restart |

---

## 9. When something goes wrong

| Symptom | What to check |
| --- | --- |
| Opens to a blank screen | Capture logcat (§9a) and send it |
| "No camera detected", or preview stays black | Another app may hold the camera — close it and reopen. Confirm the permission in Settings |
| Preview works but no prediction ever appears | Confirm the footer says **9-model**. Try a longer, clearer recording with the whole hand in frame |
| Every recording rejected with "No hands in view" | Check lighting and that the hand is fully inside the guide. If it persists in good light, that is a real finding — send the log |
| Crashes on startup | Capture the crash log (§9a) |
| Footer hash is not `995cc8d0b6f6…` | You are running a different build. Stop and tell me |

### 9a. Capture a log safely

```bash
adb logcat -d > ~/Desktop/echogesture-log.txt
```

Crash only:

```bash
adb logcat -d -b crash > ~/Desktop/echogesture-crash.txt
```

**Before sending**, check the file for anything personal — device name, location or
account identifiers can appear in system log lines. Trim those.

No cable? **Settings → Developer options → Take bug report**, then send the file.

### 9b. Capture a screenshot

On the phone: **Power + Volume Down**.

With a cable:

```bash
adb exec-out screencap -p > ~/Desktop/screenshot.png
```

---

## 10. How to report a wrong prediction

The most valuable thing you can send me.

```
Device model:
Android version:
APK hash:  6653160ea3321c35ef19b45da3f69801557e3d598739ff3f4cb526171a93d37c
Sign performed:        (ground truth)
Sign the app showed:   (prediction)
Model score:           __%
Alternative 2:         (name and score)
Frames captured:       __ over __ ms
Classifier time:       __ ms
Accepted or rejected:
Repeatable?            always / sometimes / once
Screenshot:
```

If you can, do the **same sign three times** and note all three. That tells me far
more than a single attempt.

---

## 11. Quick checklist

```
[ ] Checksum of the APK verified on the Mac
[ ] adb devices -l shows the phone as "device"
[ ] Installs with "Success"
[ ] Installed APK hash matches
[ ] Opens to the camera screen
[ ] Camera permission granted
[ ] Preview appears
[ ] Footer reads "9-model ensemble"
[ ] Glossary shows 100 of 100, search filters
[ ] Start sign begins recording
[ ] Finish sign produces a result card
[ ] Result shows sign, score, timing
[ ] Four alternatives appear
[ ] Recognise again starts fresh
[ ] Clear empties the result
[ ] A second sign does not reuse the first result
[ ] No-hand recording is rejected with a message
[ ] History shows this session's recognitions
[ ] Settings: all six controls respond
[ ] Framing guide toggle visibly changes the camera
[ ] Backgrounding and returning works
[ ] Rotation does not crash the app
```

---

## 12. Honest limits

- **No accuracy figure has been measured on a phone.** Every project number
  (85.45% corpus top-1) is a laptop measurement on the ASL Citizen test split. Your
  session is the first real on-device test. Whatever you get is the real number.
- **The app has been tested on an emulator, not a phone.** Install, launch, all four
  screens, every control, backgrounding, rotation and zero crashes are verified
  there. Live hand recognition is not.
- **The model knows 100 signs** out of a much larger vocabulary. Anything outside
  those 100 maps onto the nearest one it knows.
- **Isolated signs only.** Not continuous signing, not sentences.
- **The latency target is not met, and the numbers you will see are not the rubric's
  number.** The brief asks for ≤50 ms. The only measured per-frame detector figure
  is a *desktop* 15.3 ms p50 / 23.0 ms p95; the full Python clip-to-result path is
  about 4.65 seconds on a laptop. Neither is an on-device measurement. The
  "Classifier took N ms" on your screen times the classifier only, not the whole
  pipeline.
