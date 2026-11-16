# How to test EchoGesture on your phone

This is written for you, not for developers. No Android Studio, no terminal
needed for the phone part.

---

## 1. Find the app on your Mac

Open Terminal and paste this — it copies the APK to your Desktop and reveals it:

```bash
cp ~/AppData/EchoGesture/submission/TEST_APP_NOW/EchoGesture-Test-Ready.apk ~/Desktop/ && open -R ~/Desktop/EchoGesture-Test-Ready.apk
```

If the file is not there yet, it is because the delivery folder has not been
pulled to your working copy. The APK is committed on
`release/samsung-final-app-test`:

```bash
git -C ~/AppData/EchoGesture fetch origin release/samsung-final-app-test && git -C ~/AppData/EchoGesture show origin/release/samsung-final-app-test:submission/TEST_APP_NOW/EchoGesture-Test-Ready.apk > ~/Desktop/EchoGesture-Test-Ready.apk
```

The file on your Desktop is `EchoGesture-Test-Ready.apk`. That is the app.

Its fingerprint — check it if you want to be certain you have the right build:

```
6653160ea3321c35ef19b45da3f69801557e3d598739ff3f4cb526171a93d37c
```

---

## 2. Get it onto your phone

Pick whichever is easiest:

**Option A — USB cable (most reliable)**
1. Plug the phone into your Mac.
2. On the phone, pull down the notification shade, tap the USB notification,
   and choose **File transfer / Android Auto** (not "Charging only").
3. The phone appears in Finder under **Locations**.
4. Open it → **Downloads** → drag `EchoGesture-Test-Ready.apk` in.

**Option B — Google Drive / any cloud**
Upload the APK from your Mac, open Drive on the phone, download it.

**Option C — email it to yourself**
Some mail apps block `.apk` attachments. If yours does, use A or B.

---

## 3. Install it

1. On the phone, open **Files** (or **My Files**) → **Downloads**.
2. Tap `EchoGesture-Test-Ready.apk`.
3. Android will say *"For your security, your phone is not allowed to install
   unknown apps from this source."*
   Tap **Settings** on that prompt → turn on **Allow from this source** → tap
   **Back**.
4. Tap **Install**. If Play Protect warns you, tap **Install anyway** — this app
   is not from the Play Store, so that warning is expected.
5. Tap **Open**.

> **This is a debug build.** It installs alongside nothing and will not
> auto-update. That is intentional — it is a test artifact, not a store release.

---

## 4. Allow the camera

On first launch the app asks:

> *"Allow EchoGesture to take pictures and record video?"*

Tap **While using the app**.

This is the **only** permission the app has. It has no internet permission at
all — you can confirm this later in Settings → Apps → EchoGesture → Permissions,
where Internet will simply not be listed.

If you tap **Don't allow** by accident, the app shows a "Camera permission
needed" card with an **Allow camera access** button. Tap it to try again.

---

## 5. Wait for the models

The app loads nine neural network files (~85 MB total). On a cold start this
takes a few seconds. You'll see **"Preparing models"** and then
**"Searching for hands"**.

When the big blue **Start sign** button becomes tappable, it is ready.

Check the bottom of the screen. It should read:

> **9-model ensemble · FP16 TFLite · SHA-256 995cc8d0b6f6…**

If it says anything other than **9-model**, tell me — that would be a real bug.

---

## 6. Learn what the app can recognise

Tap **Glossary** in the bottom bar.

- There are **100 signs**. The pill at top right should say **100 of 100 signs**.
- Type in the search box, e.g. `apple`, to filter. The pill updates to **1 of 100**.
- Tap any sign to open a small sheet with its name and **Speak** / **Copy** buttons.
- **Speak** reads the label aloud. **Copy** puts it on the clipboard.

Read the note in that sheet: these are the model's *output class names*. They
tell you which sign was recognised — they are not a dictionary entry or a
description of what the sign means.

**Pick 10 signs you want to try** from this list and note them down. Good ones
to start with (they are in the vocabulary):

```
APPLE   BABY2   BANANA   BASKETBALL1   BOOK   CANDY   CHAIR
COFFEE   COMPUTER   COOK   DOG1   DRINK   FISH   FLOWER
HAT   HOUSE   KEY   LIGHT   MILK   MONEY   PHONE   PIZZA
```

---

## 7. Recognise a sign — the important bit

1. Tap **Camera** to go back.
2. Hold the phone so your **hands are fully in the guide box** and your upper
   body is visible. Good even light; avoid pointing at a bright window.
3. Tap **Start sign**.
   The button turns into **Finish sign** and a red "Recording" banner appears.
4. **Perform one complete sign**, from start to finish. One or two hands. Any
   comfortable speed.
5. Tap **Finish sign** as soon as the sign ends.
6. Wait about a second. A result card appears.

You'll see:

- the recognised sign in large text
- a **model score** percentage
- **"Classifier took N ms on this device"**
- how many frames were captured over how long
- four **alternatives**
- **Copy**, **Speak**, **Recognise again**, **Clear**

### What the score means

The card says it plainly, and so will I: the model score ranks the 100 known
signs. It is **not** a calibrated probability that the reading is correct. A
97% score does not mean 97% of the time it is right.

**Record every attempt, including the wrong ones.** The failures are more
useful to me than the successes.

---

## 8. The rules that matter when testing

- **One sign per recording.** The app analyses exactly the interval between
  Start and Finish. Do not include the pause before or after.
- **Keep hands in view the whole time.** If your hands leave the frame, the app
  will say "No hands in view" rather than guess. That is deliberate — it is not
  a bug.
- **Two seconds minimum.** Very short recordings are rejected on purpose.
- **Wait for the result.** Tapping Start again before the result appears does
  nothing; that is double-tap protection.

---

## 9. History and Settings

**History** keeps your recent recognitions *in memory for this session only* —
closing the app clears it, unless you turn on the setting below. That default is
deliberate.

**Settings** has six controls. All of them work; test them:

| Control | What should happen |
| --- | --- |
| Preferred camera | Switching to Front/Back changes which camera opens |
| Framing guide | Turning off removes the hand-placement box from the preview |
| Result text size | Small / Medium / Large changes the result sign size |
| Haptic feedback | Turning off stops the vibration on capture |
| Speak results | Turning on reads each recognised sign aloud |
| Keep history on this device | Turning on makes History survive an app restart |

If you toggle "Framing guide" off and go back to Camera, the overlay must be
gone. If you toggle it on, it must return.

---

## 10. What I want you to report

Use this template. Copy it, fill in one per problem:

```
Device:
Android version:
APK hash (see below):
Screen:
Action performed:
Expected:
Actual:
Screenshot/video:
Can reproduce (always / sometimes / once):
Severity (blocker / major / minor):
```

**APK hash** — on your Mac:

```bash
shasum -a 256 ~/AppData/EchoGesture/submission/TEST_APP_NOW/EchoGesture-Test-Ready.apk
```

If something crashes, this gets me the log (phone connected by USB):

```bash
adb logcat -d > ~/Desktop/echogesture-crash.txt
```

No USB? Use Settings → About phone → tap Build number 7 times → Developer
options → **Take bug report**, and send me the file.

---

## 11. The quick checklist

Mark each one:

```
[ ] Installs without error
[ ] Opens to the camera screen (no blank screen)
[ ] Camera permission asked, and denying it shows a clear message
[ ] After granting, live preview appears
[ ] Footer reads "9-model ensemble"
[ ] Glossary shows 100 of 100 signs
[ ] Glossary search filters the list
[ ] Glossary entry opens, Speak works, Copy works
[ ] Start sign begins a recording (button changes to Finish sign)
[ ] Finish sign produces a result card
[ ] Result shows a sign name, score, and timing
[ ] Alternatives list appears
[ ] "Recognise again" starts a fresh recording
[ ] "Clear" empties the result
[ ] A second sign does not reuse the first result
[ ] No-hands recording is rejected with a message, not a guess
[ ] History shows the recognitions from this session
[ ] Settings: all six controls respond
[ ] Framing guide toggle visibly changes the camera screen
[ ] Back button returns to Camera from other tabs
[ ] App survives switching to another app and back
[ ] Rotation does not crash the app
```

Note anything that fails, even if it seems minor. A dead button I think works is
worse than a crash I know about.

---

## 12. Honest limits of what you are testing

Read this before you judge the app:

- **No accuracy number here has been measured on a phone.** Every accuracy
  figure in the project (77.02% corpus top-1) comes from a laptop running the
  Python evaluator on the ASL Citizen test split. Your session is the first
  real on-device recognition test. Whatever you get is the real number.
- **The app has been tested on an emulator, not a phone.** Everything except
  live-hand recognition is verified there — install, launch, all four screens,
  every control, backgrounding, rotation, and zero crashes. What has *not* been
  tested is the one thing that needs a real camera and a real hand.
- **The model knows 100 signs** out of a much larger vocabulary. Anything
  outside those 100 will be mapped onto the nearest one it knows.
- **Isolated signs only.** Not continuous signing, not sentences.
- **Lighting and background matter.** This is a research prototype, not a
  product.
- **The latency target is not met.** The brief asks for ≤50 ms; the measured
  clip-to-JSON median on a laptop is ~236 ms. The on-device classifier number on
  your screen is only the classifier, not the whole pipeline.

None of that is a reason not to test it. It is the reason the test matters.
