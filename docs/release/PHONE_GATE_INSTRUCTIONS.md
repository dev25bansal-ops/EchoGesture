# Live phone recognition gate — what you must run

No physical Android device is attached to this machine (`adb devices -l` shows
only `emulator-5554`). Nothing in this file is a measurement. It is the exact
sequence that produces one.

This build uses **Start sign → perform one complete sign → Finish sign**, not
a tap after signing. Press Start just before you begin and Finish as soon as the
sign ends.

**APK:** `dist/echogesture-v4.2.apk`
SHA-256 `5490e02908159e24dac1b45dca7b21afbbe77909fb62ceab4348828e8d939dfc`
Commit `71fca9b`, branch `fix/true-video-recognition-v4`.

## How the two capture modes work

**Record (default, the button you will use).** Press **Start sign**; the button
becomes **Finish sign** and the pill shows elapsed time. Perform one complete
supported sign, then press Finish. The app reads exactly that interval: frames
are collected only between the two presses, so nothing from before Start or
after Finish can enter the sequence.

After the recording is taken, the app trims idle from the two ends before the
model sees it — this is why a second of hesitation either side does not ruin the
reading — and then samples 32 frames evenly across what remains, resamples to 24
temporal steps, and runs the four-member ensemble.

Past ~5 s of recording the app asks you to finish. Most signs are much shorter:
across 72 authorised clips the median is 2.5 s, the 90th percentile 4.3 s and
the longest 8.7 s. It does not truncate for you; Finish stays yours to press.

**Quick scan (secondary).** The rolling-window path is still present in the code
behind the same frozen model, kept as a comparison baseline. It classifies a
fixed trailing window instead of a bracketed interval. It is not wired to the
default button in this build.

See `PHONE_BUILD_IDENTITY.md` for the full build list.

---

## 1. Connect

Phone: Settings → About phone → tap **Build number** 7× → Developer options →
**USB debugging** on. Connect by USB, accept the RSA prompt.

```bash
adb devices -l
```

Requires one line ending `device` with a real model name (not `emulator-`).
Record it — it goes in the submission:

```bash
adb shell getprop ro.product.model
adb shell getprop ro.build.version.release
adb shell getprop ro.product.cpu.abi
```

## 2. Install

```bash
adb install -r dist/echogesture-v4.1.apk
```

If it fails with `INSTALL_FAILED_UPDATE_INCOMPATIBLE`, a differently-signed
build is installed:

```bash
adb uninstall org.echogesture.finalapp
```

Then reinstall. Grant camera on first launch, or:

```bash
adb shell pm grant org.echogesture.finalapp android.permission.CAMERA
adb shell am start -n org.echogesture.finalapp/.MainActivity
```

## 3. Enable diagnostics

In the app: **Settings → Capture diagnostics → on**. This deletes any previous
log and starts fresh. It writes no images, no landmarks and no identifiers, and
the app has no internet permission.

## 4. Pick 10 signs

Draw from the 100 in `models/landmark_labels.json`. Use the exact spelling.
Include at least one known-hard class so the result is informative:
**BACKPACK1, KNIGHT1, MYSELF, SPECIAL1, SQUEEZE** are the ones that failed on
their validation clip but pass on training clips.

| # | Label | Demonstration source |
| --: | --- | --- |
| 1 | | |
| 2 | | |
| 3 | | |
| 4 | | |
| 5 | | |
| 6 | | |
| 7 | | |
| 8 | | |
| 9 | | |
| 10 | | |

Perform each from an authoritative demonstration. Improvising a sign from a
written gloss measures nothing.

## 5. Five attempts per sign, varying conditions

| Attempt | Background | Lighting | Distance |
| --- | --- | --- | --- |
| 1 | plain wall | daylight | ~60 cm |
| 2 | cluttered room | indoor lamp | ~60 cm |
| 3 | plain wall | indoor lamp | ~120 cm |
| 4 | cluttered room | daylight | ~120 cm |
| 5 | your usual setup | your usual setup | your usual setup |

**Keep every failed capture.** A capture that reports "not clear enough to read"
is a data point, not something to retry away. The retry rate is part of what is
being measured.

Record-mode refusals, and what each diagnoses. Record which one you saw — they
are not interchangeable:

| Message | Meaning |
| --- | --- |
| "That recording was too short to read." | Finish was pressed within 400 ms. Nothing usable was captured. |
| "The camera did not send any frames during that recording." | No analysed frame arrived. A device or camera fault. |
| "The camera stopped during that sign." | A gap longer than 700 ms mid-recording; the sequence is broken. |
| "Hands were visible for too little of that recording." | Fewer than 10% of frames carried a hand. Re-aim the camera. |
| "No hands in view" | The capture contained no hand at all. |

Do not retry a failure silently — note it, count it, and keep going. The
rejection rate is part of what the run measures.

Keep an annotations file, one line per capture, in capture order:

```
1 EAT1 plain daylight 60
2 EAT1 cluttered lamp 120
```

## 6. Retrieve and analyse

```bash
adb shell run-as org.echogesture.finalapp cat files/diagnostics.txt > diagnostics.txt
python3 evaluation/device/accuracy_report.py diagnostics.txt annotations.txt
```

The script refuses to run if the two files disagree on capture count rather than
guessing the pairing. It prints top-1, top-5, rejection rate, accuracy at
accepted coverage, a confusion matrix and per-sign recall.

For per-class and margin detail on a corpus, use
`evaluation/device/per_class_check.py --split <csv> --clips <dir>`.

## 7. What your result will decide

The off-phone evidence is already complete and points one way:

- Android and Python agree on the identical file (EAT1 both).
- 97 of 100 genuine training clips across the previously-failing classes are
  correct.
- No class fails on every clip.
- Rejection lifts 73.6% micro top-1 to **94.1% accuracy among the 64.2% it
  accepts**.

So the decision table is:

| Phone result | Verdict | Next action |
| --- | --- | --- |
| High coverage, accurate | **FROZEN MODEL SUFFICIENT** | Ship it |
| Low `hand_frames`/`frames` | Capture problem | Camera/framing/lighting work, not the model |
| High coverage, consistently wrong signs | **PHONE-DOMAIN FINE-TUNING JUSTIFIED** | Phased 4 adaptation plan, signers held out |
| Only some signers fail | Signer-domain shift | Same as above, but per-signer reporting first |

I cannot choose between these without your data. Guessing would be the one
thing that makes this submission dishonest.

## 8. Do not

- Do not report a paused sign image on a screen as a live gesture.
- Do not claim phone accuracy from emulator results.
- Do not retry a failed capture silently.
- Do not upload video containing bystanders without consent.
- Keep `diagnostics.txt` and any recordings out of Git.